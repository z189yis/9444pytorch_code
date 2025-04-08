#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#

import sys
import time
import os
import math
import csv
import argparse
import numpy as np
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import *
from ferplus import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

emotion_table = {'neutral'  : 0, 
                 'happiness': 1, 
                 'surprise' : 2, 
                 'sadness'  : 3, 
                 'anger'    : 4, 
                 'disgust'  : 5, 
                 'fear'     : 6, 
                 'contempt' : 7}

# 创建与情绪表对应的情绪名称列表，用于绘图和输出
emotion_names = [name for name, _ in sorted(emotion_table.items(), key=lambda x: x[1])]

# List of folders for training, validation and test.
train_folders = ['FER2013Train']
valid_folders = ['FER2013Valid'] 
test_folders  = ['FER2013Test']

def cost_func(training_mode, logits, target):
    '''
    我们在大多数模式下使用交叉熵损失，除了multi-label模式
    注意：这里的实现与CNTK相匹配，直接接受logits而不是概率分布
    '''
    # 先计算softmax得到概率分布
    prediction = F.softmax(logits, dim=1)
    
    if training_mode == 'majority' or training_mode == 'probability' or training_mode == 'crossentropy': 
        # 交叉熵损失 - 与CNTK实现匹配
        return -torch.sum(target * torch.log(prediction + 1e-7), dim=1).mean()
    elif training_mode == 'multi_target':
        # Multi-target 自定义损失
        return -torch.log(torch.max(target * prediction, dim=1)[0] + 1e-7).mean()

def classification_error(logits, target):
    '''
    计算分类错误率，模仿CNTK的classification_error函数
    返回错误率，而不是准确率
    '''
    _, predicted = torch.max(logits.data, 1)
    _, targets = torch.max(target.data, 1)
    incorrect = (predicted != targets).float().mean()  # 错误率
    return incorrect

def plot_training_curves(train_losses, train_accs, val_accs, test_accs, output_folder):
    """绘制训练过程的曲线图"""
    epochs = range(1, len(train_accs) + 1)
    
    plt.figure(figsize=(12, 10))
    
    # 绘制训练损失
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_losses, 'b-', label='训练损失')
    plt.title('训练损失曲线')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # 绘制准确率
    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_accs, 'b-', label='训练准确率')
    plt.plot(epochs, val_accs, 'r-', label='验证准确率')
    
    # 如果有测试数据，绘制测试准确率
    if test_accs:
        # 创建与epochs等长的空列表，只在进行测试的epoch处填入值
        test_epochs = []
        filtered_test_accs = []
        for i, acc in enumerate(test_accs):
            if acc is not None:
                test_epochs.append(i+1)
                filtered_test_accs.append(acc)
        if filtered_test_accs:
            plt.plot(test_epochs, filtered_test_accs, 'g-', marker='o', label='测试准确率')
    
    plt.title('模型准确率曲线')
    plt.xlabel('Epochs')
    plt.ylabel('准确率 (%)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(os.path.join(output_folder, 'training_curves.png'))
    print(f"训练曲线图已保存至 {os.path.join(output_folder, 'training_curves.png')}")
    plt.close()

def plot_confusion_matrix(cm, class_names, output_path, title='混淆矩阵'):
    """
    绘制混淆矩阵图
    """
    plt.figure(figsize=(10, 8))
    
    # 创建热力图
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    # 设置x和y轴的刻度和标签
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # 在图中标注数字
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    
    # 保存图表
    plt.savefig(output_path)
    print(f"混淆矩阵已保存至 {output_path}")
    plt.close()

def calculate_metrics(all_preds, all_targets, num_classes):
    """
    计算各种评估指标
    """
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_targets, all_preds)
    
    # 计算分类报告
    report = classification_report(all_targets, all_preds, target_names=emotion_names, digits=4)
    
    # 计算准确率
    accuracy = accuracy_score(all_targets, all_preds)
    
    return cm, report, accuracy

def main(base_folder, training_mode='majority', model_name='VGG13', max_epochs=100, resume=False):
    # 确保使用GPU训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not available, using CPU instead.")
    
    # 创建需要的文件夹
    output_model_path = os.path.join(base_folder, 'models')
    output_model_folder = os.path.join(output_model_path, model_name + '_' + training_mode)
    if not os.path.exists(output_model_folder):
        os.makedirs(output_model_folder)

    # 创建日志文件 - 如果恢复训练则追加，否则覆盖
    log_mode = 'a' if resume else 'w'
    logging.basicConfig(filename=os.path.join(output_model_folder, "train.log"), 
                        filemode=log_mode, 
                        level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    
    logging.info(f"Starting with training mode {training_mode} using {model_name} model and max epochs {max_epochs}.")

    # 创建模型
    num_classes = len(emotion_table)
    model = build_model(num_classes, model_name)
    model.to(device)
    
    # 读取FER+数据集
    print("Loading data...")
    train_params = FERPlusParameters(num_classes, model.input_height, model.input_width, training_mode, False, True)
    test_and_val_params = FERPlusParameters(num_classes, model.input_height, model.input_width, "majority", True, False)

    train_data_reader = FERPlusReader.create(base_folder, train_folders, "label.csv", train_params)
    val_data_reader = FERPlusReader.create(base_folder, valid_folders, "label.csv", test_and_val_params)
    test_data_reader = FERPlusReader.create(base_folder, test_folders, "label.csv", test_and_val_params)
    
    # 打印数据摘要
    display_summary(train_data_reader, val_data_reader, test_data_reader)
    
    minibatch_size = 32
    epoch_size = train_data_reader.size()
    
    # 创建与CNTK一样的学习率和动量时间常数
    lr_per_minibatch = [model.learning_rate]*20 + [model.learning_rate / 2.0]*20 + [model.learning_rate / 10.0]*60
    mm_time_constant = -minibatch_size/np.log(0.9)
    
    # 创建优化器 - 但我们会在每个minibatch更新学习率和动量
    optimizer = optim.SGD(model.parameters(), lr=model.learning_rate, momentum=0.9)
    
    # 初始化训练状态
    start_epoch = 0
    max_val_accuracy = 0.0
    final_test_accuracy = 0.0
    best_test_accuracy = 0.0
    best_epoch = 0
    
    # 检查点文件路径
    checkpoint_path = os.path.join(output_model_folder, "checkpoint.pth")
    
    # 用于记录训练过程的列表
    train_losses = []
    train_accs = []
    val_accs = []
    test_accs = []
    
    # 如果需要恢复训练且检查点存在
    if resume and os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        max_val_accuracy = checkpoint['max_val_accuracy']
        final_test_accuracy = checkpoint['final_test_accuracy']
        best_test_accuracy = checkpoint['best_test_accuracy']
        best_epoch = checkpoint['best_epoch']
        
        if 'train_losses' in checkpoint:
            train_losses = checkpoint['train_losses']
            train_accs = checkpoint['train_accs']
            val_accs = checkpoint['val_accs']
            test_accs = checkpoint['test_accs']
        
        print(f"Resuming from epoch {start_epoch}, best validation accuracy: {max_val_accuracy*100:.2f}%")
    else:
        print("Starting fresh training...")

    print("Start training...")
    
    try:
        for epoch in range(start_epoch, max_epochs):
            train_data_reader.reset()
            val_data_reader.reset()
            test_data_reader.reset()
            
            # 训练阶段
            model.train()
            start_time = time.time()
            training_loss = 0
            training_error = 0  # 使用错误率而不是准确率，与CNTK一致
            train_samples = 0
            
            # 创建进度条
            progress = tqdm(total=train_data_reader.size()//minibatch_size, 
                          desc=f"Epoch {epoch}/{max_epochs-1}")
            
            batch_idx = 0
            while train_data_reader.has_more():
                # 按照CNTK计算当前minibatch对应的学习率和动量
                global_minibatch_idx = epoch * (epoch_size // minibatch_size) + batch_idx
                
                # 设置当前学习率
                current_lr_idx = min(global_minibatch_idx, len(lr_per_minibatch)-1)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_per_minibatch[current_lr_idx]
                
                # 设置当前动量 - 使用时间常数计算
                current_momentum = np.exp(-minibatch_size/mm_time_constant)
                for param_group in optimizer.param_groups:
                    param_group['momentum'] = current_momentum
                
                images, labels, current_batch_size = train_data_reader.next_minibatch(minibatch_size)
                
                # 转换为PyTorch张量
                images = torch.from_numpy(images).to(device)
                labels = torch.from_numpy(labels).to(device)
                
                # 梯度清零
                optimizer.zero_grad()
                
                # 前向传播
                logits = model(images)
                
                # 计算损失
                loss = cost_func(training_mode, logits, labels)
                
                # 计算错误率 - 与CNTK一致
                error = classification_error(logits, labels)
                
                # 反向传播与优化
                loss.backward()
                optimizer.step()
                
                # 统计
                training_loss += loss.item() * current_batch_size
                training_error += error.item() * current_batch_size
                train_samples += current_batch_size
                
                # 更新进度条
                progress.update(1)
                batch_idx += 1
            
            progress.close()
            
            # 计算平均损失和准确率
            training_loss /= train_samples
            training_error /= train_samples  # 错误率
            training_accuracy = 1.0 - training_error  # 转换为准确率（与CNTK一致）
            
            # 记录训练损失和准确率
            train_losses.append(training_loss)
            train_accs.append(training_accuracy * 100)
            
            # 验证阶段
            model.eval()
            val_error = 0
            val_samples = 0
            
            with torch.no_grad():
                while val_data_reader.has_more():
                    images, labels, current_batch_size = val_data_reader.next_minibatch(minibatch_size)
                    
                    # 转换为PyTorch张量
                    images = torch.from_numpy(images).to(device)
                    labels = torch.from_numpy(labels).to(device)
                    
                    # 前向传播
                    logits = model(images)
                    
                    # 计算错误率
                    error = classification_error(logits, labels)
                    val_error += error.item() * current_batch_size
                    val_samples += current_batch_size
                
            val_error /= val_samples  # 错误率
            val_accuracy = 1.0 - val_error  # 转换为准确率（与CNTK一致）
            
            # 记录验证准确率
            val_accs.append(val_accuracy * 100)
            
            # 如果验证准确率提高，计算测试准确率
            test_run = False
            if val_accuracy > max_val_accuracy:
                best_epoch = epoch
                max_val_accuracy = val_accuracy
                
                # 保存最佳模型
                torch.save(model.state_dict(), os.path.join(output_model_folder, f"best_model.pth"))
                
                test_run = True
                test_error = 0
                test_samples = 0
                
                with torch.no_grad():
                    while test_data_reader.has_more():
                        images, labels, current_batch_size = test_data_reader.next_minibatch(minibatch_size)
                        
                        # 转换为PyTorch张量
                        images = torch.from_numpy(images).to(device)
                        labels = torch.from_numpy(labels).to(device)
                        
                        # 前向传播
                        logits = model(images)
                        
                        # 计算错误率
                        error = classification_error(logits, labels)
                        test_error += error.item() * current_batch_size
                        test_samples += current_batch_size
                
                test_error /= test_samples  # 错误率
                test_accuracy = 1.0 - test_error  # 转换为准确率（与CNTK一致）
                
                final_test_accuracy = test_accuracy
                
                if final_test_accuracy > best_test_accuracy:
                    best_test_accuracy = final_test_accuracy
                
                # 记录测试准确率
                test_accs.append(test_accuracy * 100)
            else:
                # 如果没有进行测试，记录None
                test_accs.append(None)
            
            logging.info("Epoch {}: took {:.3f}s".format(epoch, time.time() - start_time))
            logging.info("  training loss:\t{:e}".format(training_loss))
            logging.info("  training accuracy:\t\t{:.2f} %".format(training_accuracy * 100))
            logging.info("  validation accuracy:\t\t{:.2f} %".format(val_accuracy * 100))
            if test_run:
                logging.info("  test accuracy:\t\t{:.2f} %".format(test_accuracy * 100))
            
            # 输出到控制台
            print("Epoch {}: took {:.3f}s".format(epoch, time.time() - start_time))
            print("  training loss:\t{:e}".format(training_loss))
            print("  training accuracy:\t\t{:.2f} %".format(training_accuracy * 100))
            print("  validation accuracy:\t\t{:.2f} %".format(val_accuracy * 100))
            if test_run:
                print("  test accuracy:\t\t{:.2f} %".format(test_accuracy * 100))
            
            # 每个epoch保存检查点以便恢复训练
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'max_val_accuracy': max_val_accuracy,
                'final_test_accuracy': final_test_accuracy,
                'best_test_accuracy': best_test_accuracy,
                'best_epoch': best_epoch,
                'train_losses': train_losses,
                'train_accs': train_accs,
                'val_accs': val_accs,
                'test_accs': test_accs
            }, checkpoint_path)
            
            # 另外每10个epoch保存一个带编号的检查点并绘制图表
            if epoch % 10 == 0 and epoch > 0:
                torch.save(model.state_dict(), os.path.join(output_model_folder, f"model_epoch_{epoch}.pth"))
                plot_training_curves(train_losses, train_accs, val_accs, test_accs, output_model_folder)
    
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving checkpoint...")
        # 保存检查点以便恢复训练
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'max_val_accuracy': max_val_accuracy,
            'final_test_accuracy': final_test_accuracy,
            'best_test_accuracy': best_test_accuracy,
            'best_epoch': best_epoch,
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'test_accs': test_accs
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}. Use --resume flag to continue training.")
        
        # 即使中断训练也绘制曲线图
        plot_training_curves(train_losses, train_accs, val_accs, test_accs, output_model_folder)
        return
    
    # 训练结束，保存最终模型
    torch.save(model.state_dict(), os.path.join(output_model_folder, "final_model.pth"))
    
    print("")
    print("Best validation accuracy:\t\t{:.2f} %, epoch {}".format(max_val_accuracy * 100, best_epoch))
    print("Test accuracy corresponding to best validation:\t\t{:.2f} %".format(final_test_accuracy * 100))
    print("Best test accuracy:\t\t{:.2f} %".format(best_test_accuracy * 100))
    
    # 绘制最终训练曲线图
    plot_training_curves(train_losses, train_accs, val_accs, test_accs, output_model_folder)
    
    # 计算并绘制混淆矩阵
    all_preds = []
    all_targets = []
    test_data_reader.reset()
    with torch.no_grad():
        while test_data_reader.has_more():
            images, labels, current_batch_size = test_data_reader.next_minibatch(minibatch_size)
            
            # 转换为PyTorch张量
            images = torch.from_numpy(images).to(device)
            labels = torch.from_numpy(labels).to(device)
            
            outputs = model(images)
            
            # 记录预测和真实标签
            _, predicted = torch.max(outputs.data, 1)
            _, targets = torch.max(labels.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    cm, report, accuracy = calculate_metrics(all_preds, all_targets, num_classes)
    plot_confusion_matrix(cm, emotion_names, os.path.join(output_model_folder, 'confusion_matrix.png'))
    print("Classification Report:\n", report)
    print("Overall Test Accuracy: {:.2f} %".format(accuracy * 100))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", 
                        "--base_folder", 
                        type=str, 
                        help="Base folder containing the training, validation and testing data.", 
                        required=True)
    parser.add_argument("-m", 
                        "--training_mode", 
                        type=str,
                        default='majority',
                        help="Specify the training mode: majority, probability, crossentropy or multi_target.")
    parser.add_argument("--model_name",
                        type=str,
                        default='VGG13',
                        help="Name of the model architecture to use.")
    parser.add_argument("--max_epochs",
                        type=int,
                        default=100,
                        help="Maximum number of training epochs.")
    parser.add_argument("--resume", 
                        action="store_true",
                        help="Resume training from checkpoint if available.")

    args = parser.parse_args()
    main(args.base_folder, args.training_mode, args.model_name, args.max_epochs, args.resume)