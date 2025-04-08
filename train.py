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

from models import *
from ferplus import *

import torch
import torch.nn as nn
import torch.optim as optim
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

# List of folders for training, validation and test.
train_folders = ['FER2013Train']
valid_folders = ['FER2013Valid'] 
test_folders  = ['FER2013Test']

def cost_func(training_mode, prediction, target):
    '''
    We use cross entropy in most mode, except for the multi-label mode, which require treating
    multiple labels exactly the same.
    '''
    if training_mode == 'majority' or training_mode == 'probability' or training_mode == 'crossentropy': 
        # Cross Entropy
        return -torch.sum(target * torch.log(prediction + 1e-7), dim=1).mean()
    elif training_mode == 'multi_target':
        # Multi-target custom loss
        return -torch.log(torch.max(target * prediction, dim=1)[0] + 1e-7).mean()
    
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

    # 创建模型
    num_classes = len(emotion_table)
    model = build_model(num_classes, model_name)
    model.to(device)
    
    # 读取FER+数据集
    print("Loading data...")
    train_params = FERPlusParameters(num_classes, model.input_height, model.input_width, training_mode, False)
    test_and_val_params = FERPlusParameters(num_classes, model.input_height, model.input_width, "majority", True)

    train_data_reader = FERPlusReader.create(base_folder, train_folders, "label.csv", train_params)
    val_data_reader = FERPlusReader.create(base_folder, valid_folders, "label.csv", test_and_val_params)
    test_data_reader = FERPlusReader.create(base_folder, test_folders, "label.csv", test_and_val_params)
    
    # 打印数据摘要
    display_summary(train_data_reader, val_data_reader, test_data_reader)
    
    minibatch_size = 32
    
    # 训练配置
    lr_schedule = [model.learning_rate] * 20 + [model.learning_rate / 2.0] * 20 + [model.learning_rate / 10.0]
    optimizer = optim.SGD(model.parameters(), lr=model.learning_rate, momentum=0.9)
    
    # 初始化训练状态
    start_epoch = 0
    max_val_accuracy = 0.0
    final_test_accuracy = 0.0
    best_test_accuracy = 0.0
    best_epoch = 0
    
    # 检查点文件路径
    checkpoint_path = os.path.join(output_model_folder, "checkpoint.pth")
    
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
        
        print(f"Resuming from epoch {start_epoch}, best validation accuracy: {max_val_accuracy*100:.2f}%")
    else:
        print("Starting fresh training...")

    print("Start training...")
    
    # 创建混合精度训练的scaler (适用于支持混合精度的GPU)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    try:
        for epoch in range(start_epoch, max_epochs):
            # 更新学习率
            if epoch < len(lr_schedule):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_schedule[epoch]
            
            train_data_reader.reset()
            val_data_reader.reset()
            test_data_reader.reset()
            
            # 训练阶段
            model.train()
            start_time = time.time()
            training_loss = 0
            training_accuracy = 0
            train_samples = 0
            
            # 创建进度条
            progress = tqdm(total=train_data_reader.size()//minibatch_size, 
                          desc=f"Epoch {epoch}/{max_epochs-1}")
            
            while train_data_reader.has_more():
                images, labels, current_batch_size = train_data_reader.next_minibatch(minibatch_size)
                
                # 转换为PyTorch张量
                images = torch.from_numpy(images).to(device)
                labels = torch.from_numpy(labels).to(device)
                
                # 梯度清零
                optimizer.zero_grad()
                
                if device.type == 'cuda':
                    # 使用混合精度训练
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        outputs_softmax = nn.functional.softmax(outputs, dim=1)
                        loss = cost_func(training_mode, outputs_softmax, labels)
                    
                    # 使用scaler进行反向传播和优化
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # CPU训练
                    outputs = model(images)
                    outputs_softmax = nn.functional.softmax(outputs, dim=1)
                    loss = cost_func(training_mode, outputs_softmax, labels)
                    loss.backward()
                    optimizer.step()
                
                # 统计
                training_loss += loss.item() * current_batch_size
                
                # 计算准确率
                _, predicted = torch.max(outputs.data, 1)
                _, targets = torch.max(labels.data, 1)
                correct = (predicted == targets).sum().item()
                training_accuracy += correct
                train_samples += current_batch_size
                
                # 更新进度条
                progress.update(1)
            
            progress.close()
            training_accuracy /= train_samples
            
            # 验证阶段
            model.eval()
            val_accuracy = 0
            val_samples = 0
            
            with torch.no_grad():
                while val_data_reader.has_more():
                    images, labels, current_batch_size = val_data_reader.next_minibatch(minibatch_size)
                    
                    # 转换为PyTorch张量
                    images = torch.from_numpy(images).to(device)
                    labels = torch.from_numpy(labels).to(device)
                    
                    outputs = model(images)
                    
                    # 计算准确率
                    _, predicted = torch.max(outputs.data, 1)
                    _, targets = torch.max(labels.data, 1)
                    correct = (predicted == targets).sum().item()
                    val_accuracy += correct
                    val_samples += current_batch_size
                
            val_accuracy /= val_samples
            
            # 如果验证准确率提高，计算测试准确率
            test_run = False
            if val_accuracy > max_val_accuracy:
                best_epoch = epoch
                max_val_accuracy = val_accuracy

                # 保存最佳模型
                torch.save(model.state_dict(), os.path.join(output_model_folder, f"best_model.pth"))

                test_run = True
                test_accuracy = 0
                test_samples = 0
                
                with torch.no_grad():
                    while test_data_reader.has_more():
                        images, labels, current_batch_size = test_data_reader.next_minibatch(minibatch_size)
                        
                        # 转换为PyTorch张量
                        images = torch.from_numpy(images).to(device)
                        labels = torch.from_numpy(labels).to(device)
                        
                        outputs = model(images)
                        
                        # 计算准确率
                        _, predicted = torch.max(outputs.data, 1)
                        _, targets = torch.max(labels.data, 1)
                        correct = (predicted == targets).sum().item()
                        test_accuracy += correct
                        test_samples += current_batch_size
                
                test_accuracy /= test_samples
                final_test_accuracy = test_accuracy
                
                if final_test_accuracy > best_test_accuracy: 
                    best_test_accuracy = final_test_accuracy
    
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
                'best_epoch': best_epoch
            }, checkpoint_path)
            
            # 另外每10个epoch保存一个带编号的检查点
            if epoch % 10 == 0 and epoch > 0:
                torch.save(model.state_dict(), os.path.join(output_model_folder, f"model_epoch_{epoch}.pth"))
    
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
            'best_epoch': best_epoch
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}. Use --resume flag to continue training.")
        return
    
    # 训练结束，保存最终模型
    torch.save(model.state_dict(), os.path.join(output_model_folder, "final_model.pth"))
    
    print("")
    print("Best validation accuracy:\t\t{:.2f} %, epoch {}".format(max_val_accuracy * 100, best_epoch))
    print("Test accuracy corresponding to best validation:\t\t{:.2f} %".format(final_test_accuracy * 100))
    print("Best test accuracy:\t\t{:.2f} %".format(best_test_accuracy * 100))
    
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