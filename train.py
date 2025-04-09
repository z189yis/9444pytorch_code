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

# Create a list of emotion names corresponding to the emotion table for plotting and output
emotion_names = [name for name, _ in sorted(emotion_table.items(), key=lambda x: x[1])]

# List of folders for training, validation and test.
train_folders = ['FER2013Train']
valid_folders = ['FER2013Valid'] 
test_folders  = ['FER2013Test']

def cost_func(training_mode, logits, target):
    '''
    We use cross entropy in most modes, except for multi-label mode which requires
    treating multiple labels exactly the same.
    Note: This implementation matches CNTK by taking logits directly rather than probabilities.
    '''
    # Calculate softmax to get probability distribution
    prediction = F.softmax(logits, dim=1)
    
    if training_mode == 'majority' or training_mode == 'probability' or training_mode == 'crossentropy': 
        # Cross entropy loss - matching CNTK implementation
        return -torch.sum(target * torch.log(prediction + 1e-7), dim=1).mean()
    elif training_mode == 'multi_target':
        # Multi-target custom loss
        return -torch.log(torch.max(target * prediction, dim=1)[0] + 1e-7).mean()

def classification_error(logits, target):
    '''
    Calculate classification error rate, mimicking CNTK's classification_error function.
    Returns error rate, not accuracy.
    '''
    _, predicted = torch.max(logits.data, 1)
    _, targets = torch.max(target.data, 1)
    incorrect = (predicted != targets).float().mean()  # Error rate
    return incorrect

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, output_folder):
    """Plot training curves for loss and accuracy"""
    epochs = range(1, len(train_accs) + 1)
    
    plt.figure(figsize=(12, 10))
    
    # Plot training and validation loss
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # Plot training and validation accuracy
    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    # Save the chart
    plt.savefig(os.path.join(output_folder, 'training_curves.png'))
    print(f"Training curves saved to {os.path.join(output_folder, 'training_curves.png')}")
    plt.close()

def plot_confusion_matrix(cm, class_names, output_path, title='Confusion Matrix'):
    """
    Plot confusion matrix
    """
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    # Set x and y tick labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations in the cells
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save the chart
    plt.savefig(output_path)
    print(f"Confusion matrix saved to {output_path}")
    plt.close()

def calculate_metrics(all_preds, all_targets, num_classes):
    """
    Calculate various evaluation metrics
    """
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Calculate classification report
    report = classification_report(all_targets, all_preds, target_names=emotion_names, digits=4)
    
    # Calculate accuracy
    accuracy = accuracy_score(all_targets, all_preds)
    
    return cm, report, accuracy

def main(base_folder, training_mode='majority', model_name='VGG13', max_epochs=100, resume=False):
    # Ensure GPU training if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not available, using CPU instead.")
    
    # Create required folders
    output_model_path = os.path.join(base_folder, 'models')
    output_model_folder = os.path.join(output_model_path, model_name + '_' + training_mode)
    if not os.path.exists(output_model_folder):
        os.makedirs(output_model_folder)

    # Create log file - append if resuming, otherwise overwrite
    log_mode = 'a' if resume else 'w'
    logging.basicConfig(filename=os.path.join(output_model_folder, "train.log"), 
                        filemode=log_mode, 
                        level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    
    logging.info(f"Starting with training mode {training_mode} using {model_name} model and max epochs {max_epochs}.")

    # Create model
    num_classes = len(emotion_table)
    model = build_model(num_classes, model_name)
    model.to(device)
    
    # Load FER+ dataset
    print("Loading data...")
    train_params = FERPlusParameters(num_classes, model.input_height, model.input_width, training_mode, False, True)
    test_and_val_params = FERPlusParameters(num_classes, model.input_height, model.input_width, "majority", True, False)

    train_data_reader = FERPlusReader.create(base_folder, train_folders, "label.csv", train_params)
    val_data_reader = FERPlusReader.create(base_folder, valid_folders, "label.csv", test_and_val_params)
    test_data_reader = FERPlusReader.create(base_folder, test_folders, "label.csv", test_and_val_params)
    
    # Print data summary
    display_summary(train_data_reader, val_data_reader, test_data_reader)
    
    minibatch_size = 32
    epoch_size = train_data_reader.size()
    
    # Create learning rate and momentum time constant similar to CNTK
    lr_per_minibatch = [model.learning_rate]*20 + [model.learning_rate / 2.0]*20 + [model.learning_rate / 10.0]*60
    mm_time_constant = -minibatch_size/np.log(0.9)
    
    # Create optimizer - we'll update learning rate and momentum for each minibatch
    optimizer = optim.SGD(model.parameters(), lr=model.learning_rate, momentum=0.9)
    
    # Initialize training state
    start_epoch = 0
    max_val_accuracy = 0.0
    best_model_epoch = 0
    
    # Checkpoint file path
    checkpoint_path = os.path.join(output_model_folder, "checkpoint.pth")
    
    # Lists to record training process
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # If resuming training and checkpoint exists
    if resume and os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        max_val_accuracy = checkpoint['max_val_accuracy']
        best_model_epoch = checkpoint['best_model_epoch']
        
        if 'train_losses' in checkpoint:
            train_losses = checkpoint['train_losses']
            val_losses = checkpoint['val_losses']
            train_accs = checkpoint['train_accs']
            val_accs = checkpoint['val_accs']
        
        print(f"Resuming from epoch {start_epoch}, best validation accuracy: {max_val_accuracy*100:.2f}%")
    else:
        print("Starting fresh training...")

    print("Start training...")
    
    try:
        for epoch in range(start_epoch, max_epochs):
            train_data_reader.reset()
            val_data_reader.reset()
            
            # Training phase
            model.train()
            start_time = time.time()
            training_loss = 0
            training_error = 0  # Use error rate instead of accuracy, consistent with CNTK
            train_samples = 0
            
            # Create progress bar
            progress = tqdm(total=train_data_reader.size()//minibatch_size, 
                          desc=f"Epoch {epoch}/{max_epochs-1}")
            
            batch_idx = 0
            while train_data_reader.has_more():
                # Calculate current minibatch learning rate and momentum according to CNTK
                global_minibatch_idx = epoch * (epoch_size // minibatch_size) + batch_idx
                
                # Set current learning rate
                current_lr_idx = min(global_minibatch_idx, len(lr_per_minibatch)-1)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_per_minibatch[current_lr_idx]
                
                # Set current momentum using time constant calculation
                current_momentum = np.exp(-minibatch_size/mm_time_constant)
                for param_group in optimizer.param_groups:
                    param_group['momentum'] = current_momentum
                
                images, labels, current_batch_size = train_data_reader.next_minibatch(minibatch_size)
                
                # Convert to PyTorch tensors
                images = torch.from_numpy(images).to(device)
                labels = torch.from_numpy(labels).to(device)
                
                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass
                logits = model(images)
                
                # Calculate loss
                loss = cost_func(training_mode, logits, labels)
                
                # Calculate error rate - consistent with CNTK
                error = classification_error(logits, labels)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                # Statistics
                training_loss += loss.item() * current_batch_size
                training_error += error.item() * current_batch_size
                train_samples += current_batch_size
                
                # Update progress bar
                progress.update(1)
                batch_idx += 1
            
            progress.close()
            
            # Calculate average loss and accuracy
            training_loss /= train_samples
            training_error /= train_samples  # Error rate
            training_accuracy = 1.0 - training_error  # Convert to accuracy (consistent with CNTK)
            
            # Record training loss and accuracy
            train_losses.append(training_loss)
            train_accs.append(training_accuracy * 100)
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_error = 0
            val_samples = 0
            
            with torch.no_grad():
                while val_data_reader.has_more():
                    images, labels, current_batch_size = val_data_reader.next_minibatch(minibatch_size)
                    
                    # Convert to PyTorch tensors
                    images = torch.from_numpy(images).to(device)
                    labels = torch.from_numpy(labels).to(device)
                    
                    # Forward pass
                    logits = model(images)
                    
                    # Calculate loss
                    loss = cost_func(training_mode, logits, labels)
                    
                    # Calculate error rate
                    error = classification_error(logits, labels)
                    
                    val_loss += loss.item() * current_batch_size
                    val_error += error.item() * current_batch_size
                    val_samples += current_batch_size
                
            val_loss /= val_samples
            val_error /= val_samples  # Error rate
            val_accuracy = 1.0 - val_error  # Convert to accuracy (consistent with CNTK)
            
            # Record validation loss and accuracy
            val_losses.append(val_loss)
            val_accs.append(val_accuracy * 100)
            
            # If validation accuracy improves, save the model
            if val_accuracy > max_val_accuracy:
                best_model_epoch = epoch
                max_val_accuracy = val_accuracy
                
                # Save best model
                torch.save(model.state_dict(), os.path.join(output_model_folder, f"best_model.pth"))
            
            # logging.info("Epoch {}: took {:.3f}s".format(epoch, time.time() - start_time))
            # logging.info("  training loss:\t{:e}".format(training_loss))
            # logging.info("  validation loss:\t{:e}".format(val_loss))
            # logging.info("  training accuracy:\t\t{:.2f} %".format(training_accuracy * 100))
            # logging.info("  validation accuracy:\t\t{:.2f} %".format(val_accuracy * 100))
            
            # Output to console
            print("Epoch {}: took {:.3f}s".format(epoch, time.time() - start_time))
            print("  training loss:\t{:e}".format(training_loss))
            print("  validation loss:\t{:e}".format(val_loss))
            print("  training accuracy:\t\t{:.2f} %".format(training_accuracy * 100))
            print("  validation accuracy:\t\t{:.2f} %".format(val_accuracy * 100))
            
            # Save checkpoint after each epoch to enable resuming training
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'max_val_accuracy': max_val_accuracy,
                'best_model_epoch': best_model_epoch,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs
            }, checkpoint_path)
            
            # Also save a numbered checkpoint and plot curves every 10 epochs
            if epoch % 10 == 0 and epoch > 0:
                torch.save(model.state_dict(), os.path.join(output_model_folder, f"model_epoch_{epoch}.pth"))
                plot_training_curves(train_losses, val_losses, train_accs, val_accs, output_model_folder)
    
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving checkpoint...")
        # Save checkpoint to enable resuming training
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'max_val_accuracy': max_val_accuracy,
            'best_model_epoch': best_model_epoch,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}. Use --resume flag to continue training.")
        
        # Plot curves even if training is interrupted
        plot_training_curves(train_losses, val_losses, train_accs, val_accs, output_model_folder)
        return
    
    # Training completed, save final model
    torch.save(model.state_dict(), os.path.join(output_model_folder, "final_model.pth"))
    
    # Load the best model for final test evaluation
    model.load_state_dict(torch.load(os.path.join(output_model_folder, "best_model.pth")))
    
    # Final evaluation on test set using the best model
    print("\nEvaluating on test set using the best model...")
    test_error = 0
    test_samples = 0
    all_preds = []
    all_targets = []
    
    # Reset test data reader
    test_data_reader.reset()
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        while test_data_reader.has_more():
            images, labels, current_batch_size = test_data_reader.next_minibatch(minibatch_size)
            
            # Convert to PyTorch tensors
            images = torch.from_numpy(images).to(device)
            labels = torch.from_numpy(labels).to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate error rate
            error = classification_error(outputs, labels)
            test_error += error.item() * current_batch_size
            test_samples += current_batch_size
            
            # Record predictions and true labels
            _, predicted = torch.max(outputs.data, 1)
            _, targets = torch.max(labels.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate final test accuracy
    test_accuracy = 1.0 - (test_error / test_samples)  # Convert error rate to accuracy
    
    print("")
    print("Best validation accuracy:\t\t{:.2f} %, epoch {}".format(max_val_accuracy * 100, best_model_epoch))
    print("Test accuracy using best model:\t\t{:.2f} %".format(test_accuracy * 100))
    
    # Plot final curves
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, output_model_folder)
    
    # Calculate and plot confusion matrix
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