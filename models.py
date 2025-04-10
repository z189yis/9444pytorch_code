#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#

import torch
import torch.nn as nn
import torch.nn.functional as F

def build_model(num_classes, model_name):
    '''
    Factory function to instantiate the model.
    '''
    model_classes = {
        'VGG13': VGG13,
        'CNN9': CNN9,
        'CNN12': CNN12
    }
    
    # 检查模型名称是否支持
    if model_name not in model_classes:
        raise ValueError(f"Model {model_name} not supported. Available models: {list(model_classes.keys())}")
    
    # 创建并返回模型
    return model_classes[model_name](num_classes)

class VGG13(nn.Module):
    '''
    A VGG13 like model (https://arxiv.org/pdf/1409.1556.pdf) tweaked for emotion data.
    '''
    def __init__(self, num_classes):
        super(VGG13, self).__init__()
        
        # First block (64 filters)
        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)  # 添加BatchNorm
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)  # 添加BatchNorm
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout2d(0.25)
        
        # Second block (128 filters)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)  # 添加BatchNorm
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)  # 添加BatchNorm
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout2d(0.25)
        
        # Third block (256 filters)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)  # 添加BatchNorm
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)  # 添加BatchNorm
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256)  # 添加BatchNorm
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout2d(0.25)
        
        # Fourth block (256 filters)
        self.conv4_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(256)  # 添加BatchNorm
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(256)  # 添加BatchNorm
        self.conv4_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4_3 = nn.BatchNorm2d(256)  # 添加BatchNorm
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop4 = nn.Dropout2d(0.25)
        
        # Calculate size after convolutions and pooling
        # 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        self.fc_input_size = 256 * 4 * 4
        
        # Fully connected layers
        self.fc5 = nn.Linear(self.fc_input_size, 1024)
        self.drop5 = nn.Dropout(0.5)
        
        self.fc6 = nn.Linear(1024, 1024)
        self.drop6 = nn.Dropout(0.5)
        
        self.fc7 = nn.Linear(1024, num_classes)
        
        # Model properties
        self.learning_rate = 0.01  # 降低学习率
        self.input_height = 64
        self.input_width = 64
        self.input_channels = 1
        
        # 初始化权重 (模仿CNTK的glorot_uniform)
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        x = self.drop1(x)
        
        # Block 2
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        x = self.drop2(x)
        
        # Block 3
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = F.relu(self.bn3_3(self.conv3_3(x)))
        x = self.pool3(x)
        x = self.drop3(x)
        
        # Block 4
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = F.relu(self.bn4_3(self.conv4_3(x)))
        x = self.pool4(x)
        x = self.drop4(x)
        
        # Flatten
        x = x.view(-1, self.fc_input_size)
        
        # Fully connected layers
        x = F.relu(self.fc5(x))
        x = self.drop5(x)
        
        x = F.relu(self.fc6(x))
        x = self.drop6(x)
        
        # Output layer (no activation - it will be applied in the loss function)
        x = self.fc7(x)
        
        return x

class CNN9(nn.Module):
    """
    A custom 9-layer CNN model for emotion recognition.
    """
    def __init__(self, num_classes):
        super(CNN9, self).__init__()
        
        # First block (64 filters)
        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout2d(0.25)
        
        # Second block (128 filters)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout2d(0.25)
        
        # Third block (256 filters)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout2d(0.25)
        
        # Fourth block (256 filters)
        self.conv4_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop4 = nn.Dropout2d(0.25)
        
        # Calculate size after convolutions and pooling
        # 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        self.fc_input_size = 256 * 4 * 4
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.drop_fc1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.drop_fc2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)


         # Model properties
        self.learning_rate = 0.01  # 降低学习率
        self.input_height = 64
        self.input_width = 64
        self.input_channels = 1

        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        x = self.drop1(x)
        
        # Block 2
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        x = self.drop2(x)
        
        # Block 3
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool3(x)
        x = self.drop3(x)
        
        # Block 4
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = self.pool4(x)
        x = self.drop4(x)
        
        # Flatten
        x = x.view(-1, self.fc_input_size)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.drop_fc1(x)
        x = F.relu(self.fc2(x))
        x = self.drop_fc2(x)
        
        # Output layer
        x = self.fc3(x)
        
        return x

class CNN12(nn.Module):
    """
    A custom 12-layer CNN model for emotion recognition.
    """
    def __init__(self, num_classes):
        super(CNN12, self).__init__()
        
        # First block (64 filters)
        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout2d(0.25)
        
        # Second block (128 filters)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout2d(0.25)
        
        # Third block (256 filters)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout2d(0.25)
        
        # Fourth block (512 filters)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop4 = nn.Dropout2d(0.25)
        
        # Fifth block (512 filters)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop5 = nn.Dropout2d(0.25)
        
        # Calculate size after convolutions and pooling
        # Assuming input size is 64x64
        self.fc_input_size = 512 * 2 * 2
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 1024)
        self.drop_fc1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.drop_fc2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, num_classes)
        
        # Model properties
        self.learning_rate = 0.01
        self.input_height = 64
        self.input_width = 64
        self.input_channels = 1
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        x = self.drop1(x)
        
        # Block 2
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        x = self.drop2(x)
        
        # Block 3
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool3(x)
        x = self.drop3(x)
        
        # Block 4
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = self.pool4(x)
        x = self.drop4(x)
        
        # Block 5
        x = F.relu(self.bn5_1(self.conv5_1(x)))
        x = F.relu(self.bn5_2(self.conv5_2(x)))
        x = self.pool5(x)
        x = self.drop5(x)
        
        # Flatten
        x = x.view(-1, self.fc_input_size)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.drop_fc1(x)
        x = F.relu(self.fc2(x))
        x = self.drop_fc2(x)
        
        # Output layer
        x = self.fc3(x)
        
        return x