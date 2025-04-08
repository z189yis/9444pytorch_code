import sys
import os
import time

print("调试信息：脚本开始运行")

try:
    print("正在导入必要的库...")
    import torch
    import numpy as np
    from PIL import Image
    import logging
    
    print("设置基础日志...")
    logging.basicConfig(level=logging.INFO)
    
    print("检查数据目录...")
    base_folder = r"C:\Users\zys18\FERPlus\data"
    required_folders = ["FER2013Train", "FER2013Valid", "FER2013Test"]
    
    for folder in required_folders:
        full_path = os.path.join(base_folder, folder)
        if not os.path.exists(full_path):
            print(f"错误: 找不到目录 {full_path}")
            sys.exit(1)
        
        label_file = os.path.join(full_path, "label.csv")
        if not os.path.exists(label_file):
            print(f"错误: 找不到文件 {label_file}")
            sys.exit(1)
        
        print(f"✓ {folder} 目录和标签文件正常")
        # 检查是否有图片文件
        image_files = [f for f in os.listdir(full_path) if f.endswith(".png")]
        print(f"  - 包含 {len(image_files)} 个图像文件")
    
    print("开始导入训练模块...")
    from models import build_model
    print("模型模块导入成功")
    
    from ferplus import FERPlusParameters, FERPlusReader
    print("数据读取模块导入成功")
    
    print("开始创建简单的训练模型...")
    num_classes = 8  # 8个表情类别
    model = build_model(num_classes, "VGG13")
    print("模型创建成功")
    
    print("测试数据加载器...")
    train_params = FERPlusParameters(num_classes, model.input_height, model.input_width, "majority", False)
    print("尝试载入少量数据...")
    reader = FERPlusReader(base_folder, ["FER2013Test"], "label.csv", train_params)  # 仅使用测试数据进行验证
    reader.load_folders("majority")
    print(f"成功加载了 {reader.size()} 个样本")
    
    if reader.size() > 0:
        print("尝试读取一个小批次...")
        inputs, targets, batch_size = reader.next_minibatch(min(32, reader.size()))
        print(f"成功读取了 {batch_size} 个样本")
        print(f"输入形状: {inputs.shape}, 输出形状: {targets.shape}")
    
    print("环境测试成功，可以开始训练")
    print("请使用以下命令运行训练脚本:")
    print("python train.py -d C:\\Users\\zys18\\FERPlus\\data")

except Exception as e:
    print(f"发生错误: {e}")
    import traceback
    traceback.print_exc()