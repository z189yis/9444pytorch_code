import os

data_path = r"C:\Users\zys18\FERPlus\data"
required_folders = ["FER2013Train", "FER2013Valid", "FER2013Test"]

for folder in required_folders:
    full_path = os.path.join(data_path, folder)
    label_file = os.path.join(full_path, "label.csv")
    if not os.path.exists(full_path):
        print(f"错误: 找不到目录 {full_path}")
    elif not os.path.exists(label_file):
        print(f"错误: 找不到文件 {label_file}")
    else:
        print(f"✓ {folder} 目录和标签文件正常")
        # 检查是否有图片文件
        image_files = [f for f in os.listdir(full_path) if f.endswith(".png")]
        print(f"  - 包含 {len(image_files)} 个图像文件")