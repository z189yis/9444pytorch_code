import torch
import numpy as np
from PIL import Image

print("Python环境正常")
print(f"GPU可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")