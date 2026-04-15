#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

print("=" * 60)
print("C²DFF-Net 环境检查")
print("=" * 60)

# 1. 检查 Python 版本
print("\n1. Python 版本:")
print(f"   Python: {sys.version}")
print(f"   路径: {sys.executable}")

# 2. 检查核心依赖库
libraries = [
    ('torch', 'PyTorch'),
    ('torchvision', 'TorchVision'),
    ('ultralytics', 'Ultralytics (YOLO)'),
    ('numpy', 'NumPy'),
    ('cv2', 'OpenCV'),
    ('matplotlib', 'Matplotlib'),
]

print("\n2. 核心依赖库检查:")
for lib_module, lib_name in libraries:
    try:
        module = __import__(lib_module)
        version = getattr(module, '__version__', 'N/A')
        print(f"   ✓ {lib_name}: {version}")
    except ImportError:
        print(f"   ✗ {lib_name}: 未安装")

# 3. 检查 CUDA 可用性
print("\n3. CUDA 检查:")
try:
    import torch
    print(f"   CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA 版本: {torch.version.cuda}")
        print(f"   GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
except ImportError:
    print("   PyTorch 未安装，无法检查 CUDA")

# 4. 检查项目文件
print("\n4. 项目文件检查:")
project_files = ['README.md', 'requirements.txt', 'train.py', 'test.py']
current_files = os.listdir('.')
print(f"   当前目录文件: {current_files}")

print("\n" + "=" * 60)
print("检查完成！")
print("=" * 60)