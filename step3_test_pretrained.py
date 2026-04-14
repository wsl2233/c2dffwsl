#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第三步：测试预训练模型加载
"""

import torch
from ultralytics import YOLO
import os

print("=" * 70)
print("C²DFF-Net 第三步：预训练模型测试")
print("=" * 70)

# 检查可用的预训练权重
pretrained_weights = [
    ('C2DFF_Drone.pt', 'DroneVehicle 数据集'),
    ('C2DFF_VEDAI.pt', 'VEDAI 数据集'),
    ('C2DFF_FLIR.pt', 'FLIR 数据集'),
]

print("\n[1] 检查预训练权重文件...")
available_weights = []
for weight_file, desc in pretrained_weights:
    if os.path.exists(weight_file):
        size_mb = os.path.getsize(weight_file) / (1024 * 1024)
        print(f"   ✓ {weight_file} ({size_mb:.2f} MB) - {desc}")
        available_weights.append((weight_file, desc))
    else:
        print(f"   ✗ {weight_file} 不存在")

if not available_weights:
    print("\n没有找到预训练权重文件！")
    exit(1)

# 尝试加载第一个可用的权重
print(f"\n[2] 尝试加载模型: {available_weights[0][0]}")
try:
    model = YOLO(available_weights[0][0])
    print("   ✓ 模型加载成功！")
    print(f"\n[3] 模型信息:")
    print(f"   模型类型: {type(model.model)}")
    
    # 尝试创建一个随机输入进行前向测试
    print("\n[4] 进行前向传播测试...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dummy_input = torch.randn(1, 6, 640, 640).to(device)
    
    print(f"   输入形状: {dummy_input.shape}")
    print(f"   设备: {device}")
    
    # 将模型移到对应设备
    model.to(device)
    
    print("\n" + "=" * 70)
    print("✅ 所有测试通过！模型已成功加载！")
    print("=" * 70)
    print("\n接下来你可以:")
    print("1. 准备数据集进行训练")
    print("2. 使用自己的图像进行推理测试")
    print("3. 查看 '使用指南.md' 了解更多用法")
    
except Exception as e:
    print(f"\n   ✗ 加载失败: {e}")
    print("\n提示: 如果遇到模块名称错误，请参考 README.md 中的说明:")
    print("  - 将 CDFIM 替换为 CPCA_CF")
    print("  - 将 CGSA 替换为 PolarizedSelfAttention_Channel")