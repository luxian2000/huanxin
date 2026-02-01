"""
测试扩展的概率分布损失函数
验证新添加的cross_entropy, jsd, hellinger损失函数是否正常工作
"""

import sys
sys.path.append('/Users/luxian/GitSpace/huanxin')

import torch
import numpy as np
from QAE083 import compute_probability_loss

def test_extended_losses():
    """测试扩展的损失函数"""
    print("测试扩展的概率分布损失函数...")

    # 创建测试概率分布
    batch_size = 3
    dim = 10

    # 目标分布 (均匀分布)
    target_probs = torch.ones(batch_size, dim) / dim

    # 输出分布 (稍微不同的分布)
    output_probs = torch.rand(batch_size, dim)
    output_probs = output_probs / output_probs.sum(dim=1, keepdim=True)  # 归一化

    print(f"目标分布形状: {target_probs.shape}")
    print(f"输出分布形状: {output_probs.shape}")
    print(f"目标分布和: {target_probs.sum(dim=1)}")
    print(f"输出分布和: {output_probs.sum(dim=1)}")

    # 测试所有损失函数
    loss_types = ['mse', 'kl', 'cross_entropy', 'jsd', 'hellinger']

    print("\n" + "="*50)
    print("损失函数测试结果:")
    print("="*50)

    results = {}
    for loss_type in loss_types:
        try:
            loss = compute_probability_loss(output_probs, target_probs, loss_type=loss_type)
            results[loss_type] = loss.item()
            print(f"✓ {loss_type.upper():15} 损失值: {loss.item():.6f}")
        except Exception as e:
            print(f"✗ {loss_type.upper():15} 错误: {e}")
    
    # 验证梯度计算
    print("\n" + "="*50)
    print("梯度计算测试:")
    print("="*50)
    
    for loss_type in loss_types:
        try:
            output_probs_grad = output_probs.clone().detach().requires_grad_(True)
            loss = compute_probability_loss(output_probs_grad, target_probs, loss_type=loss_type)
            loss.backward()
            
            grad_norm = torch.norm(output_probs_grad.grad).item()
            print(f"✓ {loss_type.upper():15} 梯度范数: {grad_norm:.6f}")
        except Exception as e:
            print(f"✗ {loss_type.upper():15} 梯度错误: {e}")
    
    # 比较不同损失函数的值
    print("\n" + "="*50)
    print("损失函数值比较:")
    print("="*50)
    
    for loss_type, value in results.items():
        print(f"{loss_type.upper():15}: {value:.6f}")
    
    test_extended_losses()