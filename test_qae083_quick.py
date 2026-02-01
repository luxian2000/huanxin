"""
QAE083快速测试脚本 - 验证训练逻辑是否正常工作
"""

import sys
sys.path.append('/Users/luxian/GitSpace/huanxin')

# 导入QAE083的主要组件
from QAE083 import (
    load_csinet_data,
    CsiNetEncoder,
    quantum_decoder_circuit,
    HybridCsiNetQuantumAutoencoder,
    prepare_target_distribution,
    compute_probability_loss
)
import torch
import torch.nn as nn
import pennylane as qml
import numpy as np

def quick_test():
    """快速测试QAE083的核心组件"""
    print("开始QAE083快速测试...")
    
    # 1. 测试数据加载
    print("\n1. 测试数据加载...")
    try:
        train_data, val_data, test_data = load_csinet_data()
        print(f"✓ 数据加载成功")
        print(f"  训练数据: {train_data.shape}")
        print(f"  验证数据: {val_data.shape}")
        print(f"  测试数据: {test_data.shape}")
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return False
    
    # 2. 测试CsiNet编码器
    print("\n2. 测试CsiNet编码器...")
    try:
        encoder = CsiNetEncoder(encoded_dim=256)
        test_input = torch.randn(2, 2, 32, 32)  # 2个样本
        encoded_output = encoder(test_input)
        print(f"✓ 编码器测试成功")
        print(f"  输入形状: {test_input.shape}")
        print(f"  输出形状: {encoded_output.shape}")
        assert encoded_output.shape == (2, 256), f"编码器输出形状错误: {encoded_output.shape}"
    except Exception as e:
        print(f"✗ 编码器测试失败: {e}")
        return False
    
    # 3. 测试量子解码器参数
    print("\n3. 测试量子解码器参数...")
    try:
        dec_shape = qml.StronglyEntanglingLayers.shape(n_layers=4, n_wires=11)
        dec_params = nn.Parameter(torch.rand(dec_shape) * 2 * np.pi - np.pi)
        print(f"✓ 量子参数初始化成功")
        print(f"  参数形状: {dec_shape}")
        print(f"  参数范围: [{dec_params.min():.3f}, {dec_params.max():.3f}]")
    except Exception as e:
        print(f"✗ 量子参数初始化失败: {e}")
        return False
    
    # 4. 测试量子电路（小规模）
    print("\n4. 测试量子电路...")
    try:
        # 使用较小的测试向量
        test_encoded = torch.randn(256)
        test_encoded = test_encoded / torch.norm(test_encoded)  # 归一化
        
        # 测试单次量子计算
        probs = quantum_decoder_circuit(test_encoded, dec_params)
        print(f"✓ 量子电路测试成功")
        print(f"  输入维度: {len(test_encoded)}")
        print(f"  输出维度: {len(probs)}")
        print(f"  概率和: {probs.sum():.6f}")
        assert len(probs) == 2048, f"输出维度错误: {len(probs)}"
        assert abs(probs.sum() - 1.0) < 1e-5, f"概率和不为1: {probs.sum()}"
    except Exception as e:
        print(f"✗ 量子电路测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. 测试完整混合模型（小批量）
    print("\n5. 测试完整混合模型...")
    try:
        # 创建小批量测试数据
        small_batch = torch.from_numpy(train_data[:3]).float()  # 3个样本
        print(f"  测试批量: {small_batch.shape}")
        
        # 创建混合模型
        hybrid_model = HybridCsiNetQuantumAutoencoder(encoder, dec_params)
        
        # 前向传播测试
        outputs = hybrid_model(small_batch)
        print(f"✓ 混合模型前向传播成功")
        print(f"  输出形状: {outputs.shape}")
        assert outputs.shape == (3, 2048), f"模型输出形状错误: {outputs.shape}"
        
        # 测试损失计算
        targets = prepare_target_distribution(small_batch)
        loss = compute_probability_loss(outputs, targets, loss_type='mse')
        print(f"✓ 损失计算成功")
        print(f"  损失值: {loss.item():.6f}")
        
    except Exception as e:
        print(f"✗ 混合模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*50)
    print("🎉 所有测试通过！QAE083可以正常训练")
    print("="*50)
    return True

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n建议：现在可以运行完整的QAE083.py进行训练")
    else:
        print("\n请检查上述错误并修复后再试")