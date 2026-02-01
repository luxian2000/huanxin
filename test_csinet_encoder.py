#!/usr/bin/env python3
"""
测试QAE083.py中新的CsiNet编码器实现
"""

import sys
import os
import torch
import numpy as np

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_csinet_encoder():
    """测试CsiNet编码器功能"""
    try:
        print("正在测试CsiNet编码器...")
        
        # 导入必要的组件
        from QAE083 import CsiNetEncoder, load_csinet_data
        
        # 加载测试数据
        print("加载测试数据...")
        train_data, val_data, test_data = load_csinet_data()
        
        # 创建编码器实例
        encoder = CsiNetEncoder(encoded_dim=512)
        print("✅ CsiNet编码器创建成功！")
        print("编码器结构:")
        print(encoder)
        
        # 测试前向传播
        print("\n测试前向传播...")
        test_batch = torch.from_numpy(train_data[:4]).float()  # 4个样本的小批次
        print(f"输入形状: {test_batch.shape}")  # 应该是 (4, 2, 32, 32)
        
        with torch.no_grad():
            encoded_output = encoder(test_batch)
        
        print(f"编码输出形状: {encoded_output.shape}")  # 应该是 (4, 512)
        print(f"输出范围: [{encoded_output.min():.4f}, {encoded_output.max():.4f}]")
        
        # 验证编码器参数
        total_params = sum(p.numel() for p in encoder.parameters())
        print(f"\n编码器总参数数: {total_params:,}")
        
        # 测试多次前向传播的一致性
        print("\n测试编码一致性...")
        with torch.no_grad():
            output1 = encoder(test_batch)
            output2 = encoder(test_batch)
        
        consistency = torch.allclose(output1, output2)
        print(f"多次前向传播一致性: {'✅ 一致' if consistency else '❌ 不一致'}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_complete_pipeline():
    """测试完整的数据处理流水线"""
    try:
        print("\n" + "="*50)
        print("测试完整数据处理流水线...")
        
        from QAE083 import load_csinet_data
        
        # 加载数据
        train_data, val_data, test_data = load_csinet_data()
        
        # 验证数据形状
        assert train_data.shape[1:] == (2, 32, 32), f"训练数据形状错误: {train_data.shape}"
        assert val_data.shape[1:] == (2, 32, 32), f"验证数据形状错误: {val_data.shape}"
        assert test_data.shape[1:] == (2, 32, 32), f"测试数据形状错误: {test_data.shape}"
        
        print("✅ 数据形状验证通过！")
        
        # 验证数据范围
        assert train_data.min() >= 0 and train_data.max() <= 1, "训练数据范围错误"
        assert val_data.min() >= 0 and val_data.max() <= 1, "验证数据范围错误"
        assert test_data.min() >= 0 and test_data.max() <= 1, "测试数据范围错误"
        
        print("✅ 数据范围验证通过！")
        
        return True
        
    except Exception as e:
        print(f"❌ 流水线测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("QAE083 CsiNet编码器测试")
    print("=" * 60)
    
    # 测试编码器
    encoder_success = test_csinet_encoder()
    
    # 测试完整流水线
    pipeline_success = test_complete_pipeline()
    
    print("\n" + "=" * 60)
    if encoder_success and pipeline_success:
        print("🎉 所有测试通过！")
        print("CsiNet编码器已成功集成到QAE083中。")
    else:
        print("❌ 部分测试失败！")
        sys.exit(1)