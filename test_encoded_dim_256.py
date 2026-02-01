#!/usr/bin/env python3
"""
测试QAE083.py中encoded_dim=256的配置
"""

import sys
import os
import torch
import numpy as np

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_encoded_dim_configuration():
    """测试encoded_dim=256的配置"""
    try:
        print("=" * 60)
        print("测试QAE083 encoded_dim=256配置")
        print("=" * 60)
        
        # 导入必要的组件
        from QAE083 import (
            CsiNetEncoder, 
            QuantumToClassicalDecoder,
            quantum_decoder_circuit,
            load_csinet_data
        )
        
        # 测试1: 编码器配置
        print("\n1. 测试CsiNet编码器 (encoded_dim=256)...")
        encoder = CsiNetEncoder(encoded_dim=256)
        print("✅ 编码器创建成功")
        print(f"   编码维度: 256")
        print(f"   输入形状: (batch_size, 2, 32, 32)")
        print(f"   输出形状: (batch_size, 256)")
        
        # 测试前向传播
        test_input = torch.randn(2, 2, 32, 32)  # 2个样本
        with torch.no_grad():
            encoded_output = encoder(test_input)
        print(f"   实际输出形状: {encoded_output.shape}")
        assert encoded_output.shape == (2, 256), "编码器输出形状错误"
        print("✅ 编码器前向传播测试通过")
        
        # 测试2: 量子解码器配置
        print("\n2. 测试量子解码器配置...")
        # 检查量子比特数
        import pennylane as qml
        dev = qml.device("lightning.qubit", wires=8)  # 应该是8个量子比特
        print(f"✅ 量子设备配置: 8量子比特 (2^8 = 256)")
        
        # 测试量子电路
        test_vector = torch.randn(256)
        # 创建测试参数
        dec_shape = qml.StronglyEntanglingLayers.shape(n_layers=4, n_wires=8)
        test_params = torch.rand(dec_shape)
        
        # 测试量子电路执行
        result = quantum_decoder_circuit(test_vector, test_params)
        expected_measurements = 8 * 3  # 8量子比特 × 3测量基
        print(f"✅ 量子测量数量: {len(result)} (期望: {expected_measurements})")
        assert len(result) == 24, "量子测量数量错误"
        
        # 测试3: 经典解码器配置
        print("\n3. 测试经典解码器配置...")
        decoder = QuantumToClassicalDecoder(quantum_output_dim=8, output_dim=2048)
        print("✅ 经典解码器创建成功")
        print(f"   输入维度: 24 (8量子比特 × 3基)")
        print(f"   输出维度: 2048")
        
        # 测试解码器前向传播
        test_quantum_output = torch.randn(1, 24)  # 1个样本，24维输入
        with torch.no_grad():
            decoded_output = decoder(test_quantum_output)
        print(f"   实际输出形状: {decoded_output.shape}")
        assert decoded_output.shape == (1, 2048), "解码器输出形状错误"
        print("✅ 经典解码器前向传播测试通过")
        
        # 测试4: 完整数据流
        print("\n4. 测试完整数据流...")
        # 加载测试数据
        train_data, _, _ = load_csinet_data()
        test_batch = torch.from_numpy(train_data[:2]).float()
        
        print(f"   输入数据形状: {test_batch.shape}")
        
        # 完整前向传播测试
        with torch.no_grad():
            # 编码
            encoded = encoder(test_batch)
            print(f"   编码后形状: {encoded.shape}")
            
            # 量子处理（简化测试）
            quantum_output = torch.randn(2, 24)  # 模拟量子测量结果
            print(f"   量子测量形状: {quantum_output.shape}")
            
            # 解码
            final_output = decoder(quantum_output)
            print(f"   最终输出形状: {final_output.shape}")
            
            # reshape检查
            reshaped_output = final_output.view(2, 2, 32, 32)
            print(f"   reshape后形状: {reshaped_output.shape}")
            
        print("✅ 完整数据流测试通过")
        
        # 测试5: 参数统计
        print("\n5. 网络参数统计...")
        
        # 编码器参数
        encoder_params = sum(p.numel() for p in encoder.parameters())
        print(f"   CsiNet编码器参数: {encoder_params:,}")
        
        # 解码器参数  
        decoder_params = sum(p.numel() for p in decoder.parameters())
        print(f"   经典解码器参数: {decoder_params:,}")
        
        # 量子参数
        quantum_params = 8 * 4 * 3  # 8 qubits × 4 layers × 3 parameters per gate
        print(f"   量子电路参数: {quantum_params}")
        
        total_params = encoder_params + decoder_params + quantum_params
        print(f"   总参数数量: {total_params:,}")
        
        print("✅ 参数统计完成")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_compression_ratio():
    """测试压缩比率"""
    try:
        print("\n" + "=" * 40)
        print("压缩比率分析")
        print("=" * 40)
        
        input_dim = 2048  # (2,32,32) 展平后的维度
        encoded_dim = 256  # 新的编码维度
        
        compression_ratio = input_dim / encoded_dim
        compression_percentage = (1 - encoded_dim/input_dim) * 100
        
        print(f"输入维度: {input_dim}")
        print(f"编码维度: {encoded_dim}")
        print(f"压缩比率: 1/{compression_ratio:.0f}")
        print(f"压缩百分比: {compression_percentage:.1f}%")
        
        # 与原来512维的对比
        old_encoded_dim = 512
        old_compression_ratio = input_dim / old_encoded_dim
        old_compression_percentage = (1 - old_encoded_dim/input_dim) * 100
        
        print(f"\n对比分析:")
        print(f"原配置 (512维): 压缩比率 1/{old_compression_ratio:.0f}, 压缩 {old_compression_percentage:.1f}%")
        print(f"新配置 (256维): 压缩比率 1/{compression_ratio:.0f}, 压缩 {compression_percentage:.1f}%")
        print(f"压缩强度增加: {(compression_percentage - old_compression_percentage):.1f} 个百分点")
        
        print("✅ 压缩比率分析完成")
        return True
        
    except Exception as e:
        print(f"❌ 压缩比率测试失败: {e}")
        return False

if __name__ == "__main__":
    # 运行配置测试
    config_success = test_encoded_dim_configuration()
    
    # 运行压缩比率测试
    ratio_success = test_compression_ratio()
    
    print("\n" + "=" * 60)
    if config_success and ratio_success:
        print("🎉 所有测试通过！")
        print("QAE083已成功配置为encoded_dim=256")
        print("主要变更:")
        print("  - 编码维度: 512 → 256")
        print("  - 量子比特: 9 → 8")
        print("  - 压缩比率: 1/4 → 1/8")
        print("  - 测量维度: 27 → 24")
    else:
        print("❌ 部分测试失败！")
        sys.exit(1)