#!/usr/bin/env python3
"""
QCNN01_gpu 训练结果分析
"""

import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np

def analyze_training_results():
    """分析训练结果"""
    print("🎯 QCNN01_gpu 训练结果分析")
    print("=" * 50)
    
    # 加载训练历史
    try:
        with open('QCNN01_gpu/training_history.pt', 'rb') as f:
            data = pickle.load(f)
        
        epochs = len(data["epoch_losses"])
        print(f"✅ 训练完成!")
        print(f"总epochs: {epochs}")
        print(f"每epoch batch数: 10个")
        print(f"总batch数: {epochs * 10}")
        
        # 提取损失数据
        train_losses = [epoch_data['avg_loss'] for epoch_data in data['epoch_losses']]
        val_losses = [data['val_mse'][i]['val_mse'] for i in range(epochs)]
        
        print(f"\n📊 损失统计:")
        print(f"初始训练损失: {train_losses[0]:.6f}")
        print(f"最终训练损失: {train_losses[-1]:.6f}")
        print(f"初始验证MSE: {val_losses[0]:.6f}")
        print(f"最终验证MSE: {val_losses[-1]:.6f}")
        print(f"损失改善幅度: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.2f}%")
        
        print(f"\n📈 详细训练历史:")
        print("Epoch | 训练损失   | 验证MSE")
        print("-" * 30)
        for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
            print(f"{i:5d} | {train_loss:.6f} | {val_loss:.6f}")
            
        # 网络配置
        config = data['network_config']
        print(f"\n🔧 网络配置:")
        print(f"• 编码维度: {config['encoded_dim']}")
        print(f"• QCNN卷积层: {config['qcnn_conv_layers']}")
        print(f"• 量子比特数: {config['quantum_wires']}")
        print(f"• 压缩率: {config['compression_ratio']}")
        print(f"• 训练设备: {config['device']}")
        
        # 数据集信息
        data_info = data['data_split_info']
        print(f"\n📊 数据集信息:")
        print(f"• 训练样本: {data_info['actual_train_used']}")
        print(f"• 验证样本: {data_info['val_size']}")
        print(f"• 测试样本: {data_info['test_size']}")
        
        # 测试结果
        try:
            test_results = torch.load('QCNN01_gpu/test_results.pt')
            print(f"\n🧪 测试结果:")
            print(f"• 测试MSE: {test_results['test_mse_loss']:.6f}")
            print(f"• 测试样本数: {test_results['n_samples']}")
        except:
            print("\n⚠️  测试结果文件未找到")
            
    except Exception as e:
        print(f"❌ 分析失败: {e}")

if __name__ == "__main__":
    analyze_training_results()