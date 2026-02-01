#!/usr/bin/env python3
"""
编码器配置管理工具
用于在不同编码器配置之间切换和比较
"""

import torch
import torch.nn as nn

class EncoderConfigManager:
    """编码器配置管理器"""
    
    @staticmethod
    def get_csinet_equivalent_config():
        """获取与CsiNet_train.py等效的配置"""
        return {
            'encoded_dim': 512,
            'activation': 'linear',
            'architecture': 'resnet_style',
            'framework': 'tensorflow',
            'description': '原始CsiNet配置'
        }
    
    @staticmethod
    def get_qcnn_current_config():
        """获取QCNN01_gpu.py当前配置"""
        return {
            'encoded_dim': 256,
            'activation': 'leaky_relu_0.3',
            'architecture': 'simple_bn',
            'framework': 'pytorch',
            'description': 'QCNN优化配置'
        }
    
    @staticmethod
    def create_compatible_encoder(target_config='qcnn'):
        """创建兼容的编码器"""
        if target_config == 'csinet':
            # 创建接近CsiNet的PyTorch版本
            return CompatibleCsiNetEncoder()
        else:
            # 创建标准QCNN编码器
            return StandardQCnnEncoder()

class CompatibleCsiNetEncoder(nn.Module):
    """兼容CsiNet的编码器(在PyTorch中实现)"""
    
    def __init__(self, encoded_dim=512):
        super(CompatibleCsiNetEncoder, self).__init__()
        self.encoded_dim = encoded_dim
        
        # 匹配CsiNet的第一层卷积
        self.conv1 = nn.Conv2d(2, 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(2)
        self.relu1 = nn.LeakyReLU(0.3)
        
        # 展平和编码层(使用线性激活)
        self.flatten = nn.Flatten()
        self.dense_encoded = nn.Linear(2048, encoded_dim)
        self.linear_activation = nn.Identity()  # 线性激活
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.flatten(x)
        x = self.dense_encoded(x)
        x = self.linear_activation(x)  # 线性输出
        return torch.clamp(x, min=1e-7, max=1e7)

class StandardQCnnEncoder(nn.Module):
    """标准QCNN编码器"""
    
    def __init__(self, encoded_dim=256):
        super(StandardQCnnEncoder, self).__init__()
        self.encoded_dim = encoded_dim
        
        self.conv1 = nn.Conv2d(2, 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(2)
        self.relu1 = nn.LeakyReLU(0.3)
        
        self.flatten = nn.Flatten()
        self.dense_encoded = nn.Linear(2048, encoded_dim)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.flatten(x)
        encoded = self.dense_encoded(x)
        encoded = torch.clamp(encoded, min=1e-7, max=1e7)
        encoded = torch.nan_to_num(encoded, nan=0.0, posinf=1.0, neginf=0.0)
        return encoded

def compare_performance_metrics():
    """比较不同配置的性能指标模板"""
    metrics_comparison = {
        'Compression Ratio': {
            'CsiNet (512)': '1/4',
            'QCNN (256)': '1/8',
            'Impact': 'QCNN压缩更强'
        },
        'Model Complexity': {
            'CsiNet (512)': 'High (ResNet)',
            'QCNN (256)': 'Low (Simple)',
            'Impact': 'QCNN计算更高效'
        },
        'Quantum Circuit Fit': {
            'CsiNet (512)': 'Needs adaptation',
            'QCNN (256)': 'Natural fit',
            'Impact': 'QCNN更适合量子处理'
        },
        'Information Retention': {
            'CsiNet (512)': 'Higher',
            'QCNN (256)': 'Lower',
            'Impact': '需实验验证'
        }
    }
    
    return metrics_comparison

def main():
    print("🔧 编码器配置管理工具")
    print("=" * 40)
    
    # 显示配置对比
    csinet_config = EncoderConfigManager.get_csinet_equivalent_config()
    qcnn_config = EncoderConfigManager.get_qcnn_current_config()
    
    print("📋 配置对比:")
    print(f"CsiNet配置: {csinet_config}")
    print(f"QCNN配置: {qcnn_config}")
    
    # 创建示例编码器
    print("\n🏗️  创建编码器实例:")
    csinet_encoder = EncoderConfigManager.create_compatible_encoder('csinet')
    qcnn_encoder = EncoderConfigManager.create_compatible_encoder('qcnn')
    
    print(f"CsiNet兼容编码器: {csinet_encoder}")
    print(f"标准QCNN编码器: {qcnn_encoder}")
    
    # 性能指标对比
    print("\n📊 性能指标对比:")
    metrics = compare_performance_metrics()
    for metric, values in metrics.items():
        print(f"\n{metric}:")
        for config, value in values.items():
            print(f"  {config}: {value}")

if __name__ == "__main__":
    main()