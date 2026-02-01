#!/usr/bin/env python3
"""
对比QCNN01_gpu.py和CsiNet_train.py中的编码器结构
"""

def analyze_csinet_train_encoder():
    """分析CsiNet_train.py的编码器结构"""
    print("🔍 CsiNet_train.py 编码器分析")
    print("=" * 40)
    
    # 从CsiNet_train.py提取的关键结构
    print("网络结构:")
    print("1. 输入层: (2, 32, 32)")
    print("2. 第一层卷积: Conv2D(2, (3, 3), padding='same')")
    print("3. 批归一化 + LeakyReLU")
    print("4. Reshape到1D: (2048,)")
    print("5. 全连接编码层: Dense(512, activation='linear')")
    print("6. 编码维度: 512")
    print()
    
    print("关键参数:")
    print("- residual_num: 2")
    print("- encoded_dim: 512")
    print("- img_channels: 2")
    print("- img_height: 32")
    print("- img_width: 32")

def analyze_qcnn01_gpu_encoder():
    """分析QCNN01_gpu.py的编码器结构"""
    print("\n🔍 QCNN01_gpu.py 编码器分析")
    print("=" * 40)
    
    # QCNN01_gpu.py中的编码器结构
    print("网络结构:")
    print("1. 输入层: (2, 32, 32)")
    print("2. 第一层卷积: Conv2d(2, 2, kernel_size=(3, 3), padding=(1, 1))")
    print("3. 批归一化: BatchNorm2d(2)")
    print("4. LeakyReLU激活: negative_slope=0.3")
    print("5. Flatten展平")
    print("6. 全连接编码层: Linear(in_features=2048, out_features=256)")
    print("7. 编码维度: 256")
    print()
    
    print("关键参数:")
    print("- encoded_dim: 256 (vs 512)")
    print("- img_channels: 2")
    print("- img_height: 32")
    print("- img_width: 32")

def compare_encoders():
    """对比两个编码器的主要差异"""
    print("\n🔄 编码器对比分析")
    print("=" * 50)
    
    differences = [
        ("编码维度", "512", "256", "QCNN01_gpu.py使用更小的编码维度"),
        ("激活函数", "linear", "LeakyReLU(0.3)", "QCNN01_gpu.py添加了非线性激活"),
        ("中间层", "ResNet残差块", "简单BN+激活", "架构复杂度不同"),
        ("框架", "TensorFlow/Keras", "PyTorch", "深度学习框架不同"),
        ("训练方式", "完整的自编码器", "编码器+量子解码器", "目标任务不同")
    ]
    
    print("主要差异对比:")
    print("属性          | CsiNet_train.py | QCNN01_gpu.py | 说明")
    print("-" * 65)
    for attr, csinet_val, qcnn_val, note in differences:
        print(f"{attr:<12} | {csinet_val:<15} | {qcnn_val:<13} | {note}")

def main():
    analyze_csinet_train_encoder()
    analyze_qcnn01_gpu_encoder()
    compare_encoders()
    
    print("\n📝 结论:")
    print("• 两个编码器都能将(2,32,32)图像压缩到低维表示")
    print("• QCNN01_gpu.py使用更小的编码维度(256 vs 512)")
    print("• 架构设计理念不同：CsiNet是完整自编码器，QCNN01是混合量子-经典网络")
    print("• 可以考虑调整QCNN01的编码维度以匹配原始CsiNet")

if __name__ == "__main__":
    main()