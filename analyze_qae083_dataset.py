#!/usr/bin/env python3
"""
QAE083数据集分析脚本

分析QAE083.py中使用的CSI数据集的详细形状信息，
包括原始数据、训练/验证/测试集划分等。
"""

import os
import numpy as np
import sys

def analyze_qae083_dataset():
    """分析QAE083.py使用的数据集"""
    
    print("=" * 60)
    print("QAE083 数据集形状分析")
    print("=" * 60)
    
    # QAE083.py中定义的数据路径
    data_paths = [
        "/Users/luxian/DataSpace/csi_cmri/CSI_channel_30km.npy",
        "./CSI_channel_30km.npy",
        "../DataSpace/csi_cmri/CSI_channel_30km.npy",
        "../../DataSpace/csi_cmri/CSI_channel_30km.npy",
    ]
    
    data_file = None
    for path in data_paths:
        if os.path.exists(path):
            data_file = path
            break
    
    if data_file is None:
        print("❌ 未找到数据文件！")
        print("请确保以下路径之一存在数据文件：")
        for path in data_paths:
            print(f"  - {path}")
        return False
    
    print(f"✅ 找到数据文件: {data_file}")
    
    try:
        # 加载数据
        print("\n正在加载数据...")
        data_30 = np.load(data_file)
        print("✅ 数据加载成功！")
        
        # 基本形状信息
        print(f"\n📊 原始数据形状分析:")
        print(f"   数据形状: {data_30.shape}")
        print(f"   数据类型: {data_30.dtype}")
        print(f"   数据大小: {data_30.size:,} 个元素")
        print(f"   内存占用: {data_30.nbytes / (1024**2):.2f} MB")
        
        # 数值范围和统计信息
        print(f"\n📈 数据统计信息:")
        print(f"   最小值: {data_30.min():.6f}")
        print(f"   最大值: {data_30.max():.6f}")
        print(f"   平均值: {data_30.mean():.6f}")
        print(f"   标准差: {data_30.std():.6f}")
        
        # QAE083.py中的数据划分参数
        print(f"\n📋 QAE083.py 数据划分参数:")
        TOTAL_SAMPLES = data_30.shape[0]
        TRAIN_RATIO = 0.70
        VAL_RATIO = 0.15
        TEST_RATIO = 0.15
        
        train_size = int(TOTAL_SAMPLES * TRAIN_RATIO)
        val_size = int(TOTAL_SAMPLES * VAL_RATIO)
        test_size = TOTAL_SAMPLES - train_size - val_size
        
        print(f"   总样本数: {TOTAL_SAMPLES:,}")
        print(f"   训练集比例: {TRAIN_RATIO*100:.1f}% → {train_size:,} 个样本")
        print(f"   验证集比例: {VAL_RATIO*100:.1f}% → {val_size:,} 个样本")
        print(f"   测试集比例: {TEST_RATIO*100:.1f}% → {test_size:,} 个样本")
        
        # 实际数据划分
        print(f"\n🔍 实际数据划分:")
        train_data = data_30[:train_size]
        val_data = data_30[train_size:train_size + val_size]
        test_data = data_30[train_size + val_size:]
        
        print(f"   训练集形状: {train_data.shape}")
        print(f"   验证集形状: {val_data.shape}")
        print(f"   测试集形状: {test_data.shape}")
        
        # 检查数据维度
        input_dim = data_30.shape[1] if len(data_30.shape) > 1 else data_30.shape[0]
        print(f"\n📐 输入维度分析:")
        print(f"   输入维度: {input_dim}")
        print(f"   是否符合QAE083要求 (2560维): {'✅ 是' if input_dim == 2560 else '❌ 否'}")
        
        if input_dim != 2560:
            print(f"   ⚠️  注意: QAE083.py期望2560维输入，但当前数据为{input_dim}维")
        
        # 网络参数相关
        ENCODED_DIM = 256
        N_LAYERS = 4
        DATA_QUBITS = int(np.ceil(np.log2(ENCODED_DIM)))
        
        print(f"\n⚙️  QAE083网络参数:")
        print(f"   编码维度: {ENCODED_DIM}")
        print(f"   量子比特数: {DATA_QUBITS}")
        print(f"   量子层数: {N_LAYERS}")
        print(f"   量子参数数量: {DATA_QUBITS * N_LAYERS * 3}")
        
        # 数据质量检查
        print(f"\n🔍 数据质量检查:")
        nan_count = np.isnan(data_30).sum()
        inf_count = np.isinf(data_30).sum()
        
        print(f"   NaN值数量: {nan_count}")
        print(f"   无穷值数量: {inf_count}")
        print(f"   数据完整性: {'✅ 良好' if (nan_count == 0 and inf_count == 0) else '❌ 存在问题'}")
        
        # 样本示例
        print(f"\n📝 样本示例 (前3个样本的前10个特征):")
        for i in range(min(3, len(data_30))):
            sample = data_30[i][:10] if len(data_30.shape) > 1 else data_30[:10]
            print(f"   样本 {i+1}: {sample}")
        
        print(f"\n" + "=" * 60)
        print("✅ 数据集分析完成！")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ 数据分析出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_with_csinet():
    """与CsiNet的数据格式进行比较"""
    print(f"\n🔄 与CsiNet数据格式比较:")
    print(f"   QAE083: 直接处理 {2560} 维向量")
    print(f"   CsiNet: 将 {2048} 维数据reshape为 (2, 32, 32) 图像格式")
    print(f"   差异: QAE083使用更高维度的数据 ({2560} vs {2048})")

if __name__ == "__main__":
    success = analyze_qae083_dataset()
    if success:
        compare_with_csinet()
    else:
        sys.exit(1)