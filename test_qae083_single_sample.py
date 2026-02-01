"""
测试脚本：针对QAE083_GPU训练好的模型进行单样本测试

功能：
1. 加载训练好的混合CsiNet-量子自编码器模型
2. 从测试数据中随机选择一个样本
3. 将样本输入模型，获取量子态在计算基下的概率分布
4. 计算样本的归一化目标概率分布
5. 打印和对比两个概率分布向量
"""

import os
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
import scipy.io as sio
import random

# 导入QAE083_gpu.py中的必要组件
from QAE083_gpu import (
    CsiNetEncoder, HybridCsiNetQuantumAutoencoder,
    load_csinet_data, prepare_target_distribution,
    compute_probability_loss, normalize_for_amplitude_embedding,
    pad_to_qubits, quantum_decoder_circuit
)

# GPU设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

def load_trained_model(model_dir="QAE083_gpu"):
    """加载训练好的模型"""
    print(f"正在加载训练好的模型从目录: {model_dir}")

    # 检查文件是否存在
    encoder_path = os.path.join(model_dir, "final_csinet_encoder.pt")
    decoder_path = os.path.join(model_dir, "final_quantum_decoder_weights.pt")

    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"找不到编码器权重文件: {encoder_path}")
    if not os.path.exists(decoder_path):
        raise FileNotFoundError(f"找不到量子解码器权重文件: {decoder_path}")

    # 初始化组件
    csinet_encoder = CsiNetEncoder(encoded_dim=256).to(device)

    # 加载量子解码器参数形状
    dec_shape = qml.StronglyEntanglingLayers.shape(n_layers=4, n_wires=11)
    dec_params = nn.Parameter(torch.rand(dec_shape, device=device))

    # 加载训练好的权重
    csinet_encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    dec_params.data = torch.load(decoder_path, map_location=device)

    print("✅ 模型权重加载成功！")

    # 创建混合模型
    model = HybridCsiNetQuantumAutoencoder(csinet_encoder, dec_params)
    model.eval()

    return model

def test_single_sample(model, test_data):
    """测试单个随机样本"""
    print("\n" + "=" * 80)
    print("🎯 单样本测试开始")
    print("=" * 80)

    # 随机选择一个测试样本
    random_idx = random.randint(0, len(test_data) - 1)
    sample = test_data[random_idx]  # (2, 32, 32)

    print(f"📊 选择的测试样本索引: {random_idx}")
    print(f"📏 样本形状: {sample.shape}")
    print(f"📈 样本数据范围: [{sample.min():.6f}, {sample.max():.6f}]")

    # 转换为tensor并移动到设备
    sample_tensor = torch.from_numpy(sample).float().unsqueeze(0).to(device)  # (1, 2, 32, 32)

    # 模型推理
    with torch.no_grad():
        model_output = model(sample_tensor)  # (1, 2048) 概率分布

    # 获取模型输出概率分布
    output_probs = model_output.squeeze(0).cpu().numpy()  # (2048,)

    # 计算样本的归一化目标概率分布
    target_probs = prepare_target_distribution(sample_tensor).squeeze(0).cpu().numpy()  # (2048,)

    print("\n🔬 模型输出结果:")
    print(f"  • 输出形状: {output_probs.shape}")
    print(f"  • 概率分布范围: [{output_probs.min():.6f}, {output_probs.max():.6f}]")
    print(f"  • 概率和 (应接近1.0): {output_probs.sum():.6f}")
    print(f"  • 非零概率数: {np.count_nonzero(output_probs)}")
    print(f"  • 最大概率值: {output_probs.max():.6f} (索引: {np.argmax(output_probs)})")

    print("\n🎯 样本归一化结果:")
    print(f"  • 目标形状: {target_probs.shape}")
    print(f"  • 概率分布范围: [{target_probs.min():.6f}, {target_probs.max():.6f}]")
    print(f"  • 概率和 (应为1.0): {target_probs.sum():.6f}")
    print(f"  • 非零概率数: {np.count_nonzero(target_probs)}")
    print(f"  • 最大概率值: {target_probs.max():.6f} (索引: {np.argmax(target_probs)})")

    # 对比分析
    print("\n📊 向量对比分析:")    # 计算各种距离度量
    mse_distance = np.mean((output_probs - target_probs) ** 2)
    kl_divergence = np.sum(target_probs * np.log((target_probs + 1e-10) / (output_probs + 1e-10)))
    cross_entropy = -np.sum(target_probs * np.log(output_probs + 1e-10))
    jsd_divergence = 0.5 * np.sum(target_probs * np.log((target_probs + 1e-10) / ((target_probs + output_probs)/2 + 1e-10))) + \
                     0.5 * np.sum(output_probs * np.log((output_probs + 1e-10) / ((target_probs + output_probs)/2 + 1e-10)))
    hellinger_distance = np.linalg.norm(np.sqrt(target_probs + 1e-10) - np.sqrt(output_probs + 1e-10)) / np.sqrt(2)

    print(f"  • MSE距离: {mse_distance:.8f}")
    print(f"  • KL散度 (目标||输出): {kl_divergence:.8f}")
    print(f"  • 交叉熵损失: {cross_entropy:.8f}")
    print(f"  • JSD散度: {jsd_divergence:.8f}")
    print(f"  • Hellinger距离: {hellinger_distance:.8f}")

    # 相关性分析
    correlation = np.corrcoef(output_probs, target_probs)[0, 1]
    print(f"  • Pearson相关系数: {correlation:.6f}")

    # Top-K 概率分析
    k = 10
    output_top_k_indices = np.argsort(output_probs)[-k:][::-1]
    target_top_k_indices = np.argsort(target_probs)[-k:][::-1]

    print(f"\n🔝 Top-{k} 概率分析:")
    print(f"  • 模型输出 Top-{k} 索引: {output_top_k_indices}")
    print(f"  • 目标分布 Top-{k} 索引: {target_top_k_indices}")
    print(f"  • Top-{k} 重叠数: {len(set(output_top_k_indices) & set(target_top_k_indices))}")

    # 打印部分概率向量（前20个和后20个）
    print("\n📋 概率分布向量详情 (前20个和后20个):")
    print("模型输出概率分布:")
    print(f"  前20: {output_probs[:20]}")
    print(f"  后20: {output_probs[-20:]}")

    print("目标概率分布:")
    print(f"  前20: {target_probs[:20]}")
    print(f"  后20: {target_probs[-20:]}")

    # 保存测试结果
    test_result = {
        "sample_index": random_idx,
        "sample_shape": sample.shape,
        "sample_range": [float(sample.min()), float(sample.max())],
        "output_probs": output_probs.tolist(),
        "target_probs": target_probs.tolist(),
        "metrics": {
            "mse_distance": float(mse_distance),
            "kl_divergence": float(kl_divergence),
            "cross_entropy": float(cross_entropy),
            "jsd_divergence": float(jsd_divergence),
            "hellinger_distance": float(hellinger_distance),
            "correlation": float(correlation)
        },
        "top_k_overlap": int(len(set(output_top_k_indices) & set(target_top_k_indices)))
    }

    torch.save(test_result, "QAE083_gpu/single_sample_test_result.pt")
    print("\n💾 测试结果已保存到: QAE083_gpu/single_sample_test_result.pt")
    print("=" * 80)

    return output_probs, target_probs

def main():
    """主函数"""
    print("=" * 80)
    print("🧪 QAE083_GPU 单样本测试脚本")
    print("=" * 80)

    try:
        # 加载测试数据
        print("正在加载测试数据...")
        _, _, test_data = load_csinet_data()
        print(f"✅ 测试数据加载成功: {test_data.shape}")

        # 加载训练好的模型
        model = load_trained_model()

        # 执行单样本测试
        output_probs, target_probs = test_single_sample(model, test_data)

        print("\n🎉 测试完成！")
        print("分析了模型输出概率分布与目标概率分布的差异")

    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()