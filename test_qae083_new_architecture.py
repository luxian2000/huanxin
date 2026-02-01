"""
测试QAE083的新架构：8量子比特编码 + 11量子比特ansatz + 2048维概率输出
"""
import torch
import torch.nn as nn
import pennylane as qml
import numpy as np

# 导入必要的组件
import sys
sys.path.append('/Users/luxian/GitSpace/huanxin')

print("=" * 70)
print("测试QAE083新架构")
print("=" * 70)

# 测试参数
print("\n1. 测试量子设备配置...")
DEV = qml.device("lightning.qubit", wires=11)
print(f"量子设备: {DEV.name}, 量子比特数: 11")

# 测试量子电路的形状
print("\n2. 测试量子解码器参数形状...")
dec_shape = qml.StronglyEntanglingLayers.shape(n_layers=4, n_wires=11)
print(f"量子解码器参数形状: {dec_shape}")
print(f"参数总数: {np.prod(dec_shape)}")

# 测试振幅编码和概率测量
print("\n3. 测试量子电路...")

def pad_to_qubits(vec, n_qubits):
    """填充向量到2^n_qubits维度"""
    target_len = 2 ** n_qubits
    if isinstance(vec, torch.Tensor):
        if len(vec) < target_len:
            return torch.nn.functional.pad(vec, (0, target_len - len(vec)))
        return vec[:target_len]
    else:
        if len(vec) < target_len:
            return np.pad(vec, (0, target_len - len(vec)), mode='constant')
        return vec[:target_len]

def normalize_for_amplitude_embedding(vec):
    """归一化向量用于幅度嵌入"""
    if isinstance(vec, torch.Tensor):
        vec = vec.detach()
    norm = torch.norm(vec) if isinstance(vec, torch.Tensor) else np.linalg.norm(vec)
    if norm < 1e-10:
        return vec
    return vec / norm

@qml.qnode(DEV, interface="torch")
def test_quantum_circuit(encoded_vec, dec_params):
    """测试量子电路"""
    # 8比特振幅编码
    encoded_padded = pad_to_qubits(encoded_vec, 8)
    encoded_normalized = normalize_for_amplitude_embedding(encoded_padded)
    
    if isinstance(encoded_normalized, torch.Tensor):
        encoded_normalized = encoded_normalized.detach().numpy()
    
    qml.AmplitudeEmbedding(encoded_normalized, wires=range(8), 
                          pad_with=0.0, normalize=True)
    
    # 11比特ansatz
    qml.StronglyEntanglingLayers(weights=dec_params, wires=range(11))
    
    # 计算基测量，返回2048维概率
    return qml.probs(wires=range(11))

# 创建测试数据
test_encoded = torch.randn(256)
test_params = torch.rand(dec_shape) * 0.1

print(f"输入编码向量形状: {test_encoded.shape}")
print(f"量子参数形状: {test_params.shape}")

# 运行量子电路
result = test_quantum_circuit(test_encoded, test_params)
print(f"量子电路输出形状: {result.shape}")
print(f"输出是否为概率分布: sum={result.sum():.6f}, min={result.min():.6f}, max={result.max():.6f}")

# 验证是否为有效的概率分布
assert result.shape[0] == 2048, f"输出维度错误: {result.shape[0]} != 2048"
assert abs(result.sum() - 1.0) < 1e-5, f"概率和不为1: {result.sum()}"
assert result.min() >= 0, f"存在负概率: {result.min()}"
print("✓ 量子电路输出验证通过！")

# 测试完整流程
print("\n4. 测试完整编码-解码流程...")

# 模拟CsiNet编码器输出
batch_size = 2
encoded_batch = torch.randn(batch_size, 256)

# 批处理量子解码
outputs = []
for i in range(batch_size):
    quantum_probs = test_quantum_circuit(encoded_batch[i], test_params)
    outputs.append(quantum_probs.unsqueeze(0))

result_batch = torch.cat(outputs, dim=0)
print(f"批处理输入形状: {encoded_batch.shape}")
print(f"批处理输出形状: {result_batch.shape}")

# Reshape到图像格式
img_result = result_batch.view(batch_size, 2, 32, 32)
print(f"图像输出形状: {img_result.shape}")

print("\n" + "=" * 70)
print("新架构测试完成！")
print("=" * 70)
print("\n架构总结:")
print("- 输入: (batch, 2, 32, 32) = 2048维")
print("- CsiNet编码器: 2048 -> 256维")
print("- 量子振幅编码: 256维 -> 8量子比特")
print("- 量子ansatz: 11量子比特 StronglyEntanglingLayers")
print("- 计算基测量: 2048个基态概率")
print("- 输出: 2048维 -> reshape到 (batch, 2, 32, 32)")
print("\n关键优势:")
print("1. 直接从量子测量得到2048维输出，无需复杂的经典解码器")
print("2. 量子态概率分布自然对应原始数据维度")
print("3. 减少了参数量和训练复杂度")
print("=" * 70)
