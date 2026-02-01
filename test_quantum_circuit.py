import torch
import pennylane as qml
import numpy as np

# 测试量子电路的基本功能
print("测试量子电路基本功能...")

# 创建设备
DEV = qml.device("lightning.qubit", wires=11)

# 定义测试电路（使用autograd接口）
@qml.qnode(DEV, interface="autograd")
def test_circuit_full(encoded_vec, dec_params):
    """完整的测试量子电路"""
    # 归一化
    encoded_padded = pad_to_qubits(encoded_vec, 8)
    encoded_normalized = normalize_for_amplitude_embedding(encoded_padded)

    # 振幅编码
    qml.AmplitudeEmbedding(encoded_normalized, wires=range(8),
                          pad_with=0.0, normalize=True)

    # 参数化层
    qml.StronglyEntanglingLayers(weights=dec_params, wires=range(11))

    # 测量
    return qml.probs(wires=range(11))
    """完整的测试量子电路"""
    # 归一化
    encoded_padded = pad_to_qubits(encoded_vec, 8)
    encoded_normalized = normalize_for_amplitude_embedding(encoded_padded)

    # 振幅编码
    qml.AmplitudeEmbedding(encoded_normalized, wires=range(8),
                          pad_with=0.0, normalize=True)

    # 参数化层
    qml.StronglyEntanglingLayers(weights=dec_params, wires=range(11))

    # 测量
    return qml.probs(wires=range(11))

def pad_to_qubits(vec, n_qubits):
    """填充向量到2^n_qubits维度"""
    target_len = 2 ** n_qubits
    if isinstance(vec, torch.Tensor):
        if len(vec) < target_len:
            return torch.nn.functional.pad(vec, (0, target_len - len(vec)))
        return vec[:target_len]
    else:
        if len(vec) < target_len:
            return np.pad(vec, (0, target_len - len(vec)))
        return vec[:target_len]

def normalize_for_amplitude_embedding(vec):
    """归一化向量用于幅度嵌入"""
    if isinstance(vec, torch.Tensor):
        vec = vec.detach()
    norm = torch.norm(vec) if isinstance(vec, torch.Tensor) else np.linalg.norm(vec)
    if norm < 1e-10:
        return vec
    return vec / norm

# 测试参数
encoded_vec = torch.randn(256)  # 256维输入
dec_shape = qml.StronglyEntanglingLayers.shape(n_layers=4, n_wires=11)
dec_params = torch.rand(dec_shape) * 2 * 3.14159 - 3.14159  # [-π, π]

print(f"输入向量形状: {encoded_vec.shape}")
print(f"解码器参数形状: {dec_params.shape}")

# 测试参数
encoded_vec = np.random.randn(256)  # 256维输入
dec_shape = qml.StronglyEntanglingLayers.shape(n_layers=4, n_wires=11)
dec_params = np.random.rand(*dec_shape) * 2 * 3.14159 - 3.14159  # [-π, π]

print(f"输入向量形状: {encoded_vec.shape}")
print(f"解码器参数形状: {dec_params.shape}")

try:
    # 前向传播测试
    print("执行前向传播...")
    result = test_circuit_full(encoded_vec, dec_params)
    print(f"输出形状: {result.shape}")
    print(f"输出和: {np.sum(result):.6f}")
    print("✅ 前向传播成功")

    # 梯度测试
    print("测试梯度计算...")
    import autograd.numpy as anp
    from autograd import grad

    def loss_fn(params):
        output = test_circuit_full(encoded_vec, params)
        return anp.sum(output)

    grad_fn = grad(loss_fn)
    gradients = grad_fn(dec_params)

    print(f"梯度形状: {gradients.shape}")
    print(f"梯度范数: {anp.linalg.norm(gradients):.6f}")
    print("✅ 梯度计算测试完成")

except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()