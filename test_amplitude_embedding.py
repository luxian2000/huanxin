import torch
import pennylane as qml
import numpy as np

# 测试振幅嵌入的梯度
print("测试振幅嵌入梯度...")

DEV = qml.device("default.qubit", wires=8)  # 使用默认设备

@qml.qnode(DEV, interface="torch")
def amplitude_test(vec):
    qml.AmplitudeEmbedding(vec, wires=range(8), pad_with=0.0, normalize=False)  # 不使用内部归一化
    return qml.probs(wires=range(8))

# 创建测试向量
test_vec = torch.randn(256, requires_grad=True)
test_vec.retain_grad()

# 归一化
norm_vec = test_vec / torch.norm(test_vec)

print(f"测试向量形状: {test_vec.shape}")
print(f"归一化向量范数: {torch.norm(norm_vec):.6f}")

try:
    # 创建目标分布
    target = torch.randn(256)
    target = target / torch.norm(target)  # 归一化
    target = torch.abs(target) ** 2  # 转换为概率分布
    target = target / torch.sum(target)  # 确保和为1

    output = amplitude_test(norm_vec)

    # 使用MSE loss
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()

    print(f"输出形状: {output.shape}")
    print(f"目标形状: {target.shape}")
    print(f"损失值: {loss.item():.6f}")
    print(f"梯度是否存在: {test_vec.grad is not None}")
    if test_vec.grad is not None:
        print(f"梯度范数: {torch.norm(test_vec.grad):.6f}")
        print(f"梯度最大值: {torch.max(torch.abs(test_vec.grad)):.6f}")
        print(f"梯度最小值: {torch.min(torch.abs(test_vec.grad)):.6f}")

except Exception as e:
    print(f"测试失败: {e}")
    import traceback
    traceback.print_exc()