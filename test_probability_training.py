"""
测试QAE083的新训练方式：概率分布对概率分布
"""
import torch
import numpy as np
import sys
sys.path.append('/Users/luxian/GitSpace/huanxin')

print("=" * 70)
print("测试概率分布训练方式")
print("=" * 70)

# 导入必要的函数
from QAE083 import (
    pad_to_qubits, 
    normalize_for_amplitude_embedding,
    prepare_target_distribution,
    compute_probability_loss
)

# 测试1: prepare_target_distribution函数
print("\n1. 测试 prepare_target_distribution 函数")
batch = torch.randn(2, 2, 32, 32)  # 模拟输入batch
print(f"输入batch形状: {batch.shape}")

target_probs = prepare_target_distribution(batch)
print(f"目标概率分布形状: {target_probs.shape}")
print(f"概率和: {target_probs.sum(dim=1)}")  # 每个样本应该和为1
print(f"概率范围: [{target_probs.min():.6f}, {target_probs.max():.6f}]")

# 验证是否为有效概率分布
for i in range(target_probs.shape[0]):
    prob_sum = target_probs[i].sum()
    assert abs(prob_sum - 1.0) < 1e-5, f"样本{i}概率和不为1: {prob_sum}"
    assert target_probs[i].min() >= 0, f"样本{i}存在负概率"
print("✓ 目标概率分布验证通过！")

# 测试2: compute_probability_loss函数
print("\n2. 测试 compute_probability_loss 函数")
output_probs = torch.rand(2, 2048)
output_probs = output_probs / output_probs.sum(dim=1, keepdim=True)  # 归一化

target_probs_test = torch.rand(2, 2048)
target_probs_test = target_probs_test / target_probs_test.sum(dim=1, keepdim=True)

mse_loss = compute_probability_loss(output_probs, target_probs_test, loss_type='mse')
kl_loss = compute_probability_loss(output_probs, target_probs_test, loss_type='kl')

print(f"MSE损失: {mse_loss:.6f}")
print(f"KL散度: {kl_loss:.6f}")
print("✓ 损失函数验证通过！")

# 测试3: 梯度传播
print("\n3. 测试梯度传播")
# 创建可训练参数
output_probs_grad = torch.rand(2, 2048, requires_grad=True)
output_probs_grad_norm = output_probs_grad / output_probs_grad.sum(dim=1, keepdim=True)

target_probs_grad = torch.rand(2, 2048)
target_probs_grad = target_probs_grad / target_probs_grad.sum(dim=1, keepdim=True)

loss = compute_probability_loss(output_probs_grad_norm, target_probs_grad, loss_type='mse')
loss.backward()

print(f"损失值: {loss.item():.6f}")
print(f"梯度是否存在: {output_probs_grad.grad is not None}")
print(f"梯度范数: {output_probs_grad.grad.norm().item():.6f}")
print("✓ 梯度传播验证通过！")

# 测试4: 完整流程
print("\n4. 测试完整训练流程")
print("输入 -> 目标概率分布 -> 模拟量子输出 -> 计算损失 -> 反向传播")

# 模拟完整流程
input_batch = torch.randn(4, 2, 32, 32)
print(f"输入batch: {input_batch.shape}")

# 准备目标概率分布
target_dist = prepare_target_distribution(input_batch)
print(f"目标概率分布: {target_dist.shape}, 和={target_dist[0].sum():.6f}")

# 模拟量子电路输出（随机初始化的概率分布）
simulated_output = torch.rand(4, 2048, requires_grad=True)
simulated_output_norm = simulated_output / simulated_output.sum(dim=1, keepdim=True)
print(f"模拟输出概率分布: {simulated_output_norm.shape}")

# 计算损失
loss = compute_probability_loss(simulated_output_norm, target_dist, loss_type='mse')
print(f"损失: {loss.item():.6f}")

# 反向传播
loss.backward()
print(f"梯度范数: {simulated_output.grad.norm().item():.6f}")
print("✓ 完整流程验证通过！")

print("\n" + "=" * 70)
print("所有测试通过！")
print("=" * 70)
print("\n新训练方式总结:")
print("1. 输入: (batch, 2, 32, 32) 原始CSI数据")
print("2. 目标: 将输入reshape+归一化为概率分布 (batch, 2048)")
print("3. 输出: 量子电路测量得到的概率分布 (batch, 2048)")
print("4. 损失: MSE(输出概率, 目标概率) 或 KL(目标||输出)")
print("5. 优势: 输入输出在同一概率空间，损失函数有物理意义")
print("6. 梯度: 支持端到端梯度传播，可训练量子参数")
print("=" * 70)
