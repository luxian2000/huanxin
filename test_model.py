import torch
import numpy as np
import pennylane as qml
import os
from QAE_08 import qae_circuit, validate_model, load_data  # 导入数据加载函数

# 加载训练好的模型参数
try:
    enc_params = torch.load('model_parameters/final_qae_encoder_weights.pt')
    dec_params = torch.load('model_parameters/final_qae_decoder_weights.pt')
    print("模型参数加载成功")
except Exception as e:
    print(f"加载模型参数失败: {e}")
    exit()

# 加载测试数据（使用与训练相同的逻辑）
data_30 = load_data()  # shape=(80000, 2560)

# 数据集分割（与训练脚本相同）
TOTAL_SAMPLES = data_30.shape[0]
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

train_size = int(TOTAL_SAMPLES * TRAIN_RATIO)
val_size = int(TOTAL_SAMPLES * VAL_RATIO)
test_size = TOTAL_SAMPLES - train_size - val_size

test_data = data_30[train_size + val_size:]

print(f"测试集大小: {len(test_data)} 个样本")

def test_model(test_samples, enc_params, dec_params, n_test=100):
    """
    测试训练好的 QAE 模型
    """
    fidelities = []
    losses = []

    for i, sample in enumerate(test_samples[:n_test]):
        if isinstance(sample, np.ndarray):
            sample = torch.from_numpy(sample).float()

        # 计算保真度
        fid = qae_circuit(sample, enc_params, dec_params)
        fidelities.append(fid.item())

        # 计算损失 (1 - fidelity)
        loss = 1.0 - fid
        losses.append(loss.item())

        if (i + 1) % 10 == 0:
            print(f"测试样本 {i+1}: 保真度 = {fid.item():.6f}, 损失 = {loss.item():.6f}")

    avg_fidelity = np.mean(fidelities)
    avg_loss = np.mean(losses)

    print("\n测试结果:")
    print(f"平均保真度: {avg_fidelity:.6f}")
    print(f"平均损失: {avg_loss:.6f}")
    print(f"测试样本数: {len(fidelities)}")

    return avg_fidelity, avg_loss

# 运行测试
if __name__ == "__main__":
    print("开始测试训练好的 QAE 模型...")
    avg_fid, avg_loss = test_model(test_data, enc_params, dec_params, n_test=500)  # 测试前500个样本
    print("测试完成！")

# 示例调用（需要提供 test_data）
# test_data = ...  # 从 pg08_QAE.py 中获取
# avg_fid, avg_loss = test_model(test_data, enc_params, dec_params)

print("请确保 test_data 已定义，然后取消注释测试调用。")
print("或者在 pg08_QAE.py 中添加测试函数并调用。")