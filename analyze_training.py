import torch
import matplotlib.pyplot as plt
import numpy as np

# 加载训练历史
try:
    history = torch.load('model_parameters/qae_training_history.pt')
    print("训练历史键:", list(history.keys()))
except Exception as e:
    print(f"加载训练历史失败: {e}")
    exit()

# 提取数据
epoch_losses = history.get('epoch_losses', [])
val_fidelity = history.get('val_fidelity', [])
batch_losses = history.get('batch_losses', [])

if epoch_losses:
    epochs = [h['epoch'] for h in epoch_losses]
    losses = [h['avg_loss'] for h in epoch_losses]
    print(f"训练了 {len(epochs)} 个 epoch")
    print(f"最终 epoch 损失: {losses[-1]:.6f}")
    print(f"损失变化: 初始 {losses[0]:.6f} -> 最终 {losses[-1]:.6f}")

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, 'b-', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('QAE Training Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss_curve.png')
    plt.show()

if val_fidelity:
    val_epochs = [h['epoch'] for h in val_fidelity]
    val_fids = [h['val_fidelity'] for h in val_fidelity]
    print(f"验证集保真度: 初始 {val_fids[0]:.6f} -> 最终 {val_fids[-1]:.6f}")

    # 绘制保真度曲线
    plt.figure(figsize=(10, 6))
    plt.plot(val_epochs, val_fids, 'r-', label='Validation Fidelity')
    plt.xlabel('Epoch')
    plt.ylabel('Fidelity')
    plt.title('QAE Validation Fidelity over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('validation_fidelity_curve.png')
    plt.show()

# 分析参数变化
try:
    initial_enc = torch.load('model_parameters/initial_quantum_encoder_weights.pt')
    final_enc = torch.load('model_parameters/final_qae_encoder_weights.pt')
    initial_dec = torch.load('model_parameters/initial_quantum_decoder_weights.pt')
    final_dec = torch.load('model_parameters/final_qae_decoder_weights.pt')

    enc_change = torch.norm(final_enc - initial_enc).item()
    dec_change = torch.norm(final_dec - initial_dec).item()

    print(f"编码器参数变化范数: {enc_change:.6f}")
    print(f"解码器参数变化范数: {dec_change:.6f}")

except Exception as e:
    print(f"加载参数失败: {e}")

# 分析收敛性
if len(losses) > 1:
    loss_diff = np.diff(losses)
    if np.mean(loss_diff[-5:]) < 0.001:  # 最后5个epoch平均变化小于阈值
        print("模型似乎已收敛")
    else:
        print("模型仍在收敛中")

print("分析完成。")