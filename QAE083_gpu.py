"""
QAE083_gpu: 混合经典-量子编码解码神经网络 (使用CsiNet编码器，encoded_dim=256) - GPU版本

网络架构：
1. 经典编码器：使用CsiNet卷积编码器将(2,32,32)图像压缩到256维
2. 量子态映射：将256维经典向量映射为量子态（幅度嵌入）
3. 量子解码器：使用参数化量子线路解码量子态，恢复到2048维
4. 经典解码器：将2048维向量重塑为(2,32,32)图像格式

数据流：
输入(2,32,32) -> CsiNet编码器 -> 256维 -> 量子态 -> 量子解码器 -> 2048维 -> (2,32,32)

GPU训练：使用PennyLane + PyTorch在GPU上训练
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
import csv
import scipy.io as sio

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# GPU设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 创建输出目录
OUTPUT_DIR = "QAE083_gpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 图像参数 (匹配CsiNet)
img_height = 32
img_width = 32
img_channels = 2
img_total = img_height * img_width * img_channels  # 2048
encoded_dim = 256  # 压缩率1/8

def load_csinet_data():
    """加载CsiNet格式的.mat数据文件并reshape为图像格式"""
    data_paths = {
        'train': "/home/luxian/csinet/data/DATA_Htrainin.mat",
        'val': "/home/luxian/csinet/data/DATA_Hvalin.mat",
        'test': "/home/luxian/csinet/data/DATA_Htestin.mat"
    }

    print("正在加载CsiNet数据集...")
    datasets = {}

    for key, path in data_paths.items():
        try:
            mat_data = sio.loadmat(path)
            x = mat_data['HT'].astype('float32')
            # 归一化到[0,1]
            x = (x - x.min()) / (x.max() - x.min())
            # reshape为图像格式
            x = np.reshape(x, (len(x), img_channels, img_height, img_width))
            datasets[key] = x
            print(f"{key}数据加载成功: {x.shape}")
        except Exception as e:
            raise FileNotFoundError(f"无法加载{key}数据 {path}: {e}")

    print(f"数据范围: [{datasets['train'].min():.4f}, {datasets['train'].max():.4f}]")
    return datasets['train'], datasets['val'], datasets['test']

# ============================================================================
# 1. CsiNet编码器（基于Keras实现转换为PyTorch）
# ============================================================================

class CsiNetEncoder(nn.Module):
    """CsiNet编码器：将(2,32,32)图像压缩到256维向量"""

    def __init__(self, encoded_dim=256):  # 修改：默认参数改为256
        super(CsiNetEncoder, self).__init__()
        self.encoded_dim = encoded_dim

        # 第一层卷积
        self.conv1 = nn.Conv2d(2, 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(2)
        self.relu1 = nn.LeakyReLU(0.3)

        # 全连接层进行压缩
        self.flatten = nn.Flatten()
        self.dense_encoded = nn.Linear(img_total, encoded_dim)

    def forward(self, x):
        """
        前向传播
        x: (batch_size, 2, 32, 32)
        """
        # 第一层卷积 + BN + LeakyReLU
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # flatten并压缩到编码维度
        x = self.flatten(x)
        encoded = self.dense_encoded(x)

        # 确保输出是有效的实数向量（用于量子振幅嵌入）
        encoded = torch.clamp(encoded, min=1e-7, max=1e7)  # 防止极端值
        encoded = torch.nan_to_num(encoded, nan=0.0, posinf=1.0, neginf=0.0)  # 处理NaN和无穷大

        return encoded

# ============================================================================
# 2. 量子态映射和量子解码器
# ============================================================================

def normalize_for_amplitude_embedding(vec):
    """归一化向量用于幅度嵌入（保持梯度流）"""
    # 确保向量是实数且有效
    if isinstance(vec, torch.Tensor):
        vec = torch.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=0.0)
        vec = torch.clamp(vec, min=0.0, max=1e7)  # 确保非负
        norm = torch.norm(vec, p=2)
    else:
        vec = np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=0.0)
        vec = np.clip(vec, 0.0, 1e7)
        norm = np.linalg.norm(vec, ord=2)

    # 如果范数太小，返回均匀分布
    if norm < 1e-10:
        if isinstance(vec, torch.Tensor):
            return torch.ones_like(vec) / torch.sqrt(torch.tensor(float(len(vec))))
        else:
            return np.ones_like(vec) / np.sqrt(len(vec))

    return vec / norm

def pad_to_qubits(vec, n_qubits):
    """填充向量到2^n_qubits维度"""
    target_len = 2 ** n_qubits
    if isinstance(vec, torch.Tensor):
        # 确保向量是有效的
        vec = torch.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=0.0)
        vec = torch.clamp(vec, min=0.0, max=1e7)
        if len(vec) < target_len:
            return torch.nn.functional.pad(vec, (0, target_len - len(vec)))
        return vec[:target_len]
    else:
        vec = np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=0.0)
        vec = np.clip(vec, 0.0, 1e7)
        if len(vec) < target_len:
            return np.pad(vec, (0, target_len - len(vec)))
        return vec[:target_len]

# ============================================================================
# 2. 量子态映射和量子解码器
# ============================================================================

# Quantum device - GPU版本
DEV = qml.device("default.qubit", wires=11)

@qml.qnode(DEV, interface="torch")
def quantum_decoder_circuit(encoded_vec, dec_params):
    """
    量子自编码器解码器电路
    
    Args:
        encoded_vec: 经典编码器输出的256维向量
        dec_params: 量子解码器参数
        
    Returns:
        2048维概率分布（对应11个量子比特在计算基下的测量概率）
    """
    # 1. 将经典编码向量嵌入为量子态（使用前8个量子比特）
    encoded_padded = pad_to_qubits(encoded_vec, 8)
    encoded_normalized = normalize_for_amplitude_embedding(encoded_padded)
    
    # 额外确保归一化（双重保险）
    encoded_normalized = encoded_normalized / (torch.norm(encoded_normalized, p=2) + 1e-10)
    
    # 在前8个量子比特上进行振幅编码
    qml.AmplitudeEmbedding(encoded_normalized, wires=range(8), 
                          pad_with=0.0, normalize=True)
    
    # 后3个量子比特初始化为|0>态（默认已是|0>，无需额外操作）
    
    # 2. 应用参数化量子解码层（作用于全部11个量子比特）
    qml.StronglyEntanglingLayers(weights=dec_params, wires=range(11))
    
    # 3. 计算基测量，返回2048个基态的概率分布
    return qml.probs(wires=range(11))

# ============================================================================
# 3. 完整的混合网络

class HybridCsiNetQuantumAutoencoder(nn.Module):
    """
    完整的混合CsiNet-量子自编码器

    流程：
    1. CsiNet编码器压缩图像到256维
    2. 量子态嵌入（8个量子比特）和量子解码器变换（11个量子比特）
    3. 计算基测量得到2048维概率分布（直接作为输出）
    """
    def __init__(self, csinet_encoder, dec_params):
        super(HybridCsiNetQuantumAutoencoder, self).__init__()
        self.csinet_encoder = csinet_encoder
        self.dec_params = dec_params

    def forward(self, x):
        """
        前向传播
        x: (batch_size, 2, 32, 32)
        返回: (batch_size, 2048) 概率分布
        """
        batch_size = x.shape[0]

        # 1. CsiNet编码器处理
        encoded_batch = self.csinet_encoder(x)  # (batch_size, 256)

        # 2. 量子解码（逐个样本处理）
        outputs = []
        for i in range(batch_size):
            # 获取单个编码向量
            encoded_vec = encoded_batch[i]  # (256,)

            # 量子解码（返回2048个概率值）
            quantum_probs = quantum_decoder_circuit(encoded_vec, self.dec_params)  # (2048,)
            # 确保输出在 GPU 上
            quantum_probs = quantum_probs.to(device)
            outputs.append(quantum_probs.unsqueeze(0))

        # 合并所有样本，返回概率分布
        result = torch.cat(outputs, dim=0)  # (batch_size, 2048)

        return result

# ============================================================================
# 4. 训练和测试函数
# ============================================================================

def prepare_target_distribution(batch):
    """
    将原始输入batch转换为归一化的目标概率分布

    Args:
        batch: (batch_size, 2, 32, 32) 原始输入

    Returns:
        (batch_size, 2048) 归一化的概率分布
    """
    batch_size = batch.shape[0]
    targets = []

    for i in range(batch_size):
        # 将图像展平为2048维向量
        vec = batch[i].view(-1)  # (2048,)

        # 填充到2^11=2048（这里已经是2048，无需填充）
        vec_padded = pad_to_qubits(vec, 11)

        # 归一化为概率分布
        vec_normalized = normalize_for_amplitude_embedding(vec_padded)

        # 直接使用归一化向量作为概率分布（数据已预处理为[0,1]正实数）
        prob_dist = vec_normalized

        # 归一化确保和为1
        prob_dist = prob_dist / (prob_dist.sum() + 1e-10)

        targets.append(prob_dist.unsqueeze(0))

    return torch.cat(targets, dim=0)

def save_initial_parameters(csinet_encoder, dec_params):
    """保存初始参数"""
    torch.save(csinet_encoder.state_dict(),
              f"{OUTPUT_DIR}/initial_csinet_encoder.pt")
    torch.save(dec_params,
              f"{OUTPUT_DIR}/initial_quantum_decoder_weights.pt")
    print("初始参数已保存！")

def compute_probability_loss(output_probs, target_probs, loss_type='cross_entropy'):
    """
    计算两个概率分布之间的损失

    Args:
        output_probs: 模型输出的概率分布 (batch_size, 2048)
        target_probs: 目标概率分布 (batch_size, 2048)
        loss_type: 'kl', 'mse', 'cross_entropy', 'jsd', 'hellinger' (默认交叉熵)

    Returns:
        损失值
    """
    epsilon = 1e-10  # 避免log(0)和除零

    if loss_type == 'mse':
        # 概率分布的均方误差
        return torch.mean((output_probs - target_probs) ** 2)

    elif loss_type == 'kl':
        # KL散度: KL(target || output)
        kl_div = target_probs * torch.log((target_probs + epsilon) / (output_probs + epsilon))
        return torch.mean(torch.sum(kl_div, dim=1))

    elif loss_type == 'cross_entropy':
        # 交叉熵: -sum(target * log(output))
        ce_loss = -target_probs * torch.log(output_probs + epsilon)
        return torch.mean(torch.sum(ce_loss, dim=1))

    elif loss_type == 'jsd':
        # Jensen-Shannon散度: 1/2 * KL(target||M) + 1/2 * KL(output||M)
        # 其中 M = (target + output) / 2
        M = (target_probs + output_probs) / 2
        kl_target_M = target_probs * torch.log((target_probs + epsilon) / (M + epsilon))
        kl_output_M = output_probs * torch.log((output_probs + epsilon) / (M + epsilon))
        jsd = 0.5 * torch.sum(kl_target_M + kl_output_M, dim=1)
        return torch.mean(jsd)

    elif loss_type == 'hellinger':
        # Hellinger距离: (1/√2) * ||√target - √output||_2
        sqrt_target = torch.sqrt(target_probs + epsilon)
        sqrt_output = torch.sqrt(output_probs + epsilon)
        hellinger_dist = torch.norm(sqrt_target - sqrt_output, p=2, dim=1) / torch.sqrt(torch.tensor(2.0))
        return torch.mean(hellinger_dist)

    else:
        raise ValueError(f"不支持的损失类型: {loss_type}. 支持: 'mse', 'kl', 'cross_entropy', 'jsd', 'hellinger'")

def validate_model(model, val_data, val_samples=500, loss_type='cross_entropy'):
    """验证模型，默认使用交叉熵"""
    model.eval()
    try:
        subset = torch.from_numpy(val_data[:min(val_samples, len(val_data))]).float().to(device)
        with torch.no_grad():
            outputs = model(subset)  # (batch, 2048) 概率分布
            targets = prepare_target_distribution(subset)  # (batch, 2048) 目标概率分布
            loss = compute_probability_loss(outputs, targets, loss_type=loss_type)
        model.train()
        return float(loss)
    except Exception as e:
        print(f"验证错误: {e}")
        model.train()
        return float("nan")

def format_time(seconds):
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}分{secs:.0f}秒"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}小时{int(minutes)}分"

def train_hybrid_model():
    """训练混合CsiNet-量子自编码器 - GPU版本 (继续训练)"""
    try:
        print("\n" + "=" * 80)
        print("🚀 继续QAE083混合量子自编码器训练 (GPU版本 - epoch 20-39)")
        print("=" * 80)

        # 初始化组件
        csinet_encoder = CsiNetEncoder(encoded_dim=256).to(device)
        print("📋 CsiNet编码器结构:")
        print(csinet_encoder)

        # 初始化量子解码器参数：使用11个量子比特进行ansatz变换
        dec_shape = qml.StronglyEntanglingLayers.shape(n_layers=4, n_wires=11)
        dec_params = nn.Parameter(torch.rand(dec_shape, device=device) * 2 * 3.14159 - 3.14159)  # 初始化为[-π, π]范围，直接在GPU上创建
        print(f"\n⚛️  量子解码器配置:")
        print(f"  • 参数形状: {dec_shape}")
        print(f"  • 量子比特: 11 (8个用于256维编码，11个用于ansatz和测量)")
        print(f"  • 设备: {device}")

        # 保存初始参数
        save_initial_parameters(csinet_encoder, dec_params)

        # 创建混合模型
        hybrid_model = HybridCsiNetQuantumAutoencoder(csinet_encoder, dec_params)
        print(f"\n🤖 混合模型创建完成: CsiNet编码器 + 量子解码器")

        # 优化器
        quantum_optimizer = torch.optim.Adam([dec_params], lr=0.001)
        classical_optimizer = torch.optim.Adam(csinet_encoder.parameters(), lr=0.001)

        # 训练参数
        n_epochs = 40  # 总共训练40个epoch (继续训练20个epoch)
        batch_size = 50  # 每个batch处理50个样本
        n_samples = 500  # 保持500个样本，每个epoch有10个batch (500/50=10)
        samples = torch.from_numpy(train_data[:n_samples]).float().to(device)

        # 加载之前训练的模型权重 (从epoch 19开始继续训练)
        try:
            csinet_encoder.load_state_dict(torch.load(f"{OUTPUT_DIR}/csinet_encoder_epoch_19.pt"))
            dec_params.data = torch.load(f"{OUTPUT_DIR}/quantum_decoder_epoch_19.pt")
            print("✅ 已加载epoch 19的模型权重作为起点")
        except FileNotFoundError:
            print("⚠️  未找到epoch 19的权重文件，将从头开始训练")

        # 训练历史
        training_history = {
            "epoch_losses": [],
            "val_mse": [],
            "batch_losses": [],
            "data_split_info": {
                "train_size": len(train_data),
                "val_size": len(val_data),
                "test_size": len(test_data),
                "actual_train_used": n_samples,
            },
            "network_config": {
                "encoded_dim": 256,
                "quantum_encoding_qubits": 8,
                "quantum_ansatz_qubits": 11,
                "quantum_layers": 4,
                "output_dim": 2048,
                "compression_ratio": "1/8",
                "device": str(device)
            }
        }

        # CSV文件记录
        csv_file = f"{OUTPUT_DIR}/hybrid_batch_losses.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'batch', 'loss', 'dec_params_norm'])

        print(f"\n🎯 训练配置概览:")
        print(f"  • 编码维度: 256 (压缩率 1/8)")
        print(f"  • 量子比特: 11 (8编码 + 3辅助)")
        print(f"  • 量子层数: 4")
        print(f"  • 输出维度: 2048 (概率分布)")
        print(f"  • 总epochs: {n_epochs}")
        print(f"  • 训练样本: {n_samples}")
        print(f"  • 设备: {device}")

        start_time = time.time()
        print(f"\n⏰ 继续训练开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        for epoch in range(20, n_epochs):  # 从epoch 20开始训练到epoch 39
            hybrid_model.train()
            epoch_loss = 0.0
            batch_count = 0

            # 随机打乱训练数据
            indices = torch.randperm(n_samples)
            samples_shuffled = samples[indices]

            epoch_start_time = time.time()
            batch_losses = []  # 记录本轮所有batch的损失

            for i in range(0, n_samples, batch_size):
                batch = samples_shuffled[i:i + batch_size]
                actual_batch_size = batch.shape[0]

                if actual_batch_size < 1:  # 修改条件，允许batch_size=1
                    continue

                # 清零梯度
                classical_optimizer.zero_grad()
                quantum_optimizer.zero_grad()

                # 前向传播 - 得到概率分布
                outputs = hybrid_model(batch)  # (batch_size, 2048)

                # 准备目标概率分布
                targets = prepare_target_distribution(batch)  # (batch_size, 2048)

                # 计算概率分布之间的损失（默认使用交叉熵）
                loss = compute_probability_loss(outputs, targets)

                # 反向传播
                loss.backward()

                # 记录参数范数
                dec_params_norm = torch.norm(dec_params).item()

                # 更新参数
                classical_optimizer.step()
                quantum_optimizer.step()

                current_loss = loss.item()
                epoch_loss += current_loss * actual_batch_size
                batch_count += actual_batch_size
                batch_losses.append(current_loss)

                # 记录batch loss
                training_history["batch_losses"].append({
                    "epoch": epoch,
                    "batch": i // batch_size,
                    "loss": float(current_loss),
                    "dec_params_norm": float(dec_params_norm)
                })

                # 写入CSV
                with open(csv_file, 'a', newline='') as f:
                    csv.writer(f).writerow([epoch, i // batch_size, current_loss, dec_params_norm])

                if (i // batch_size) % 10 == 0:
                    print(f"  Batch {(i//batch_size)+1:2d}/{(n_samples//batch_size):2d}: "
                          f"Loss = {current_loss:.8f}")

            if batch_count > 0:
                avg_epoch_loss = epoch_loss / batch_count
                epoch_time = time.time() - epoch_start_time
                val_mse = validate_model(hybrid_model, val_data, val_samples=200)

                training_history["epoch_losses"].append({"epoch": epoch, "avg_loss": float(avg_epoch_loss)})
                training_history["val_mse"].append({"epoch": epoch, "val_mse": float(val_mse)})

                # 保存epoch权重
                torch.save(csinet_encoder.state_dict(),
                          f"{OUTPUT_DIR}/csinet_encoder_epoch_{epoch}.pt")
                torch.save(dec_params.clone().detach(),
                          f"{OUTPUT_DIR}/quantum_decoder_epoch_{epoch}.pt")

                # 保存中间训练历史
                torch.save(training_history,
                          f"{OUTPUT_DIR}/training_history_epoch_{epoch}.pt")

                # 详细的epoch结束信息打印
                print("\n" + "=" * 80)
                print(f"🎉 EPOCH {epoch} 训练完成!")
                print("=" * 80)

                # 训练统计
                print(f"📊 训练统计:")
                print(f"  • 平均训练损失: {avg_epoch_loss:.8f}")
                print(f"  • 验证集KL散度: {val_mse:.8f}")
                print(f"  • 处理样本数: {batch_count}")
                print(f"  • Epoch耗时: {format_time(epoch_time)}")
                print(f"  • 平均批处理时间: {epoch_time/batch_count:.4f}秒/样本")

                # 损失详情
                print(f"\n📉 损失分析:")
                print(f"  • 最小batch损失: {min(batch_losses):.8f}")
                print(f"  • 最大batch损失: {max(batch_losses):.8f}")
                print(f"  • 损失标准差: {np.std(batch_losses):.8f}")
                print(f"  • 损失改善率: {((batch_losses[0] - batch_losses[-1])/batch_losses[0]*100):.2f}%")

                # 参数状态
                print(f"\n⚙️  参数状态:")
                print(f"  • 量子参数范数: {dec_params_norm:.4f}")
                print(f"  • 经典编码器参数数: {sum(p.numel() for p in csinet_encoder.parameters()):,}")
                print(f"  • 量子解码器参数数: {dec_params.numel():,}")
                print(f"  • 总可训练参数: {sum(p.numel() for p in csinet_encoder.parameters()) + dec_params.numel():,}")

                # 保存信息
                print(f"\n💾 保存状态:")
                print(f"  • 编码器权重: csinet_encoder_epoch_{epoch}.pt")
                print(f"  • 量子参数: quantum_decoder_epoch_{epoch}.pt")
                print(f"  • 训练历史: training_history_epoch_{epoch}.pt")

                # 进度条
                progress = (epoch + 1) / n_epochs * 100
                bar_length = 30
                filled_length = int(bar_length * progress // 100)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                print(f"\n🔄 总体进度: |{bar}| {progress:.1f}% ({epoch + 1}/{n_epochs})")
                print("=" * 80)

        total_time = time.time() - start_time
        print(f"\n🏆 训练圆满完成!")
        print("=" * 80)
        print(f"⏱️  总训练时间: {format_time(total_time)}")
        print(f"📅 训练结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # 保存最终模型
        torch.save(csinet_encoder.state_dict(),
                  f"{OUTPUT_DIR}/final_csinet_encoder.pt")
        torch.save(dec_params,
                  f"{OUTPUT_DIR}/final_quantum_decoder_weights.pt")
        torch.save(training_history,
                  f"{OUTPUT_DIR}/training_history.pt")

        print(f"\n💾 最终模型保存:")
        print(f"  • 最终编码器: final_csinet_encoder.pt")
        print(f"  • 最终量子参数: final_quantum_decoder_weights.pt")
        print(f"  • 完整训练历史: training_history.pt")
        print("=" * 80)

        return hybrid_model, training_history

    except Exception as e:
        print(f"训练过程错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_trained_model(model, test_data, test_samples=500):
    """测试训练好的模型"""
    print("\n" + "=" * 70)
    print("在测试集上评估模型...")
    print("=" * 70)
    try:
        model.eval()
        subset = torch.from_numpy(test_data[:min(test_samples, len(test_data))]).float().to(device)

        with torch.no_grad():
            # 模型输出概率分布
            outputs = model(subset)  # (batch, 2048)
            # 准备目标概率分布
            targets = prepare_target_distribution(subset)  # (batch, 2048)
            # 计算概率分布之间的损失（使用默认交叉熵）
            prob_loss = compute_probability_loss(outputs, targets)

        print(f"测试集交叉熵损失（{len(subset)} 个样本）: {prob_loss:.6f}")

        # 计算其他损失函数作为对比指标
        with torch.no_grad():
            mse_loss = compute_probability_loss(outputs, targets, loss_type='mse')
            kl_loss = compute_probability_loss(outputs, targets, loss_type='kl')
            jsd_loss = compute_probability_loss(outputs, targets, loss_type='jsd')
            hellinger_loss = compute_probability_loss(outputs, targets, loss_type='hellinger')

        print(f"测试集 MSE损失: {mse_loss:.6f}")
        print(f"测试集 KL散度: {kl_loss:.6f}")
        print(f"测试集 JSD散度: {jsd_loss:.6f}")
        print(f"测试集 Hellinger距离: {hellinger_loss:.6f}")

        # 保存测试结果
        test_results = {
            "test_cross_entropy_loss": float(prob_loss),
            "test_mse_loss": float(mse_loss),
            "test_kl_divergence": float(kl_loss),
            "test_jsd_divergence": float(jsd_loss),
            "test_hellinger_distance": float(hellinger_loss),
            "n_samples": len(subset),
            "encoded_dim": 256,
            "loss_function_used": "cross_entropy",  # 记录使用的损失函数
            "device": str(device)
        }
        torch.save(test_results, f"{OUTPUT_DIR}/test_results.pt")
        print("测试结果已保存！")

        return float(prob_loss)
    except Exception as e:
        print(f"测试错误: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# 5. 主程序
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("🔬 QAE083_gpu: 混合CsiNet-量子编码解码神经网络 (encoded_dim=256) - GPU版本")
    print("=" * 80)

    # Data loading
    train_data, val_data, test_data = load_csinet_data()

    print(f"\n📊 数据集信息:")
    print(f"  • 训练集样本数: {len(train_data):,}")
    print(f"  • 验证集样本数: {len(val_data):,}")
    print(f"  • 测试集样本数: {len(test_data):,}")
    print(f"  • 输入形状: {train_data.shape[1:]}")
    print(f"  • 数据范围: [{train_data.min():.4f}, {train_data.max():.4f}]")
    print(f"  • 训练设备: {device}")

    print(f"\n🏗️  网络架构 (encoded_dim=256):")
    print("  1. CsiNet编码器: (2,32,32) → 256维 (压缩率1/8)")
    print("  2. 量子态嵌入: 256维向量映射为量子态 (8量子比特振幅编码)")
    print("  3. 量子解码器: 11量子比特参数化量子线路 (StronglyEntanglingLayers)")
    print("  4. 计算基测量: 直接得到2048维概率分布")
    print("  5. 损失函数: 输出概率分布 vs 输入归一化概率分布的KL散度 (可选: mse, cross_entropy, jsd, hellinger)")
    print("  6. 训练设备: GPU (PennyLane + PyTorch)")

    print("=" * 80)
    print("🚀 开始训练流程...")

    # 训练模型
    trained_model, history = train_hybrid_model()

    if trained_model is not None:
        # 测试模型
        test_loss = test_trained_model(trained_model, test_data, test_samples=500)

        print("\n" + "=" * 70)
        print("训练和测试完成！")
        print("=" * 70)
        print(f"所有结果保存在目录: {OUTPUT_DIR}/")
        print(f"配置详情: 概率分布对概率分布训练, 8比特编码+11比特ansatz, GPU加速")
    else:
        print("\n训练失败！")
