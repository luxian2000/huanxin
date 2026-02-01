"""
QCNN01_gpu: 混合经典-量子卷积神经网络解码器 (基于CsiNet编码器，encoded_dim=256) - GPU版本

网络架构：
1. 经典编码器：使用CsiNet卷积编码器将(2,32,32)图像压缩到256维
2. 量子态映射：将256维经典向量映射为量子态（幅度嵌入）
3. QCNN解码器：使用量子卷积神经网络解码量子态，恢复到2048维概率分布
4. 输出重塑：将2048维概率分布重塑为(2,32,32)图像格式

数据流：
输入(2,32,32) -> CsiNet编码器 -> 256维 -> 量子态 -> QCNN解码器 -> 2048维概率 -> (2,32,32)

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
OUTPUT_DIR = "QCNN01_gpu"
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
# 1. CsiNet编码器（保持不变）
# ============================================================================

class CsiNetEncoder(nn.Module):
    """CsiNet编码器：将(2,32,32)图像压缩到256维向量"""

    def __init__(self, encoded_dim=256):
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
        x: (batch_size, 2, 32, 32)
        """
        # 第一层卷积 + BN + LeakyReLU
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # flatten并压缩到编码维度
        x = self.flatten(x)
        encoded = self.dense_encoded(x)

        # 确保输出是有效的实数向量
        encoded = torch.clamp(encoded, min=1e-7, max=1e7)
        encoded = torch.nan_to_num(encoded, nan=0.0, posinf=1.0, neginf=0.0)

        return encoded

# ============================================================================
# 2. 量子态映射和QCNN解码器
# ============================================================================

def normalize_for_amplitude_embedding(vec):
    """归一化向量用于幅度嵌入"""
    if isinstance(vec, torch.Tensor):
        vec = torch.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=0.0)
        vec = torch.clamp(vec, min=0.0, max=1e7)
        norm = torch.norm(vec, p=2)
    else:
        vec = np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=0.0)
        vec = np.clip(vec, 0.0, 1e7)
        norm = np.linalg.norm(vec, ord=2)

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
# QCNN解码器电路
# ============================================================================

# Quantum device - GPU版本
DEV = qml.device("lightning.gpu", wires=11)

def qconv_layer(wires, params):
    """量子卷积层：对相邻量子比特应用RY和CNOT"""
    n_wires = len(wires)
    for i in range(n_wires):
        qml.RY(params[i], wires[i])
    for i in range(n_wires - 1):
        qml.CNOT(wires=[wires[i], wires[i+1]])

def qpool_layer(source_wires, sink_wires, params):
    """量子池化层：将源量子比特池化到目标量子比特"""
    for source, sink in zip(source_wires, sink_wires):
        qml.RY(params[0], source)
        qml.RY(params[1], sink)
        qml.CNOT(source, sink)

@qml.qnode(DEV, interface="torch")
def qcnn_decoder_circuit(encoded_vec, conv_params, pool_params):
    """
    QCNN解码器电路
    
    Args:
        encoded_vec: 256维编码向量
        conv_params: 卷积层参数
        pool_params: 池化层参数
        
    Returns:
        2048维概率分布
    """
    # 1. 将编码向量嵌入到前8个量子比特
    encoded_padded = pad_to_qubits(encoded_vec, 8)
    encoded_normalized = normalize_for_amplitude_embedding(encoded_padded)
    
    encoded_normalized = encoded_normalized / (torch.norm(encoded_normalized, p=2) + 1e-10)
    
    qml.AmplitudeEmbedding(encoded_normalized, wires=range(8), pad_with=0.0, normalize=True)
    
    # 后3个量子比特初始化为|0>
    
    # 2. QCNN解码层（逐步混合和"上采样"）
    all_wires = list(range(11))
    
    # 层1：卷积所有11个量子比特
    qconv_layer(all_wires, conv_params[0])
    
    # 层2：再次卷积
    qconv_layer(all_wires, conv_params[1])
    
    # 层3：卷积
    qconv_layer(all_wires, conv_params[2])
    
    # 注意：传统QCNN有池化减少比特，但这里为解码保持所有比特
    
    # 3. 计算基测量，返回2048个概率值
    return qml.probs(wires=all_wires)

# ============================================================================
# 3. 完整的混合网络
# ============================================================================

class HybridCsiNetQCNN(nn.Module):
    """
    完整的混合CsiNet-QCNN解码器

    流程：
    1. CsiNet编码器压缩图像到256维
    2. QCNN解码器将256维解码为2048维概率分布
    3. 重塑为(2,32,32)图像
    """
    def __init__(self, csinet_encoder, conv_params, pool_params):
        super(HybridCsiNetQCNN, self).__init__()
        self.csinet_encoder = csinet_encoder
        self.conv_params = conv_params
        self.pool_params = pool_params

    def forward(self, x):
        """
        x: (batch_size, 2, 32, 32)
        返回: (batch_size, 2, 32, 32) 重构图像
        """
        batch_size = x.shape[0]

        # 1. CsiNet编码器
        encoded_batch = self.csinet_encoder(x)  # (batch_size, 256)

        # 2. QCNN解码（逐个样本）
        outputs = []
        for i in range(batch_size):
            encoded_vec = encoded_batch[i]
            
            # QCNN解码
            probs = qcnn_decoder_circuit(encoded_vec, self.conv_params, self.pool_params)  # (2048,)
            probs = probs.to(device)
            
            # 重塑为图像格式 (2, 32, 32)
            image = probs.view(2, 32, 32)
            outputs.append(image.unsqueeze(0))

        return torch.cat(outputs, dim=0)

# ============================================================================
# 4. 训练和测试函数
# ============================================================================

def compute_mse_loss(output, target):
    """计算MSE损失"""
    return torch.mean((output - target) ** 2)

def validate_model(model, val_data, val_samples=500):
    """验证模型"""
    model.eval()
    try:
        subset = torch.from_numpy(val_data[:min(val_samples, len(val_data))]).float().to(device)
        with torch.no_grad():
            outputs = model(subset)
            loss = compute_mse_loss(outputs, subset)
        model.train()
        return float(loss)
    except Exception as e:
        print(f"验证错误: {e}")
        model.train()
        return float("nan")

def save_initial_parameters(csinet_encoder, conv_params, pool_params):
    """保存初始参数"""
    torch.save(csinet_encoder.state_dict(), f"{OUTPUT_DIR}/initial_csinet_encoder.pt")
    torch.save(conv_params, f"{OUTPUT_DIR}/initial_conv_params.pt")
    torch.save(pool_params, f"{OUTPUT_DIR}/initial_pool_params.pt")
    print("初始参数已保存！")

def format_time(seconds):
    """格式化时间"""
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
    """训练混合CsiNet-QCNN模型 - GPU版本"""
    try:
        print("\n" + "=" * 80)
        print("🚀 QCNN01_gpu: 混合CsiNet-QCNN解码器训练")
        print("=" * 80)

        # 初始化组件
        csinet_encoder = CsiNetEncoder(encoded_dim=256).to(device)
        print("📋 CsiNet编码器结构:")
        print(csinet_encoder)

        # 初始化QCNN参数
        n_conv_layers = 3
        conv_params = nn.ParameterList([
            nn.Parameter(torch.rand(11) * 2 * 3.14159 - 3.14159) for _ in range(n_conv_layers)
        ])
        pool_params = nn.Parameter(torch.rand(2) * 2 * 3.14159 - 3.14159)  # 虽然未使用，但保留
        
        print(f"\n⚛️  QCNN解码器配置:")
        print(f"  • 卷积层数: {n_conv_layers}")
        print(f"  • 量子比特: 11 (8编码 + 3辅助)")
        print(f"  • 输出维度: 2048 (概率分布 -> (2,32,32))")
        print(f"  • 设备: {device}")

        # 保存初始参数
        save_initial_parameters(csinet_encoder, conv_params, pool_params)

        # 创建混合模型
        hybrid_model = HybridCsiNetQCNN(csinet_encoder, conv_params, pool_params)
        print(f"\n🤖 混合模型创建完成: CsiNet编码器 + QCNN解码器")

        # 优化器
        classical_optimizer = torch.optim.Adam(csinet_encoder.parameters(), lr=0.001)
        quantum_optimizer = torch.optim.Adam(list(conv_params) + [pool_params], lr=0.001)

        # 训练参数
        n_epochs = 20
        batch_size = 50
        n_samples = 500
        samples = torch.from_numpy(train_data[:n_samples]).float().to(device)

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
                "qcnn_conv_layers": n_conv_layers,
                "quantum_wires": 11,
                "output_shape": (2, 32, 32),
                "compression_ratio": "1/8",
                "device": str(device)
            }
        }

        # CSV文件
        csv_file = f"{OUTPUT_DIR}/hybrid_batch_losses.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'batch', 'loss'])

        print(f"\n🎯 训练配置:")
        print(f"  • 编码维度: 256")
        print(f"  • QCNN卷积层: {n_conv_layers}")
        print(f"  • 总epochs: {n_epochs}")
        print(f"  • 训练样本: {n_samples}")
        print(f"  • 设备: {device}")

        start_time = time.time()
        print(f"\n⏰ 训练开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        for epoch in range(n_epochs):
            hybrid_model.train()
            epoch_loss = 0.0
            batch_count = 0

            indices = torch.randperm(n_samples)
            samples_shuffled = samples[indices]

            epoch_start_time = time.time()
            batch_losses = []

            print(f"\n🚀 开始EPOCH {epoch} 训练...")

            for i in range(0, n_samples, batch_size):
                batch = samples_shuffled[i:i + batch_size]
                actual_batch_size = batch.shape[0]

                if actual_batch_size < 1:
                    continue

                classical_optimizer.zero_grad()
                quantum_optimizer.zero_grad()

                # 前向传播
                outputs = hybrid_model(batch)

                # 计算MSE损失
                loss = compute_mse_loss(outputs, batch)

                loss.backward()

                classical_optimizer.step()
                quantum_optimizer.step()

                current_loss = loss.item()
                epoch_loss += current_loss * actual_batch_size
                batch_count += actual_batch_size
                batch_losses.append(current_loss)

                with open(csv_file, 'a', newline='') as f:
                    csv.writer(f).writerow([epoch, i // batch_size, current_loss])

                # 每batch打印损失
                print(f"  Batch {(i//batch_size)+1:2d}/{(n_samples//batch_size):2d}: Loss = {current_loss:.8f}")

                # 在epoch中间（batch 5）打印中间信息
                if (i // batch_size) == 4:  # batch 5 (0-based 4)
                    mid_time = time.time() - epoch_start_time
                    print(f"  📍 EPOCH {epoch} 中间检查点 (batch 5/10): 平均损失 = {np.mean(batch_losses):.8f}, 耗时 = {format_time(mid_time)}")


            if batch_count > 0:
                avg_epoch_loss = epoch_loss / batch_count
                epoch_time = time.time() - epoch_start_time
                val_mse = validate_model(hybrid_model, val_data, val_samples=200)

                training_history["epoch_losses"].append({"epoch": epoch, "avg_loss": float(avg_epoch_loss)})
                training_history["val_mse"].append({"epoch": epoch, "val_mse": float(val_mse)})

                torch.save(csinet_encoder.state_dict(), f"{OUTPUT_DIR}/csinet_encoder_epoch_{epoch}.pt")
                torch.save([p.clone().detach() for p in conv_params], f"{OUTPUT_DIR}/conv_params_epoch_{epoch}.pt")
                torch.save(pool_params.clone().detach(), f"{OUTPUT_DIR}/pool_params_epoch_{epoch}.pt")
                torch.save(training_history, f"{OUTPUT_DIR}/training_history_epoch_{epoch}.pt")

                print("\n" + "=" * 80)
                print(f"🎉 EPOCH {epoch} 完成!")
                print("=" * 80)
                print(f"📊 平均训练损失: {avg_epoch_loss:.8f}")
                print(f"📊 验证MSE: {val_mse:.8f}")
                print(f"⏱️  Epoch耗时: {format_time(epoch_time)}")
                print(f"📈 损失改善: {((batch_losses[0] - batch_losses[-1])/batch_losses[0]*100):.2f}%")
                print(f"📋 本轮所有batch训练损失: {batch_losses}")

                progress = (epoch + 1) / n_epochs * 100
                bar_length = 30
                filled_length = int(bar_length * progress // 100)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                print(f"🔄 进度: |{bar}| {progress:.1f}% ({epoch + 1}/{n_epochs})")
                print("=" * 80)

        total_time = time.time() - start_time
        print(f"\n🏆 训练完成!")
        print("=" * 80)
        print(f"⏱️  总时间: {format_time(total_time)}")

        torch.save(csinet_encoder.state_dict(), f"{OUTPUT_DIR}/final_csinet_encoder.pt")
        torch.save([p for p in conv_params], f"{OUTPUT_DIR}/final_conv_params.pt")
        torch.save(pool_params, f"{OUTPUT_DIR}/final_pool_params.pt")
        torch.save(training_history, f"{OUTPUT_DIR}/training_history.pt")

        print(f"\n💾 最终模型保存到: {OUTPUT_DIR}/")
        print("=" * 80)

        return hybrid_model, training_history

    except Exception as e:
        print(f"训练错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_trained_model(model, test_data, test_samples=500):
    """测试模型"""
    print("\n" + "=" * 70)
    print("测试模型...")
    print("=" * 70)
    try:
        model.eval()
        subset = torch.from_numpy(test_data[:min(test_samples, len(test_data))]).float().to(device)

        with torch.no_grad():
            outputs = model(subset)
            mse_loss = compute_mse_loss(outputs, subset)

        print(f"测试MSE损失 ({len(subset)}样本): {mse_loss:.6f}")

        test_results = {
            "test_mse_loss": float(mse_loss),
            "n_samples": len(subset),
            "encoded_dim": 256,
            "device": str(device)
        }
        torch.save(test_results, f"{OUTPUT_DIR}/test_results.pt")
        print("测试结果已保存！")

        return float(mse_loss)
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
    print("🔬 QCNN01_gpu: 混合CsiNet-QCNN解码器 (encoded_dim=256) - GPU版本")
    print("=" * 80)

    # 加载数据
    train_data, val_data, test_data = load_csinet_data()

    print(f"\n📊 数据集:")
    print(f"  • 训练: {len(train_data):,}")
    print(f"  • 验证: {len(val_data):,}")
    print(f"  • 测试: {len(test_data):,}")
    print(f"  • 形状: {train_data.shape[1:]}")
    print(f"  • 范围: [{train_data.min():.4f}, {train_data.max():.4f}]")
    print(f"  • 设备: {device}")

    print(f"\n🏗️  架构:")
    print("  1. CsiNet编码器: (2,32,32) → 256维")
    print("  2. 量子态嵌入: 256维 → 8量子比特振幅编码")
    print("  3. QCNN解码器: 11量子比特卷积层")
    print("  4. 输出: 2048维概率 → (2,32,32)图像")
    print("  5. 损失: MSE")
    print("  6. 训练: GPU")

    print("=" * 80)
    print("🚀 开始训练...")

    # 训练
    trained_model, history = train_hybrid_model()

    if trained_model is not None:
        test_loss = test_trained_model(trained_model, test_data, test_samples=500)

        print("\n" + "=" * 70)
        print("完成！")
        print("=" * 70)
        print(f"结果保存在: {OUTPUT_DIR}/")
    else:
        print("\n训练失败！")