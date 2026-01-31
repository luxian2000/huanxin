"""
QAE083: 混合经典-量子编码解码神经网络

网络架构：
1. 经典编码器：使用CsiNet架构将2560维输入压缩到256维
2. 量子态映射：将256维经典向量映射为量子态（幅度嵌入）
3. 量子解码器：使用参数化量子线路解码量子态，恢复到2560维

数据流：
输入(2560维) -> 经典编码器 -> 256维 -> 量子态 -> 量子解码器 -> 输出(2560维)
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
import csv

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 创建输出目录
OUTPUT_DIR = "QAE083"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(data_path="/Users/luxian/DataSpace/csi_cmri/CSI_channel_30km.npy"):
    """安全地加载数据文件或生成模拟数据"""
    possible_paths = [
        data_path,
        "./CSI_channel_30km.npy",
        f"../DataSpace/csi_cmri/CSI_channel_30km.npy",
        f"../../DataSpace/csi_cmri/CSI_channel_30km.npy",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Loading data from: {path}")
            return np.load(path)
    
    # 如果没有找到数据文件，则创建模拟数据
    print("Data file not found. Creating simulated CSI data...")
    simulated_data = np.random.randn(80000, 2560).astype(np.float32)
    print(f"Generated simulated data with shape: {simulated_data.shape}")
    return simulated_data

# Data loading
print("=" * 70)
print("QAE083: 混合经典-量子编码解码神经网络")
print("=" * 70)
data_30 = load_data()  # shape=(80000, 2560)

# Dataset split
TOTAL_SAMPLES = data_30.shape[0]
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

train_size = int(TOTAL_SAMPLES * TRAIN_RATIO)
val_size = int(TOTAL_SAMPLES * VAL_RATIO)
test_size = TOTAL_SAMPLES - train_size - val_size

train_data = data_30[:train_size]
val_data = data_30[train_size:train_size + val_size]
test_data = data_30[train_size + val_size:]

print("\n数据划分结果:")
print(f"训练集: {len(train_data)} 个样本 ({TRAIN_RATIO*100:.1f}%)")
print(f"验证集: {len(val_data)} 个样本 ({VAL_RATIO*100:.1f}%)")
print(f"测试集: {len(test_data)} 个样本 ({TEST_RATIO*100:.1f}%)")

# Network parameters
INPUT_DIM = 2560  # 输入维度
ENCODED_DIM = 256  # 经典编码器输出维度（压缩维度）
OUTPUT_DIM = 2560  # 最终输出维度（应与输入相同）

# Quantum parameters
N_LAYERS = 4
DATA_QUBITS = int(np.ceil(np.log2(ENCODED_DIM)))  # 8 qubits to host 256-dim classical encoding

print(f"\n网络架构:")
print(f"输入维度: {INPUT_DIM}")
print(f"经典编码维度: {ENCODED_DIM}")
print(f"量子比特数: {DATA_QUBITS}")
print(f"量子层数: {N_LAYERS}")
print(f"输出维度: {OUTPUT_DIM}")

# ============================================================================
# 1. 经典编码器（基于CsiNet架构）
# ============================================================================

class ClassicalEncoder(nn.Module):
    """经典编码器：将2560维输入压缩到256维（无BatchNorm版本）"""
    def __init__(self, input_dim=2560, encoded_dim=256):
        super(ClassicalEncoder, self).__init__()
        self.input_dim = input_dim
        self.encoded_dim = encoded_dim
        
        # 多层感知机编码器（移除BatchNorm避免单样本问题）
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            
            nn.Linear(512, encoded_dim),
            nn.Tanh()  # 使用Tanh激活，输出范围[-1, 1]
        )
        
    def forward(self, x):
        """编码过程"""
        return self.encoder(x)

# 初始化经典编码器
classical_encoder = ClassicalEncoder(INPUT_DIM, ENCODED_DIM)
print(f"\n经典编码器结构:")
print(classical_encoder)

# ============================================================================
# 2. 量子态映射和量子解码器
# ============================================================================

def normalize_for_amplitude_embedding(vec):
    """归一化向量用于幅度嵌入"""
    if isinstance(vec, torch.Tensor):
        vec = vec.detach()
    norm = torch.norm(vec) if isinstance(vec, torch.Tensor) else np.linalg.norm(vec)
    if norm < 1e-10:
        return vec
    return vec / norm

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

# Quantum device
DEV = qml.device("lightning.qubit", wires=DATA_QUBITS)

@qml.qnode(DEV, interface="torch")
def quantum_decoder_circuit(encoded_vec, dec_params):
    """
    量子自编码器解码器电路
    
    使用标准量子自编码器架构：
    1. 将经典编码嵌入为量子态
    2. 应用参数化解码层恢复量子态信息
    3. 在多个测量基下测量以提取完整量子态信息
    
    Args:
        encoded_vec: 经典编码器输出的256维向量
        dec_params: 量子解码器参数
        
    Returns:
        量子态的多基测量结果（PauliX, PauliY, PauliZ）
    """
    # 1. 将经典编码向量嵌入为量子态
    encoded_padded = pad_to_qubits(encoded_vec, DATA_QUBITS)
    encoded_normalized = normalize_for_amplitude_embedding(encoded_padded)
    
    if isinstance(encoded_normalized, torch.Tensor):
        encoded_normalized = encoded_normalized.detach().numpy()
    
    qml.AmplitudeEmbedding(encoded_normalized, wires=range(DATA_QUBITS), 
                          pad_with=0.0, normalize=True)
    
    # 2. 应用参数化量子解码层（标准量子自编码器解码器）
    # 使用多层强纠缠层来恢复量子态信息
    qml.StronglyEntanglingLayers(weights=dec_params, wires=range(DATA_QUBITS))
    
    # 3. 多基测量提取完整量子态信息
    # 测量PauliX, PauliY, PauliZ获得更多信息用于重构
    measurements = []
    
    # PauliZ测量
    for i in range(DATA_QUBITS):
        measurements.append(qml.expval(qml.PauliZ(i)))
    
    # PauliX测量
    for i in range(DATA_QUBITS):
        measurements.append(qml.expval(qml.PauliX(i)))
    
    # PauliY测量
    for i in range(DATA_QUBITS):
        measurements.append(qml.expval(qml.PauliY(i)))
    
    return measurements

# ============================================================================
# 3. 量子输出到经典解码器（将量子测量结果映射回2560维）
# ============================================================================

class QuantumToClassicalDecoder(nn.Module):
    """
    将量子解码器输出映射回原始空间（2560维）（无BatchNorm版本）
    
    量子解码器输出3*DATA_QUBITS个测量值（X, Y, Z基测量）
    通过多层神经网络将量子测量信息重构为原始数据
    """
    def __init__(self, quantum_output_dim, output_dim=2560):
        super(QuantumToClassicalDecoder, self).__init__()
        
        # 量子输出维度是 3 * DATA_QUBITS (因为有X, Y, Z三个基的测量)
        self.quantum_input_dim = quantum_output_dim * 3
        
        self.decoder = nn.Sequential(
            nn.Linear(self.quantum_input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            
            nn.Linear(1024, output_dim),
            nn.Tanh()  # 输出范围[-1, 1]
        )
        
    def forward(self, x):
        """解码过程"""
        return self.decoder(x)

# 初始化量子到经典的解码器
q2c_decoder = QuantumToClassicalDecoder(DATA_QUBITS, OUTPUT_DIM)
print(f"\n量子到经典解码器结构:")
print(f"量子测量维度: {DATA_QUBITS * 3} (X, Y, Z基)")
print(q2c_decoder)

# ============================================================================
# 4. 完整的混合网络
# ============================================================================

class HybridClassicalQuantumAutoencoder(nn.Module):
    """
    完整的混合经典-量子自编码器
    
    流程：
    1. 经典编码器压缩输入
    2. 量子态嵌入和量子解码器变换
    3. 多基测量提取量子态信息
    4. 经典解码器重构原始数据
    """
    def __init__(self, classical_encoder, q2c_decoder, dec_params):
        super(HybridClassicalQuantumAutoencoder, self).__init__()
        self.classical_encoder = classical_encoder
        self.q2c_decoder = q2c_decoder
        self.dec_params = dec_params
    
    def forward(self, x):
        """
        前向传播（批量处理版本）
        x: (batch_size, 2560)
        """
        batch_size = x.shape[0]
        
        # 1. 经典编码器批量处理
        encoded_batch = self.classical_encoder(x)  # (batch_size, 256)
        
        # 2. 量子解码和经典解码（逐个样本处理）
        outputs = []
        for i in range(batch_size):
            # 获取单个编码向量
            encoded_vec = encoded_batch[i]  # (256,)
            
            # 量子解码（返回3*DATA_QUBITS个测量值）
            quantum_measurements = quantum_decoder_circuit(encoded_vec, self.dec_params)
            quantum_output = torch.stack(quantum_measurements)  # (3*DATA_QUBITS,)
            
            # 量子到经典解码
            decoded = self.q2c_decoder(quantum_output.unsqueeze(0))  # (1, 2560)
            outputs.append(decoded)
        
        return torch.cat(outputs, dim=0)

# ============================================================================
# 5. 训练函数
# ============================================================================

def save_initial_parameters(classical_encoder, dec_params, q2c_decoder):
    """保存初始参数"""
    torch.save(classical_encoder.state_dict(), 
              f"{OUTPUT_DIR}/initial_classical_encoder.pt")
    torch.save(dec_params, 
              f"{OUTPUT_DIR}/initial_quantum_decoder_weights.pt")
    torch.save(q2c_decoder.state_dict(), 
              f"{OUTPUT_DIR}/initial_q2c_decoder.pt")
    print("初始参数已保存！")

def compute_mse(output, target):
    """计算均方误差"""
    return torch.mean((output - target) ** 2)

def validate_model(model, val_samples=500):
    """验证模型"""
    model.eval()  # 切换到评估模式
    try:
        subset = torch.from_numpy(val_data[:min(val_samples, len(val_data))]).float()
        with torch.no_grad():
            outputs = model(subset)
            mse = compute_mse(outputs, subset)
        model.train()  # 切换回训练模式
        return float(mse)
    except Exception as e:
        print(f"验证错误: {e}")
        model.train()  # 确保回到训练模式
        return float("nan")

def train_hybrid_model():
    """训练混合经典-量子自编码器"""
    try:
        # 初始化量子解码器参数（使用Parameter确保是叶子张量）
        dec_shape = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, 
                                                       n_wires=DATA_QUBITS)
        # 使用nn.Parameter确保是叶子张量
        dec_params = nn.Parameter(torch.rand(dec_shape) * 0.1)
        
        # 保存初始参数
        save_initial_parameters(classical_encoder, dec_params, q2c_decoder)
        
        # 创建混合模型
        hybrid_model = HybridClassicalQuantumAutoencoder(
            classical_encoder, q2c_decoder, dec_params
        )
        
        # 优化器 - 分别优化经典和量子参数
        classical_params = list(classical_encoder.parameters()) + \
                          list(q2c_decoder.parameters())
        
        # 单独创建量子参数优化器
        quantum_optimizer = torch.optim.Adam([dec_params], lr=0.01)
        classical_optimizer = torch.optim.Adam(classical_params, lr=0.001)
        
        # 训练参数
        n_epochs = 120
        batch_size = 20  # 增加批量大小避免BatchNorm问题
        n_samples = 1000  # 使用部分训练数据
        samples = torch.from_numpy(train_data[:n_samples]).float()
        
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
        }
        
        # CSV文件记录batch losses
        csv_file = f"{OUTPUT_DIR}/hybrid_batch_losses.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'batch', 'loss', 'dec_params_norm'])
        
        print("\n" + "=" * 70)
        print("开始训练混合经典-量子自编码器...")
        print("=" * 70)
        start_time = time.time()
        
        for epoch in range(n_epochs):
            hybrid_model.train()
            epoch_loss = 0.0
            batch_count = 0
            
            # 随机打乱训练数据
            indices = torch.randperm(n_samples)
            samples_shuffled = samples[indices]
            
            for i in range(0, n_samples, batch_size):
                batch = samples_shuffled[i:i + batch_size]
                actual_batch_size = batch.shape[0]
                
                # 跳过太小的批次
                if actual_batch_size < 2:
                    continue
                    
                # 清零梯度
                classical_optimizer.zero_grad()
                quantum_optimizer.zero_grad()
                
                # 前向传播
                outputs = hybrid_model(batch)
                
                # 计算损失（重构误差）
                loss = compute_mse(outputs, batch)
                
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
                
                # 记录batch loss
                batch_info = {
                    "epoch": epoch,
                    "batch": i // batch_size,
                    "loss": float(current_loss),
                    "dec_params_norm": float(dec_params_norm)
                }
                training_history["batch_losses"].append(batch_info)
                
                # 写入CSV
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, i // batch_size, current_loss, dec_params_norm])
                
                if (i // batch_size) % 5 == 0:
                    print(f"Epoch {epoch}, Batch {i//batch_size}: "
                          f"MSE Loss = {current_loss:.6f}")
            
            if batch_count > 0:
                avg_epoch_loss = epoch_loss / batch_count
                
                # 验证（临时切换到评估模式）
                hybrid_model.eval()
                val_mse = validate_model(hybrid_model, val_samples=200)
                hybrid_model.train()  # 切换回训练模式
                
                training_history["epoch_losses"].append({
                    "epoch": epoch,
                    "avg_loss": float(avg_epoch_loss)
                })
                training_history["val_mse"].append({
                    "epoch": epoch,
                    "val_mse": float(val_mse)
                })
                
                # 保存epoch权重
                torch.save(classical_encoder.state_dict(), 
                          f"{OUTPUT_DIR}/classical_encoder_epoch_{epoch}.pt")
                torch.save(dec_params.clone().detach(), 
                          f"{OUTPUT_DIR}/quantum_decoder_epoch_{epoch}.pt")
                torch.save(q2c_decoder.state_dict(), 
                          f"{OUTPUT_DIR}/q2c_decoder_epoch_{epoch}.pt")
                
                print(f"Epoch {epoch} 完成: "
                      f"训练 MSE = {avg_epoch_loss:.6f}, "
                      f"验证 MSE = {val_mse:.6f}")
                print("-" * 70)
        
        total_time = time.time() - start_time
        print(f"\n训练完成！总用时: {total_time:.2f} 秒")
        
        # 保存最终模型
        torch.save(classical_encoder.state_dict(), 
                  f"{OUTPUT_DIR}/final_classical_encoder.pt")
        torch.save(dec_params, 
                  f"{OUTPUT_DIR}/final_quantum_decoder_weights.pt")
        torch.save(q2c_decoder.state_dict(), 
                  f"{OUTPUT_DIR}/final_q2c_decoder.pt")
        torch.save(training_history, 
                  f"{OUTPUT_DIR}/training_history.pt")
        print("最终模型和训练历史已保存！")
        
        return hybrid_model, training_history
        
    except Exception as e:
        print(f"训练过程错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# ============================================================================
# 6. 测试函数
# ============================================================================

def test_trained_model(model, test_samples=500):
    """测试训练好的模型"""
    print("\n" + "=" * 70)
    print("在测试集上评估模型...")
    print("=" * 70)
    try:
        model.eval()
        subset = torch.from_numpy(test_data[:min(test_samples, len(test_data))]).float()
        
        with torch.no_grad():
            outputs = model(subset)
            mse = compute_mse(outputs, subset)
            
        print(f"测试集 MSE（{len(subset)} 个样本）: {mse:.6f}")
        
        # 保存测试结果
        test_results = {
            "test_mse": float(mse),
            "n_samples": len(subset)
        }
        torch.save(test_results, f"{OUTPUT_DIR}/test_results.pt")
        print("测试结果已保存！")
        
        return float(mse)
    except Exception as e:
        print(f"测试错误: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# 7. 主程序
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("QAE083: 混合经典-量子编码解码神经网络")
    print("=" * 70)
    print("网络架构:")
    print("1. 经典编码器: 2560 -> 256 维")
    print("2. 量子态嵌入: 256维向量映射为量子态")
    print("3. 量子自编码器解码器: 参数化量子线路变换")
    print("4. 多基测量: X, Y, Z基测量提取量子态信息 (24维)")
    print("5. 经典解码器: 24维器 (QAE083)")
    print("=" * 70)
    print("\n架构说明:")
    print("1. 经典编码器: 2560 -> 256 维")
    print("2. 量子态嵌入: 256维向量映射为量子态")
    print("3. 量子解码器: 参数化量子线路")
    print("4. 经典解码器: 量子测量 -> 2560 维")
    print("\n开始训练...")
    
    # 训练模型
    trained_model, history = train_hybrid_model()
    
    if trained_model is not None:
        # 测试模型
        test_mse = test_trained_model(trained_model, test_samples=500)
        
        print("\n" + "=" * 70)
        print("训练和测试完成！")
        print("=" * 70)
        print(f"所有结果保存在目录: {OUTPUT_DIR}/")
    else:
        print("\n训练失败！")
