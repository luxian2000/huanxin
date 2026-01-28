import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from sklearn.preprocessing import StandardScaler

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 定义保存目录
SAVE_DIR = "QAE_09"

def load_data(data_path="/Users/luxian/DataSpace/csi_cmri/CSI_channel_30km.npy"):
    """安全地加载数据文件或生成模拟数据 - 与pg08_QAE.py相同的逻辑"""
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
    # 创建模拟的CSI数据 (80000, 2560)
    simulated_data = np.random.randn(80000, 2560).astype(np.float32)
    print(f"Generated simulated data with shape: {simulated_data.shape}")
    return simulated_data

def load_and_preprocess_data():
    """加载并预处理数据 - 使用与pg08_QAE.py相同的数据划分逻辑"""
    # 加载原始数据
    data_30 = load_data()
    
    # 数据划分 - 与pg08_QAE.py完全一致
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

    print("数据划分结果 (与pg08_QAE.py相同):")
    print(f"训练集: {len(train_data)} 个样本 ({TRAIN_RATIO*100:.1f}%)")
    print(f"验证集: {len(val_data)} 个样本 ({VAL_RATIO*100:.1f}%)")
    print(f"测试集: {len(test_data)} 个样本 ({TEST_RATIO*100:.1f}%)")
    
    # 不进行标准化，保持与pg08_QAE.py一致
    print(f"数据预处理完成 (无标准化，与pg08_QAE.py保持一致)")
    
    return train_data, val_data, test_data

# 与pg08_QAE.py兼容的函数
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def normalize(x):
    norm = torch.norm(x)
    if norm == 0:
        return x
    return x / norm

def pad_to_qubits(vec, n_qubits):
    target_len = 2 ** n_qubits
    if len(vec) < target_len:
        return F.pad(vec, (0, target_len - len(vec)))
    return vec[:target_len]

# 定义类
class ClassicalEncoder(nn.Module):
    """改进的经典编码器 - 与pg08_QAE.py兼容"""
    def __init__(self, input_dim=2560, output_dim=256):
        super(ClassicalEncoder, self).__init__()
        
        # 直接使用与pg08_QAE.py类似的权重和偏置
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        # 如果输入是单个样本，添加批次维度
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        output = self.linear(x)
        output = torch.sigmoid(output)
        
        # 对每个样本进行归一化
        output = F.normalize(output, p=2, dim=1)
        
        # 如果输入是单个样本，移除批次维度
        if output.size(0) == 1:
            output = output.squeeze(0)
        
        return output

class QuantumAutoencoder(nn.Module):
    """改进的量子自编码器"""
    def __init__(self, data_qubits=8, n_layers=4):  # 使用与pg08_QAE.py相同的层数
        super(QuantumAutoencoder, self).__init__()
        self.data_qubits = data_qubits
        self.n_layers = n_layers
        
        # 量子设备
        self.dev = qml.device("lightning.qubit", wires=data_qubits)
        
        # 初始化参数 - 使用nn.Parameter
        self.init_parameters()
        
    def init_parameters(self):
        """初始化量子电路参数"""
        # 编码器参数 - 使用nn.Parameter
        # StronglyEntanglingLayers 参数形状: (n_layers, n_wires, 3)
        enc_shape = (self.n_layers, self.data_qubits, 3)
        self.enc_params = nn.Parameter(torch.rand(enc_shape) * 0.01)
        
        # 解码器参数 - 使用nn.Parameter
        dec_shape = (self.n_layers, self.data_qubits, 3)
        self.dec_params = nn.Parameter(torch.rand(dec_shape) * 0.01)
        
    def forward(self, classical_vector):
        """量子电路前向传播"""
        @qml.qnode(self.dev, interface="torch")
        def circuit(enc_params, dec_params, x):
            # 确保输入向量是归一化的，处理零向量的情况
            x_norm = torch.linalg.norm(x)
            if x_norm == 0:
                # 如果向量是零向量，创建一个单位向量
                x_normalized = torch.zeros_like(x)
                x_normalized[0] = 1.0
            else:
                x_normalized = x / x_norm
            
            # 1. 振幅编码
            qml.AmplitudeEmbedding(x_normalized, wires=range(self.data_qubits), pad_with=0.0, normalize=True)
            
            # 2. 编码器层
            qml.StronglyEntanglingLayers(weights=enc_params, wires=range(self.data_qubits))
            
            # 3. 解码器层
            qml.StronglyEntanglingLayers(weights=dec_params, wires=range(self.data_qubits))
            
            # 4. 逆振幅嵌入以计算保真度
            qml.adjoint(qml.AmplitudeEmbedding)(x_normalized, wires=range(self.data_qubits), pad_with=0.0, normalize=True)
            
            # 5. 返回测量|0...0>的概率
            return qml.expval(qml.Projector([0]*self.data_qubits, wires=range(self.data_qubits)))
        
        return circuit(self.enc_params, self.dec_params, classical_vector)

class ImprovedQAE:
    """改进的量子自编码器训练器"""
    def __init__(self, train_data, val_data, test_data):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        
        # 初始化组件 - 使用与pg08_QAE.py相同的参数
        self.classical_encoder = ClassicalEncoder(input_dim=2560, output_dim=256)
        self.quantum_ae = QuantumAutoencoder(data_qubits=8, n_layers=4)
        
        # 优化器
        self.optimizer = torch.optim.Adam(list(self.classical_encoder.parameters()) + 
                                         list(self.quantum_ae.parameters()), lr=0.005)
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )
        
    def compute_fidelity(self, batch):
        """计算批量的保真度"""
        fidelities = []
        
        for sample in batch:
            # 转换为tensor
            if isinstance(sample, np.ndarray):
                sample = torch.from_numpy(sample).float()
            
            # 经典编码
            classical_vec = self.classical_encoder(sample)
            
            # 填充到2^n长度
            target_len = 2 ** 8  # 8 qubits
            if len(classical_vec) < target_len:
                padded_vec = F.pad(classical_vec, (0, target_len - len(classical_vec)))
            else:
                padded_vec = classical_vec[:target_len]
            
            # 量子电路计算保真度
            fidelity = self.quantum_ae(padded_vec)
            fidelities.append(fidelity)
        
        return torch.stack(fidelities)
    
    def train_epoch(self, epoch, batch_size=50):
        """训练一个epoch"""
        self.classical_encoder.train()
        self.quantum_ae.train()
        
        n_samples = min(1000, len(self.train_data))  # 与pg08_QAE.py相同
        indices = np.random.choice(len(self.train_data), n_samples, replace=False)
        samples = [self.train_data[i] for i in indices]
        
        epoch_loss = 0.0
        batch_count = 0
        
        for i in range(0, n_samples, batch_size):
            batch = samples[i:i+batch_size]
            batch_tensor = torch.from_numpy(np.array(batch)).float()
            
            # 前向传播
            self.optimizer.zero_grad()
            fidelities = self.compute_fidelity(batch_tensor)
            
            # 计算损失
            loss = 1.0 - torch.mean(fidelities)
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            self.optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
            if (i // batch_size) % 10 == 0:
                avg_fidelity = torch.mean(fidelities).item()
                print(f"Epoch {epoch}, Batch {i//batch_size}: Loss = {loss.item():.6f}, Fidelity = {avg_fidelity:.6f}")
        
        return epoch_loss / batch_count if batch_count > 0 else 0.0
    
    def validate(self, n_samples=500):
        """验证模型"""
        self.classical_encoder.eval()
        self.quantum_ae.eval()
        
        with torch.no_grad():
            subset = self.val_data[:min(n_samples, len(self.val_data))]
            subset_tensor = torch.from_numpy(subset).float()
            
            fidelities = []
            for i in range(0, len(subset_tensor), 50):
                batch = subset_tensor[i:i+50]
                batch_fidelities = self.compute_fidelity(batch)
                fidelities.extend(batch_fidelities)
            
            if fidelities:
                all_fidelities = torch.stack(fidelities)
                avg_fidelity = torch.mean(all_fidelities).item()
                return avg_fidelity
        
        return 0.0
    
    def test(self, n_samples=500):
        """测试模型"""
        self.classical_encoder.eval()
        self.quantum_ae.eval()
        
        with torch.no_grad():
            subset = self.test_data[:min(n_samples, len(self.test_data))]
            subset_tensor = torch.from_numpy(subset).float()
            
            fidelities = []
            for i in range(0, len(subset_tensor), 50):
                batch = subset_tensor[i:i+50]
                batch_fidelities = self.compute_fidelity(batch)
                fidelities.extend(batch_fidelities)
            
            if fidelities:
                all_fidelities = torch.stack(fidelities)
                avg_fidelity = torch.mean(all_fidelities).item()
                std_fidelity = torch.std(all_fidelities).item()
                
                print(f"测试结果:")
                print(f"平均保真度: {avg_fidelity:.6f}")
                print(f"标准差: {std_fidelity:.6f}")
                print(f"样本数: {len(all_fidelities)}")
                
                return avg_fidelity, std_fidelity
        
        return 0.0, 0.0
    
    def train(self, n_epochs=5):  # 减少epochs数量以便快速测试
        """训练模型"""
        print("开始训练改进的量子自编码器...")
        print(f"训练样本数: {len(self.train_data)}")
        print(f"验证样本数: {len(self.val_data)}")
        print(f"测试样本数: {len(self.test_data)}")
        print("=" * 60)
        
        best_val_fidelity = 0.0
        training_history = {
            'train_loss': [],
            'val_fidelity': [],
            'learning_rates': []
        }
        
        start_time = time.time()
        
        for epoch in range(n_epochs):
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_fidelity = self.validate()
            
            # 更新学习率
            self.scheduler.step(val_fidelity)
            
            # 记录历史
            training_history['train_loss'].append(train_loss)
            training_history['val_fidelity'].append(val_fidelity)
            training_history['learning_rates'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            print(f"Epoch {epoch+1}/{n_epochs}:")
            print(f"  训练损失: {train_loss:.6f}")
            print(f"  验证保真度: {val_fidelity:.6f}")
            print(f"  学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
            print("-" * 50)
            
            # 保存最佳模型
            if val_fidelity > best_val_fidelity:
                best_val_fidelity = val_fidelity
                self.save_model(f"best_model_epoch_{epoch+1}.pth")
                print(f"  保存最佳模型 (保真度: {val_fidelity:.6f})")
        
        total_time = time.time() - start_time
        print(f"训练完成! 总时间: {total_time:.2f}秒")
        print(f"最佳验证保真度: {best_val_fidelity:.6f}")
        
        # 最终测试
        test_fidelity, test_std = self.test()
        print(f"最终测试保真度: {test_fidelity:.6f} ± {test_std:.6f}")
        
        return training_history
    
    def save_model(self, filename):
        """保存模型"""
        os.makedirs(SAVE_DIR, exist_ok=True)
        torch.save({
            'classical_encoder_state': self.classical_encoder.state_dict(),
            'quantum_enc_params': self.quantum_ae.enc_params,
            'quantum_dec_params': self.quantum_ae.dec_params,
            'optimizer_state': self.optimizer.state_dict()
        }, os.path.join(SAVE_DIR, filename))
    
    def load_model(self, filename):
        """加载模型"""
        checkpoint = torch.load(os.path.join(SAVE_DIR, filename))
        self.classical_encoder.load_state_dict(checkpoint['classical_encoder_state'])
        self.quantum_ae.enc_params = checkpoint['quantum_enc_params']
        self.quantum_ae.dec_params = checkpoint['quantum_dec_params']
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])

def main():
    """主函数"""
    # 加载和预处理数据
    print("加载和预处理数据...")
    train_data, val_data, test_data = load_and_preprocess_data()
    
    # 创建改进的QAE
    print("初始化改进的量子自编码器...")
    qae = ImprovedQAE(train_data, val_data, test_data)
    
    # 训练
    history = qae.train(n_epochs=5)  # 减少epochs以便快速测试
    
    # 保存训练历史
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(history, os.path.join(SAVE_DIR, "training_history.pth"))
    print(f"训练历史已保存到 {SAVE_DIR}/training_history.pth")
    
    return qae, history

if __name__ == "__main__":
    main()
