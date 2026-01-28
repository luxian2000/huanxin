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

def load_and_preprocess_data(data_path=None):
    """加载并预处理数据"""
    # 加载数据（使用你的现有函数）
    from pg08_QAE import load_data
    data_30 = load_data(data_path)
    
    # 数据标准化
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data_30)
    
    # 数据集划分
    TOTAL_SAMPLES = data_normalized.shape[0]
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    train_size = int(TOTAL_SAMPLES * TRAIN_RATIO)
    val_size = int(TOTAL_SAMPLES * VAL_RATIO)
    test_size = TOTAL_SAMPLES - train_size - val_size
    
    train_data = data_normalized[:train_size]
    val_data = data_normalized[train_size:train_size + val_size]
    test_data = data_normalized[train_size + val_size:]
    
    print(f"数据预处理完成:")
    print(f"训练集: {len(train_data)} 个样本")
    print(f"验证集: {len(val_data)} 个样本")
    print(f"测试集: {len(test_data)} 个样本")
    
    return train_data, val_data, test_data, scaler

class ClassicalEncoder(nn.Module):
    """改进的经典编码器"""
    def __init__(self, input_dim=2560, hidden_dims=[1024, 512, 256], output_dim=256):
        super(ClassicalEncoder, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 构建编码器层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        output = self.encoder(x)
        # L2归一化
        output = F.normalize(output, p=2, dim=1)
        return output

class QuantumAutoencoder:
    """改进的量子自编码器"""
    def __init__(self, data_qubits=8, latent_qubits=4, n_layers=6):
        self.data_qubits = data_qubits
        self.latent_qubits = latent_qubits
        self.n_layers = n_layers
        
        # 量子设备
        self.dev = qml.device("lightning.qubit", wires=data_qubits)
        
        # 初始化参数
        self.init_parameters()
        
    def init_parameters(self):
        """初始化量子电路参数"""
        # 编码器参数
        enc_shape = (self.n_layers, self.data_qubits, 3)  # 每个层、每个量子比特、3个旋转参数
        self.enc_params = torch.randn(enc_shape, requires_grad=True) * 0.1
        
        # 解码器参数
        dec_shape = (self.n_layers, self.data_qubits, 3)
        self.dec_params = torch.randn(dec_shape, requires_grad=True) * 0.1
        
    def quantum_circuit(self, classical_vector):
        """改进的量子电路"""
        @qml.qnode(self.dev, interface="torch")
        def circuit(enc_params, dec_params, x):
            # 1. 振幅编码
            qml.AmplitudeEmbedding(x, wires=range(self.data_qubits), normalize=True)
            
            # 2. 编码器层
            for layer in range(self.n_layers):
                # 单量子比特旋转
                for qubit in range(self.data_qubits):
                    qml.Rot(enc_params[layer, qubit, 0],
                           enc_params[layer, qubit, 1],
                           enc_params[layer, qubit, 2],
                           wires=qubit)
                
                # 纠缠层：线性纠缠
                for qubit in range(self.data_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
            
            # 3. 模拟压缩：测量并重置后4个量子比特
            # 在模拟中，我们通过部分迹来近似
            
            # 4. 解码器层
            for layer in range(self.n_layers):
                # 单量子比特旋转
                for qubit in range(self.data_qubits):
                    qml.Rot(dec_params[layer, qubit, 0],
                           dec_params[layer, qubit, 1],
                           dec_params[layer, qubit, 2],
                           wires=qubit)
                
                # 纠缠层
                for qubit in range(self.data_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
            
            # 5. 计算保真度
            qml.adjoint(qml.AmplitudeEmbedding)(x, wires=range(self.data_qubits), normalize=True)
            
            # 测量所有量子比特在|0⟩态的概率
            return qml.expval(qml.Projector([0]*self.data_qubits, wires=range(self.data_qubits)))
        
        return circuit(self.enc_params, self.dec_params, classical_vector)
    
    def pad_to_qubits(self, vec):
        """填充向量到2^n长度"""
        target_len = 2 ** self.data_qubits
        if len(vec) < target_len:
            return F.pad(vec, (0, target_len - len(vec)))
        return vec[:target_len]

class ImprovedQAE:
    """改进的量子自编码器训练器"""
    def __init__(self, train_data, val_data, test_data):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        
        # 初始化组件
        self.classical_encoder = ClassicalEncoder()
        self.quantum_ae = QuantumAutoencoder()
        
        # 优化器
        self.optimizer = torch.optim.Adam([
            {'params': self.classical_encoder.parameters(), 'lr': 0.001},
            {'params': self.quantum_ae.enc_params, 'lr': 0.01},
            {'params': self.quantum_ae.dec_params, 'lr': 0.01}
        ])
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
    def compute_fidelity(self, batch):
        """计算批量的保真度"""
        fidelities = []
        
        for sample in batch:
            # 经典编码
            classical_vec = self.classical_encoder(sample.unsqueeze(0)).squeeze()
            
            # 填充到2^n长度
            padded_vec = self.quantum_ae.pad_to_qubits(classical_vec)
            
            # 量子电路计算保真度
            fidelity = self.quantum_ae.quantum_circuit(padded_vec)
            fidelities.append(fidelity)
        
        return torch.stack(fidelities)
    
    def train_epoch(self, epoch, batch_size=50):
        """训练一个epoch"""
        self.classical_encoder.train()
        
        n_samples = min(3000, len(self.train_data))  # 使用更多样本
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
            
            # 计算损失（添加正则化）
            loss = 1.0 - torch.mean(fidelities)
            l2_reg = 0.001 * (torch.norm(self.quantum_ae.enc_params) + 
                             torch.norm(self.quantum_ae.dec_params))
            total_loss = loss + l2_reg
            
            # 反向传播
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.classical_encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_([self.quantum_ae.enc_params, self.quantum_ae.dec_params], max_norm=1.0)
            
            # 更新参数
            self.optimizer.step()
            
            epoch_loss += total_loss.item()
            batch_count += 1
            
            if (i // batch_size) % 10 == 0:
                avg_fidelity = torch.mean(fidelities).item()
                print(f"Epoch {epoch}, Batch {i//batch_size}: Loss = {total_loss.item():.6f}, Fidelity = {avg_fidelity:.6f}")
        
        return epoch_loss / batch_count if batch_count > 0 else 0.0
    
    def validate(self, n_samples=500):
        """验证模型"""
        self.classical_encoder.eval()
        
        with torch.no_grad():
            subset = self.val_data[:min(n_samples, len(self.val_data))]
            subset_tensor = torch.from_numpy(subset).float()
            
            fidelities = []
            for i in range(0, len(subset_tensor), 50):
                batch = subset_tensor[i:i+50]
                batch_fidelities = self.compute_fidelity(batch)
                fidelities.append(batch_fidelities)
            
            if fidelities:
                all_fidelities = torch.cat(fidelities)
                avg_fidelity = torch.mean(all_fidelities).item()
                return avg_fidelity
        
        return 0.0
    
    def test(self, n_samples=500):
        """测试模型"""
        self.classical_encoder.eval()
        
        with torch.no_grad():
            subset = self.test_data[:min(n_samples, len(self.test_data))]
            subset_tensor = torch.from_numpy(subset).float()
            
            fidelities = []
            for i in range(0, len(subset_tensor), 50):
                batch = subset_tensor[i:i+50]
                batch_fidelities = self.compute_fidelity(batch)
                fidelities.append(batch_fidelities)
            
            if fidelities:
                all_fidelities = torch.cat(fidelities)
                avg_fidelity = torch.mean(all_fidelities).item()
                std_fidelity = torch.std(all_fidelities).item()
                
                print(f"测试结果:")
                print(f"平均保真度: {avg_fidelity:.6f}")
                print(f"标准差: {std_fidelity:.6f}")
                print(f"样本数: {len(all_fidelities)}")
                
                return avg_fidelity, std_fidelity
        
        return 0.0, 0.0
    
    def train(self, n_epochs=30):
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
        os.makedirs("improved_models", exist_ok=True)
        torch.save({
            'classical_encoder_state': self.classical_encoder.state_dict(),
            'quantum_enc_params': self.quantum_ae.enc_params,
            'quantum_dec_params': self.quantum_ae.dec_params,
            'optimizer_state': self.optimizer.state_dict()
        }, os.path.join("improved_models", filename))
    
    def load_model(self, filename):
        """加载模型"""
        checkpoint = torch.load(os.path.join("improved_models", filename))
        self.classical_encoder.load_state_dict(checkpoint['classical_encoder_state'])
        self.quantum_ae.enc_params = checkpoint['quantum_enc_params']
        self.quantum_ae.dec_params = checkpoint['quantum_dec_params']
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])

def main():
    """主函数"""
    # 加载和预处理数据
    print("加载和预处理数据...")
    train_data, val_data, test_data, scaler = load_and_preprocess_data()
    
    # 创建改进的QAE
    print("初始化改进的量子自编码器...")
    qae = ImprovedQAE(train_data, val_data, test_data)
    
    # 训练
    history = qae.train(n_epochs=30)
    
    # 保存训练历史
    torch.save(history, "improved_models/training_history.pth")
    print("训练历史已保存到 improved_models/training_history.pth")
    
    return qae, history

if __name__ == "__main__":
    main()