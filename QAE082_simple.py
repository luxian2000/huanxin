import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from sklearn.preprocessing import StandardScaler

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

class ExperimentConfig:
    """实验配置管理类"""
    def __init__(self):
        self.config = {
            'model_name': 'QAE082_Simple',
            'input_dim': 2560,
            'compressed_dim': 512,
            'output_dim': 2560,
            'n_qubits': 9,              # 使用9量子比特 (2^9=512)
            'n_layers': 4,
            'n_epochs': 100,
            'batch_size': 100,
            'learning_rate': 0.001,
            'train_samples': 2000,
            'val_samples': 500,
            'test_samples': 500,
            'regularization_weight': 0.01
        }
    
    def save_config(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get_config(self):
        return self.config

def load_data(data_path="/Users/luxian/DataSpace/csi_cmri/CSI_channel_30km.npy"):
    """安全地加载数据文件"""
    possible_paths = [
        data_path,
        "./CSI_channel_30km.npy",
        f"../DataSpace/csi_cmri/CSI_channel_30km.npy",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Loading data from: {path}")
            data = np.load(path)[:10000]  # 使用前10000个样本进行快速测试
            return data.astype(np.float32)
    
    print("Data file not found. Creating small test dataset...")
    return np.random.randn(10000, 2560).astype(np.float32)

# CsiNet经典编码器
class CsiNetEncoder(nn.Module):
    def __init__(self, input_dim=2560, compressed_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, compressed_dim),
            nn.BatchNorm1d(compressed_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        return self.encoder(x)

# 简化的量子解码器
class SimpleQuantumDecoder(nn.Module):
    def __init__(self, compressed_dim=512, output_dim=2560, n_qubits=9, n_layers=4):
        super().__init__()
        self.compressed_dim = compressed_dim
        self.output_dim = output_dim
        self.n_qubits = n_qubits
        
        # 量子参数
        self.quantum_weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3) * 0.1
        )
        
        # 经典扩展层
        self.expansion_layer = nn.Linear(2**n_qubits, output_dim)
    
    def quantum_circuit(self, inputs, weights):
        # 创建量子设备和节点
        dev = qml.device("lightning.qubit", wires=self.n_qubits)
        
        @qml.qnode(dev, interface="torch")
        def circuit(input_data, weight_params):
            # 振幅编码
            qml.AmplitudeEmbedding(input_data, wires=range(self.n_qubits), pad_with=0.0, normalize=True)
            
            # 参数化层
            qml.StronglyEntanglingLayers(weights=weight_params, wires=range(self.n_qubits))
            
            # 测量所有量子比特
            return qml.probs(wires=range(self.n_qubits))
        
        return circuit(inputs, weights)
    
    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        batch_size = x.shape[0]
        outputs = []
        
        # 为每个样本执行量子计算
        for i in range(batch_size):
            sample = x[i]
            # pad到512维以适应9量子比特
            if sample.shape[0] < 512:
                padded_sample = F.pad(sample, (0, 512-sample.shape[0]))
            else:
                padded_sample = sample[:512]
            
            # 量子处理
            quantum_probs = self.quantum_circuit(padded_sample.detach().numpy(), self.quantum_weights)
            quantum_output = torch.tensor(quantum_probs, dtype=torch.float32)
            outputs.append(quantum_output)
        
        # 合并结果
        if batch_size > 1:
            quantum_batch = torch.stack(outputs)
        else:
            quantum_batch = outputs[0].unsqueeze(0)
        
        # 经典扩展到目标维度
        final_output = self.expansion_layer(quantum_batch)
        
        return final_output

# 数据加载
data = load_data()
config = ExperimentConfig()
cfg = config.get_config()

# 简单数据划分
train_data = data[:8000]
val_data = data[8000:9000]
test_data = data[9000:10000]

print(f"数据集大小: 训练={len(train_data)}, 验证={len(val_data)}, 测试={len(test_data)}")

# 初始化模型
encoder = CsiNetEncoder(cfg['input_dim'], cfg['compressed_dim'])
decoder = SimpleQuantumDecoder(cfg['compressed_dim'], cfg['output_dim'], 
                              cfg['n_qubits'], cfg['n_layers'])

def compute_loss(reconstructed, original):
    return F.mse_loss(reconstructed, original)

def train_step(batch_x):
    # 编码
    compressed = encoder(batch_x)
    # 解码
    reconstructed = decoder(compressed)
    # 计算损失
    loss = compute_loss(reconstructed, batch_x)
    return loss, reconstructed

def validate_model():
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        val_batch = torch.from_numpy(val_data[:cfg['val_samples']]).float()
        compressed = encoder(val_batch)
        reconstructed = decoder(compressed)
        loss = compute_loss(reconstructed, val_batch)
    encoder.train()
    decoder.train()
    return loss.item()

# 训练循环
optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=cfg['learning_rate']
)

print("开始简化版QAE082训练...")
print(f"配置: {cfg}")

for epoch in range(cfg['n_epochs']):
    epoch_loss = 0
    num_batches = 0
    
    # 随机采样训练数据
    indices = np.random.choice(len(train_data), cfg['train_samples'], replace=False)
    train_batch = torch.from_numpy(train_data[indices]).float()
    
    # 分批处理
    for i in range(0, cfg['train_samples'], cfg['batch_size']):
        batch_x = train_batch[i:i+cfg['batch_size']]
        
        optimizer.zero_grad()
        loss, _ = train_step(batch_x)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1
    
    avg_loss = epoch_loss / num_batches
    val_loss = validate_model()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Train Loss = {avg_loss:.6f}, Val Loss = {val_loss:.6f}")

print("训练完成!")

# 测试
encoder.eval()
decoder.eval()
with torch.no_grad():
    test_batch = torch.from_numpy(test_data[:cfg['test_samples']]).float()
    compressed = encoder(test_batch)
    reconstructed = decoder(compressed)
    test_loss = compute_loss(reconstructed, test_batch)
    print(f"测试损失: {test_loss.item():.6f}")

# 保存模型
os.makedirs("QAE082", exist_ok=True)
torch.save(encoder.state_dict(), "QAE082/csinet_encoder_final.pt")
torch.save(decoder.state_dict(), "QAE082/quantum_decoder_final.pt")
print("模型已保存到QAE082目录")