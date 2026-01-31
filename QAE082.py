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
            'model_name': 'QAE082_Hybrid',
            'input_dim': 2560,
            'compressed_dim': 512,
            'output_dim': 2560,
            'n_qubits': 9,
            'n_layers': 4,
            'n_epochs': 100,
            'batch_size': 100,
            'learning_rate': 0.001,
            'train_samples': 2000,
            'val_samples': 500,
            'test_samples': 500,
            'regularization_weight': 0.01,
            'gradient_clip_norm': 5.0
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
            data = np.load(path)[:10000]  # 使用前10000个样本
            return data.astype(np.float32)
    
    print("Data file not found. Creating test dataset...")
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

# 量子解码器
class QuantumDecoder(nn.Module):
    def __init__(self, compressed_dim=512, output_dim=2560, n_qubits=9, n_layers=4):
        super().__init__()
        self.compressed_dim = compressed_dim
        self.output_dim = output_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
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
            # pad到512维以适应9量子比特 (2^9=512)
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

# 数据加载和划分
data = load_data()
config = ExperimentConfig()
cfg = config.get_config()

train_data = data[:8000]
val_data = data[8000:9000]
test_data = data[9000:10000]

print(f"数据集大小: 训练={len(train_data)}, 验证={len(val_data)}, 测试={len(test_data)}")

# 初始化模型
encoder = CsiNetEncoder(cfg['input_dim'], cfg['compressed_dim'])
decoder = QuantumDecoder(cfg['compressed_dim'], cfg['output_dim'], 
                        cfg['n_qubits'], cfg['n_layers'])

def compute_loss(reconstructed, original):
    return F.mse_loss(reconstructed, original)

def train_model():
    """训练混合模型"""
    try:
        # 保存配置和初始参数
        config.save_config('experiment_config.json')
        output_dir = "QAE082"
        os.makedirs(output_dir, exist_ok=True)
        
        torch.save(encoder.state_dict(), f"{output_dir}/initial_csinet_encoder.pt")
        torch.save(decoder.state_dict(), f"{output_dir}/initial_quantum_decoder.pt")
        print("初始参数已保存!")
        
        # 优化器
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=cfg['learning_rate']
        )
        
        print("开始QAE082混合模型训练...")
        print(f"配置: {cfg}")
        start_time = time.time()
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 20
        
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
                # 编码
                compressed = encoder(batch_x)
                # 解码
                reconstructed = decoder(compressed)
                # 计算损失
                loss = compute_loss(reconstructed, batch_x)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(decoder.parameters()), 
                    max_norm=cfg['gradient_clip_norm']
                )
                
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            
            # 验证
            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                val_indices = np.random.choice(len(val_data), cfg['val_samples'], replace=False)
                val_batch = torch.from_numpy(val_data[val_indices]).float()
                compressed = encoder(val_batch)
                reconstructed = decoder(compressed)
                val_loss = compute_loss(reconstructed, val_batch)
            encoder.train()
            decoder.train()
            
            print(f"Epoch {epoch}: Train Loss = {avg_loss:.6f}, Val Loss = {val_loss:.6f}")
            
            # 保存检查点
            if epoch % 20 == 0:
                torch.save(encoder.state_dict(), f"{output_dir}/csinet_encoder_epoch_{epoch}.pt")
                torch.save(decoder.state_dict(), f"{output_dir}/quantum_decoder_epoch_{epoch}.pt")
            
            # 早停机制
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(encoder.state_dict(), f"{output_dir}/best_csinet_encoder.pt")
                torch.save(decoder.state_dict(), f"{output_dir}/best_quantum_decoder.pt")
            else:
                patience_counter += 1
                
            if patience_counter >= max_patience:
                print(f"早停触发，最佳验证损失: {best_val_loss:.6f}")
                break
        
        total_time = time.time() - start_time
        print(f"训练完成! 总时间: {total_time:.2f}秒")
        print(f"最佳验证损失: {best_val_loss:.6f}")
        
        # 保存最终结果
        torch.save(encoder.state_dict(), f"{output_dir}/final_csinet_encoder.pt")
        torch.save(decoder.state_dict(), f"{output_dir}/final_quantum_decoder.pt")
        print("最终模型已保存!")
        
        return encoder, decoder
        
    except Exception as e:
        print(f"训练错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_model(models=None):
    """测试模型"""
    print("\n测试QAE082混合模型...")
    try:
        if models is None:
            # 加载最佳模型
            output_dir = "QAE082"
            encoder_path = os.path.join(output_dir, "best_csinet_encoder.pt")
            decoder_path = os.path.join(output_dir, "best_quantum_decoder.pt")
            
            if os.path.exists(encoder_path) and os.path.exists(decoder_path):
                encoder.load_state_dict(torch.load(encoder_path))
                decoder.load_state_dict(torch.load(decoder_path))
                print("加载最佳模型权重")
            else:
                print("未找到训练好的模型")
                return None
        else:
            encoder, decoder = models
        
        # 测试
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            test_batch = torch.from_numpy(test_data[:cfg['test_samples']]).float()
            compressed = encoder(test_batch)
            reconstructed = decoder(compressed)
            test_loss = compute_loss(reconstructed, test_batch)
            
        print(f"测试损失: {test_loss.item():.6f}")
        return test_loss.item()
        
    except Exception as e:
        print(f"测试错误: {e}")
        return None

def main():
    """主函数"""
    print("=== QAE082 混合量子-经典解码器 ===")
    print("模型架构:")
    print("1. 编码器: CsiNet经典神经网络 (2560→512)")
    print("2. 解码器: 量子全连接层 (512→512量子态→2560)")
    print("3. 量子电路: 9量子比特, 4层强纠缠结构")
    print("=" * 50)
    
    # 检查是否已有训练权重
    output_dir = "QAE082"
    best_encoder_path = f"{output_dir}/best_csinet_encoder.pt"
    best_decoder_path = f"{output_dir}/best_quantum_decoder.pt"
    
    if os.path.exists(best_encoder_path) and os.path.exists(best_decoder_path):
        print("检测到已有的训练权重，直接测试...")
        test_result = test_model()
    else:
        print("开始训练模型...")
        models = train_model()
        if models[0] is not None:
            test_result = test_model(models)
            
            # 显示保存的文件
            print(f"\n保存的文件在 '{output_dir}' 目录中:")
            if os.path.exists(output_dir):
                for file in sorted(os.listdir(output_dir)):
                    file_path = os.path.join(output_dir, file)
                    size = os.path.getsize(file_path)
                    print(f"  - {file} ({size/1024:.1f}KB)")

if __name__ == "__main__":
    main()