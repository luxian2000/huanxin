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
            'model_name': 'QAE_Improved',
            'input_dim': 2560,
            'hidden_dims': [1024, 512],
            'latent_dim': 256,
            'n_qubits': 8,
            'n_layers': 6,  # 从4层增加到6层
            'n_epochs': 300,
            'batch_size': 200,  # 增加批量大小
            'learning_rate': 0.001,  # 降低学习率
            'train_samples': 5000,   # 增加训练样本
            'val_samples': 1000,
            'test_samples': 1000,
            'regularization_weight': 0.01,
            'gradient_clip_norm': 5.0,
            'early_stopping_patience': 30
        }
    
    def save_config(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get_config(self):
        return self.config

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

# 改进的经典编码器
class ImprovedClassicalEncoder(nn.Module):
    """深层经典编码器with BatchNorm和Dropout"""
    def __init__(self, input_dim=2560, hidden_dims=[1024, 512], output_dim=256):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        # 构建深层网络
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),  # 批归一化
                nn.ReLU(),                   # ReLU激活
                nn.Dropout(0.2)              # Dropout防过拟合
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        # 保存原始维度信息
        original_dim = x.dim()
        # 如果输入是单个样本，添加批次维度
        if original_dim == 1:
            x = x.unsqueeze(0)
        
        # 设置BatchNorm为评估模式以处理单样本
        if x.size(0) == 1:
            self.network.eval()
            with torch.no_grad():
                output = self.network(x)
            self.network.train()
        else:
            output = self.network(x)
        
        # L2归一化
        output = F.normalize(output, p=2, dim=1)
        
        # 如果原始输入是单个样本，移除批次维度
        if original_dim == 1:
            output = output.squeeze(0)
        
        return output

# 数据加载和预处理
data_30 = load_data()
config = ExperimentConfig()
cfg = config.get_config()

# 改进的数据划分
TRAIN_RATIO = 0.75  # 增加训练比例
VAL_RATIO = 0.125
TEST_RATIO = 0.125

train_size = int(len(data_30) * TRAIN_RATIO)
val_size = int(len(data_30) * VAL_RATIO)
test_size = len(data_30) - train_size - val_size

train_data = data_30[:train_size]
val_data = data_30[train_size:train_size + val_size]
test_data = data_30[train_size + val_size:]

print("改进的数据划分结果:")
print(f"训练集: {len(train_data)} 个样本 ({TRAIN_RATIO*100:.1f}%)")
print(f"验证集: {len(val_data)} 个样本 ({VAL_RATIO*100:.1f}%)")
print(f"测试集: {len(test_data)} 个样本 ({TEST_RATIO*100:.1f}%)")

# 模型参数
INPUT_DIM = cfg['input_dim']
HIDDEN_DIMS = cfg['hidden_dims']
OUTPUT_DIM = cfg['latent_dim']
N_LAYERS = cfg['n_layers']
DATA_QUBITS = cfg['n_qubits']

print(f"改进参数: DATA_QUBITS={DATA_QUBITS}, N_LAYERS={N_LAYERS}")

# 初始化改进的经典编码器
classical_encoder = ImprovedClassicalEncoder(INPUT_DIM, HIDDEN_DIMS, OUTPUT_DIM)

# 确保批量大小至少为2，避免BatchNorm错误
MIN_BATCH_SIZE = 2
if cfg['batch_size'] < MIN_BATCH_SIZE:
    cfg['batch_size'] = MIN_BATCH_SIZE
    print(f"调整批量大小至最小值: {MIN_BATCH_SIZE}")

# 量子设备
DEV = qml.device("lightning.qubit", wires=DATA_QUBITS)

@qml.qnode(DEV, interface="torch")
def improved_qae_circuit(classical_params, enc_params, dec_params):
    """改进的量子自编码器电路"""
    # 振幅编码
    qml.AmplitudeEmbedding(classical_params, wires=range(DATA_QUBITS), pad_with=0.0, normalize=True)
    
    # 改进的编码器层
    qml.StronglyEntanglingLayers(weights=enc_params, wires=range(DATA_QUBITS))
    
    # 改进的解码器层
    qml.StronglyEntanglingLayers(weights=dec_params, wires=range(DATA_QUBITS))
    
    # 保真度计算
    qml.adjoint(qml.AmplitudeEmbedding)(classical_params, wires=range(DATA_QUBITS), pad_with=0.0, normalize=True)
    return qml.expval(qml.Projector([0]*DATA_QUBITS, wires=range(DATA_QUBITS)))

def pad_to_qubits(vec, n_qubits):
    """填充向量到指定量子比特数"""
    target_len = 2 ** n_qubits
    if len(vec) < target_len:
        return F.pad(vec, (0, target_len - len(vec)))
    return vec[:target_len]

def process_batch(batch, enc_params, dec_params):
    """处理批次数据"""
    fidelities = []
    for sample in batch:
        if isinstance(sample, np.ndarray):
            sample = torch.from_numpy(sample).float()
        
        # 使用改进的经典编码器
        classical_vec = classical_encoder(sample)
        classical_vec_padded = pad_to_qubits(classical_vec, DATA_QUBITS)
        
        # 量子电路计算
        fid = improved_qae_circuit(classical_vec_padded.detach().numpy(), enc_params, dec_params)
        fidelities.append(fid)
    
    return torch.stack(fidelities)

def validate_model(enc_params, dec_params, val_samples=cfg['val_samples']):
    """验证模型性能"""
    try:
        subset = val_data[:min(val_samples, len(val_data))]
        fids = process_batch(subset, enc_params, dec_params)
        return float(torch.mean(fids))
    except Exception as e:
        print(f"Validation error: {e}")
        return float("nan")

def improved_loss_function(fidelities, enc_params, dec_params, reg_weight=cfg['regularization_weight']):
    """改进的损失函数with正则化"""
    # 主损失
    main_loss = 1.0 - torch.mean(fidelities)
    
    # L2正则化
    l2_reg = reg_weight * (torch.norm(enc_params) + torch.norm(dec_params))
    
    # 梯度范数惩罚
    grad_penalty = 0
    if enc_params.grad is not None:
        grad_penalty += torch.norm(enc_params.grad)
    if dec_params.grad is not None:
        grad_penalty += torch.norm(dec_params.grad)
    
    return main_loss + l2_reg + 0.001 * grad_penalty

def monitor_gradients(enc_params, dec_params):
    """监控梯度信息"""
    grad_info = {
        'enc_grad_norm': torch.norm(enc_params.grad).item() if enc_params.grad is not None else 0,
        'dec_grad_norm': torch.norm(dec_params.grad).item() if dec_params.grad is not None else 0,
        'enc_param_norm': torch.norm(enc_params).item(),
        'dec_param_norm': torch.norm(dec_params).item()
    }
    return grad_info

def get_training_batch(epoch, batch_size=cfg['batch_size']):
    """动态批量采样"""
    indices = np.random.choice(len(train_data), batch_size, replace=False)
    return [train_data[i] for i in indices]

def train_improved():
    """改进的训练函数"""
    try:
        # 保存配置
        config.save_config('experiment_config.json')
        # 使用QAE_08相同的目录结构
        output_dir = "QAE_081"
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化参数
        enc_shape = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=DATA_QUBITS)
        dec_shape = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=DATA_QUBITS)
        enc_params = torch.rand(enc_shape, requires_grad=True)
        dec_params = torch.rand(dec_shape, requires_grad=True)
        
        # 保存初始参数（按照QAE_08格式）
        torch.save(classical_encoder.state_dict(), f"{output_dir}/initial_weight.pt")
        torch.save(enc_params, f"{output_dir}/initial_quantum_encoder_weights.pt")
        torch.save(dec_params, f"{output_dir}/initial_quantum_decoder_weights.pt")
        print("初始参数已保存!")
        
        # 改进的优化器和调度器
        opt = torch.optim.Adam([
            {'params': classical_encoder.parameters(), 'lr': cfg['learning_rate']},
            {'params': [enc_params, dec_params], 'lr': cfg['learning_rate']}
        ])
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='max', factor=0.5, patience=15
        )
        
        # 早停机制
        best_val_fid = 0
        patience_counter = 0
        max_patience = cfg['early_stopping_patience']
        
        n_epochs = 10  # 训练10个完整epoch
        batch_size = cfg['batch_size']
        
        training_history = {
            "epoch_losses": [],
            "val_fidelity": [],
            "batch_losses": [],
            "gradient_info": [],
            "weights_history": [],
            "config": cfg,
            "data_split_info": {
                "train_size": len(train_data),
                "val_size": len(val_data),
                "test_size": len(test_data),
                "actual_train_used": cfg['train_samples'],
            },
        }
        
        print("开始改进的QAE训练...")
        print(f"配置: {cfg}")
        start_time = time.time()
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            # 每个epoch使用新的随机样本
            samples = get_training_batch(epoch, cfg['train_samples'])
            
            for i in range(0, len(samples), batch_size):
                batch = samples[i:i+batch_size]
                
                # 前向传播
                opt.zero_grad()
                fidelities = process_batch(batch, enc_params, dec_params)
                loss = improved_loss_function(fidelities, enc_params, dec_params)
                loss.backward()
                
                # 梯度监控和裁剪
                grad_info = monitor_gradients(enc_params, dec_params)
                if grad_info['enc_grad_norm'] > 10 or grad_info['dec_grad_norm'] > 10:
                    torch.nn.utils.clip_grad_norm_([enc_params, dec_params], max_norm=cfg['gradient_clip_norm'])
                
                # 更新参数
                opt.step()
                
                current_loss = loss.item()
                epoch_loss += current_loss
                batch_count += 1
                
                # 记录批次信息
                training_history["batch_losses"].append({
                    "epoch": epoch,
                    "batch": i // batch_size,
                    "loss": float(current_loss),
                    "grad_info": grad_info
                })
                
                if (i // batch_size) % 5 == 0:
                    print(f"Epoch {epoch}, Batch {i//batch_size}: loss = {current_loss:.6f}")
                    print(f"  梯度范数 - 编码器: {grad_info['enc_grad_norm']:.4f}, 解码器: {grad_info['dec_grad_norm']:.4f}")
            
            if batch_count > 0:
                avg_epoch_loss = epoch_loss / batch_count
                val_fid = validate_model(enc_params, dec_params, val_samples=cfg['val_samples'])
                
                # 更新学习率
                scheduler.step(val_fid)
                
                # 记录epoch信息
                training_history["epoch_losses"].append({
                    "epoch": epoch, 
                    "avg_loss": float(avg_epoch_loss)
                })
                training_history["val_fidelity"].append({
                    "epoch": epoch, 
                    "val_fidelity": float(val_fid)
                })
                
                # 保存权重快照（按照QAE_08格式）
                enc_snapshot = enc_params.clone().detach()
                dec_snapshot = dec_params.clone().detach()
                training_history["weights_history"].append({
                    "encoder": enc_snapshot.numpy(),
                    "decoder": dec_snapshot.numpy(),
                })
                
                torch.save(enc_snapshot, f"{output_dir}/qae_encoder_epoch_{epoch}.pt")
                torch.save(dec_snapshot, f"{output_dir}/qae_decoder_epoch_{epoch}.pt")
                
                print(f"Epoch {epoch} completed:")
                print(f"  训练损失: {avg_epoch_loss:.6f}")
                print(f"  验证保真度: {val_fid:.6f}")
                print(f"  学习率: {opt.param_groups[0]['lr']:.6f}")
                print("-" * 50)
                
                # 早停检查
                if val_fid > best_val_fid:
                    best_val_fid = val_fid
                    patience_counter = 0
                    # 保存最佳模型
                    torch.save(enc_params, f"{output_dir}/best_qae_encoder_weights.pt")
                    torch.save(dec_params, f"{output_dir}/best_qae_decoder_weights.pt")
                    print(f"  保存最佳模型 (保真度: {val_fid:.6f})")
                else:
                    patience_counter += 1
                    
                if patience_counter >= max_patience:
                    print(f"早停触发，最佳验证保真度: {best_val_fid:.6f}")
                    break
        
        total_time = time.time() - start_time
        print(f"改进训练完成! 总时间: {total_time:.2f}秒")
        print(f"最佳验证保真度: {best_val_fid:.6f}")
        
        # 保存最终结果（按照QAE_08格式）
        torch.save(enc_params, f"{output_dir}/final_qae_encoder_weights.pt")
        torch.save(dec_params, f"{output_dir}/final_qae_decoder_weights.pt")
        torch.save(training_history, f"{output_dir}/qae_training_history.pt")
        print("最终模型和训练历史已保存!")
        
        return (enc_params, dec_params), training_history
        
    except Exception as e:
        print(f"训练错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def continue_training_from_saved_weights(starting_epoch=0, additional_epochs=9):
    """从已保存的权重继续训练"""
    try:
        # 使用QAE_08相同的目录结构
        output_dir = "QAE_081"
        
        # 检查是否存在最佳权重文件
        best_encoder_path = f"{output_dir}/best_qae_encoder_weights.pt"
        best_decoder_path = f"{output_dir}/best_qae_decoder_weights.pt"
        
        if os.path.exists(best_encoder_path) and os.path.exists(best_decoder_path):
            print("检测到已有的最佳权重，将从这些权重继续训练...")
            enc_params = torch.load(best_encoder_path)
            dec_params = torch.load(best_decoder_path)
            print(f"加载最佳权重: 编码器形状 {enc_params.shape}, 解码器形状 {dec_params.shape}")
        else:
            print("未检测到最佳权重文件，将从初始权重开始训练...")
            # 加载初始权重
            enc_params = torch.load(f"{output_dir}/initial_quantum_encoder_weights.pt")
            dec_params = torch.load(f"{output_dir}/initial_quantum_decoder_weights.pt")
        
        # 确保参数需要梯度
        enc_params.requires_grad_(True)
        dec_params.requires_grad_(True)
        
        # 加载经典编码器权重
        classical_encoder.load_state_dict(torch.load(f"{output_dir}/initial_weight.pt"))
        
        # 改进的优化器和调度器
        opt = torch.optim.Adam([
            {'params': classical_encoder.parameters(), 'lr': cfg['learning_rate']},
            {'params': [enc_params, dec_params], 'lr': cfg['learning_rate']}
        ])
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='max', factor=0.5, patience=15
        )
        
        # 早停机制
        best_val_fid = 0
        patience_counter = 0
        max_patience = cfg['early_stopping_patience']
        
        n_epochs = additional_epochs
        batch_size = cfg['batch_size']
        
        print(f"开始延续训练: 从epoch {starting_epoch} 开始，训练 {additional_epochs} 个额外epoch")
        print(f"配置: {cfg}")
        start_time = time.time()
        
        for epoch in range(n_epochs):
            global_epoch = starting_epoch + epoch
            epoch_loss = 0.0
            batch_count = 0
            
            # 每个epoch使用新的随机样本
            samples = get_training_batch(global_epoch, cfg['train_samples'])
            
            for i in range(0, len(samples), batch_size):
                batch = samples[i:i+batch_size]
                
                # 前向传播
                opt.zero_grad()
                fidelities = process_batch(batch, enc_params, dec_params)
                loss = improved_loss_function(fidelities, enc_params, dec_params)
                loss.backward()
                
                # 梯度监控和裁剪
                grad_info = monitor_gradients(enc_params, dec_params)
                if grad_info['enc_grad_norm'] > 10 or grad_info['dec_grad_norm'] > 10:
                    torch.nn.utils.clip_grad_norm_([enc_params, dec_params], max_norm=cfg['gradient_clip_norm'])
                
                # 更新参数
                opt.step()
                
                current_loss = loss.item()
                epoch_loss += current_loss
                batch_count += 1
                
                if (i // batch_size) % 5 == 0:
                    print(f"Epoch {global_epoch}, Batch {i//batch_size}: loss = {current_loss:.6f}")
                    print(f"  梯度范数 - 编码器: {grad_info['enc_grad_norm']:.4f}, 解码器: {grad_info['dec_grad_norm']:.4f}")
            
            if batch_count > 0:
                avg_epoch_loss = epoch_loss / batch_count
                val_fid = validate_model(enc_params, dec_params, val_samples=cfg['val_samples'])
                
                # 更新学习率
                scheduler.step(val_fid)
                
                print(f"Epoch {global_epoch} completed:")
                print(f"  训练损失: {avg_epoch_loss:.6f}")
                print(f"  验证保真度: {val_fid:.6f}")
                print(f"  学习率: {opt.param_groups[0]['lr']:.6f}")
                print("-" * 50)
                
                # 保存权重快照（按照QAE_08格式）
                enc_snapshot = enc_params.clone().detach()
                dec_snapshot = dec_params.clone().detach()
                torch.save(enc_snapshot, f"{output_dir}/qae_encoder_epoch_{global_epoch}.pt")
                torch.save(dec_snapshot, f"{output_dir}/qae_decoder_epoch_{global_epoch}.pt")
                
                # 早停检查
                if val_fid > best_val_fid:
                    best_val_fid = val_fid
                    patience_counter = 0
                    # 保存最佳模型
                    torch.save(enc_params, f"{output_dir}/best_qae_encoder_weights.pt")
                    torch.save(dec_params, f"{output_dir}/best_qae_decoder_weights.pt")
                    print(f"  保存最佳模型 (保真度: {val_fid:.6f})")
                else:
                    patience_counter += 1
                    
                if patience_counter >= max_patience:
                    print(f"早停触发，最佳验证保真度: {best_val_fid:.6f}")
                    break
        
        total_time = time.time() - start_time
        print(f"延续训练完成! 总时间: {total_time:.2f}秒")
        print(f"最佳验证保真度: {best_val_fid:.6f}")
        
        # 保存最终结果（按照QAE_08格式）
        torch.save(enc_params, f"{output_dir}/final_qae_encoder_weights.pt")
        torch.save(dec_params, f"{output_dir}/final_qae_decoder_weights.pt")
        print("最终模型已保存!")
        
        return (enc_params, dec_params)
        
    except Exception as e:
        print(f"延续训练错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_trained_model(weights=None, test_samples=cfg['test_samples'], use_saved_weights=True):
    """测试训练好的模型"""
    print("\n测试改进的QAE模型...")
    try:
        # 如果没有提供权重且要求使用保存的权重，则从QAE_081目录加载
        if use_saved_weights and weights is None:
            output_dir = "QAE_081"
            enc_path = os.path.join(output_dir, "final_qae_encoder_weights.pt")
            dec_path = os.path.join(output_dir, "final_qae_decoder_weights.pt")
            
            if os.path.exists(enc_path) and os.path.exists(dec_path):
                enc_params = torch.load(enc_path)
                dec_params = torch.load(dec_path)
                print(f"从 {output_dir} 目录加载权重文件")
            else:
                print(f"权重文件未找到，使用随机初始化权重")
                enc_shape = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=DATA_QUBITS)
                dec_shape = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=DATA_QUBITS)
                enc_params = torch.rand(enc_shape, requires_grad=True)
                dec_params = torch.rand(dec_shape, requires_grad=True)
        elif weights is not None:
            enc_params, dec_params = weights
        else:
            # 默认使用随机权重
            enc_shape = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=DATA_QUBITS)
            dec_shape = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=DATA_QUBITS)
            enc_params = torch.rand(enc_shape, requires_grad=True)
            dec_params = torch.rand(dec_shape, requires_grad=True)
        
        subset = test_data[:min(test_samples, len(test_data))]
        fidelities = process_batch(subset, enc_params, dec_params)
        avg_fid = torch.mean(fidelities).item()
        std_fid = torch.std(fidelities).item()
        print(f"测试集平均保真度 ({len(subset)} 样本): {avg_fid:.6f} (标准差: {std_fid:.6f})")
        return fidelities
    except Exception as e:
        print(f"测试错误: {e}")
        return None

def main():
    """主函数"""
    print("=== QAE_081 改进版量子自编码器 ===")
    print("实施的改进:")
    print("1. 深层经典编码器 (BatchNorm + Dropout)")
    print("2. 增加量子电路层数 (4→6层)")
    print("3. 增加训练样本 (1000→5000)")
    print("4. 改进损失函数 (L2正则化 + 梯度惩罚)")
    print("5. 动态学习率调度")
    print("6. 早停机制")
    print("7. 完整的实验配置管理")
    print("=" * 50)
    
    # 检查是否已有训练过的权重
    output_dir = "QAE_081"
    best_encoder_path = f"{output_dir}/best_qae_encoder_weights.pt"
    best_decoder_path = f"{output_dir}/best_qae_decoder_weights.pt"
    
    if os.path.exists(best_encoder_path) and os.path.exists(best_decoder_path):
        print("检测到已有的训练权重，将使用延续训练模式...")
        print("在现有10个epoch基础上再训练10个epoch")
        weights = continue_training_from_saved_weights(starting_epoch=10, additional_epochs=10)
    else:
        print("未检测到已有的训练权重，将从头开始训练...")
        weights, history = train_improved()
        
    if weights is not None:
        # 测试训练好的模型
        test_results = test_trained_model(weights)
        if test_results is not None:
            print(f"\n最终测试结果: 平均保真度 = {torch.mean(test_results).item():.6f}")
        
        # 显示保存的文件
        print(f"\n保存的文件在 '{output_dir}' 目录中:")
        if os.path.exists(output_dir):
            for file in sorted(os.listdir(output_dir)):
                file_path = os.path.join(output_dir, file)
                size = os.path.getsize(file_path)
                print(f"  - {file} ({size/1024:.1f}KB)")
    else:
        print("训练失败.")

if __name__ == "__main__":
    main()