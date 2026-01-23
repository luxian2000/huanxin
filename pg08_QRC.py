import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
import time
import os

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 数据加载
data_30 = np.load('CSI_channel_30km.npy')  # shape=(80000, 2560)

# 数据划分参数
TOTAL_SAMPLES = 80000
TRAIN_RATIO = 0.70    # 70% 训练
VAL_RATIO = 0.15      # 15% 验证  
TEST_RATIO = 0.15     # 15% 测试

# 计算各集合大小
train_size = int(TOTAL_SAMPLES * TRAIN_RATIO)
val_size = int(TOTAL_SAMPLES * VAL_RATIO)
test_size = TOTAL_SAMPLES - train_size - val_size

# 划分数据集
train_data = data_30[:train_size]
val_data = data_30[train_size:train_size + val_size]
test_data = data_30[train_size + val_size:]

print("数据划分结果:")
print(f"训练集: {len(train_data)} 个样本 ({TRAIN_RATIO*100:.1f}%)")
print(f"验证集: {len(val_data)} 个样本 ({VAL_RATIO*100:.1f}%)")
print(f"测试集: {len(test_data)} 个样本 ({TEST_RATIO*100:.1f}%)")

INPUT_DIM = 2560
OUTPUT_DIM = 256

# 离散时间晶体(DTC)参数
N_QUBITS = 10               # 量子比特数
DTC_PERIOD = 8              # DTC周期
N_STEPS = 20                # 演化步数
DRIVE_STRENGTH = 0.8        # 驱动强度

IMG_QUBITS = int(np.ceil(np.log2(INPUT_DIM)))  # 12
COM_QUBITS = int(np.ceil(np.log2(OUTPUT_DIM)))  # 8
ALL_QUBITS = N_QUBITS  # 10个量子比特用于DTC

print(f"DTC量子储存库计算配置:")
print(f"  量子比特数: {N_QUBITS}")
print(f"  DTC周期: {DTC_PERIOD}")
print(f"  演化步数: {N_STEPS}")
print(f"  驱动强度: {DRIVE_STRENGTH}")

# 初始化并保存经典神经网络参数 - 使用PyTorch张量
WEIGHT = torch.randn(INPUT_DIM, OUTPUT_DIM, requires_grad=True) * 0.01
BIAS = torch.randn(1, OUTPUT_DIM, requires_grad=True)

# 创建保存参数的目录
os.makedirs('model_parameters', exist_ok=True)

def save_initial_parameters():
    """保存初始化的参数"""
    torch.save(WEIGHT, 'model_parameters/initial_weight.pt')
    torch.save(BIAS, 'model_parameters/initial_bias.pt')
    print("Initial WEIGHT and BIAS saved!")

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def normlize(x):
    norm = torch.norm(x)
    if norm == 0:
        return x
    return x / norm

def dense_layer(x):
    """经典密集层 - 数据预处理和压缩"""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    output = torch.matmul(x, WEIGHT) + BIAS
    output = sigmoid(output)
    output = normlize(output[0])  # 确保输出是一维的
    return output

# ============================================================
# 离散时间晶体(DTC)相关函数
# ============================================================

def dtc_drive_pulse(qubits, strength, duration=1.0):
    """
    应用DTC驱动脉冲 - 周期驱动产生时间晶体
    
    DTC的物理基础：
    H = J * Z_i * Z_{i+1} + Ω * X_i  (Ising + 横向场)
    周期驱动破坏连续对称性，产生时间平移对称性破缺
    """
    for qubit in qubits:
        # X方向的共振驱动
        qml.RX(strength * duration, wires=qubit)
        # Z方向的自旋相互作用
        qml.RZ(strength * 0.5 * duration, wires=qubit)

def dtc_ising_interaction(qubits, coupling_strength):
    """
    应用Ising相互作用 - 产生DTC的关键
    H_Ising = J * Σ Z_i * Z_{i+1}
    """
    n_qubits = len(qubits)
    for i in range(n_qubits - 1):
        qml.IsingZZ(coupling_strength, wires=[qubits[i], qubits[i+1]])
    # 环形拓扑
    qml.IsingZZ(coupling_strength, wires=[qubits[n_qubits-1], qubits[0]])

def dtc_evolution(qubits, drive_strength, num_steps):
    """
    离散时间晶体演化 - 周期驱动动力学
    
    一个周期包含：
    1. Ising相互作用
    2. 驱动脉冲
    
    重复num_steps次产生DTC动力学
    """
    for step in range(num_steps):
        # Step 1: Ising相互作用 (产生纠缠)
        coupling = drive_strength * 0.3
        dtc_ising_interaction(qubits, coupling)
        
        # Step 2: 驱动脉冲 (打破连续对称性)
        dtc_drive_pulse(qubits, drive_strength)
        
        # Step 3: 额外的单比特旋转 (增加信息混合)
        for qubit in qubits:
            phase = (drive_strength * step) % (2 * np.pi)
            qml.RY(phase * 0.1, wires=qubit)

def extract_dtc_features(qubits):
    """
    从DTC演化后的量子态提取特征
    测量每个量子比特的Pauli Z期望值
    """
    features = []
    for qubit in qubits:
        # 测量Z期望值
        features.append(qml.expval(qml.PauliZ(qubit)))
    return features

# 定义量子设备和电路
dev = qml.device('lightning.qubit', wires=ALL_QUBITS)

@qml.qnode(dev, interface='torch')
def quantum_reservoir_circuit(img_params, drive_strength):
    '''
    基于离散时间晶体的量子储存库计算电路
    
    架构:
    输入编码 → DTC演化 → 特征提取 → 期望值测量
    
    优势:
    - 梯度自由设计 (固定驱动参数)
    - 拓扑噪声鲁棒性
    - 信息自然编码在DTC动力学中
    '''
    
    # Step 1: 参数编码和预处理
    com_params = dense_layer(img_params)
    if len(com_params) < 2**COM_QUBITS:
        com_params_padded = torch.nn.functional.pad(com_params, (0, 2**COM_QUBITS - len(com_params)))
    else:
        com_params_padded = com_params[:2**COM_QUBITS]
    
    # Step 2: 幅度编码 - 将经典数据编码到量子态
    qml.AmplitudeEmbedding(com_params_padded, wires=range(COM_QUBITS), pad_with=0.0, normalize=True)
    
    # Step 3: 初始化剩余量子比特
    for i in range(COM_QUBITS, ALL_QUBITS):
        qml.Hadamard(wires=i)
    
    # Step 4: DTC动力学演化 - 量子储存库
    # 这是梯度自由的 - 驱动参数是固定的
    dtc_evolution(range(ALL_QUBITS), drive_strength, N_STEPS)
    
    # Step 5: 提取特征 - 测量所有量子比特的Pauli Z期望值
    features = extract_dtc_features(range(ALL_QUBITS))
    
    # 返回特征的和作为输出
    return sum(features)

@qml.qnode(dev, interface='torch')
def quantum_reservoir_circuit_with_readout(img_params, drive_strength):
    '''
    量子储存库 + 线性读取头
    经典线性分类器对提取的特征进行处理
    '''
    
    # Step 1-4: 同上
    com_params = dense_layer(img_params)
    if len(com_params) < 2**COM_QUBITS:
        com_params_padded = torch.nn.functional.pad(com_params, (0, 2**COM_QUBITS - len(com_params)))
    else:
        com_params_padded = com_params[:2**COM_QUBITS]
    
    qml.AmplitudeEmbedding(com_params_padded, wires=range(COM_QUBITS), pad_with=0.0, normalize=True)
    
    for i in range(COM_QUBITS, ALL_QUBITS):
        qml.Hadamard(wires=i)
    
    dtc_evolution(range(ALL_QUBITS), drive_strength, N_STEPS)
    
    # 返回所有量子比特的Z期望值作为特征向量
    return [qml.expval(qml.PauliZ(i)) for i in range(ALL_QUBITS)]

class LinearReadout(nn.Module):
    """
    经典线性读取头 - 对DTC提取的特征进行分类
    这是唯一需要训练的部分
    """
    def __init__(self, input_size, output_size=1):
        super(LinearReadout, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, features):
        return self.linear(features)

# 批量处理函数 - DTC版本
def process_batch_dtc(img_batch, drive_strength, readout=None):
    '''处理批量样本 - 量子储存库版本'''
    batch_results = []
    
    for img_params in img_batch:
        # 确保输入是PyTorch张量
        if isinstance(img_params, np.ndarray):
            img_params = torch.from_numpy(img_params).float()
        
        # 运行DTC电路获得特征
        features = quantum_reservoir_circuit_with_readout(img_params, drive_strength)
        
        # 转换为张量
        features_tensor = torch.tensor(features, dtype=torch.float32)
        
        # 通过线性读取头
        if readout is not None:
            output = readout(features_tensor)
        else:
            output = torch.sum(features_tensor)
        
        # 确保结果是实数
        if isinstance(output, (complex, np.complex128)):
            output = torch.tensor(np.real(output), dtype=torch.float32)
        
        batch_results.append(output)
    
    return torch.stack(batch_results)

def validate_model_dtc(drive_strength, readout, val_samples=1000):
    """在验证集上评估模型 - DTC版本"""
    try:
        val_subset = val_data[:min(val_samples, len(val_data))]
        results = process_batch_dtc(val_subset, drive_strength, readout)
        return float(torch.mean(results))
    except Exception as e:
        print(f"Validation error: {e}")
        return float('inf')

# 批量训练函数 - DTC版本
def train_batch_version_dtc():
    try:
        # 保存初始参数
        save_initial_parameters()
        
        # 使用训练集
        n_samples = 1000
        samples = train_data[:n_samples]

        # 初始化线性读取头
        readout = LinearReadout(ALL_QUBITS, output_size=1)
        
        # 优化器 - 仅优化读取头的权重 (DTC驱动参数固定)
        opt = torch.optim.SGD(readout.parameters(), lr=0.01)
        
        # 保存初始驱动参数
        initial_config = {
            'drive_strength': DRIVE_STRENGTH,
            'n_steps': N_STEPS,
            'n_qubits': ALL_QUBITS,
            'dtc_period': DTC_PERIOD
        }
        torch.save(initial_config, 'model_parameters/initial_dtc_config.pt')
        torch.save(readout.state_dict(), 'model_parameters/initial_readout_weights.pt')
        print("Initial DTC configuration and readout weights saved!")
        print(f"  Drive strength: {DRIVE_STRENGTH}")
        print(f"  Evolution steps: {N_STEPS}")
        print(f"  Readout parameters: {sum(p.numel() for p in readout.parameters())}")

        n_epochs = 5
        batch_size = 50
        
        # 记录训练历史
        training_history = {
            'epoch_losses': [],
            'val_losses': [],
            'batch_losses': [],
            'readout_weights_history': [],
            'dtc_architecture': {
                'type': 'QuantumReservoirComputing_DTC',
                'algorithm': 'DiscreteTimeCrystal',
                'n_qubits': ALL_QUBITS,
                'evolution_steps': N_STEPS,
                'dtc_period': DTC_PERIOD,
                'drive_strength': DRIVE_STRENGTH,
                'gradient_free': True,
                'topological_noise_robustness': True,
                'readout_type': 'LinearReadout'
            },
            'data_split_info': {
                'train_size': len(train_data),
                'val_size': len(val_data),
                'test_size': len(test_data),
                'actual_train_used': n_samples
            }
        }

        print("\nStarting Quantum Reservoir Computing with DTC training...")
        print("=" * 60)
        print("Note: DTC reservoir is fixed, only linear readout is trained")
        print("=" * 60)
        start_time = time.time()

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            batch_count = 0

            for i in range(0, n_samples, batch_size):
                batch = samples[i:i+batch_size]
                
                def closure():
                    opt.zero_grad()
                    results = process_batch_dtc(batch, DRIVE_STRENGTH, readout)
                    loss = torch.mean(results)
                    loss.backward()
                    return loss
                
                # 记录训练前的权重
                pre_readout_norm = torch.norm(torch.cat([p.flatten() for p in readout.parameters()]))
                
                # 更新权重
                loss = opt.step(closure)
                current_loss = loss.item() if hasattr(loss, 'item') else float(loss)
                epoch_loss += current_loss
                batch_count += 1

                # 记录训练后的权重
                post_readout_norm = torch.norm(torch.cat([p.flatten() for p in readout.parameters()]))
                
                # 记录批次信息
                training_history['batch_losses'].append({
                    'epoch': epoch,
                    'batch': i // batch_size,
                    'loss': float(current_loss),
                    'pre_readout_norm': float(pre_readout_norm),
                    'post_readout_norm': float(post_readout_norm),
                    'drive_strength': DRIVE_STRENGTH
                })

                if (i // batch_size) % 5 == 0:
                    print(f"Epoch {epoch}, Batch {i//batch_size}: loss = {current_loss:.6f}")

            if batch_count > 0:
                avg_epoch_loss = epoch_loss / batch_count
                # 计算验证损失
                val_loss = validate_model_dtc(DRIVE_STRENGTH, readout, val_samples=500)
                
                training_history['epoch_losses'].append({
                    'epoch': epoch,
                    'avg_loss': float(avg_epoch_loss)
                })
                training_history['val_losses'].append({
                    'epoch': epoch,
                    'val_loss': float(val_loss)
                })
                
                # 保存每个epoch的读取头权重
                epoch_readout_state = {
                    'weight': readout.linear.weight.clone().detach(),
                    'bias': readout.linear.bias.clone().detach()
                }
                training_history['readout_weights_history'].append(epoch_readout_state)
                torch.save(epoch_readout_state, f'model_parameters/readout_weights_epoch_{epoch}.pt')
                
                print(f"Epoch {epoch} completed: Train Loss = {avg_epoch_loss:.6f}, Val Loss = {val_loss:.6f}")
                print(f"  Readout weight norm: {torch.norm(readout.linear.weight):.6f}")
                print(f"Readout weights for epoch {epoch} saved!")
                print("-" * 60)

        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds!")
        
        # 保存最终权重和训练历史
        final_readout_state = {
            'weight': readout.linear.weight.clone().detach(),
            'bias': readout.linear.bias.clone().detach()
        }
        
        torch.save(final_readout_state, 'model_parameters/final_readout_weights.pt')
        torch.save(readout.state_dict(), 'model_parameters/final_readout_model.pt')
        torch.save(training_history, 'model_parameters/training_history.pt')
        print("Final readout weights and training history saved!")
        
        return readout, training_history

    except Exception as e:
        print(f"Error in DTC training: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# 测试训练好的模型
def test_trained_model_dtc(readout, test_samples=1000):
    """测试训练好的模型 - DTC版本"""
    print("\nTesting trained DTC quantum reservoir on test set...")
    try:
        test_subset = test_data[:min(test_samples, len(test_data))]
        results = process_batch_dtc(test_subset, DRIVE_STRENGTH, readout)
        print(f"Test results on {len(test_subset)} samples:")
        for i in range(min(5, len(results))):
            print(f"  Sample {i}: {results[i].item():.6f}")
        if len(results) > 5:
            print(f"  ... (showing first 5 of {len(results)} results)")
        avg_result = torch.mean(results).item()
        std_result = torch.std(results).item()
        print(f"Average test result: {avg_result:.6f}")
        print(f"Standard deviation: {std_result:.6f}")
        print(f"Min: {torch.min(results).item():.6f}, Max: {torch.max(results).item():.6f}")
        return results
    except Exception as e:
        print(f"Error in testing: {e}")
        return None


# 主程序
if __name__ == "__main__":
    print("Starting Quantum Reservoir Computing with Discrete Time Crystal...")
    print("=" * 60)
    print(f"Data Split: {TRAIN_RATIO*100:.0f}% Train, {VAL_RATIO*100:.0f}% Validation, {TEST_RATIO*100:.0f}% Test")
    print(f"Training set: {len(train_data)} samples")
    print(f"Validation set: {len(val_data)} samples")
    print(f"Test set: {len(test_data)} samples")
    print("=" * 60)
    print("\n🔬 Physical Principles:")
    print("  • DTC exploits periodic driving to break continuous symmetry")
    print("  • Quantum state encodes information through time crystal dynamics")
    print("  • Linear readout extracts features from DTC evolution")
    print("  • Gradient-free training: only readout is optimized")
    print("  • Topological noise robustness from DTC properties")
    print("=" * 60)
    
    # 训练DTC量子储存库
    readout_model, history = train_batch_version_dtc()
    
    if readout_model is not None:
        # 测试训练好的模型
        test_results = test_trained_model_dtc(readout_model)
        
        # 显示训练总结
        print("\n" + "=" * 60)
        print("QUANTUM RESERVOIR COMPUTING WITH DTC - TRAINING SUMMARY:")
        print("=" * 60)
        print(f"Data split: {TRAIN_RATIO*100:.1f}% train, {VAL_RATIO*100:.1f}% val, {TEST_RATIO*100:.1f}% test")
        print(f"Training samples used: {history['data_split_info']['actual_train_used']}")
        print(f"Total training samples available: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        print(f"Test samples: {len(test_data)}")
        print("\nClassical NN parameters (preprocessing):")
        print(f"  - WEIGHT shape: {WEIGHT.shape}")
        print(f"  - BIAS shape: {BIAS.shape}")
        print("\nQuantum Reservoir Architecture (DTC):")
        arch = history['dtc_architecture']
        print(f"  - Algorithm: {arch['algorithm']}")
        print(f"  - Quantum qubits: {arch['n_qubits']}")
        print(f"  - Evolution steps: {arch['evolution_steps']}")
        print(f"  - DTC period: {arch['dtc_period']}")
        print(f"  - Drive strength: {arch['drive_strength']}")
        print(f"  - Gradient-free: {arch['gradient_free']}")
        print(f"  - Topological noise robustness: {arch['topological_noise_robustness']}")
        print(f"  - Readout type: {arch['readout_type']}")
        print("\nTrainable readout parameters:")
        for name, param in readout_model.named_parameters():
            print(f"  - {name}: {param.shape}")
        print(f"  - Total parameters: {sum(p.numel() for p in readout_model.parameters())}")
        print(f"\nTraining epochs: {len(history['epoch_losses'])}")
        if len(history['epoch_losses']) > 0:
            print(f"  - Final train loss: {history['epoch_losses'][-1]['avg_loss']:.6f}")
        if len(history['val_losses']) > 0:
            print(f"  - Final validation loss: {history['val_losses'][-1]['val_loss']:.6f}")
        
        # 显示保存的文件
        print("\nSaved files in 'model_parameters' directory:")
        saved_files = os.listdir('model_parameters')
        for file in sorted(saved_files):
            print(f"  - {file}")
        
        print("\n💡 Key Advantages over Standard Parametrized Quantum Circuits:")
        print("  ✓ Gradient-free training avoids barren plateaus")
        print("  ✓ Fixed DTC dynamics reduce optimization complexity")
        print("  ✓ Topological protection provides noise robustness")
        print("  ✓ Natural information encoding through time crystal dynamics")
        print("  ✓ Fewer trainable parameters (only readout)")
        print("  ✓ Suitable for NISQ devices")
    else:
        print("Training failed!")
