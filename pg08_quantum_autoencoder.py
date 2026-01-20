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

N_LAYERS = 4
IMG_QUBITS = int(np.ceil(np.log2(INPUT_DIM)))  # 12
COM_QUBITS = int(np.ceil(np.log2(OUTPUT_DIM)))  # 8
ALL_QUBITS = IMG_QUBITS  # 12个量子比特

# 量子自编码器参数
N_ENCODER_LAYERS = 2
N_DECODER_LAYERS = 2
N_HIDDEN_LAYERS = 2

print(f"IMG_QUBITS: {IMG_QUBITS}, COM_QUBITS: {COM_QUBITS}, ALL_QUBITS: {ALL_QUBITS}")
print(f"Quantum Autoencoder: {N_ENCODER_LAYERS} encoder layers, {N_HIDDEN_LAYERS} hidden layers, {N_DECODER_LAYERS} decoder layers")

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
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    output = torch.matmul(x, WEIGHT) + BIAS
    output = sigmoid(output)
    output = normlize(output[0])  # 确保输出是一维的
    return output

# ============================================================
# 量子自编码器相关函数
# ============================================================

def encoder_circuit(input_qubits, hidden_qubits, weights):
    """
    量子自编码器编码器
    
    Args:
        input_qubits: 输入量子比特列表
        hidden_qubits: 隐藏(压缩)量子比特列表
        weights: 编码器参数 shape: (N_ENCODER_LAYERS, len(input_qubits), 3)
    """
    n_wires = len(input_qubits)
    
    for layer in range(weights.shape[0]):
        # 单参数旋转门
        for i, qubit in enumerate(input_qubits):
            qml.RY(weights[layer, i, 0], wires=qubit)
            qml.RZ(weights[layer, i, 1], wires=qubit)
        
        # 纠缠层 - 邻近CNOT
        for i in range(n_wires - 1):
            qml.CNOT(wires=[input_qubits[i], input_qubits[i+1]])
        qml.CNOT(wires=[input_qubits[n_wires-1], input_qubits[0]])  # 环形连接
        
        # 额外的双参数旋转
        for i, qubit in enumerate(input_qubits):
            qml.RY(weights[layer, i, 2], wires=qubit)


def decoder_circuit(hidden_qubits, output_qubits, weights):
    """
    量子自编码器解码器
    
    Args:
        hidden_qubits: 隐藏(压缩)量子比特列表
        output_qubits: 输出量子比特列表
        weights: 解码器参数 shape: (N_DECODER_LAYERS, len(output_qubits), 3)
    """
    n_wires = len(output_qubits)
    
    for layer in range(weights.shape[0]):
        # 单参数旋转门
        for i, qubit in enumerate(output_qubits):
            qml.RY(weights[layer, i, 0], wires=qubit)
            qml.RZ(weights[layer, i, 1], wires=qubit)
        
        # 纠缠层 - 邻近CNOT
        for i in range(n_wires - 1):
            qml.CNOT(wires=[output_qubits[i], output_qubits[i+1]])
        qml.CNOT(wires=[output_qubits[n_wires-1], output_qubits[0]])  # 环形连接
        
        # 额外的双参数旋转
        for i, qubit in enumerate(output_qubits):
            qml.RY(weights[layer, i, 2], wires=qubit)


def hidden_circuit(hidden_qubits, weights):
    """
    量子自编码器隐藏层(中间层处理)
    
    Args:
        hidden_qubits: 隐藏量子比特列表
        weights: 隐藏层参数 shape: (N_HIDDEN_LAYERS, len(hidden_qubits), 3)
    """
    n_wires = len(hidden_qubits)
    
    for layer in range(weights.shape[0]):
        # 旋转门
        for i, qubit in enumerate(hidden_qubits):
            qml.RX(weights[layer, i, 0], wires=qubit)
            qml.RY(weights[layer, i, 1], wires=qubit)
            qml.RZ(weights[layer, i, 2], wires=qubit)
        
        # 纠缠 - 对角线连接
        for i in range(n_wires - 1):
            qml.CNOT(wires=[hidden_qubits[i], hidden_qubits[i+1]])


coe = [-1] * ALL_QUBITS
obs_list = [qml.PauliZ(0)] * ALL_QUBITS
hamiltonian = qml.Hamiltonian(coe, observables=obs_list)

dev = qml.device('lightning.qubit', wires=ALL_QUBITS)

@qml.qnode(dev, interface='torch')
def total_circuit_qae(img_params, encoder_weights, hidden_weights, decoder_weights):
    '''
    量子自编码器完整电路
    
    结构:
    输入编码 → 编码器 → 隐藏层 → 解码器 → 输出 → 测量
    '''
    
    # 定义量子比特划分
    input_qubits = list(range(ALL_QUBITS))  # 所有12个量子比特用于输入
    hidden_qubits = list(range(COM_QUBITS))  # 前8个用作隐藏层
    output_qubits = list(range(ALL_QUBITS))  # 所有12个用于输出
    
    # Step 1: 压缩参数编码
    com_params = dense_layer(img_params)
    if len(com_params) < 2**COM_QUBITS:
        com_params_padded = torch.nn.functional.pad(com_params, (0, 2**COM_QUBITS - len(com_params)))
    else:
        com_params_padded = com_params[:2**COM_QUBITS]
    
    # 幅度编码到输入量子比特
    qml.AmplitudeEmbedding(com_params_padded, wires=input_qubits, pad_with=0.0, normalize=True)
    
    # Step 2: 编码器 - 从全量子比特到隐藏层压缩
    encoder_circuit(input_qubits, hidden_qubits, encoder_weights)
    
    # Step 3: 隐藏层 - 在压缩空间中的处理
    hidden_circuit(hidden_qubits, hidden_weights)
    
    # Step 4: 解码器 - 从隐藏层恢复到全量子比特
    decoder_circuit(hidden_qubits, output_qubits, decoder_weights)
    
    # Step 5: 逆编码 - 回到参数空间
    img_params_norm = normlize(img_params)
    if len(img_params_norm) < 2**IMG_QUBITS:
        img_params_padded = torch.nn.functional.pad(img_params_norm, (0, 2**IMG_QUBITS - len(img_params_norm)))
    else:
        img_params_padded = img_params_norm[:2**IMG_QUBITS]
    
    qml.adjoint(qml.AmplitudeEmbedding(img_params_padded, wires=input_qubits, pad_with=0.0, normalize=True))
    
    # Step 6: 测量期望值
    return qml.expval(hamiltonian)


# 批量处理函数 - 分别处理每个样本
def process_batch_qae(img_batch, encoder_weights, hidden_weights, decoder_weights):
    '''处理批量的样本 - 量子自编码器版本'''
    batch_results = []
    for img_params in img_batch:
        # 确保输入是PyTorch张量
        if isinstance(img_params, np.ndarray):
            img_params = torch.from_numpy(img_params).float()
        result = total_circuit_qae(img_params, encoder_weights, hidden_weights, decoder_weights)
        # 确保结果是实数类型
        if isinstance(result, (complex, np.complex128)):
            result = torch.tensor(np.real(result), dtype=torch.float32)
        batch_results.append(result)
    return torch.stack(batch_results)


def validate_model_qae(encoder_weights, hidden_weights, decoder_weights, val_samples=1000):
    """在验证集上评估模型 - 量子自编码器版本"""
    try:
        val_subset = val_data[:min(val_samples, len(val_data))]
        results = process_batch_qae(val_subset, encoder_weights, hidden_weights, decoder_weights)
        return float(torch.mean(results))
    except Exception as e:
        print(f"Validation error: {e}")
        return float('inf')


# 批量训练函数 - 量子自编码器版本
def train_batch_version_qae():
    try:
        # 保存初始参数
        save_initial_parameters()
        
        # 使用训练集
        n_samples = 1000
        samples = train_data[:n_samples]

        # 计算参数形状
        encoder_shape = (N_ENCODER_LAYERS, ALL_QUBITS, 3)
        hidden_shape = (N_HIDDEN_LAYERS, COM_QUBITS, 3)
        decoder_shape = (N_DECODER_LAYERS, ALL_QUBITS, 3)
        
        # 初始化量子自编码器权重
        encoder_weights = torch.rand(encoder_shape, requires_grad=True) * 0.1
        hidden_weights = torch.rand(hidden_shape, requires_grad=True) * 0.1
        decoder_weights = torch.rand(decoder_shape, requires_grad=True) * 0.1
        
        # 保存初始量子权重
        torch.save({
            'encoder': encoder_weights.clone().detach(),
            'hidden': hidden_weights.clone().detach(),
            'decoder': decoder_weights.clone().detach()
        }, 'model_parameters/initial_quantum_weights.pt')
        print("Initial quantum autoencoder weights saved!")
        print(f"  Encoder shape: {encoder_shape}")
        print(f"  Hidden shape: {hidden_shape}")
        print(f"  Decoder shape: {decoder_shape}")

        # 优化器 - 为所有权重创建优化器
        opt = torch.optim.Adam(
            [encoder_weights, hidden_weights, decoder_weights], 
            lr=0.01
        )
        
        n_epochs = 5
        batch_size = 50
        
        # 记录训练历史
        training_history = {
            'epoch_losses': [],
            'val_losses': [],
            'batch_losses': [],
            'weights_history': [],
            'autoencoder_architecture': {
                'type': 'QuantumAutoencoder',
                'n_encoder_layers': N_ENCODER_LAYERS,
                'n_hidden_layers': N_HIDDEN_LAYERS,
                'n_decoder_layers': N_DECODER_LAYERS,
                'input_qubits': ALL_QUBITS,
                'hidden_qubits': COM_QUBITS,
                'total_qubits': ALL_QUBITS
            },
            'data_split_info': {
                'train_size': len(train_data),
                'val_size': len(val_data),
                'test_size': len(test_data),
                'actual_train_used': n_samples
            }
        }

        print("Starting quantum autoencoder training...")
        print("=" * 60)
        start_time = time.time()

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            batch_count = 0

            for i in range(0, n_samples, batch_size):
                batch = samples[i:i+batch_size]
                
                def closure():
                    opt.zero_grad()
                    results = process_batch_qae(batch, encoder_weights, hidden_weights, decoder_weights)
                    loss = torch.mean(results)
                    loss.backward()
                    return loss
                
                # 记录训练前的权重
                pre_encoder_norm = torch.norm(encoder_weights)
                pre_hidden_norm = torch.norm(hidden_weights)
                pre_decoder_norm = torch.norm(decoder_weights)
                pre_total_norm = pre_encoder_norm + pre_hidden_norm + pre_decoder_norm
                
                # 更新权重
                loss = opt.step(closure)
                current_loss = loss.item() if hasattr(loss, 'item') else float(loss)
                epoch_loss += current_loss
                batch_count += 1

                # 记录训练后的权重
                post_encoder_norm = torch.norm(encoder_weights)
                post_hidden_norm = torch.norm(hidden_weights)
                post_decoder_norm = torch.norm(decoder_weights)
                post_total_norm = post_encoder_norm + post_hidden_norm + post_decoder_norm
                
                # 记录批次信息
                training_history['batch_losses'].append({
                    'epoch': epoch,
                    'batch': i // batch_size,
                    'loss': float(current_loss),
                    'pre_weights_norm': float(pre_total_norm),
                    'post_weights_norm': float(post_total_norm),
                    'encoder_norm': float(post_encoder_norm),
                    'hidden_norm': float(post_hidden_norm),
                    'decoder_norm': float(post_decoder_norm)
                })

                if (i // batch_size) % 5 == 0:
                    print(f"Epoch {epoch}, Batch {i//batch_size}: loss = {current_loss:.6f}")

            if batch_count > 0:
                avg_epoch_loss = epoch_loss / batch_count
                # 计算验证损失
                val_loss = validate_model_qae(encoder_weights, hidden_weights, decoder_weights, val_samples=500)
                
                training_history['epoch_losses'].append({
                    'epoch': epoch,
                    'avg_loss': float(avg_epoch_loss)
                })
                training_history['val_losses'].append({
                    'epoch': epoch,
                    'val_loss': float(val_loss)
                })
                
                # 保存每个epoch的权重
                epoch_weights = {
                    'encoder': encoder_weights.clone().detach(),
                    'hidden': hidden_weights.clone().detach(),
                    'decoder': decoder_weights.clone().detach()
                }
                training_history['weights_history'].append({
                    'encoder': epoch_weights['encoder'].numpy(),
                    'hidden': epoch_weights['hidden'].numpy(),
                    'decoder': epoch_weights['decoder'].numpy()
                })
                
                torch.save(epoch_weights, f'model_parameters/quantum_weights_epoch_{epoch}.pt')
                print(f"Epoch {epoch} completed: Train Loss = {avg_epoch_loss:.6f}, Val Loss = {val_loss:.6f}")
                print(f"  Encoder norm: {torch.norm(encoder_weights):.6f}, "
                      f"Hidden norm: {torch.norm(hidden_weights):.6f}, "
                      f"Decoder norm: {torch.norm(decoder_weights):.6f}")
                print(f"Quantum autoencoder weights for epoch {epoch} saved!")
                print("-" * 60)

        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds!")
        
        # 保存最终权重和训练历史
        final_weights = {
            'encoder': encoder_weights.clone().detach(),
            'hidden': hidden_weights.clone().detach(),
            'decoder': decoder_weights.clone().detach()
        }
        
        torch.save(final_weights, 'model_parameters/final_quantum_weights.pt')
        torch.save(training_history, 'model_parameters/training_history.pt')
        print("Final quantum autoencoder weights and training history saved!")
        
        return final_weights, training_history

    except Exception as e:
        print(f"Error in quantum autoencoder training: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# 测试训练好的模型
def test_trained_model_qae(weights, test_samples=1000):
    """测试训练好的模型 - 量子自编码器版本"""
    print("\nTesting trained quantum autoencoder on test set...")
    try:
        test_subset = test_data[:min(test_samples, len(test_data))]
        results = process_batch_qae(
            test_subset, 
            weights['encoder'], 
            weights['hidden'], 
            weights['decoder']
        )
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
    print("Starting Quantum Autoencoder training with CSI channel data...")
    print("=" * 60)
    print(f"Data Split: {TRAIN_RATIO*100:.0f}% Train, {VAL_RATIO*100:.0f}% Validation, {TEST_RATIO*100:.0f}% Test")
    print(f"Training set: {len(train_data)} samples")
    print(f"Validation set: {len(val_data)} samples")
    print(f"Test set: {len(test_data)} samples")
    print("=" * 60)
    
    # 训练量子自编码器
    final_weights, history = train_batch_version_qae()
    
    if final_weights is not None:
        # 测试训练好的模型
        test_results = test_trained_model_qae(final_weights)
        
        # 显示训练总结
        print("\n" + "=" * 60)
        print("QUANTUM AUTOENCODER TRAINING SUMMARY:")
        print("=" * 60)
        print(f"Data split: {TRAIN_RATIO*100:.1f}% train, {VAL_RATIO*100:.1f}% val, {TEST_RATIO*100:.1f}% test")
        print(f"Training samples used: {history['data_split_info']['actual_train_used']}")
        print(f"Total training samples available: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        print(f"Test samples: {len(test_data)}")
        print("\nClassical NN parameters saved:")
        print(f"  - WEIGHT shape: {WEIGHT.shape}")
        print(f"  - BIAS shape: {BIAS.shape}")
        print("\nQuantum Autoencoder Architecture:")
        arch = history['autoencoder_architecture']
        print(f"  - Type: {arch['type']}")
        print(f"  - Encoder layers: {arch['n_encoder_layers']}")
        print(f"  - Hidden layers: {arch['n_hidden_layers']}")
        print(f"  - Decoder layers: {arch['n_decoder_layers']}")
        print(f"  - Input qubits: {arch['input_qubits']}")
        print(f"  - Hidden qubits: {arch['hidden_qubits']}")
        print(f"  - Total qubits: {arch['total_qubits']}")
        print("\nQuantum weights shapes:")
        print(f"  - Encoder shape: {final_weights['encoder'].shape}")
        print(f"  - Hidden shape: {final_weights['hidden'].shape}")
        print(f"  - Decoder shape: {final_weights['decoder'].shape}")
        print(f"  - Number of epochs: {len(history['epoch_losses'])}")
        if len(history['epoch_losses']) > 0:
            print(f"  - Final train loss: {history['epoch_losses'][-1]['avg_loss']:.6f}")
        if len(history['val_losses']) > 0:
            print(f"  - Final validation loss: {history['val_losses'][-1]['val_loss']:.6f}")
        
        # 显示保存的文件
        print("\nSaved files in 'model_parameters' directory:")
        saved_files = os.listdir('model_parameters')
        for file in sorted(saved_files):
            print(f"  - {file}")
    else:
        print("Training failed!")
