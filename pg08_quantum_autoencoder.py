import pennylane as qml
import torch
import torch.nn.functional as F
import numpy as np
import time
import os

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 数据加载
data_30 = np.load('../DataSpace/csi_cmri/CSI_channel_30km.npy')  # shape=(80000, 2560)

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

COM_QUBITS = int(np.ceil(np.log2(OUTPUT_DIM)))
N_DECODER_LAYERS = 2

# 初始化并保存经典神经网络参数 - 使用PyTorch张量
WEIGHT = torch.randn(INPUT_DIM, OUTPUT_DIM, requires_grad=True)
WEIGHT.data.mul_(0.01)
BIAS = torch.randn(1, OUTPUT_DIM, requires_grad=True)
RECON_WEIGHT = torch.randn(COM_QUBITS, INPUT_DIM, requires_grad=True)
RECON_WEIGHT.data.mul_(0.01)
RECON_BIAS = torch.randn(INPUT_DIM, requires_grad=True)
RECON_BIAS.data.mul_(0.01)

# 创建保存参数的目录
os.makedirs('model_parameters', exist_ok=True)

def save_initial_parameters():
    """保存初始化的参数"""
    torch.save(WEIGHT, 'model_parameters/initial_weight.pt')
    torch.save(BIAS, 'model_parameters/initial_bias.pt')
    torch.save(RECON_WEIGHT, 'model_parameters/initial_recon_weight.pt')
    torch.save(RECON_BIAS, 'model_parameters/initial_recon_bias.pt')
    print("Initial weights saved for compression and reconstruction!")


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
    if output.dim() > 1 and output.shape[0] == 1:
        output = output.squeeze(0)
    output = sigmoid(output)
    return normlize(output)
# ============================================================
# Hybrid classical-quantum autoencoder components
# ============================================================

hidden_qubits = list(range(COM_QUBITS))
dev = qml.device('lightning.qubit', wires=COM_QUBITS)

@qml.qnode(dev, interface='torch')
def quantum_decoder_circuit(com_params, decoder_weights):
    padded_length = 2 ** COM_QUBITS
    if len(com_params) < padded_length:
        padded = F.pad(com_params, (0, padded_length - len(com_params)))
    else:
        padded = com_params[:padded_length]
    qml.AmplitudeEmbedding(padded, wires=hidden_qubits, pad_with=0.0, normalize=True)
    qml.templates.StronglyEntanglingLayers(decoder_weights, wires=hidden_qubits)
    return [qml.expval(qml.PauliZ(qubit)) for qubit in hidden_qubits]

def classical_reconstruct(measurements):
    return torch.matmul(measurements, RECON_WEIGHT) + RECON_BIAS

def forward_pass(img_params, decoder_weights):
    compressed = dense_layer(img_params)
    measurements = quantum_decoder_circuit(compressed, decoder_weights)
    if isinstance(measurements, list):
        measurements = torch.stack(measurements)
    elif isinstance(measurements, np.ndarray):
        measurements = torch.from_numpy(measurements).float()
    return classical_reconstruct(measurements)

def process_batch(img_batch, decoder_weights):
    batch_tensor = torch.from_numpy(np.array(img_batch)).float()
    reconstructions = []
    for sample in batch_tensor:
        recon = forward_pass(sample, decoder_weights)
        reconstructions.append(recon)
    reconstructions = torch.stack(reconstructions)
    return reconstructions, batch_tensor

def validate_model(decoder_weights, val_samples=1000):
    try:
        subset = val_data[:min(val_samples, len(val_data))]
        reconstructions, targets = process_batch(subset, decoder_weights)
        loss = F.mse_loss(reconstructions, targets)
        return float(loss)
    except Exception as e:
        print(f"Validation error: {e}")
        return float('inf')

def train_batch_version_qae():
    try:
        save_initial_parameters()
        n_samples = int(os.getenv("QAE_TRAIN_SAMPLES", "1000"))
        n_samples = min(n_samples, len(train_data))
        samples = train_data[:n_samples]

        decoder_shape = (N_DECODER_LAYERS, COM_QUBITS, 3)
        decoder_weights = torch.rand(decoder_shape, requires_grad=True)
        decoder_weights.data.mul_(0.1)

        torch.save(decoder_weights.clone().detach(), 'model_parameters/initial_quantum_weights.pt')
        print(f"Initial quantum decoder weights saved (shape {decoder_shape})")

        opt = torch.optim.Adam(
            [WEIGHT, BIAS, RECON_WEIGHT, RECON_BIAS, decoder_weights],
            lr=0.01
        )

        n_epochs = int(os.getenv("QAE_EPOCHS", "5"))
        batch_size = 50

        training_history = {
            'epoch_losses': [],
            'val_losses': [],
            'batch_losses': [],
            'weights_history': [],
            'autoencoder_architecture': {
                'type': 'HybridClassical-QuantumDecoder',
                'quantum_layers': N_DECODER_LAYERS,
                'hidden_qubits': COM_QUBITS,
                'compression_dim': OUTPUT_DIM,
                'reconstruction_dim': INPUT_DIM
            },
            'data_split_info': {
                'train_size': len(train_data),
                'val_size': len(val_data),
                'test_size': len(test_data),
                'actual_train_used': n_samples
            }
        }

        print("Starting hybrid autoencoder training...")
        print("=" * 60)
        start_time = time.time()

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            batch_count = 0

            for i in range(0, n_samples, batch_size):
                batch = samples[i:i+batch_size]
                opt.zero_grad()

                reconstructions, targets = process_batch(batch, decoder_weights)
                loss = F.mse_loss(reconstructions, targets)

                pre_decoder_norm = torch.norm(decoder_weights)
                pre_classical_norm = torch.norm(RECON_WEIGHT) + torch.norm(RECON_BIAS)
                pre_total_norm = pre_decoder_norm + pre_classical_norm + torch.norm(WEIGHT) + torch.norm(BIAS)

                loss.backward()
                opt.step()

                post_decoder_norm = torch.norm(decoder_weights)
                post_classical_norm = torch.norm(RECON_WEIGHT) + torch.norm(RECON_BIAS)
                post_total_norm = post_decoder_norm + post_classical_norm + torch.norm(WEIGHT) + torch.norm(BIAS)

                current_loss = loss.item()
                epoch_loss += current_loss
                batch_count += 1

                training_history['batch_losses'].append({
                    'epoch': epoch,
                    'batch': i // batch_size,
                    'loss': float(current_loss),
                    'pre_weights_norm': float(pre_total_norm),
                    'post_weights_norm': float(post_total_norm),
                    'decoder_norm': float(post_decoder_norm),
                    'reconstruction_norm': float(post_classical_norm)
                })

                if (i // batch_size) % 5 == 0:
                    print(f"Epoch {epoch}, Batch {i//batch_size}: loss = {current_loss:.6f}")

            if batch_count > 0:
                avg_epoch_loss = epoch_loss / batch_count
                val_loss = validate_model(decoder_weights, val_samples=500)

                training_history['epoch_losses'].append({
                    'epoch': epoch,
                    'avg_loss': float(avg_epoch_loss)
                })
                training_history['val_losses'].append({
                    'epoch': epoch,
                    'val_loss': float(val_loss)
                })

                epoch_weights = {
                    'decoder': decoder_weights.clone().detach(),
                    'compression': {
                        'weight': WEIGHT.clone().detach(),
                        'bias': BIAS.clone().detach()
                    },
                    'reconstruction': {
                        'weight': RECON_WEIGHT.clone().detach(),
                        'bias': RECON_BIAS.clone().detach()
                    }
                }
                training_history['weights_history'].append({
                    'decoder': epoch_weights['decoder'].numpy(),
                    'compression_weight': epoch_weights['compression']['weight'].numpy(),
                    'compression_bias': epoch_weights['compression']['bias'].numpy(),
                    'reconstruction_weight': epoch_weights['reconstruction']['weight'].numpy(),
                    'reconstruction_bias': epoch_weights['reconstruction']['bias'].numpy()
                })

                torch.save(epoch_weights, f'model_parameters/quantum_weights_epoch_{epoch}.pt')
                print(f"Epoch {epoch} completed: Train Loss = {avg_epoch_loss:.6f}, Val Loss = {val_loss:.6f}")
                print(f"  Decoder norm: {torch.norm(decoder_weights):.6f}")
                print(f"  Reconstruction norm: {post_classical_norm:.6f}")
                print(f"Quantum decoder weights for epoch {epoch} saved!")
                print("-" * 60)

        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds!")

        final_weights = {
            'decoder': decoder_weights.clone().detach(),
            'compression': {
                'weight': WEIGHT.clone().detach(),
                'bias': BIAS.clone().detach()
            },
            'reconstruction': {
                'weight': RECON_WEIGHT.clone().detach(),
                'bias': RECON_BIAS.clone().detach()
            }
        }

        torch.save(final_weights, 'model_parameters/final_quantum_weights.pt')
        torch.save(training_history, 'model_parameters/training_history.pt')
        print("Final hybrid weights and training history saved!")
        return final_weights, training_history

    except Exception as e:
        print(f"Error in hybrid training: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# 测试训练好的模型
def test_trained_model_qae(weights, test_samples=1000):
    print("\nTesting trained hybrid autoencoder on test set...")
    try:
        decoder_weights = weights['decoder']
        test_subset = test_data[:min(test_samples, len(test_data))]
        reconstructions, targets = process_batch(test_subset, decoder_weights)
        mse = F.mse_loss(reconstructions, targets).item()
        print(f"Test samples: {len(test_subset)}")
        print(f"Reconstruction MSE: {mse:.6f}")
        print("Sample-wise summary (first 5):")
        for i in range(min(5, reconstructions.shape[0])):
            diff = torch.mean((reconstructions[i] - targets[i]) ** 2).item()
            print(f"  Sample {i}: MSE = {diff:.6f}")
        if reconstructions.shape[0] > 5:
            print(f"  ... (showing first 5 of {reconstructions.shape[0]} samples)")
        avg_result = torch.mean(reconstructions).item()
        std_result = torch.std(reconstructions).item()
        print(f"Reconstruction mean: {avg_result:.6f}")
        print(f"Reconstruction std: {std_result:.6f}")
        return reconstructions
    except Exception as e:
        print(f"Error in testing: {e}")
        return None

# 主程序
if __name__ == "__main__":
    print("Starting Hybrid Classical-Quantum Autoencoder training with CSI channel data...")
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
        print("HYBRID AUTOENCODER TRAINING SUMMARY:")
        print("=" * 60)
        print(f"Data split: {TRAIN_RATIO*100:.1f}% train, {VAL_RATIO*100:.1f}% val, {TEST_RATIO*100:.1f}% test")
        print(f"Training samples used: {history['data_split_info']['actual_train_used']}")
        print(f"Total training samples available: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        print(f"Test samples: {len(test_data)}")
        print("\nClassical compression parameters saved:")
        print(f"  - WEIGHT shape: {WEIGHT.shape}")
        print(f"  - BIAS shape: {BIAS.shape}")
        print("\nClassical reconstruction head:")
        print(f"  - RECON_WEIGHT shape: {RECON_WEIGHT.shape}")
        print(f"  - RECON_BIAS shape: {RECON_BIAS.shape}")
        print("\nQuantum decoder architecture:")
        arch = history['autoencoder_architecture']
        print(f"  - Type: {arch['type']}")
        print(f"  - Quantum layers: {arch['quantum_layers']}")
        print(f"  - Hidden qubits: {arch['hidden_qubits']}")
        print(f"  - Compression dim: {arch['compression_dim']}")
        print(f"  - Reconstruction dim: {arch['reconstruction_dim']}")
        print("\nFinal weight shapes:")
        print(f"  - Compression weight: {final_weights['compression']['weight'].shape}")
        print(f"  - Reconstruction weight: {final_weights['reconstruction']['weight'].shape}")
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
