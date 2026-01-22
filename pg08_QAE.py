import os
import time
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

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
    # 创建模拟的CSI数据 (80000, 2560)
    simulated_data = np.random.randn(80000, 2560).astype(np.float32)
    print(f"Generated simulated data with shape: {simulated_data.shape}")
    return simulated_data

# Data loading
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

print("数据划分结果:")
print(f"训练集: {len(train_data)} 个样本 ({TRAIN_RATIO*100:.1f}%)")
print(f"验证集: {len(val_data)} 个样本 ({VAL_RATIO*100:.1f}%)")
print(f"测试集: {len(test_data)} 个样本 ({TEST_RATIO*100:.1f}%)")

# Classical NN parameters
INPUT_DIM = 2560
OUTPUT_DIM = 256  # compressed latent size from classical NN

N_LAYERS = 4
DATA_QUBITS = int(np.ceil(np.log2(OUTPUT_DIM)))  # 8 qubits to host the 256-dim classical latent
LATENT_QUBITS = 4  # target compressed size in the quantum autoencoder

print(f"DATA_QUBITS: {DATA_QUBITS}, LATENT_QUBITS: {LATENT_QUBITS}")

WEIGHT = torch.randn(INPUT_DIM, OUTPUT_DIM, requires_grad=True) * 0.01
BIAS = torch.randn(1, OUTPUT_DIM, requires_grad=True)

os.makedirs("model_parameters", exist_ok=True)

def save_initial_parameters():
    torch.save(WEIGHT, "model_parameters/initial_weight.pt")
    torch.save(BIAS, "model_parameters/initial_bias.pt")
    print("Initial WEIGHT and BIAS saved!")


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def normalize(x):
    norm = torch.norm(x)
    if norm == 0:
        return x
    return x / norm


def dense_layer(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    output = torch.matmul(x, WEIGHT) + BIAS
    output = sigmoid(output)
    output = normalize(output[0])
    return output


def pad_to_qubits(vec, n_qubits):
    target_len = 2 ** n_qubits
    if len(vec) < target_len:
        return torch.nn.functional.pad(vec, (0, target_len - len(vec)))
    return vec[:target_len]


# Quantum device
DEV = qml.device("lightning.qubit", wires=DATA_QUBITS)


@qml.qnode(DEV, interface="torch")
def qae_circuit(img_params, enc_params, dec_params):
    """Quantum autoencoder circuit returning reconstruction fidelity.

    Steps:
    1) Classical encoder outputs a 256-dim vector -> amplitude embedded on DATA_QUBITS.
    2) Trainable encoder unitary (enc_params).
    3) Trainable decoder unitary (dec_params).
    4) Apply adjoint amplitude embedding of the same classical code to estimate fidelity.
    5) Fidelity is the probability of measuring |0...0> after the adjoint embedding.
    """
    com_params = dense_layer(img_params)
    com_params_padded = pad_to_qubits(com_params, DATA_QUBITS)

    qml.AmplitudeEmbedding(com_params_padded.detach().numpy() if isinstance(com_params_padded, torch.Tensor) else com_params_padded, 
                          wires=range(DATA_QUBITS), pad_with=0.0, normalize=True)
    qml.StronglyEntanglingLayers(weights=enc_params, wires=range(DATA_QUBITS))
    qml.StronglyEntanglingLayers(weights=dec_params, wires=range(DATA_QUBITS))

    # Fidelity with the input state: use Projector to |0...0>
    qml.adjoint(qml.AmplitudeEmbedding)(com_params_padded.detach().numpy() if isinstance(com_params_padded, torch.Tensor) else com_params_padded, 
                                       wires=range(DATA_QUBITS), pad_with=0.0, normalize=True)
    return qml.expval(qml.Projector([0]*DATA_QUBITS, wires=range(DATA_QUBITS)))


def process_batch(img_batch, enc_params, dec_params):
    # 只在主循环中做自动微分，这里不保留计算图
    fidelities = []
    for img_params in img_batch:
        if isinstance(img_params, np.ndarray):
            img_params = torch.from_numpy(img_params).float()
        # 用 .detach() 避免多次反向传播
        fid = qae_circuit(img_params, enc_params, dec_params)  # 移除 .detach()
        fidelities.append(fid)
    return torch.stack(fidelities)


def validate_model(enc_params, dec_params, val_samples=500):
    try:
        subset = val_data[: min(val_samples, len(val_data))]
        fids = process_batch(subset, enc_params, dec_params)
        return float(torch.mean(fids))
    except Exception as e:
        print(f"Validation error: {e}")
        return float("nan")


def train_batch_version():
    try:
        save_initial_parameters()

        n_samples = 1000
        samples = train_data[:n_samples]

        enc_shape = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=DATA_QUBITS)
        dec_shape = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=DATA_QUBITS)
        enc_params = torch.rand(enc_shape, requires_grad=True)
        dec_params = torch.rand(dec_shape, requires_grad=True)

        torch.save(enc_params, "model_parameters/initial_quantum_encoder_weights.pt")
        torch.save(dec_params, "model_parameters/initial_quantum_decoder_weights.pt")
        print("Initial quantum encoder/decoder weights saved!")

        opt = torch.optim.Adam([enc_params, dec_params], lr=0.01)
        n_epochs = 5
        batch_size = 50

        training_history = {
            "epoch_losses": [],
            "val_fidelity": [],
            "batch_losses": [],
            "weights_history": [],
            "data_split_info": {
                "train_size": len(train_data),
                "val_size": len(val_data),
                "test_size": len(test_data),
                "actual_train_used": n_samples,
            },
        }

        print("Starting QAE training...")
        start_time = time.time()

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            batch_count = 0

            for i in range(0, n_samples, batch_size):
                batch = samples[i : i + batch_size]

                # 直接计算梯度而不是使用闭包
                opt.zero_grad()
                fidelities = []
                for img_params in batch:
                    if isinstance(img_params, np.ndarray):
                        img_params = torch.from_numpy(img_params).float()
                    fid = qae_circuit(img_params, enc_params, dec_params)
                    fidelities.append(fid)
                fidelities = torch.stack(fidelities)
                loss = 1.0 - torch.mean(fidelities)
                loss.backward()

                pre_enc = enc_params.clone().detach()
                pre_dec = dec_params.clone().detach()
                
                # 更新参数
                opt.step()
                
                current_loss = loss.item()
                epoch_loss += current_loss
                batch_count += 1

                training_history["batch_losses"].append(
                    {
                        "epoch": epoch,
                        "batch": i // batch_size,
                        "loss": float(current_loss),
                        "enc_norm": float(torch.norm(pre_enc)),
                        "dec_norm": float(torch.norm(pre_dec)),
                    }
                )

                if (i // batch_size) % 5 == 0:
                    print(f"Epoch {epoch}, Batch {i//batch_size}: loss = {current_loss:.6f}")

            if batch_count > 0:
                avg_epoch_loss = epoch_loss / batch_count
                val_fid = validate_model(enc_params, dec_params, val_samples=500)

                training_history["epoch_losses"].append({"epoch": epoch, "avg_loss": float(avg_epoch_loss)})
                training_history["val_fidelity"].append({"epoch": epoch, "val_fidelity": float(val_fid)})

                enc_snapshot = enc_params.clone().detach()
                dec_snapshot = dec_params.clone().detach()
                training_history["weights_history"].append(
                    {
                        "encoder": enc_snapshot.numpy(),
                        "decoder": dec_snapshot.numpy(),
                    }
                )
                torch.save(enc_snapshot, f"model_parameters/qae_encoder_epoch_{epoch}.pt")
                torch.save(dec_snapshot, f"model_parameters/qae_decoder_epoch_{epoch}.pt")
                print(
                    f"Epoch {epoch} completed: Train Loss = {avg_epoch_loss:.6f}, Val Fidelity = {val_fid:.6f}"
                )
                print("-" * 50)

        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds!")
        torch.save(enc_params, "model_parameters/final_qae_encoder_weights.pt")
        torch.save(dec_params, "model_parameters/final_qae_decoder_weights.pt")
        torch.save(training_history, "model_parameters/qae_training_history.pt")
        print("Final QAE weights and training history saved!")
        return (enc_params, dec_params), training_history

    except Exception as e:
        print(f"Error in batch training: {e}")
        import traceback

        traceback.print_exc()
        return None, None


def train_batch_version_extended():
    try:
        save_initial_parameters()

        n_samples = 1000  # 使用相同数量的样本
        samples = train_data[:n_samples]

        enc_shape = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=DATA_QUBITS)
        dec_shape = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=DATA_QUBITS)
        enc_params = torch.rand(enc_shape, requires_grad=True)
        dec_params = torch.rand(dec_shape, requires_grad=True)

        torch.save(enc_params, "model_parameters/initial_quantum_encoder_weights.pt")
        torch.save(dec_params, "model_parameters/initial_quantum_decoder_weights.pt")
        print("Initial quantum encoder/decoder weights saved!")

        opt = torch.optim.Adam([enc_params, dec_params], lr=0.005)  # 稍微降低学习率
        n_epochs = 10  # 增加到10个epoch
        batch_size = 100  # 增大批量大小

        training_history = {
            "epoch_losses": [],
            "val_fidelity": [],
            "batch_losses": [],
            "weights_history": [],
            "data_split_info": {
                "train_size": len(train_data),
                "val_size": len(val_data),
                "test_size": len(test_data),
                "actual_train_used": n_samples,
            },
        }

        print("Starting extended QAE training...")
        print(f"Training for {n_epochs} epochs with batch size {batch_size}")
        start_time = time.time()

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            batch_count = 0

            for i in range(0, n_samples, batch_size):
                batch = samples[i : i + batch_size]

                # 直接计算梯度而不是使用闭包
                opt.zero_grad()
                fidelities = []
                for img_params in batch:
                    if isinstance(img_params, np.ndarray):
                        img_params = torch.from_numpy(img_params).float()
                    fid = qae_circuit(img_params, enc_params, dec_params)
                    fidelities.append(fid)
                fidelities = torch.stack(fidelities)
                loss = 1.0 - torch.mean(fidelities)
                loss.backward()

                pre_enc = enc_params.clone().detach()
                pre_dec = dec_params.clone().detach()
                
                # 更新参数
                opt.step()
                
                current_loss = loss.item()
                epoch_loss += current_loss
                batch_count += 1

                training_history["batch_losses"].append(
                    {
                        "epoch": epoch,
                        "batch": i // batch_size,
                        "loss": float(current_loss),
                        "enc_norm": float(torch.norm(pre_enc)),
                        "dec_norm": float(torch.norm(pre_dec)),
                    }
                )

                if (i // batch_size) % 3 == 0:  # 调整打印频率
                    print(f"Epoch {epoch}, Batch {i//batch_size}: loss = {current_loss:.6f}")

            if batch_count > 0:
                avg_epoch_loss = epoch_loss / batch_count
                val_fid = validate_model(enc_params, dec_params, val_samples=500)

                training_history["epoch_losses"].append({"epoch": epoch, "avg_loss": float(avg_epoch_loss)})
                training_history["val_fidelity"].append({"epoch": epoch, "val_fidelity": float(val_fid)})

                enc_snapshot = enc_params.clone().detach()
                dec_snapshot = dec_params.clone().detach()
                training_history["weights_history"].append(
                    {
                        "encoder": enc_snapshot.numpy(),
                        "decoder": dec_snapshot.numpy(),
                    }
                )
                torch.save(enc_snapshot, f"model_parameters/qae_encoder_epoch_{epoch}.pt")
                torch.save(dec_snapshot, f"model_parameters/qae_decoder_epoch_{epoch}.pt")
                print(
                    f"Epoch {epoch} completed: Train Loss = {avg_epoch_loss:.6f}, Val Fidelity = {val_fid:.6f}"
                )
                print("-" * 50)

        total_time = time.time() - start_time
        print(f"Extended training completed in {total_time:.2f} seconds!")
        torch.save(enc_params, "model_parameters/final_qae_encoder_weights.pt")
        torch.save(dec_params, "model_parameters/final_qae_decoder_weights.pt")
        torch.save(training_history, "model_parameters/qae_training_history.pt")
        print("Final QAE weights and training history saved!")
        return (enc_params, dec_params), training_history

    except Exception as e:
        print(f"Error in extended batch training: {e}")
        import traceback

        traceback.print_exc()
        return None, None


def test_trained_model(weights, test_samples=500):
    print("\nTesting trained QAE on test set...")
    try:
        enc_params, dec_params = weights
        subset = test_data[: min(test_samples, len(test_data))]
        fidelities = process_batch(subset, enc_params, dec_params)
        avg_fid = torch.mean(fidelities).item()
        std_fid = torch.std(fidelities).item()
        print(f"Average fidelity on {len(subset)} samples: {avg_fid:.6f} (std: {std_fid:.6f})")
        return fidelities
    except Exception as e:
        print(f"Error in testing: {e}")
        return None


def continue_training_from_saved_weights(starting_epoch=10, additional_epochs=5):
    """
    从保存的权重继续训练
    starting_epoch: 开始继续训练的epoch编号
    additional_epochs: 额外训练的epoch数
    """
    try:
        # 加载最新的权重
        latest_encoder_path = f"model_parameters/qae_encoder_epoch_{starting_epoch-1}.pt"
        latest_decoder_path = f"model_parameters/qae_decoder_epoch_{starting_epoch-1}.pt"
        
        if not (os.path.exists(latest_encoder_path) and os.path.exists(latest_decoder_path)):
            print(f"无法找到起始权重文件，无法继续训练")
            return None, None
            
        enc_params = torch.load(latest_encoder_path, map_location=torch.device('cpu'))
        dec_params = torch.load(latest_decoder_path, map_location=torch.device('cpu'))
        
        # 确保参数设置为可训练
        enc_params.requires_grad_(True)
        dec_params.requires_grad_(True)
        
        print(f"从第{starting_epoch}个epoch开始继续训练，额外训练{additional_epochs}个epoch")
        
        n_samples = 1000  # 使用相同数量的样本
        samples = train_data[:n_samples]

        # 使用与之前相同的优化器设置
        opt = torch.optim.Adam([enc_params, dec_params], lr=0.005)  # 保持相同的学习率
        batch_size = 100  # 保持相同的批量大小

        # 加载之前的训练历史
        prev_history_path = "model_parameters/qae_training_history.pt"
        if os.path.exists(prev_history_path):
            prev_history = torch.load(prev_history_path, weights_only=False)
            training_history = prev_history
            # 清空当前epoch之后的记录，因为我们将在后面追加新记录
            training_history["epoch_losses"] = training_history["epoch_losses"][:starting_epoch]
            training_history["val_fidelity"] = training_history["val_fidelity"][:starting_epoch]
            training_history["weights_history"] = training_history["weights_history"][:starting_epoch]
        else:
            # 如果没有找到之前的训练历史，则创建新的
            training_history = {
                "epoch_losses": [],
                "val_fidelity": [],
                "batch_losses": [],
                "weights_history": [],
                "data_split_info": {
                    "train_size": len(train_data),
                    "val_size": len(val_data),
                    "test_size": len(test_data),
                    "actual_train_used": n_samples,
                },
            }

        print(f"Starting continued QAE training from epoch {starting_epoch}...")
        start_time = time.time()

        for epoch in range(starting_epoch, starting_epoch + additional_epochs):
            epoch_loss = 0.0
            batch_count = 0

            for i in range(0, n_samples, batch_size):
                batch = samples[i : i + batch_size]

                # 直接计算梯度而不是使用闭包
                opt.zero_grad()
                fidelities = []
                for img_params in batch:
                    if isinstance(img_params, np.ndarray):
                        img_params = torch.from_numpy(img_params).float()
                    fid = qae_circuit(img_params, enc_params, dec_params)
                    fidelities.append(fid)
                fidelities = torch.stack(fidelities)
                loss = 1.0 - torch.mean(fidelities)
                loss.backward()

                pre_enc = enc_params.clone().detach()
                pre_dec = dec_params.clone().detach()
                
                # 更新参数
                opt.step()
                
                current_loss = loss.item()
                epoch_loss += current_loss
                batch_count += 1

                training_history["batch_losses"].append(
                    {
                        "epoch": epoch,
                        "batch": i // batch_size,
                        "loss": float(current_loss),
                        "enc_norm": float(torch.norm(pre_enc)),
                        "dec_norm": float(torch.norm(pre_dec)),
                    }
                )

                if (i // batch_size) % 3 == 0:  # 调整打印频率
                    print(f"Epoch {epoch}, Batch {i//batch_size}: loss = {current_loss:.6f}")

            if batch_count > 0:
                avg_epoch_loss = epoch_loss / batch_count
                val_fid = validate_model(enc_params, dec_params, val_samples=500)

                training_history["epoch_losses"].append({"epoch": epoch, "avg_loss": float(avg_epoch_loss)})
                training_history["val_fidelity"].append({"epoch": epoch, "val_fidelity": float(val_fid)})

                enc_snapshot = enc_params.clone().detach()
                dec_snapshot = dec_params.clone().detach()
                training_history["weights_history"].append(
                    {
                        "encoder": enc_snapshot.numpy(),
                        "decoder": dec_snapshot.numpy(),
                    }
                )
                torch.save(enc_snapshot, f"model_parameters/qae_encoder_epoch_{epoch}.pt")
                torch.save(dec_snapshot, f"model_parameters/qae_decoder_epoch_{epoch}.pt")
                print(
                    f"Epoch {epoch} completed: Train Loss = {avg_epoch_loss:.6f}, Val Fidelity = {val_fid:.6f}"
                )
                print("-" * 50)

        total_time = time.time() - start_time
        print(f"Continued training completed in {total_time:.2f} seconds!")
        torch.save(enc_params, "model_parameters/final_qae_encoder_weights.pt")
        torch.save(dec_params, "model_parameters/final_qae_decoder_weights.pt")
        torch.save(training_history, "model_parameters/qae_training_history.pt")
        print("Final QAE weights and updated training history saved!")
        return (enc_params, dec_params), training_history

    except Exception as e:
        print(f"Error in continued batch training: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def train_with_dynamic_lr(starting_epoch=40, additional_epochs=10):
    """
    使用动态学习率的训练函数
    starting_epoch: 开始继续训练的epoch编号
    additional_epochs: 额外训练的epoch数
    """
    try:
        # 加载最新的权重
        latest_encoder_path = f"model_parameters/qae_encoder_epoch_{starting_epoch-1}.pt"
        latest_decoder_path = f"model_parameters/qae_decoder_epoch_{starting_epoch-1}.pt"
        
        if not (os.path.exists(latest_encoder_path) and os.path.exists(latest_decoder_path)):
            print(f"无法找到起始权重文件，无法继续训练")
            return None, None
            
        enc_params = torch.load(latest_encoder_path, map_location=torch.device('cpu'))
        dec_params = torch.load(latest_decoder_path, map_location=torch.device('cpu'))
        
        # 确保参数设置为可训练
        enc_params.requires_grad_(True)
        dec_params.requires_grad_(True)
        
        print(f"从第{starting_epoch}个epoch开始使用动态学习率训练，额外训练{additional_epochs}个epoch")
        
        n_samples = 1000  # 使用相同数量的样本
        samples = train_data[:n_samples]

        # 使用Adam优化器
        opt = torch.optim.Adam([enc_params, dec_params], lr=0.001)  # 使用较小的初始学习率
        
        # 定义学习率调度器 - 每隔一定epoch将学习率乘以0.9
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.9)
        
        batch_size = 100  # 保持相同的批量大小

        # 加载之前的训练历史
        prev_history_path = "model_parameters/qae_training_history.pt"
        if os.path.exists(prev_history_path):
            prev_history = torch.load(prev_history_path, weights_only=False)
            training_history = prev_history
            # 清空当前epoch之后的记录，因为我们将在后面追加新记录
            training_history["epoch_losses"] = training_history["epoch_losses"][:starting_epoch]
            training_history["val_fidelity"] = training_history["val_fidelity"][:starting_epoch]
            training_history["weights_history"] = training_history["weights_history"][:starting_epoch]
        else:
            # 如果没有找到之前的训练历史，则创建新的
            training_history = {
                "epoch_losses": [],
                "val_fidelity": [],
                "batch_losses": [],
                "weights_history": [],
                "data_split_info": {
                    "train_size": len(train_data),
                    "val_size": len(val_data),
                    "test_size": len(test_data),
                    "actual_train_used": n_samples,
                },
            }

        print(f"Starting QAE training with dynamic learning rate from epoch {starting_epoch}...")
        start_time = time.time()

        for epoch in range(starting_epoch, starting_epoch + additional_epochs):
            epoch_loss = 0.0
            batch_count = 0

            for i in range(0, n_samples, batch_size):
                batch = samples[i : i + batch_size]

                # 直接计算梯度而不是使用闭包
                opt.zero_grad()
                fidelities = []
                for img_params in batch:
                    if isinstance(img_params, np.ndarray):
                        img_params = torch.from_numpy(img_params).float()
                    fid = qae_circuit(img_params, enc_params, dec_params)
                    fidelities.append(fid)
                fidelities = torch.stack(fidelities)
                loss = 1.0 - torch.mean(fidelities)
                loss.backward()

                pre_enc = enc_params.clone().detach()
                pre_dec = dec_params.clone().detach()
                
                # 更新参数
                opt.step()
                
                current_loss = loss.item()
                epoch_loss += current_loss
                batch_count += 1

                training_history["batch_losses"].append(
                    {
                        "epoch": epoch,
                        "batch": i // batch_size,
                        "loss": float(current_loss),
                        "enc_norm": float(torch.norm(pre_enc)),
                        "dec_norm": float(torch.norm(pre_dec)),
                        "lr": opt.param_groups[0]['lr']  # 记录当前学习率
                    }
                )

                if (i // batch_size) % 3 == 0:  # 调整打印频率
                    print(f"Epoch {epoch}, Batch {i//batch_size}: loss = {current_loss:.6f}, LR = {opt.param_groups[0]['lr']:.6f}")

            if batch_count > 0:
                avg_epoch_loss = epoch_loss / batch_count
                val_fid = validate_model(enc_params, dec_params, val_samples=500)

                training_history["epoch_losses"].append({"epoch": epoch, "avg_loss": float(avg_epoch_loss)})
                training_history["val_fidelity"].append({"epoch": epoch, "val_fidelity": float(val_fid)})

                enc_snapshot = enc_params.clone().detach()
                dec_snapshot = dec_params.clone().detach()
                training_history["weights_history"].append(
                    {
                        "encoder": enc_snapshot.numpy(),
                        "decoder": dec_snapshot.numpy(),
                    }
                )
                torch.save(enc_snapshot, f"model_parameters/qae_encoder_epoch_{epoch}.pt")
                torch.save(dec_snapshot, f"model_parameters/qae_decoder_epoch_{epoch}.pt")
                print(
                    f"Epoch {epoch} completed: Train Loss = {avg_epoch_loss:.6f}, Val Fidelity = {val_fid:.6f}, LR = {opt.param_groups[0]['lr']:.6f}"
                )
                
                # 更新学习率
                scheduler.step()
                
                print("-" * 50)

        total_time = time.time() - start_time
        print(f"Dynamic LR training completed in {total_time:.2f} seconds!")
        torch.save(enc_params, "model_parameters/final_qae_encoder_weights.pt")
        torch.save(dec_params, "model_parameters/final_qae_decoder_weights.pt")
        torch.save(training_history, "model_parameters/qae_training_history.pt")
        print("Final QAE weights and updated training history saved!")
        return (enc_params, dec_params), training_history

    except Exception as e:
        print(f"Error in dynamic LR batch training: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    print("Starting quantum-classical hybrid model with quantum autoencoder...")
    print("=" * 60)
    print(f"Data Split: {TRAIN_RATIO*100:.0f}% Train, {VAL_RATIO*100:.0f}% Validation, {TEST_RATIO*100:.0f}% Test")
    print(f"Training set: {len(train_data)} samples")
    print(f"Validation set: {len(val_data)} samples")
    print(f"Test set: {len(test_data)} samples")
    print("=" * 60)

    # 检查是否已有训练过的权重，决定是从头开始还是继续训练
    latest_encoder_path = "model_parameters/final_qae_encoder_weights.pt"
    latest_decoder_path = "model_parameters/final_qae_decoder_weights.pt"
    
    if os.path.exists(latest_encoder_path) and os.path.exists(latest_decoder_path):
        print("检测到已有的训练权重，将从最新权重继续训练...")
        weights, history = continue_training_from_saved_weights(starting_epoch=60, additional_epochs=10)
    else:
        print("未检测到已有的训练权重，将从头开始训练...")
        weights, history = train_batch_version_extended()
        
    if weights is not None:
        test_trained_model(weights)
        print("\nSaved files in 'model_parameters' directory:")
        for file in sorted(os.listdir("model_parameters")):
            print(f"  - {file}")
    else:
        print("Training failed.")
