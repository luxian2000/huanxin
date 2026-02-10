import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def load_training_history(history_path):
    """加载训练历史数据"""
    try:
        data = torch.load(history_path, map_location='cpu', weights_only=False)
        return data
    except Exception as e:
        print(f"加载训练历史失败: {e}")
        return None

def load_test_results(test_path):
    """加载测试结果数据"""
    try:
        data = torch.load(test_path, map_location='cpu', weights_only=False)
        return data
    except Exception as e:
        print(f"加载测试结果失败: {e}")
        return None

def analyze_training_progress(history_data):
    """分析训练进度"""
    if not history_data:
        return
    
    print("=== QAE083 训练结果分析 ===\n")
    
    # 提取训练数据
    epoch_losses = history_data.get('epoch_losses', [])
    val_mse = history_data.get('val_mse', [])
    batch_losses = history_data.get('batch_losses', [])
    data_split_info = history_data.get('data_split_info', {})
    network_config = history_data.get('network_config', {})
    
    print("1. 数据集配置:")
    print(f"   - 训练样本数: {data_split_info.get('train_size', 'N/A')}")
    print(f"   - 验证样本数: {data_split_info.get('val_size', 'N/A')}")
    print(f"   - 测试样本数: {data_split_info.get('test_size', 'N/A')}")
    print(f"   - 实际使用的训练样本数: {data_split_info.get('actual_train_used', 'N/A')}")
    
    print("\n2. 网络配置:")
    print(f"   - 编码维度: {network_config.get('encoded_dim', 'N/A')}")
    print(f"   - 量子编码量子比特数: {network_config.get('quantum_encoding_qubits', 'N/A')}")
    print(f"   - 量子ansatz量子比特数: {network_config.get('quantum_ansatz_qubits', 'N/A')}")
    print(f"   - 量子层数: {network_config.get('quantum_layers', 'N/A')}")
    print(f"   - 输出维度: {network_config.get('output_dim', 'N/A')}")
    print(f"   - 压缩比: {network_config.get('compression_ratio', 'N/A')}")
    
    print("\n3. 训练进度分析:")
    if epoch_losses:
        epochs = len(epoch_losses)
        avg_losses = [epoch_data['avg_loss'] for epoch_data in epoch_losses]
        
        print(f"   - 总训练轮数: {epochs}")
        print(f"   - 最终平均损失: {avg_losses[-1]:.6f}")
        print(f"   - 最佳平均损失: {min(avg_losses):.6f} (第{np.argmin(avg_losses)}轮)")
        print(f"   - 损失改善幅度: {(avg_losses[0] - avg_losses[-1]) / avg_losses[0] * 100:.2f}%")
        
        # 损失收敛性分析
        if epochs >= 5:
            recent_avg = np.mean(avg_losses[-5:])
            early_avg = np.mean(avg_losses[:5])
            print(f"   - 前5轮平均损失: {early_avg:.6f}")
            print(f"   - 后5轮平均损失: {recent_avg:.6f}")
            print(f"   - 近期改善: {(early_avg - recent_avg) / early_avg * 100:.2f}%")
            
            # 收敛判断
            loss_diff = abs(avg_losses[-1] - avg_losses[-2]) if epochs > 1 else 0
            if loss_diff < 0.001:
                print("   - 状态: 基本收敛")
            elif avg_losses[-1] < avg_losses[-2]:
                print("   - 状态: 持续下降")
            else:
                print("   - 状态: 可能过拟合或震荡")
    
    print("\n4. 验证性能分析:")
    if val_mse:
        # 尝试不同的键名
        mse_values = []
        for val_data in val_mse:
            # 尝试多种可能的键名
            mse_val = val_data.get('mse') or val_data.get('val_mse') or val_data.get('validation_mse')
            if mse_val is not None:
                mse_values.append(mse_val)
        
        if mse_values:
            print(f"   - 最终验证MSE: {mse_values[-1]:.6f}")
            print(f"   - 最佳验证MSE: {min(mse_values):.6f} (第{np.argmin(mse_values)}轮)")
            print(f"   - MSE改善幅度: {(mse_values[0] - mse_values[-1]) / mse_values[0] * 100:.2f}%")
            
            # 过拟合检测
            if epoch_losses and len(avg_losses) == len(mse_values):
                train_final = avg_losses[-1]
                val_final = mse_values[-1]
                gap = abs(val_final - train_final)
                if gap > train_final * 0.1:  # 10%差距认为有过拟合风险
                    print("   - 过拟合风险: 较高")
                elif gap > train_final * 0.05:  # 5%差距认为有轻微过拟合
                    print("   - 过拟合风险: 轻微")
                else:
                    print("   - 过拟合风险: 较低")
        else:
            print("   - 验证数据结构未知，请检查数据格式")
    else:
        print("   - 无验证数据")

def analyze_batch_level_performance(history_data):
    """分析批次级别性能"""
    if not history_data:
        return
        
    batch_losses = history_data.get('batch_losses', [])
    if not batch_losses:
        return
        
    print("\n5. 批次级别性能分析:")
    
    # 按epoch组织批次损失
    epoch_batches = {}
    for batch_data in batch_losses:
        epoch = batch_data['epoch']
        if epoch not in epoch_batches:
            epoch_batches[epoch] = []
        epoch_batches[epoch].append(batch_data['loss'])
    
    # 分析每个epoch的批次表现
    epoch_stats = []
    for epoch, losses in epoch_batches.items():
        if losses:
            stats = {
                'epoch': epoch,
                'mean_loss': np.mean(losses),
                'std_loss': np.std(losses),
                'min_loss': np.min(losses),
                'max_loss': np.max(losses),
                'batch_count': len(losses)
            }
            epoch_stats.append(stats)
    
    if epoch_stats:
        # 最稳定的epoch
        min_std_idx = np.argmin([stat['std_loss'] for stat in epoch_stats])
        most_stable = epoch_stats[min_std_idx]
        print(f"   - 最稳定epoch: 第{most_stable['epoch']}轮 (标准差: {most_stable['std_loss']:.6f})")
        
        # 波动最大的epoch
        max_std_idx = np.argmax([stat['std_loss'] for stat in epoch_stats])
        most_volatile = epoch_stats[max_std_idx]
        print(f"   - 波动最大epoch: 第{most_volatile['epoch']}轮 (标准差: {most_volatile['std_loss']:.6f})")
        
        # 损失分布趋势
        mean_losses = [stat['mean_loss'] for stat in epoch_stats]
        std_losses = [stat['std_loss'] for stat in epoch_stats]
        
        if len(mean_losses) >= 3:
            early_std = np.mean(std_losses[:len(std_losses)//3])
            late_std = np.mean(std_losses[-len(std_losses)//3:])
            if late_std < early_std * 0.8:
                print("   - 训练稳定性: 逐渐改善")
            elif late_std > early_std * 1.2:
                print("   - 训练稳定性: 逐渐恶化")
            else:
                print("   - 训练稳定性: 基本稳定")

def analyze_test_results(test_data):
    """分析测试结果"""
    if not test_data:
        return
        
    print("\n6. 测试集性能评估:")
    
    test_ce_loss = test_data.get('test_cross_entropy_loss')
    test_mse_loss = test_data.get('test_mse_loss')
    test_kl_div = test_data.get('test_kl_divergence')
    test_jsd_div = test_data.get('test_jsd_divergence')
    test_hellinger = test_data.get('test_hellinger_distance')
    n_samples = test_data.get('n_samples')
    encoded_dim = test_data.get('encoded_dim')
    loss_function = test_data.get('loss_function_used')
    
    if test_ce_loss is not None:
        print(f"   - 测试交叉熵损失: {test_ce_loss:.6f}")
    if test_mse_loss is not None:
        print(f"   - 测试MSE损失: {test_mse_loss:.6f}")
    if test_kl_div is not None:
        print(f"   - KL散度: {test_kl_div:.6f}")
    if test_jsd_div is not None:
        print(f"   - JS散度: {test_jsd_div:.6f}")
    if test_hellinger is not None:
        print(f"   - Hellinger距离: {test_hellinger:.6f}")
    
    print(f"   - 测试样本数: {n_samples}")
    print(f"   - 编码维度: {encoded_dim}")
    print(f"   - 使用的损失函数: {loss_function}")

def plot_training_curves(history_data, save_path="./"):
    """绘制训练曲线"""
    if not history_data:
        return
        
    epoch_losses = history_data.get('epoch_losses', [])
    val_mse = history_data.get('val_mse', [])
    
    if not epoch_losses:
        return
    
    epochs = range(len(epoch_losses))
    avg_losses = [epoch_data['avg_loss'] for epoch_data in epoch_losses]
    
    plt.figure(figsize=(15, 5))
    
    # 训练损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, avg_losses, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss Curve')
    plt.grid(True, alpha=0.3)
    
    # 验证MSE曲线
    if val_mse:
        # 尝试提取验证MSE值
        mse_values = []
        for val_data in val_mse:
            mse_val = val_data.get('mse') or val_data.get('val_mse') or val_data.get('validation_mse')
            if mse_val is not None:
                mse_values.append(mse_val)
        
        if mse_values:
            val_epochs = range(len(mse_values))
            plt.subplot(1, 2, 2)
            plt.plot(val_epochs, mse_values, 'r-', linewidth=2, marker='s', markersize=4)
            plt.xlabel('Epoch')
            plt.ylabel('Validation MSE')
            plt.title('Validation MSE Curve')
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(save_path, "qae083_training_analysis.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n训练曲线已保存到: {plot_file}")

def main():
    # 设置路径
    qae083_dir = "./QAE083"
    
    # 加载数据
    history_path = os.path.join(qae083_dir, "training_history.pt")
    test_path = os.path.join(qae083_dir, "test_results.pt")
    
    print("正在加载QAE083训练结果...")
    
    history_data = load_training_history(history_path)
    test_data = load_test_results(test_path)
    
    # 分析结果
    analyze_training_progress(history_data)
    analyze_batch_level_performance(history_data)
    analyze_test_results(test_data)
    
    # 绘制曲线
    plot_training_curves(history_data)
    
    print("\n=== 分析完成 ===")

if __name__ == "__main__":
    main()