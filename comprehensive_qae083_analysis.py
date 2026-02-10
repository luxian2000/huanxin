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
        print(f"加载训练历史失败 {history_path}: {e}")
        return None

def extract_metrics(history_data):
    """从训练历史中提取关键指标"""
    if not history_data:
        return None
    
    metrics = {
        'epochs': len(history_data.get('epoch_losses', [])),
        'epoch_losses': [epoch_data['avg_loss'] for epoch_data in history_data.get('epoch_losses', [])],
        'val_mse': [],
        'batch_losses': history_data.get('batch_losses', []),
        'config': history_data.get('network_config', {}),
        'data_info': history_data.get('data_split_info', {})
    }
    
    # 提取验证MSE
    val_mse_data = history_data.get('val_mse', [])
    for val_data in val_mse_data:
        mse_val = val_data.get('mse') or val_data.get('val_mse') or val_data.get('validation_mse')
        if mse_val is not None:
            metrics['val_mse'].append(mse_val)
    
    return metrics

def compare_training_phases(initial_metrics, continued_metrics):
    """比较初始训练和继续训练的结果"""
    print("=== QAE083 训练阶段对比分析 ===\n")
    
    if not initial_metrics or not continued_metrics:
        print("缺少必要的训练数据")
        return
    
    print("1. 训练配置对比:")
    print(f"   初始训练轮数: {initial_metrics['epochs']}")
    print(f"   继续训练轮数: {continued_metrics['epochs']}")
    print(f"   总训练轮数: {initial_metrics['epochs'] + continued_metrics['epochs']}")
    
    print(f"\n   编码维度: {initial_metrics['config'].get('encoded_dim', 'N/A')}")
    print(f"   量子比特数: {initial_metrics['config'].get('quantum_ansatz_qubits', 'N/A')}")
    print(f"   量子层数: {initial_metrics['config'].get('quantum_layers', 'N/A')}")
    
    print("\n2. 损失性能对比:")
    
    # 初始训练性能
    initial_losses = initial_metrics['epoch_losses']
    continued_losses = continued_metrics['epoch_losses']
    
    print("   初始训练阶段:")
    print(f"     - 起始损失: {initial_losses[0]:.6f}")
    print(f"     - 最终损失: {initial_losses[-1]:.6f}")
    print(f"     - 改善幅度: {(initial_losses[0] - initial_losses[-1]) / initial_losses[0] * 100:.2f}%")
    
    print("   继续训练阶段:")
    print(f"     - 起始损失: {continued_losses[0]:.6f}")
    print(f"     - 最终损失: {continued_losses[-1]:.6f}")
    print(f"     - 改善幅度: {(continued_losses[0] - continued_losses[-1]) / continued_losses[0] * 100:.2f}%")
    
    print("   整体表现:")
    overall_start = initial_losses[0]
    overall_end = continued_losses[-1] if continued_losses else initial_losses[-1]
    overall_improvement = (overall_start - overall_end) / overall_start * 100
    print(f"     - 总体改善幅度: {overall_improvement:.2f}%")
    
    # 验证性能对比
    if initial_metrics['val_mse'] and continued_metrics['val_mse']:
        initial_val = initial_metrics['val_mse']
        continued_val = continued_metrics['val_mse']
        
        print("\n3. 验证性能对比:")
        print(f"   初始验证MSE: {initial_val[-1]:.6f}")
        print(f"   继续验证MSE: {continued_val[-1]:.6f}")
        print(f"   验证性能改善: {(initial_val[-1] - continued_val[-1]) / initial_val[-1] * 100:.2f}%")
    
    print("\n4. 收敛性分析:")
    
    # 计算损失变化率
    def calculate_convergence_rate(losses):
        if len(losses) < 2:
            return 0
        rates = [(losses[i-1] - losses[i]) / losses[i-1] for i in range(1, len(losses))]
        return np.mean(rates) * 100
    
    initial_rate = calculate_convergence_rate(initial_losses)
    continued_rate = calculate_convergence_rate(continued_losses) if continued_losses else 0
    
    print(f"   初始阶段平均收敛率: {initial_rate:.4f}% per epoch")
    if continued_losses:
        print(f"   继续阶段平均收敛率: {continued_rate:.4f}% per epoch")
        
        if continued_rate > initial_rate:
            print("   - 收敛速度: 加快")
        elif continued_rate < initial_rate:
            print("   - 收敛速度: 减缓")
        else:
            print("   - 收敛速度: 基本稳定")
    
    print("\n5. 稳定性分析:")
    
    # 计算损失标准差
    initial_std = np.std(initial_losses)
    continued_std = np.std(continued_losses) if continued_losses else 0
    
    print(f"   初始阶段损失标准差: {initial_std:.6f}")
    if continued_losses:
        print(f"   继续阶段损失标准差: {continued_std:.6f}")
        
        if continued_std < initial_std:
            print("   - 训练稳定性: 提升")
        elif continued_std > initial_std:
            print("   - 训练稳定性: 下降")
        else:
            print("   - 训练稳定性: 基本不变")

def plot_comprehensive_comparison(initial_metrics, continued_metrics, save_path="./"):
    """绘制综合对比图"""
    if not initial_metrics:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 训练损失对比
    ax1 = axes[0, 0]
    initial_epochs = range(len(initial_metrics['epoch_losses']))
    continued_epochs = range(len(initial_metrics['epoch_losses']), 
                           len(initial_metrics['epoch_losses']) + len(continued_metrics['epoch_losses'])) if continued_metrics['epoch_losses'] else []
    
    ax1.plot(initial_epochs, initial_metrics['epoch_losses'], 'b-o', label='Initial Training', linewidth=2)
    if continued_epochs:
        ax1.plot(continued_epochs, continued_metrics['epoch_losses'], 'r-s', label='Continued Training', linewidth=2)
        # 连接点
        ax1.plot([initial_epochs[-1], continued_epochs[0]], 
                [initial_metrics['epoch_losses'][-1], continued_metrics['epoch_losses'][0]], 
                'g--', alpha=0.7, linewidth=1)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Average Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 验证MSE对比
    ax2 = axes[0, 1]
    if initial_metrics['val_mse']:
        ax2.plot(initial_epochs, initial_metrics['val_mse'], 'b-o', label='Initial Validation', linewidth=2)
    if continued_metrics['val_mse']:
        ax2.plot(continued_epochs, continued_metrics['val_mse'], 'r-s', label='Continued Validation', linewidth=2)
        if initial_metrics['val_mse']:
            ax2.plot([initial_epochs[-1], continued_epochs[0]], 
                    [initial_metrics['val_mse'][-1], continued_metrics['val_mse'][0]], 
                    'g--', alpha=0.7, linewidth=1)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation MSE')
    ax2.set_title('Validation Performance Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 损失改善趋势
    ax3 = axes[1, 0]
    if len(initial_metrics['epoch_losses']) >= 2:
        initial_improvements = [0] + [(initial_metrics['epoch_losses'][i-1] - initial_metrics['epoch_losses'][i]) / initial_metrics['epoch_losses'][i-1] * 100 
                                    for i in range(1, len(initial_metrics['epoch_losses']))]
        ax3.plot(initial_epochs, initial_improvements, 'b-', alpha=0.7, label='Initial Training Rate')
    
    if continued_metrics['epoch_losses'] and len(continued_metrics['epoch_losses']) >= 2:
        continued_improvements = [0] + [(continued_metrics['epoch_losses'][i-1] - continued_metrics['epoch_losses'][i]) / continued_metrics['epoch_losses'][i-1] * 100 
                                      for i in range(1, len(continued_metrics['epoch_losses']))]
        continued_full_epochs = range(len(initial_metrics['epoch_losses']), 
                                    len(initial_metrics['epoch_losses']) + len(continued_improvements))
        ax3.plot(continued_full_epochs, continued_improvements, 'r-', alpha=0.7, label='Continued Training Rate')
    
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Improvement Rate (%)')
    ax3.set_title('Convergence Rate Analysis')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 损失分布箱线图
    ax4 = axes[1, 1]
    data_to_plot = [initial_metrics['epoch_losses']]
    labels = ['Initial']
    
    if continued_metrics['epoch_losses']:
        data_to_plot.append(continued_metrics['epoch_losses'])
        labels.append('Continued')
    
    box_plot = ax4.boxplot(data_to_plot, labels=labels, patch_artist=True)
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    ax4.set_ylabel('Loss Values')
    ax4.set_title('Loss Distribution Comparison')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(save_path, "qae083_comprehensive_analysis.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n综合分析图表已保存到: {plot_file}")

def analyze_test_performance():
    """分析测试性能"""
    test_path = "./QAE083/test_results.pt"
    try:
        test_data = torch.load(test_path, map_location='cpu', weights_only=False)
        print("\n6. 测试集最终性能:")
        print(f"   - 测试交叉熵损失: {test_data.get('test_cross_entropy_loss', 'N/A'):.6f}")
        print(f"   - KL散度: {test_data.get('test_kl_divergence', 'N/A'):.6f}")
        print(f"   - JS散度: {test_data.get('test_jsd_divergence', 'N/A'):.6f}")
        print(f"   - Hellinger距离: {test_data.get('test_hellinger_distance', 'N/A'):.6f}")
        print(f"   - 测试样本数: {test_data.get('n_samples', 'N/A')}")
    except Exception as e:
        print(f"加载测试结果失败: {e}")

def main():
    # 加载数据
    print("正在加载QAE083训练数据...")
    
    initial_history = load_training_history("./QAE083/training_history.pt")
    continued_history = load_training_history("./QAE083/training_history_continued.pt")
    
    # 提取指标
    initial_metrics = extract_metrics(initial_history)
    continued_metrics = extract_metrics(continued_history)
    
    # 分析对比
    compare_training_phases(initial_metrics, continued_metrics)
    
    # 绘制对比图
    plot_comprehensive_comparison(initial_metrics, continued_metrics)
    
    # 分析测试性能
    analyze_test_performance()
    
    print("\n=== 综合分析完成 ===")

if __name__ == "__main__":
    main()