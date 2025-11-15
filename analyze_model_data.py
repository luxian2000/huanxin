import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体以支持中文标签
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_training_data(model_path='model_30km_1k'):
    """
    加载训练历史数据
    """
    training_history_path = os.path.join(model_path, 'training_history.pt')
    
    if not os.path.exists(training_history_path):
        raise FileNotFoundError(f"找不到训练历史文件: {training_history_path}")
    
    # 加载训练历史数据，设置weights_only=False以避免加载问题
    training_history = torch.load(training_history_path, weights_only=False)
    
    return training_history

def plot_training_curves(training_history):
    """
    绘制训练和验证损失曲线
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 提取训练损失数据
    epoch_losses = training_history['epoch_losses']
    val_losses = training_history['val_losses']
    batch_losses = training_history['batch_losses']
    
    epochs = [item['epoch'] for item in epoch_losses]
    train_losses = [item['avg_loss'] for item in epoch_losses]
    val_losses_vals = [item['val_loss'] for item in val_losses]
    
    # 绘制epoch级别的训练和验证损失
    ax1.plot(epochs, train_losses, 'o-', label='训练损失', color='blue')
    ax1.plot(epochs, val_losses_vals, 's-', label='验证损失', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('损失值')
    ax1.set_title('训练和验证损失曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 绘制batch级别的训练损失
    batch_epochs = [item['epoch'] for item in batch_losses]
    batch_vals = [item['loss'] for item in batch_losses]
    
    ax2.plot(batch_vals, label='Batch损失', color='green', alpha=0.7)
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('损失值')
    ax2.set_title('Batch级别训练损失')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_loss_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_weight_changes(training_history):
    """
    绘制权重变化情况
    """
    batch_losses = training_history['batch_losses']
    
    pre_weights_norms = [item['pre_weights_norm'] for item in batch_losses]
    post_weights_norms = [item['post_weights_norm'] for item in batch_losses]
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(pre_weights_norms, label='更新前权重范数', alpha=0.7)
    plt.plot(post_weights_norms, label='更新后权重范数', alpha=0.7)
    
    plt.xlabel('Batch')
    plt.ylabel('权重范数')
    plt.title('训练过程中权重范数变化')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('weight_changes.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_batch_loss_detailed(training_history):
    """
    绘制每个epoch内batch损失的详细曲线
    """
    # 提取batch损失数据
    batch_losses = training_history['batch_losses']
    
    # 按epoch组织数据
    epoch_data = {}
    for item in batch_losses:
        epoch = item['epoch']
        if epoch not in epoch_data:
            epoch_data[epoch] = []
        epoch_data[epoch].append(item['loss'])
    
    # 创建图表
    plt.figure(figsize=(15, 8))
    
    # 为每个epoch绘制损失曲线
    colors = plt.cm.tab10(np.linspace(0, 1, len(epoch_data)))
    for i, (epoch, losses) in enumerate(sorted(epoch_data.items())):
        batches = list(range(len(losses)))
        plt.plot(batches, losses, color=colors[i], 
                label=f'Epoch {epoch}', linewidth=2, alpha=0.8)
    
    plt.xlabel('Batch索引')
    plt.ylabel('损失值')
    plt.title('各Epoch内Batch损失函数曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    all_losses = [item['loss'] for item in batch_losses]
    mean_loss = np.mean(all_losses)
    min_loss = np.min(all_losses)
    max_loss = np.max(all_losses)
    
    plt.text(0.02, 0.98, f'总Batch数: {len(all_losses)}\n平均损失: {mean_loss:.6f}\n最小损失: {min_loss:.6f}\n最大损失: {max_loss:.6f}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('detailed_batch_loss_by_epoch.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_epoch_batch_loss_heatmap(training_history):
    """
    绘制每个epoch内batch损失的热力图
    """
    # 提取batch损失数据
    batch_losses = training_history['batch_losses']
    
    # 按epoch组织数据
    epoch_data = {}
    for item in batch_losses:
        epoch = item['epoch']
        if epoch not in epoch_data:
            epoch_data[epoch] = []
        epoch_data[epoch].append(item['loss'])
    
    # 转换为矩阵形式
    max_batches = max(len(batches) for batches in epoch_data.values())
    loss_matrix = np.full((len(epoch_data), max_batches), np.nan)
    
    for i, (epoch, losses) in enumerate(sorted(epoch_data.items())):
        loss_matrix[i, :len(losses)] = losses
    
    # 绘制热力图
    plt.figure(figsize=(15, 6))
    im = plt.imshow(loss_matrix, cmap='viridis', aspect='auto', interpolation='none')
    
    plt.xlabel('Batch索引')
    plt.ylabel('Epoch')
    plt.title('各Epoch内Batch损失热力图')
    
    # 设置y轴标签
    plt.yticks(range(len(epoch_data)), [f'Epoch {i}' for i in sorted(epoch_data.keys())])
    
    # 添加颜色条
    cbar = plt.colorbar(im)
    cbar.set_label('损失值')
    
    plt.tight_layout()
    plt.savefig('epoch_batch_loss_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_quantum_weights(model_path='model_30km_1k'):
    """
    分析量子权重的变化
    """
    # 加载初始和最终权重
    initial_weights_path = os.path.join(model_path, 'initial_quantum_weights.pt')
    final_weights_path = os.path.join(model_path, 'final_quantum_weights.pt')
    
    if os.path.exists(initial_weights_path) and os.path.exists(final_weights_path):
        initial_weights = torch.load(initial_weights_path, weights_only=False)
        final_weights = torch.load(final_weights_path, weights_only=False)
        
        # 计算权重变化
        weight_diff = final_weights - initial_weights
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # 可视化初始权重
        im1 = ax1.imshow(initial_weights.numpy(), cmap='viridis', aspect='auto')
        ax1.set_title('初始量子权重')
        ax1.set_xlabel('参数索引')
        ax1.set_ylabel('层索引')
        plt.colorbar(im1, ax=ax1)
        
        # 可视化最终权重
        im2 = ax2.imshow(final_weights.numpy(), cmap='viridis', aspect='auto')
        ax2.set_title('最终权重')
        ax2.set_xlabel('参数索引')
        ax2.set_ylabel('层索引')
        plt.colorbar(im2, ax=ax2)
        
        # 可视化权重变化
        im3 = ax3.imshow(weight_diff.numpy(), cmap='coolwarm', aspect='auto')
        ax3.set_title('权重变化')
        ax3.set_xlabel('参数索引')
        ax3.set_ylabel('层索引')
        plt.colorbar(im3, ax=ax3)
        
        plt.tight_layout()
        plt.savefig('quantum_weights_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"初始权重统计: 均值={np.mean(initial_weights.numpy()):.6f}, 标准差={np.std(initial_weights.numpy()):.6f}")
        print(f"最终权重统计: 均值={np.mean(final_weights.numpy()):.6f}, 标准差={np.std(final_weights.numpy()):.6f}")
        print(f"权重变化统计: 均值={np.mean(weight_diff.numpy()):.6f}, 标准差={np.std(weight_diff.numpy()):.6f}")

def print_training_summary(training_history):
    """
    打印训练摘要信息
    """
    print("=" * 60)
    print("训练摘要")
    print("=" * 60)
    
    data_info = training_history['data_split_info']
    print(f"数据集划分信息:")
    print(f"  训练集大小: {data_info['train_size']}")
    print(f"  验证集大小: {data_info['val_size']}")
    print(f"  测试集大小: {data_info['test_size']}")
    print(f"  实际使用训练样本数: {data_info['actual_train_used']}")
    
    epoch_losses = training_history['epoch_losses']
    val_losses = training_history['val_losses']
    
    if epoch_losses:
        print(f"\n训练结果:")
        print(f"  训练轮数: {len(epoch_losses)}")
        print(f"  最终训练损失: {epoch_losses[-1]['avg_loss']:.6f}")
        print(f"  最终验证损失: {val_losses[-1]['val_loss']:.6f}")
        
        # 找到最佳验证损失
        best_val_loss = min(val_losses, key=lambda x: x['val_loss'])
        print(f"  最佳验证损失: {best_val_loss['val_loss']:.6f} (Epoch {best_val_loss['epoch']})")

def main():
    """
    主函数
    """
    try:
        # 加载训练数据
        training_history = load_training_data()
        
        # 打印训练摘要
        print_training_summary(training_history)
        
        # 绘制训练曲线
        plot_training_curves(training_history)
        
        # 绘制权重变化
        plot_weight_changes(training_history)
        
        # 绘制每个epoch内batch损失的详细曲线
        plot_batch_loss_detailed(training_history)
        
        # 绘制epoch-batch损失热力图
        plot_epoch_batch_loss_heatmap(training_history)
        
        # 分析量子权重
        analyze_quantum_weights()
        
        print("\n分析完成！已生成以下图表:")
        print("1. training_loss_curves.png - 训练和验证损失曲线")
        print("2. weight_changes.png - 权重变化情况")
        print("3. detailed_batch_loss_by_epoch.png - 各Epoch内Batch损失函数曲线")
        print("4. epoch_batch_loss_heatmap.png - 各Epoch内Batch损失热力图")
        print("5. quantum_weights_analysis.png - 量子权重分析")
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()