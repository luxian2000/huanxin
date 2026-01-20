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

def plot_batch_loss_curve(training_history):
    """
    绘制batch级别的损失函数曲线
    """
    # 提取batch损失数据
    batch_losses = training_history['batch_losses']
    batch_vals = [item['loss'] for item in batch_losses]
    epochs = [item['epoch'] for item in batch_losses]
    batches = list(range(len(batch_vals)))
    
    # 创建图表
    plt.figure(figsize=(15, 8))
    
    # 绘制所有batch的损失曲线
    plt.plot(batches, batch_vals, color='blue', alpha=0.7, linewidth=0.8, label='Batch损失')
    
    # 标记每个epoch的开始点
    epoch_boundaries = []
    for i, epoch in enumerate(epochs):
        if i == 0 or epoch != epochs[i-1]:
            epoch_boundaries.append((i, epoch))
    
    # 绘制epoch边界线
    for batch_idx, epoch in epoch_boundaries:
        plt.axvline(x=batch_idx, color='red', linestyle='--', alpha=0.7, linewidth=1)
        plt.text(batch_idx, plt.ylim()[1], f'Epoch {epoch}', 
                rotation=90, verticalalignment='top', 
                color='red', fontsize=9)
    
    plt.xlabel('Batch索引')
    plt.ylabel('损失值')
    plt.title('Batch级别损失函数曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    mean_loss = np.mean(batch_vals)
    min_loss = np.min(batch_vals)
    max_loss = np.max(batch_vals)
    
    plt.text(0.02, 0.98, f'平均损失: {mean_loss:.6f}\n最小损失: {min_loss:.6f}\n最大损失: {max_loss:.6f}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('batch_loss_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Batch损失统计:")
    print(f"  总Batch数: {len(batch_vals)}")
    print(f"  平均损失: {mean_loss:.6f}")
    print(f"  最小损失: {min_loss:.6f}")
    print(f"  最大损失: {max_loss:.6f}")
    print(f"  最终损失: {batch_vals[-1]:.6f}")

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

def main():
    """
    主函数
    """
    try:
        # 加载训练数据
        training_history = load_training_data()
        
        # 绘制batch损失曲线
        plot_batch_loss_curve(training_history)
        
        # 绘制epoch-batch损失热力图
        plot_epoch_batch_loss_heatmap(training_history)
        
        print("\n图表已生成:")
        print("1. batch_loss_curve.png - Batch级别损失函数曲线")
        print("2. epoch_batch_loss_heatmap.png - 各Epoch内Batch损失热力图")
        
    except Exception as e:
        print(f"绘制过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()