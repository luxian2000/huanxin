import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# 设置中文字体以支持中文标签
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def draw_hybrid_network_architecture():
    """
    绘制经典-量子混合神经网络结构图
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # 设置坐标轴范围
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 绘制输入层 (CSI数据)
    input_rect = patches.Rectangle((1, 7), 2, 1, linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.7)
    ax.add_patch(input_rect)
    ax.text(2, 7.5, '输入CSI数据\n(2560维)', ha='center', va='center', fontsize=12, weight='bold')
    
    # 绘制经典神经网络层
    classical_rect = patches.Rectangle((4, 7), 3, 1, linewidth=2, edgecolor='green', facecolor='lightgreen', alpha=0.7)
    ax.add_patch(classical_rect)
    ax.text(5.5, 7.5, '经典全连接层\n(2560→256)', ha='center', va='center', fontsize=12, weight='bold')
    
    # 绘制激活函数
    sigmoid_rect = patches.Rectangle((7.5, 7), 1.5, 1, linewidth=2, edgecolor='orange', facecolor='lightyellow', alpha=0.7)
    ax.add_patch(sigmoid_rect)
    ax.text(8.25, 7.5, 'Sigmoid\n激活函数', ha='center', va='center', fontsize=10)
    
    # 绘制归一化层
    norm_rect = patches.Rectangle((9.5, 7), 1.5, 1, linewidth=2, edgecolor='purple', facecolor='plum', alpha=0.7)
    ax.add_patch(norm_rect)
    ax.text(10.25, 7.5, '归一化\n层', ha='center', va='center', fontsize=10)
    
    # 绘制量子编码层
    amplitude_embedding_rect = patches.Rectangle((6, 5), 3, 1, linewidth=2, edgecolor='cyan', facecolor='lightcyan', alpha=0.7)
    ax.add_patch(amplitude_embedding_rect)
    ax.text(7.5, 5.5, '幅度编码\n(256→2^8=256)', ha='center', va='center', fontsize=10)
    
    # 绘制量子线路
    quantum_circuit_rect = patches.Rectangle((5, 3), 5, 1.5, linewidth=2, edgecolor='red', facecolor='lightcoral', alpha=0.7)
    ax.add_patch(quantum_circuit_rect)
    ax.text(7.5, 3.75, '强纠缠量子线路\n(12量子比特, 4层)', ha='center', va='center', fontsize=12, weight='bold')
    
    # 绘制量子测量
    measurement_rect = patches.Rectangle((6.5, 1.5), 2, 1, linewidth=2, edgecolor='magenta', facecolor='violet', alpha=0.7)
    ax.add_patch(measurement_rect)
    ax.text(7.5, 2, '量子测量\n(Hamiltonian)', ha='center', va='center', fontsize=10)
    
    # 绘制输出
    output_rect = patches.Rectangle((6.5, 0), 2, 1, linewidth=2, edgecolor='black', facecolor='white', alpha=0.7)
    ax.add_patch(output_rect)
    ax.text(7.5, 0.5, '输出\n(标量)', ha='center', va='center', fontsize=12, weight='bold')
    
    # 绘制连接箭头
    # 输入到经典层
    ax.annotate('', xy=(4, 7.5), xytext=(3, 7.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # 经典层到激活函数
    ax.annotate('', xy=(7.5, 7.5), xytext=(7, 7.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # 激活函数到归一化
    ax.annotate('', xy=(9.5, 7.5), xytext=(9, 7.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # 归一化到量子编码
    ax.annotate('', xy=(7.5, 6.5), xytext=(10.25, 7),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # 量子编码到量子线路
    ax.annotate('', xy=(7.5, 4.5), xytext=(7.5, 5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # 量子线路到测量
    ax.annotate('', xy=(7.5, 2.5), xytext=(7.5, 3),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # 测量到输出
    ax.annotate('', xy=(7.5, 1.5), xytext=(7.5, 1),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # 添加反馈连接（图像参数编码）
    img_params_rect = patches.Rectangle((12, 7), 2, 1, linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.7)
    ax.add_patch(img_params_rect)
    ax.text(13, 7.5, '图像参数\n(2560维)', ha='center', va='center', fontsize=10)
    
    # 图像参数到量子线路的反馈连接
    ax.annotate('', xy=(10, 3.75), xytext=(12, 7.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue', linestyle='dashed'))
    
    # 标题
    ax.text(8, 9.5, '经典-量子混合神经网络架构图', ha='center', va='center', fontsize=18, weight='bold')
    
    # 添加说明文字
    ax.text(1, 1, '网络结构说明:\n'
             '1. 输入层: 2560维CSI信道状态信息\n'
             '2. 经典层: 全连接层将2560维数据压缩到256维\n'
             '3. 激活函数: Sigmoid激活函数\n'
             '4. 归一化: 对输出进行归一化处理\n'
             '5. 量子编码: 将256维数据编码到8个量子比特中\n'
             '6. 量子线路: 12量子比特4层强纠缠结构\n'
             '7. 反馈连接: 原始2560维数据作为图像参数编码到量子线路\n'
             '8. 测量: 通过Hamiltonian测量得到标量输出',
             fontsize=10, verticalalignment='bottom', bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat"))
    
    plt.tight_layout()
    plt.savefig('hybrid_network_architecture.png', dpi=300, bbox_inches='tight')
    plt.show()

def draw_quantum_circuit_detail():
    """
    绘制详细的量子线路结构图
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # 设置坐标轴范围
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 绘制量子比特线
    n_qubits = 12
    for i in range(n_qubits):
        y_pos = 8 - i * 0.6
        ax.plot([1, 15], [y_pos, y_pos], 'black', lw=1)
        ax.text(0.5, y_pos, f'Qubit {i}', ha='center', va='center', fontsize=10)
    
    # 绘制幅度编码部分
    ax.text(2, 9, '幅度编码层', ha='center', va='center', fontsize=14, weight='bold')
    for i in range(8):  # 前8个量子比特用于幅度编码
        y_pos = 8 - i * 0.6
        circle = patches.Circle((2, y_pos), 0.1, color='blue', alpha=0.7)
        ax.add_patch(circle)
    
    # 绘制强纠缠层 (4层)
    layer_positions = [4, 6, 8, 10]
    layer_names = ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4']
    
    for idx, (pos, name) in enumerate(zip(layer_positions, layer_names)):
        ax.text(pos, 9, name, ha='center', va='center', fontsize=12, weight='bold')
        
        # 绘制旋转门
        for i in range(n_qubits):
            y_pos = 8 - i * 0.6
            rect = patches.Rectangle((pos-0.2, y_pos-0.15), 0.4, 0.3, 
                                   linewidth=1, edgecolor='red', facecolor='lightcoral', alpha=0.7)
            ax.add_patch(rect)
            ax.text(pos, y_pos, f'R{i}', ha='center', va='center', fontsize=8)
        
        # 绘制纠缠门
        for i in range(n_qubits):
            y_pos1 = 8 - i * 0.6
            y_pos2 = 8 - ((i + 1) % n_qubits) * 0.6
            # 画控制点
            circle = patches.Circle((pos+0.4, y_pos1), 0.05, color='black')
            ax.add_patch(circle)
            # 画目标点
            rect = patches.Rectangle((pos+0.35, y_pos2-0.1), 0.1, 0.2, 
                                   linewidth=1, edgecolor='black', facecolor='black')
            ax.add_patch(rect)
            # 连接线
            ax.plot([pos+0.4, pos+0.4], [y_pos1, y_pos2], 'black', lw=1)
    
    # 绘制图像参数编码反馈
    ax.text(12, 9, '图像参数编码\n(反向)', ha='center', va='center', fontsize=12, weight='bold')
    for i in range(n_qubits):
        y_pos = 8 - i * 0.6
        triangle = patches.Polygon([[12, y_pos-0.1], [11.8, y_pos+0.1], [12.2, y_pos+0.1]], 
                                 color='green', alpha=0.7)
        ax.add_patch(triangle)
    
    # 绘制测量部分
    ax.text(14, 9, '测量', ha='center', va='center', fontsize=14, weight='bold')
    for i in range(n_qubits):
        y_pos = 8 - i * 0.6
        meter = patches.Rectangle((13.8, y_pos-0.15), 0.4, 0.3, 
                                linewidth=1, edgecolor='purple', facecolor='violet', alpha=0.7)
        ax.add_patch(meter)
        ax.text(14, y_pos, 'M', ha='center', va='center', fontsize=10)
    
    # 标题
    ax.text(8, 9.7, '量子线路详细结构图 (12量子比特, 4层强纠缠)', ha='center', va='center', fontsize=16, weight='bold')
    
    # 添加说明文字
    ax.text(1, 1, '量子线路说明:\n'
             '1. 前8个量子比特 (Q0-Q7): 用于幅度编码256维经典输出\n'
             '2. 所有12个量子比特 (Q0-Q11): 参与强纠缠层计算\n'
             '3. 每层包含:\n'
             '   - 单量子比特旋转门 (R0-R11)\n'
             '   - 相邻量子比特之间的CNOT纠缠门\n'
             '4. 图像参数编码: 原始2560维数据通过反向幅度编码注入\n'
             '5. 最终测量: 通过Hamiltonian测量得到期望值',
             fontsize=10, verticalalignment='bottom', bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat"))
    
    plt.tight_layout()
    plt.savefig('quantum_circuit_detail.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_network_info():
    """
    打印网络结构信息
    """
    print("=" * 60)
    print("经典-量子混合神经网络结构信息")
    print("=" * 60)
    
    print("1. 输入层:")
    print("   - 输入维度: 2560 (CSI信道状态信息)")
    
    print("\n2. 经典神经网络部分:")
    print("   - 全连接层: 2560 → 256")
    print("   - 激活函数: Sigmoid")
    print("   - 归一化: L2归一化")
    
    print("\n3. 量子神经网络部分:")
    print("   - 量子比特数: 12")
    print("   - 编码量子比特: 8 (用于256维数据的幅度编码)")
    print("   - 线路层数: 4")
    print("   - 线路类型: 强纠缠层 (StronglyEntanglingLayers)")
    print("   - 编码方式: 幅度编码 + 图像参数反向编码")
    
    print("\n4. 输出层:")
    print("   - 测量方式: Hamiltonian期望值测量")
    print("   - 输出维度: 1 (标量)")
    
    print("\n5. 训练参数:")
    print("   - 训练样本数: 1000")
    print("   - 验证样本数: 12000")
    print("   - 测试样本数: 12000")
    print("   - Epoch数: 5")
    print("   - Batch大小: 50")
    print("   - 优化器: SGD")
    print("   - 学习率: 0.01")

def main():
    """
    主函数
    """
    try:
        # 打印网络结构信息
        print_network_info()
        
        # 绘制混合网络架构图
        draw_hybrid_network_architecture()
        
        # 绘制量子线路详细结构图
        draw_quantum_circuit_detail()
        
        print("\n网络结构可视化完成！已生成以下图表:")
        print("1. hybrid_network_architecture.png - 经典-量子混合神经网络架构图")
        print("2. quantum_circuit_detail.png - 量子线路详细结构图")
        
    except Exception as e:
        print(f"可视化过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()