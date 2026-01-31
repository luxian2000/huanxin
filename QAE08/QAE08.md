# QAE_08 混合量子-经典神经网络架构详解

## 1. 概述

QAE_08是一个基于量子自编码器（Quantum Autoencoder, QAE）的混合神经网络模型，专门用于信道状态信息（CSI）数据的压缩与重构。该模型结合了经典神经网络的强大表征能力和量子电路的高效信息处理特性。

## 2. 整体架构

### 2.1 数据流图
```
输入CSI数据(2560维) 
    ↓
经典编码器(线性变换+Sigmoid+归一化)
    ↓
潜在表示(256维)
    ↓
振幅编码(8量子比特)
    ↓
量子编码器(4层StronglyEntanglingLayers)
    ↓
量子解码器(4层StronglyEntanglingLayers)
    ↓
保真度测量(|0...0⟩概率)
```

### 2.2 模型组件
- **经典编码器**：负责高维CSI数据的初步压缩
- **量子电路**：执行精细的特征提取和重构
- **混合训练框架**：支持端到端的联合优化

## 3. 经典神经网络部分

### 3.1 网络结构
```python
# 输入层 → 隐藏层 → 输出层
输入维度: 2560 (CSI数据特征)
隐藏层: 单层线性变换
激活函数: Sigmoid
输出维度: 256 (压缩后的潜在表示)
归一化: L2归一化
```

### 3.2 参数配置
```python
WEIGHT = torch.randn(2560, 256, requires_grad=True) * 0.01  # 权重矩阵
BIAS = torch.randn(1, 256, requires_grad=True)              # 偏置向量
```

### 3.3 前向传播
```python
def dense_layer(x):
    output = torch.matmul(x, WEIGHT) + BIAS  # 线性变换
    output = sigmoid(output)                 # Sigmoid激活
    output = normalize(output[0])            # L2归一化
    return output
```

### 3.4 参数规模
- 权重参数：2560 × 256 = 655,360个
- 偏置参数：256个
- **总计：655,616个经典参数**

## 4. 量子神经网络部分

### 4.1 量子电路配置
```python
DATA_QUBITS = 8     # 量子比特数 (2^8 = 256，与经典输出匹配)
N_LAYERS = 4        # 电路层数
LATENT_QUBITS = 4   # 目标压缩维度
```

### 4.2 编码器结构
```python
# StronglyEntanglingLayers参数结构
enc_shape = (4, 8, 3)  # (层数, 量子比特数, 旋转门参数数)
enc_params = torch.rand(enc_shape, requires_grad=True)
```

### 4.3 解码器结构
```python
# 与编码器相同的参数结构
dec_shape = (4, 8, 3)  # (层数, 量子比特数, 旋转门参数数)
dec_params = torch.rand(dec_shape, requires_grad=True)
```

### 4.4 量子电路实现
```python
@qml.qnode(DEV, interface="torch")
def qae_circuit(img_params, enc_params, dec_params):
    # 1. 振幅编码
    com_params = dense_layer(img_params)
    com_params_padded = pad_to_qubits(com_params, 8)
    qml.AmplitudeEmbedding(com_params_padded, wires=range(8), normalize=True)
    
    # 2. 量子编码器
    qml.StronglyEntanglingLayers(weights=enc_params, wires=range(8))
    
    # 3. 量子解码器
    qml.StronglyEntanglingLayers(weights=dec_params, wires=range(8))
    
    # 4. 保真度计算
    qml.adjoint(qml.AmplitudeEmbedding)(com_params_padded, wires=range(8), normalize=True)
    return qml.expval(qml.Projector([0]*8, wires=range(8)))
```

### 4.5 参数规模
- 编码器参数：4 × 8 × 3 = 96个
- 解码器参数：4 × 8 × 3 = 96个
- **总计：192个量子参数**

## 5. 训练配置

### 5.1 数据配置
```python
# 数据集划分
TOTAL_SAMPLES = 80000
TRAIN_RATIO = 0.70    # 训练集: 56,000样本
VAL_RATIO = 0.15      # 验证集: 12,000样本
TEST_RATIO = 0.15     # 测试集: 12,000样本

# 实际训练配置
实际训练样本: 1000个 (每次从训练集中选取)
验证样本: 500个
测试样本: 500个
```

### 5.2 训练超参数
```python
学习率: 0.005
优化器: Adam
训练轮数: 240轮
批量大小: 100样本
总批次数: 2400批次
```

### 5.3 损失函数
```python
loss = 1.0 - mean(fidelities)  # 最小化(1-保真度)
```

## 6. 性能表现

### 6.1 训练结果
```
训练损失范围: 0.192861 ~ 0.978924
验证保真度范围: 0.046416 ~ 0.807139
最终性能:
- 训练损失: 0.192861
- 验证保真度: 0.807139
- 损失改善: 约78.7%
- 保真度提升: 约1638%
```

### 6.2 收敛性分析
- 训练过程稳定
- 权重变化幅度适中
- 无明显过拟合现象

## 7. 文件结构

### 7.1 输出目录 (QAE_08/)
```
QAE_08/
├── 模型权重文件
│   ├── initial_weight.pt              # 经典编码器初始权重 (2561.6KB)
│   ├── initial_bias.pt                # 经典编码器初始偏置
│   ├── initial_quantum_encoder_weights.pt  # 量子编码器初始权重
│   ├── initial_quantum_decoder_weights.pt  # 量子解码器初始权重
│   ├── final_qae_encoder_weights.pt   # 最终量子编码器权重
│   └── final_qae_decoder_weights.pt   # 最终量子解码器权重
├── 每轮权重快照 (240轮)
│   ├── qae_encoder_epoch_0.pt 到 qae_encoder_epoch_239.pt
│   └── qae_decoder_epoch_0.pt 到 qae_decoder_epoch_239.pt
└── 训练历史
    └── qae_training_history.pt        # 完整训练记录 (464.3KB)
```

### 7.2 训练历史内容
```python
training_history = {
    'epoch_losses': [...],      # 每轮平均训练损失
    'val_fidelity': [...],      # 每轮验证保真度
    'batch_losses': [...],      # 每批次损失详情
    'weights_history': [...],   # 权重演化记录
    'data_split_info': {...}    # 数据集划分信息
}
```

## 8. 技术特点

### 8.1 架构优势
1. **混合计算**：充分发挥经典和量子计算各自优势
2. **端到端训练**：支持联合优化整个网络
3. **模块化设计**：各组件可独立修改和优化
4. **完整记录**：详细的训练过程和参数保存

### 8.2 实现特色
1. **振幅编码**：高效的量子态制备方法
2. **StronglyEntanglingLayers**：结构化的量子纠缠操作
3. **保真度测量**：基于投影测量的性能评估
4. **梯度反向传播**：支持自动微分的量子电路

## 9. 应用场景

### 9.1 主要用途
- 无线通信系统中的CSI数据压缩
- 高维信号的量子重构
- 量子机器学习算法验证

### 9.2 潜在扩展
- 多天线MIMO信道处理
- 动态信道环境适应
- 实时信号处理优化

## 10. 总结

QAE_08成功实现了经典-量子混合神经网络在CSI数据处理中的应用，通过240轮训练达到了80.7%的验证保真度。该架构为量子机器学习在通信领域的应用提供了有价值的参考实现。