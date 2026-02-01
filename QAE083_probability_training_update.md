# QAE083.py 概率分布训练方式更新

## 更新时间
2026年1月31日

## 问题分析
原始代码存在的问题：
- 输入：未归一化的原始CSI数据（任意实数）
- 输出：量子电路测量得到的概率分布（和为1，非负）
- 损失函数：直接对两者做MSE
- **问题**：输入和输出不在同一数值空间，损失函数无物理意义，训练难以收敛

## 解决方案
实现"概率分布对概率分布"的训练方式：
1. 将原始输入数据归一化为概率分布
2. 量子电路输出概率分布
3. 两个概率分布在同一空间做损失计算
4. 支持梯度传播和端到端训练

## 主要修改

### 1. 新增函数：prepare_target_distribution
```python
def prepare_target_distribution(batch):
    """
    将原始输入batch转换为归一化的目标概率分布
    
    流程:
    1. 将(batch, 2, 32, 32)展平为(batch, 2048)
    2. 填充到2^11=2048维（这里已经是2048，无需填充）
    3. 归一化为概率分布（L2范数归一化后取模方）
    4. 确保和为1
    
    Returns: (batch_size, 2048) 归一化的概率分布
    """
```

**关键步骤**：
- `pad_to_qubits(vec, 11)`: 确保向量长度为2048
- `normalize_for_amplitude_embedding(vec)`: L2归一化
- `|vec|^2`: 取模方得到概率分布
- 重新归一化确保和为1

### 2. 修改损失函数：compute_probability_loss
```python
def compute_probability_loss(output_probs, target_probs, loss_type='mse'):
    """
    计算两个概率分布之间的损失
    
    支持两种损失类型:
    - 'mse': 概率分布的均方误差
    - 'kl': KL散度 KL(target || output)
    """
```

**MSE损失**：
```
L = mean((output_probs - target_probs)^2)
```

**KL散度损失**：
```
L = mean(Σ target_probs * log(target_probs / output_probs))
```

### 3. 简化混合模型
**修改前**：
```python
class HybridCsiNetQuantumAutoencoder(nn.Module):
    def __init__(self, csinet_encoder, prob_mapper, dec_params):
        # 包含ProbabilityToDataMapper
        self.prob_mapper = prob_mapper
    
    def forward(self, x):
        # 量子输出 -> prob_mapper -> reshape到图像
        return result.view(batch_size, 2, 32, 32)
```

**修改后**：
```python
class HybridCsiNetQuantumAutoencoder(nn.Module):
    def __init__(self, csinet_encoder, dec_params):
        # 移除prob_mapper
    
    def forward(self, x):
        # 直接返回概率分布，不reshape
        return result  # (batch_size, 2048)
```

**优势**：
- 减少参数量（移除了prob_mapper的所有参数）
- 简化架构，减少信息损失
- 输出直接是概率分布，便于损失计算

### 4. 更新训练循环
**修改前**：
```python
outputs = hybrid_model(batch)  # (batch, 2, 32, 32)
loss = compute_mse(outputs, batch)
```

**修改后**：
```python
outputs = hybrid_model(batch)  # (batch, 2048) 概率分布
targets = prepare_target_distribution(batch)  # (batch, 2048) 目标概率
loss = compute_probability_loss(outputs, targets, loss_type='mse')
```

### 5. 更新验证和测试函数
```python
def validate_model(model, val_data, val_samples=500, loss_type='mse'):
    outputs = model(subset)  # 概率分布
    targets = prepare_target_distribution(subset)  # 目标概率分布
    loss = compute_probability_loss(outputs, targets, loss_type=loss_type)
    return float(loss)
```

### 6. 移除ProbabilityToDataMapper
- 不再需要概率映射层
- 训练和推理都在概率空间进行
- 参数从 ~4百万 减少到 0

## 训练流程对比

### 修改前（不合理）
```
输入: (batch, 2, 32, 32) 原始CSI数据
  ↓
CsiNet编码: 2048 → 256维
  ↓
量子编码: 256维 → 8量子比特
  ↓
量子ansatz: 11量子比特
  ↓
概率测量: 2048维概率分布 [0,1] 和为1
  ↓
ProbabilityToDataMapper: 2048 → 2048维
  ↓
Reshape: (batch, 2, 32, 32)
  ↓
MSE损失: vs 原始输入 (任意实数)
❌ 问题: 概率分布 vs 任意实数，数值空间不匹配
```

### 修改后（合理）
```
输入: (batch, 2, 32, 32) 原始CSI数据
  ↓ (两条路径)
  
路径1: 目标概率分布
  Flatten: (batch, 2048)
  ↓
  Pad & Normalize: 归一化为概率分布
  ↓
  目标: (batch, 2048) 概率分布 [0,1] 和为1

路径2: 模型输出
  CsiNet编码: 2048 → 256维
  ↓
  量子编码: 256维 → 8量子比特
  ↓
  量子ansatz: 11量子比特
  ↓
  概率测量: (batch, 2048) 概率分布 [0,1] 和为1
  
  ↓ (两条路径汇合)
损失计算: MSE(输出概率, 目标概率) 或 KL(目标||输出)
✓ 正确: 两个概率分布在同一数值空间
```

## 理论优势

### 1. 数值空间一致性
- **输入和输出都是概率分布**
- 都在[0,1]区间，和为1
- 损失函数有明确物理意义

### 2. 量子物理对应
- 输入概率分布 ↔ 目标量子态的Born概率
- 输出概率分布 ↔ 实际量子测量的概率
- 损失函数 ↔ 量子态保真度

### 3. 训练稳定性
- 避免数值范围不匹配导致的梯度问题
- 概率分布天然归一化，有助于训练稳定
- 减少参数量，降低过拟合风险

### 4. 信息论基础
- MSE损失：欧氏距离，几何意义明确
- KL散度：信息熵差异，统计意义明确
- 两者都是概率分布间的标准度量

## 验证结果

运行 `test_probability_training.py`：
- ✅ 目标概率分布生成正确（和为1，非负）
- ✅ MSE和KL散度损失计算正确
- ✅ 梯度传播正常工作
- ✅ 完整训练流程验证通过

## 使用说明

### 训练新模型
```bash
cd /Users/luxian/GitSpace/huanxin
python QAE083.py
```

### 关键配置
- `loss_type='mse'`: 默认使用MSE损失（也可改为'kl'）
- 输入数据会自动转换为目标概率分布
- 输出是概率分布，不再reshape回图像格式

### 保存的模型文件
- `initial_csinet_encoder.pt`
- `initial_quantum_decoder_weights.pt`
- `final_csinet_encoder.pt`
- `final_quantum_decoder_weights.pt`
- `training_history.pt`
- `test_results.pt` （包含prob_loss和KL散度）

## 代码示例

### 推理使用
```python
# 加载模型
csinet_encoder = CsiNetEncoder(encoded_dim=256)
csinet_encoder.load_state_dict(torch.load('final_csinet_encoder.pt'))
dec_params = torch.load('final_quantum_decoder_weights.pt')

# 创建模型
model = HybridCsiNetQuantumAutoencoder(csinet_encoder, dec_params)
model.eval()

# 推理
input_data = torch.randn(1, 2, 32, 32)
output_probs = model(input_data)  # (1, 2048) 概率分布

# 如果需要转回图像格式（用于可视化）
# 注意：概率分布不能直接当作像素值，需要额外映射
```

### 自定义损失函数
```python
# 使用KL散度
loss = compute_probability_loss(outputs, targets, loss_type='kl')

# 混合损失
mse_loss = compute_probability_loss(outputs, targets, loss_type='mse')
kl_loss = compute_probability_loss(outputs, targets, loss_type='kl')
combined_loss = 0.7 * mse_loss + 0.3 * kl_loss
```

## 注意事项

1. **概率分布不是原始数据**：
   - 模型输出是概率分布，不能直接当作重构的CSI数据
   - 如需重构数据，需要额外的反映射层或后处理

2. **评估指标**：
   - 主要指标是概率损失和KL散度
   - 不再使用传统的NMSE（因为不在数据空间）

3. **数据归一化**：
   - 原始CSI数据需要合理缩放
   - 避免极值导致的数值问题

4. **批大小**：
   - 量子电路逐样本处理，建议batch_size=16-32
   - 过大的batch_size会增加计算时间

## 未来改进

1. **添加反映射层**：
   - 在推理时将概率分布映射回数据空间
   - 探索可训练的反映射方法

2. **优化损失函数**：
   - 尝试Wasserstein距离
   - 设计物理约束的损失项

3. **提高效率**：
   - 量子电路批处理并行化
   - 优化概率分布计算

4. **理论分析**：
   - 概率分布表示能力分析
   - 重构保真度理论界限

## 参考文献
- 量子自编码器: arXiv:1612.02806
- 概率分布度量: KL散度、Wasserstein距离
- 量子态保真度: Nielsen & Chuang, Quantum Computation and Quantum Information
