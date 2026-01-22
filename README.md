model_parameters: 
训练数据集： CSI_channel_30km
样本数：1K
epoch: 5
batch: 50*20
encoder为2560->256全连接层
decoder为 12qubits 4层*strong_entangling_layer

## pg08_QAE.py 训练效果分析

基于 `model_parameters` 目录中的训练中间数据，分析结果如下：

输出中间参数：model_parameters
分析中间lost function: analyze_training.py

- **训练周期**: 共 240 个 epoch
- **训练损失**: 从 0.978924 下降到 0.192861，损失下降约 80.3%，模型已收敛
- **验证集保真度**: 从 0.046416 提升到 0.807138，提升约 16.4 倍，模型学习效果显著
- **参数变化**: 编码器参数变化范数 8.55，解码器参数变化范数 8.65，参数调整充分
- **可视化结果**:
	- `training_loss_curve.png`：训练损失随 epoch 变化曲线
	- `validation_fidelity_curve.png`：验证集保真度随 epoch 变化曲线

**结论**：
模型训练效果良好，损失收敛，验证集保真度显著提升，参数调整充分。建议后续可进一步分析 batch 级损失细节或测试集表现。

## 如何测试训练好的模型

1. **加载模型参数**：
   - 从 `model_parameters/final_qae_encoder_weights.pt` 加载编码器参数
   - 从 `model_parameters/final_qae_decoder_weights.pt` 加载解码器参数

2. **准备测试数据**：
   - 使用测试集数据（test_data），通常为训练脚本中的 `data_30[train_size + val_size:]`
   - 确保数据格式为 torch.Tensor 或 numpy.ndarray

3. **运行测试**：
   - 使用 `test_model.py` 脚本进行测试
   - 脚本会计算每个测试样本的重构保真度和损失
   - 输出平均保真度和平均损失

4. **评估指标**：
   - **保真度 (Fidelity)**: 衡量重构质量，越接近 1 越好
   - **损失 (Loss)**: 1 - 保真度，越小越好

5. **示例代码**：
   ```python
   from test_model import test_model
   # 假设 test_data 已定义
   avg_fidelity, avg_loss = test_model(test_data, enc_params, dec_params, n_test=100)
   ```

## 测试结果

运行 `test_model.py` 对训练好的模型进行测试，结果如下：

- **测试样本数**: 500 个
- **平均保真度**: 0.807138
- **平均损失**: 0.192862

**结论**：
模型在测试集上的表现与验证集一致，保真度约0.81，损失约0.19，表明模型具有良好的泛化能力，没有出现明显的过拟合现象。

6. **停止训练条件**：
   - 当测试集保真度稳定或开始下降时
   - 损失收敛且无显著提升时
   - 达到预设 epoch 数或资源限制时
