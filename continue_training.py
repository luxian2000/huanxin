#!/usr/bin/env python3
"""
直接继续训练QAE083模型的脚本
从第10个epoch开始继续训练10个epoch
"""

import os
import sys
import torch
import numpy as np
import pennylane as qml
import scipy.io as sio
import csv
import time

# 添加当前目录到路径
sys.path.append('/Users/luxian/GitSpace/huanxin')

# 导入QAE083模块
from QAE083 import (
    continue_training_from_saved_weights,
    test_trained_model,
    load_csinet_data
)

def main():
    print("=" * 80)
    print("🔄 直接继续训练QAE083模型")
    print("=" * 80)

    # 加载数据
    train_data, val_data, test_data = load_csinet_data()

    # 继续训练
    continued_model, continued_history = continue_training_from_saved_weights(
        starting_epoch=10, additional_epochs=10
    )

    if continued_model is not None:
        # 测试继续训练后的模型
        final_test_loss = test_trained_model(continued_model, test_data, test_samples=500)

        print("\n" + "=" * 70)
        print("继续训练和测试完成！")
        print("=" * 70)
        print("所有结果保存在目录: QAE083/")
    else:
        print("\n继续训练失败！")

if __name__ == "__main__":
    main()