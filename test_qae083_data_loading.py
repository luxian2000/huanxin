#!/usr/bin/env python3
"""
测试QAE083.py的数据加载功能
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_data_loading():
    """测试数据加载功能"""
    try:
        print("正在测试QAE083数据加载功能...")
        
        # 导入QAE083模块中的数据加载函数
        from QAE083 import load_csinet_data
        
        # 测试数据加载
        train_data, val_data, test_data = load_csinet_data()
        
        print("\n✅ 数据加载测试成功！")
        print(f"训练数据形状: {train_data.shape}")
        print(f"验证数据形状: {val_data.shape}")
        print(f"测试数据形状: {test_data.shape}")
        print(f"数据类型: {train_data.dtype}")
        
        # 显示数据范围
        print(f"\n数据范围:")
        print(f"训练数据: [{train_data.min():.6f}, {train_data.max():.6f}]")
        print(f"验证数据: [{val_data.min():.6f}, {val_data.max():.6f}]")
        print(f"测试数据: [{test_data.min():.6f}, {test_data.max():.6f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_data_loading()
    if not success:
        sys.exit(1)