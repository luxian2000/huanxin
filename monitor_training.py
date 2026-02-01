#!/usr/bin/env python3
"""
训练监控脚本 - 实时查看QCNN训练状态
"""

import os
import time
import subprocess

def monitor_training():
    """监控训练进程"""
    print("🔍 开始监控训练进程...")
    print("=" * 50)
    
    # 检查输出目录
    output_dir = "QCNN01_gpu"
    
    while True:
        # 检查是否有新的模型文件生成
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            model_files = [f for f in files if f.endswith('.pt')]
            csv_files = [f for f in files if f.endswith('.csv')]
            
            print(f"\n📁 输出目录内容 ({time.strftime('%H:%M:%S')}):")
            print(f"模型文件: {len(model_files)} 个")
            print(f"CSV文件: {len(csv_files)} 个")
            
            if model_files:
                print("最新模型文件:")
                for f in sorted(model_files)[-5:]:  # 显示最新的5个文件
                    print(f"  - {f}")
                    
            if csv_files:
                print("CSV文件:")
                for f in csv_files:
                    print(f"  - {f}")
        
        # 检查训练进程是否还在运行
        try:
            # 检查是否有python进程在运行QCNN01_gpu.py
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            if 'QCNN01_gpu.py' in result.stdout:
                print("✅ 训练进程仍在运行")
            else:
                print("⚠️  训练进程可能已完成或停止")
                break
                
        except Exception as e:
            print(f"监控出错: {e}")
        
        print("-" * 30)
        time.sleep(30)  # 每30秒检查一次

if __name__ == "__main__":
    monitor_training()