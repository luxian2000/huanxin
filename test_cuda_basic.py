import subprocess
import sys

def test_cuda_availability():
    """测试CUDA的基本可用性"""
    print("=== CUDA环境检查 ===")
    
    # 检查nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ NVIDIA驱动和CUDA工具包正常工作")
            print(result.stdout.split('\n')[0])  # 显示第一行版本信息
        else:
            print("❌ nvidia-smi执行失败")
            return False
    except Exception as e:
        print(f"❌ 无法执行nvidia-smi: {e}")
        return False
    
    # 检查nvcc版本
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'release' in line:
                    print(f"✅ CUDA编译器版本: {line.strip()}")
                    break
        else:
            print("❌ nvcc执行失败")
    except Exception as e:
        print(f"❌ 无法执行nvcc: {e}")
    
    # 检查CUDA库路径
    try:
        import os
        cuda_paths = []
        if 'CUDA_HOME' in os.environ:
            cuda_paths.append(os.environ['CUDA_HOME'])
        if 'CUDA_PATH' in os.environ:
            cuda_paths.append(os.environ['CUDA_PATH'])
        
        # 常见的CUDA安装路径
        common_paths = ['/usr/local/cuda', '/opt/cuda']
        for path in common_paths:
            if os.path.exists(path):
                cuda_paths.append(path)
        
        if cuda_paths:
            print(f"✅ 找到CUDA路径: {cuda_paths}")
        else:
            print("⚠️  未找到明确的CUDA路径环境变量")
            
    except Exception as e:
        print(f"❌ 检查CUDA路径时出错: {e}")
    
    return True

def test_python_cuda_libraries():
    """测试Python中可用的CUDA相关库"""
    print("\n=== Python CUDA库检查 ===")
    
    # 测试cupy
    try:
        import cupy as cp
        print(f"✅ CuPy版本: {cp.__version__}")
        # 简单的GPU计算测试
        x = cp.array([1, 2, 3])
        y = cp.array([4, 5, 6])
        z = x + y
        print(f"✅ CuPy GPU计算测试通过: {z}")
    except ImportError:
        print("❌ CuPy未安装")
    except Exception as e:
        print(f"❌ CuPy测试失败: {e}")
    
    # 测试numba CUDA
    try:
        from numba import cuda
        print("✅ Numba CUDA支持可用")
        # 检查CUDA设备
        if cuda.is_available():
            print(f"✅ CUDA设备数量: {cuda.list_devices().count}")
        else:
            print("❌ CUDA设备不可用")
    except ImportError:
        print("❌ Numba未安装")
    except Exception as e:
        print(f"❌ Numba CUDA测试失败: {e}")

if __name__ == "__main__":
    test_cuda_availability()
    test_python_cuda_libraries()