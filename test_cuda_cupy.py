import cupy as cp

def test_cupy_cuda():
    print("CuPy 版本:", cp.__version__)
    print("可用 GPU 数量:", cp.cuda.runtime.getDeviceCount())
    # 在 GPU 上做一次简单计算
    a = cp.array([1, 2, 3])
    b = cp.array([4, 5, 6])
    c = a + b
    print("CuPy 张量加法结果:", cp.asnumpy(c))
    print("计算设备:", cp.cuda.Device())

if __name__ == "__main__":
    test_cupy_cuda()
