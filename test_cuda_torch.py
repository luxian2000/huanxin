import torch

def test_cuda():
    print("PyTorch 版本:", torch.__version__)
    print("CUDA 是否可用:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU 数量:", torch.cuda.device_count())
        print("GPU 名称:", torch.cuda.get_device_name(0))
        # 在 GPU 上做一次简单计算
        a = torch.tensor([1.0, 2.0, 3.0]).cuda()
        b = torch.tensor([4.0, 5.0, 6.0]).cuda()
        c = a + b
        print("张量加法结果:", c.cpu().numpy())
    else:
        print("未检测到可用的 CUDA 设备")

if __name__ == "__main__":
    test_cuda()
