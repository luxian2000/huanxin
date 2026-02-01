import torch
from QAE083 import compute_probability_loss

print('🔍 交叉熵与概率分布相似性分析')
print('=' * 50)

# 创建测试概率分布
target = torch.tensor([0.5, 0.3, 0.2])  # 目标分布

# 不同的预测分布，从完全匹配到完全不同
test_distributions = [
    ('完全匹配', torch.tensor([0.5, 0.3, 0.2])),
    ('略有差异', torch.tensor([0.4, 0.4, 0.2])),
    ('中等差异', torch.tensor([0.3, 0.3, 0.4])),
    ('较大差异', torch.tensor([0.1, 0.2, 0.7])),
    ('完全相反', torch.tensor([0.2, 0.3, 0.5])),  # 重新排列
]

print('目标分布:', target.tolist())
print()

for name, pred in test_distributions:
    ce_loss = compute_probability_loss(pred.unsqueeze(0), target.unsqueeze(0), 'cross_entropy')
    kl_loss = compute_probability_loss(pred.unsqueeze(0), target.unsqueeze(0), 'kl')

    print(f'{name:10s}: CE={ce_loss:.4f}, KL={kl_loss:.4f}')
    print(f'   预测分布: {pred.tolist()}')
    print()

print('📊 结论:')
print('• 交叉熵越小 → 两个概率分布越接近')
print('• 交叉熵=熵时 → 两个分布完全相同')
print('• 交叉熵越大 → 两个分布差异越大')