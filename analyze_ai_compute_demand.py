"""
analyze_ai_compute_demand.py
人工智能未来算力需求增加分析图

分析并可视化人工智能模型训练所需算力的历史增长趋势与未来预测。
数据来源：Epoch AI、OpenAI Scaling 论文及公开研究报告。
"""

import numpy as np
import matplotlib.pyplot as plt

# ── 字体与负号配置（支持中文） ──────────────────────────────────────────
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ══════════════════════════════════════════════════════════════════════════════
# 1. 历史 AI 训练算力数据（单位：FLOPs）
#    来源：Epoch AI 数据库 / 公开论文
# ══════════════════════════════════════════════════════════════════════════════
# (年份, 模型名称, 训练算力 FLOPs, 分类)
AI_MODELS = [
    (2012, "AlexNet",          4.3e18,   "视觉模型"),
    (2014, "VGGNet",           1.9e19,   "视觉模型"),
    (2015, "ResNet-50",        3.2e18,   "视觉模型"),
    (2016, "NASNet",           1.8e21,   "自动搜索"),
    (2017, "Transformer",      5.3e18,   "语言模型"),
    (2018, "BERT-Large",       6.4e20,   "语言模型"),
    (2019, "GPT-2",            1.5e21,   "语言模型"),
    (2019, "T5-11B",           2.0e22,   "语言模型"),
    (2020, "GPT-3",            3.1e23,   "大语言模型"),
    (2021, "Switch-C",         1.3e23,   "大语言模型"),
    (2021, "Megatron-Turing",  2.0e23,   "大语言模型"),
    (2022, "PaLM-540B",        2.5e24,   "大语言模型"),
    (2022, "Chinchilla",       5.8e23,   "大语言模型"),
    (2023, "GPT-4",            2.1e25,   "大语言模型"),
    (2023, "Llama 2-70B",      6.9e23,   "开源模型"),
    (2023, "Gemini Ultra",     5.0e25,   "大语言模型"),
    (2024, "Llama 3-70B",      6.3e24,   "开源模型"),
    (2024, "GPT-4o",           6.0e25,   "大语言模型"),
]

# 摩尔定律参考算力（以 2012 年 AlexNet 为基准，每 2 年翻倍）
MOORE_BASE_YEAR    = 2012
MOORE_BASE_FLOPS   = 4.3e18
MOORE_DOUBLING     = 2.0       # 每 N 年翻倍

# 未来预测终止年份
FUTURE_END         = 2035

# 相对算力倍数基准（GPT-4 发布年份）
GPT4_BASELINE_YEAR = 2023

# ══════════════════════════════════════════════════════════════════════════════
# 2. 数据准备
# ══════════════════════════════════════════════════════════════════════════════
years      = np.array([m[0] for m in AI_MODELS], dtype=float)
flops      = np.array([m[2] for m in AI_MODELS], dtype=float)
names      = [m[1] for m in AI_MODELS]
categories = [m[3] for m in AI_MODELS]

# 历史拟合：对数线性回归（log10 FLOPs ~ year）
log_flops  = np.log10(flops)
coeffs     = np.polyfit(years, log_flops, 1)           # [斜率, 截距]
slope, intercept = coeffs
if slope <= 0:
    print("⚠️  警告：拟合斜率为非正值，趋势预测结果无效。")
    exit(1)
doubling_months  = np.log10(2) / slope * 12            # 翻倍所需月数

print("=" * 60)
print("📊 AI 算力需求增长分析")
print("=" * 60)
print(f"  历史拟合斜率    : {slope:.4f} (log₁₀ FLOPs / 年)")
print(f"  算力翻倍周期    : {doubling_months:.1f} 个月")
print(f"  对比摩尔定律    : {MOORE_DOUBLING * 12:.0f} 个月")
print()

# 预测曲线
fit_years   = np.linspace(2012, FUTURE_END, 300)
fit_log     = np.polyval(coeffs, fit_years)

# 摩尔定律曲线
moore_log   = np.log10(MOORE_BASE_FLOPS) + (fit_years - MOORE_BASE_YEAR) / MOORE_DOUBLING * np.log10(2)

# GPT-4 基准对数值（用于相对倍数计算）
gpt4_log    = np.polyval(coeffs, GPT4_BASELINE_YEAR)


def relative_compute(year):
    """计算指定年份相对于 GPT-4 的算力倍数。"""
    return 10 ** (np.polyval(coeffs, year) - gpt4_log)

# ══════════════════════════════════════════════════════════════════════════════
# 3. 绘图
# ══════════════════════════════════════════════════════════════════════════════
CATEGORY_COLORS = {
    "视觉模型":   "#4C9BE8",
    "语言模型":   "#F5A623",
    "大语言模型": "#E84C4C",
    "开源模型":   "#7ED321",
    "自动搜索":   "#9B59B6",
}

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle("人工智能未来算力需求增加分析", fontsize=18, fontweight='bold', y=0.98)

# ── 子图1：历史散点 + 趋势线 ──────────────────────────────────────────────
ax1 = axes[0, 0]
for cat, color in CATEGORY_COLORS.items():
    mask = [c == cat for c in categories]
    ax1.scatter(years[mask], log_flops[mask],
                color=color, label=cat, s=80, zorder=5, edgecolors='white', linewidths=0.8)

ax1.plot(fit_years[fit_years <= 2024], np.polyval(coeffs, fit_years[fit_years <= 2024]),
         'k--', linewidth=1.5, label='历史趋势拟合', alpha=0.7)
ax1.plot(fit_years[fit_years >= 2024], np.polyval(coeffs, fit_years[fit_years >= 2024]),
         'r--', linewidth=2.0, label='未来趋势预测', alpha=0.9)
ax1.plot(fit_years, moore_log,
         'b:', linewidth=1.5, label=f'摩尔定律 (×2/{MOORE_DOUBLING:.0f}年)', alpha=0.6)

# 标注关键模型
KEY_LABELS = {"AlexNet", "BERT-Large", "GPT-3", "GPT-4", "Gemini Ultra", "GPT-4o"}
for i, name in enumerate(names):
    if name in KEY_LABELS:
        ax1.annotate(name, xy=(years[i], log_flops[i]),
                     xytext=(3, 4), textcoords='offset points',
                     fontsize=7.5, color='#333333')

ax1.set_xlabel("年份", fontsize=12)
ax1.set_ylabel("训练算力 (log₁₀ FLOPs)", fontsize=12)
ax1.set_title("AI 模型训练算力历史增长与未来趋势", fontsize=13, fontweight='bold')
ax1.legend(fontsize=8, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(2011, FUTURE_END + 1)
ax1.axvline(x=2024, color='gray', linestyle=':', alpha=0.5, label='当前时间')

# ── 子图2：未来十年预测柱状图 ──────────────────────────────────────────────
ax2 = axes[0, 1]
future_years = np.arange(2025, FUTURE_END + 1)
future_log   = np.polyval(coeffs, future_years)

# 以 GPT-4 为基准计算相对倍数
relative_mul = np.array([relative_compute(y) for y in future_years])

bars = ax2.bar(future_years, relative_mul,
               color=plt.cm.Reds(np.linspace(0.4, 0.9, len(future_years))),
               edgecolor='white', linewidth=0.8)
for bar, val in zip(bars, relative_mul):
    if val > 5:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{val:.0f}×', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax2.set_xlabel("年份", fontsize=12)
ax2.set_ylabel("相对 GPT-4 算力倍数", fontsize=12)
ax2.set_title("未来 AI 算力需求预测\n（相对 GPT-4 训练算力）", fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_xticks(future_years)
ax2.set_xticklabels([str(y) for y in future_years], rotation=45, ha='right', fontsize=9)

# ── 子图3：算力年增长率 ────────────────────────────────────────────────────
ax3 = axes[1, 0]

# 每年算力增长率（实际数据：年份取整后的最大算力）
year_max = {}
for y, f in zip(years, flops):
    yr = int(y)
    if yr not in year_max or f > year_max[yr]:
        year_max[yr] = f

sorted_years  = sorted(year_max.keys())
sorted_flops  = [year_max[y] for y in sorted_years]
growth_rates  = [sorted_flops[i] / sorted_flops[i - 1]
                 for i in range(1, len(sorted_flops))]
growth_years  = sorted_years[1:]

bars3 = ax3.bar(growth_years, growth_rates,
                color=['#E84C4C' if r > 10 else '#F5A623' if r > 3 else '#4C9BE8'
                       for r in growth_rates],
                edgecolor='white', linewidth=0.8)
for bar, val in zip(bars3, growth_rates):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
             f'{val:.1f}×', ha='center', va='bottom', fontsize=8.5)

ax3.axhline(y=np.mean(growth_rates), color='navy', linestyle='--', linewidth=1.5,
            label=f'年均增长倍数 {np.mean(growth_rates):.1f}×')
ax3.set_xlabel("年份", fontsize=12)
ax3.set_ylabel("算力年增长倍数", fontsize=12)
ax3.set_title("AI 训练算力年增长率分析", fontsize=13, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_xticks(growth_years)
ax3.set_xticklabels([str(y) for y in growth_years], rotation=45, ha='right', fontsize=9)

# ── 子图4：算力需求驱动因素与量子计算潜力 ──────────────────────────────────
ax4 = axes[1, 1]

# 水平条形图：各驱动因素贡献权重（示意性数据，基于业界研究）
factors = ["模型参数量扩大", "训练数据量增加", "算法复杂度提升",
           "多模态融合训练", "强化学习微调", "安全对齐训练"]
weights = [35, 25, 15, 12, 8, 5]
colors4 = ['#E84C4C', '#F5A623', '#4C9BE8', '#7ED321', '#9B59B6', '#1ABC9C']

bars4 = ax4.barh(factors, weights, color=colors4, edgecolor='white', linewidth=0.8)
for bar, val in zip(bars4, weights):
    ax4.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
             f'{val}%', va='center', fontsize=10, fontweight='bold')

ax4.set_xlabel("贡献占比 (%)", fontsize=12)
ax4.set_title("AI 算力需求增长驱动因素分析", fontsize=13, fontweight='bold')
ax4.set_xlim(0, 45)
ax4.grid(True, alpha=0.3, axis='x')
ax4.invert_yaxis()

# ── 全局信息框 ────────────────────────────────────────────────────────────
info_text = (
    f"📈 关键指标摘要\n"
    f"• AI 算力年均增长: {10**slope:.0f}× / 年\n"
    f"• 算力翻倍周期:  {doubling_months:.1f} 个月\n"
    f"• 摩尔定律翻倍: {MOORE_DOUBLING * 12:.0f} 个月\n"
    f"• AI 算力增速超摩尔定律 {MOORE_DOUBLING * 12 / doubling_months:.1f}×\n"
    f"• 预测 2030 年需求: GPT-4 的 {relative_compute(2030):.0f}×"
)
fig.text(0.5, 0.01, info_text, ha='center', va='bottom', fontsize=10,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                   edgecolor='#F5A623', alpha=0.9))

plt.tight_layout(rect=[0, 0.09, 1, 0.97])
output_path = 'ai_compute_demand_analysis.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ 分析图已保存至: {output_path}")
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# 4. 控制台报告
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("🎯 未来算力需求预测报告")
print("=" * 60)
print(f"{'年份':<8} {'预测算力 (log₁₀ FLOPs)':<28} {'相对 GPT-4 倍数':<20}")
print("-" * 60)
for yr in range(2025, FUTURE_END + 1):
    log_val  = np.polyval(coeffs, yr)
    rel      = relative_compute(yr)
    print(f"  {yr:<6} {log_val:<28.2f} {rel:<20.1f}×")
print("=" * 60)
print("⚠️  注：以上预测基于历史趋势线性外推，实际算力受技术突破、")
print("     能耗限制、量子计算发展等多种因素影响。")
print("=" * 60)
