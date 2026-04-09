"""
Tapio Decoupling States Classification — Academic Style
"""

import numpy as np
import matplotlib.pyplot as plt

# ── 配色方案（学术风格：低饱和、高对比）──
LINE_BETA08  = '#C45C3C'   # 暗砖红
LINE_BETA12  = '#2B5B8A'   # 深钢蓝
TEXT_NORMAL  = '#2D3436'   # 近黑灰
TEXT_STRONG  = '#8B1A1A'   # 暗红（强脱钩 / 强负脱钩高亮）
AXIS_COLOR   = '#4A4A4A'
BG_COLOR     = '#FAFAFA'

# 区域底色（极淡，区分四象限）
FILL_Q1 = '#FFF3E0'  # 右上 浅暖橙
FILL_Q2 = '#FCE4EC'  # 左上 浅粉
FILL_Q3 = '#E8F5E9'  # 左下 浅绿
FILL_Q4 = '#E3F2FD'  # 右下 浅蓝

fig, ax = plt.subplots(figsize=(10, 10))
fig.patch.set_facecolor('white')
ax.set_facecolor(BG_COLOR)

lim = 1.5

# ── 象限底色 ──
ax.fill_between([0, lim],  0,  lim, color=FILL_Q1, zorder=0)
ax.fill_between([-lim, 0], 0,  lim, color=FILL_Q2, zorder=0)
ax.fill_between([-lim, 0], -lim, 0, color=FILL_Q3, zorder=0)
ax.fill_between([0, lim],  -lim, 0, color=FILL_Q4, zorder=0)

# ── 坐标轴线 ──
ax.axhline(0, color=AXIS_COLOR, linewidth=1.0, zorder=2)
ax.axvline(0, color=AXIS_COLOR, linewidth=1.0, zorder=2)

# ── β = 0.8 和 β = 1.2 线 ──
x = np.linspace(-lim, lim, 300)
ax.plot(x, 0.8 * x, '--', color=LINE_BETA08, linewidth=2.5, label='β = 0.8', zorder=3)
ax.plot(x, 1.2 * x, '--', color=LINE_BETA12, linewidth=2.5, label='β = 1.2', zorder=3)

# ── 标注工具 ──
def add_label(ax, x, y, title, subtitle,
              fontsize_title=11, fontsize_sub=9,
              color=TEXT_NORMAL, ha='center'):
    ax.text(x, y, title,
            fontsize=fontsize_title, fontweight='bold',
            color=color, ha=ha, va='center', linespacing=1.4, zorder=5)
    if subtitle:
        ax.text(x, y - 0.08 * title.count('\n') - 0.14, subtitle,
                fontsize=fontsize_sub, color=color, ha=ha, va='top',
                linespacing=1.35, zorder=5, style='italic')

# ── 右上象限（GDP>0, CE>0）──
add_label(ax, 0.85, 0.22,
          'Weak\nDecoupling (WD)',
          'ΔCO₂>0, ΔGDP>0\n0<β<0.8')

add_label(ax, 1.08, 0.95,
          'Expansive\nCoupling (EC)',
          'ΔCO₂>0, ΔGDP>0\n0.8<β<1.2')

add_label(ax, 0.55, 1.15,
          'Expansive Negative\nDecoupling (END)',
          'ΔCO₂>0, ΔGDP>0\nβ>1.2')

# ── 左上象限（GDP<0, CE>0）──
add_label(ax, -0.75, 0.85,
          'Strong Negative\nDecoupling (SND)',
          'ΔCO₂>0, ΔGDP<0\nβ<0',
          fontsize_title=13, color=TEXT_STRONG)

# ── 右下象限（GDP>0, CE<0）──
add_label(ax, 0.75, -0.80,
          'Strong\nDecoupling (SD)',
          'ΔCO₂<0, ΔGDP>0\nβ<0',
          fontsize_title=13, color=TEXT_STRONG)

# ── 左下象限（GDP<0, CE<0）──
add_label(ax, -0.80, -0.30,
          'Weak Negative\nDecoupling (WND)',
          'ΔCO₂<0, ΔGDP<0\n0<β<0.8')

add_label(ax, -1.10, -0.90,
          'Recessive\nCoupling (RC)',
          'ΔCO₂<0, ΔGDP<0\n0.8<β<1.2')

add_label(ax, -0.45, -1.20,
          'Recessive\nDecoupling (RD)',
          'ΔCO₂<0, ΔGDP<0\nβ>1.2')

# ── 坐标轴设置 ──
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_xlabel('ΔGDP (Relative Change)', fontsize=13, labelpad=8, color=TEXT_NORMAL)
ax.set_ylabel('ΔCO₂ (Relative Change)', fontsize=13, labelpad=8, color=TEXT_NORMAL)
ax.set_title('Decoupling States Classification',
             fontsize=16, fontweight='bold', color=TEXT_NORMAL, pad=15)

ax.set_xticks(np.arange(-1.5, 2.0, 0.5))
ax.set_yticks(np.arange(-1.5, 2.0, 0.5))
ax.tick_params(colors=AXIS_COLOR, labelsize=10)
ax.set_aspect('equal')

for spine in ax.spines.values():
    spine.set_color(AXIS_COLOR)
    spine.set_linewidth(0.8)

# ── 图例 ──
leg = ax.legend(loc='upper right', fontsize=11, framealpha=0.95,
                edgecolor='#CCCCCC', fancybox=False)
leg.get_frame().set_linewidth(0.8)

plt.tight_layout()
plt.savefig('/home/claude/decoupling_states_academic.png',
            dpi=250, bbox_inches='tight', facecolor='white')
plt.close()
print("Done")
