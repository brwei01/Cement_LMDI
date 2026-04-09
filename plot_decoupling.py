"""
Tapio Decoupling Scatter Plot (按区域 + 全球趋势线)
- 不同颜色 = 不同区域 (Region)
- 颜色深浅 = 年份（浅→深 表示早→晚）
- 不同形状 = 不同脱钩状态
- 黑色连线 + 大标记 = 全球脱钩趋势
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ── 读取数据 ──
df = pd.read_csv('Data1/results_updated/GDP_CE_Decoupling_5yrs_interval.csv', encoding='utf-8-sig')
df_global = pd.read_csv('Data1/results_updated/Global_GDP_CE_Decoupling_5yrs_interval.csv', encoding='utf-8-sig')

# 去除无效行
df = df.dropna(subset=['Decoupling_State'])
df = df[df['Decoupling_State'] != 'N/A']
df_global = df_global.dropna(subset=['Decoupling_State'])
df_global = df_global[df_global['Decoupling_State'] != 'N/A']

# 去除 Unknown 区域
df = df[~df['Region'].isin(['Unknown'])]

# ── 年份排序 ──
years_sorted = sorted(df['Years'].unique())
year_to_idx = {y: i for i, y in enumerate(years_sorted)}
n_years = len(years_sorted)

# ── 脱钩状态 → 标记形状 ──
state_markers = {
    'Strong Decoupling':              's',   # 方块
    'Weak Decoupling':                'o',   # 圆
    'Expansive Coupling':             'D',   # 菱形
    'Expansive Negative Decoupling':  '^',   # 上三角
    'Strong Negative Decoupling':     'v',   # 下三角
    'Weak Negative Decoupling':       'P',   # 加号（粗）
    'Recessive Coupling':             'X',   # X形
    'Recessive Decoupling':           '*',   # 星形
}

# ── 为每个区域分配一个色系 ──
region_cmaps = {
    'Africa':   'Oranges',
    'Americas': 'Blues',
    'Asia':     'Reds',
    'Europe':   'Greens',
    'Oceania':  'Purples',
}

regions = sorted(df['Region'].unique())
n_regions = len(regions)

# 对未定义的区域分配默认 colormap
_fallback_cmaps = ['YlOrBr', 'BuGn', 'OrRd', 'PuRd', 'YlGn']
for i, r in enumerate(regions):
    if r not in region_cmaps:
        region_cmaps[r] = _fallback_cmaps[i % len(_fallback_cmaps)]

region_cmap_objs = {r: plt.colormaps.get_cmap(region_cmaps[r]) for r in regions}

# ── 画图 ──
fig, ax = plt.subplots(figsize=(14, 10))

for _, row in df.iterrows():
    region  = row['Region']
    state   = row['Decoupling_State']
    year    = row['Years']
    x       = row['pct_GDP']
    y       = row['pct_CE']

    # 颜色：根据年份在 colormap 中取值（0.3~0.9 避免太浅/太深）
    t = year_to_idx[year] / max(n_years - 1, 1)
    color = region_cmap_objs[region](0.3 + 0.55 * t)

    marker = state_markers.get(state, 'o')

    ax.scatter(x, y, c=[color], marker=marker, s=80, edgecolors='white',
               linewidths=0.3, zorder=3, alpha=0.75)

# ── 全球趋势线（带箭头的路径）──
df_global = df_global.sort_values('Years')
global_x = df_global['pct_GDP'].values
global_y = df_global['pct_CE'].values
global_years = df_global['Years'].values

# 画带箭头的线段（浅灰→深灰渐变，表示时间推进）
for i in range(len(global_x) - 1):
    shade = 0.3 + 0.5 * (i / max(len(global_x) - 2, 1))
    ax.annotate('', xy=(global_x[i+1], global_y[i+1]),
                xytext=(global_x[i], global_y[i]),
                arrowprops=dict(arrowstyle='->', color=str(1 - shade),
                                lw=1.8, connectionstyle='arc3,rad=0.1'),
                zorder=4)

# 小圆点 + 年份标签
for i in range(len(global_x)):
    shade = 0.3 + 0.5 * (i / max(len(global_x) - 1, 1))
    ax.scatter(global_x[i], global_y[i], c=str(1 - shade), s=40,
               edgecolors='white', linewidths=0.6, zorder=5)
    label = global_years[i].split(' - ')[-1]
    # 交替偏移方向避免标签重叠
    offset = (8, 8) if i % 2 == 0 else (-8, -12)
    ax.annotate(label, (global_x[i], global_y[i]),
                textcoords='offset points', xytext=offset,
                fontsize=7, color=str(1 - shade),
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7),
                zorder=6)

# ── 坐标轴 ──
ax.axhline(0, color='black', linewidth=0.8, zorder=1)
ax.axvline(0, color='black', linewidth=0.8, zorder=1)
ax.set_xlabel('ΔGDP (Relative Change)', fontsize=13)
ax.set_ylabel('ΔCO₂ (Relative Change)', fontsize=13)
ax.set_title('Tapio Decoupling — by Region', fontsize=15, fontweight='bold')
ax.grid(True, alpha=0.15)

# ── 图例1：脱钩状态（形状）──
legend_markers = []
for state, marker in state_markers.items():
    legend_markers.append(
        Line2D([0], [0], marker=marker, color='grey', linestyle='None',
               markersize=8, label=state)
    )
leg1 = ax.legend(handles=legend_markers, loc='upper right', title='Decoupling Type',
                 fontsize=9, title_fontsize=10, framealpha=0.9, markerscale=1.2)
ax.add_artist(leg1)

# ── 图例：全球趋势 ──
global_legend = Line2D([0], [0], color='grey', marker='o', markersize=5,
                       linewidth=1.5, label='Global Trend →',
                       markerfacecolor='grey', markeredgecolor='white')
leg_global = ax.legend(handles=[global_legend], loc='lower right', fontsize=9, framealpha=0.9)
ax.add_artist(leg_global)
ax.add_artist(leg1)

# ── 图例2：区域颜色色带 ──
cbar_width = 0.11
cbar_height = 0.035 * n_regions
ax_inset = fig.add_axes([0.22, 0.72, cbar_width, cbar_height])
ax_inset.set_xlim(0, n_years)
ax_inset.set_ylim(0, n_regions)

for ri, region in enumerate(regions):
    cmap = region_cmap_objs[region]
    for yi in range(n_years):
        t = yi / max(n_years - 1, 1)
        color = cmap(0.3 + 0.55 * t)
        ax_inset.add_patch(plt.Rectangle((yi, n_regions - 1 - ri), 1, 1,
                                          facecolor=color, edgecolor='white',
                                          linewidth=0.3))

ax_inset.set_yticks(np.arange(n_regions) + 0.5)
ax_inset.set_yticklabels(list(reversed(regions)), fontsize=7.5)
ax_inset.set_xticks([])
ax_inset.set_title(f'light → dark ({years_sorted[0][:4]} → {years_sorted[-1][-4:]})',
                    fontsize=8, pad=3)
ax_inset.tick_params(axis='y', length=0, pad=2)
for spine in ax_inset.spines.values():
    spine.set_visible(False)

plt.subplots_adjust(left=0.18, right=0.95, top=0.93, bottom=0.08)
plt.savefig('GDP-CE decoupling_result_plot.png', dpi=200, bbox_inches='tight')
plt.close()
print("图已保存: GDP-CE decoupling_result_plot.png")
