"""
Tapio Decoupling Scatter Plot
- 不同颜色 = 不同国家/城市
- 颜色深浅 = 年份（浅→深 表示早→晚）
- 不同形状 = 不同脱钩状态
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ── 读取数据 ──
df = pd.read_csv('/mnt/project/GDP_CE_Decoupling_5yrs_interval.csv', encoding='utf-8-sig')

# 去除无效行
df = df.dropna(subset=['Decoupling_State'])
df = df[df['Decoupling_State'] != 'N/A']

# ── 可选：筛选国家（74个国家太多，建议选取感兴趣的子集）──
# 如果你想画全部国家，注释掉下面这行；如果只画部分国家，修改列表
SELECTED_COUNTRIES = [
    'China', 'United States', 'Germany', 'Japan', 'India', 'United Kingdom',
    'France', 'Brazil', 'South Korea', 'Canada', 'Australia', 'Russia'
]
# SELECTED_COUNTRIES = None  # 取消注释此行则画全部国家

if SELECTED_COUNTRIES:
    df = df[df['Countries'].isin(SELECTED_COUNTRIES)]

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

# ── 为每个国家分配一个色系（colormap）──
# 使用 tab20 + tab20b + tab20c 等调色板组合
base_cmaps = [
    'Blues', 'Oranges', 'Greens', 'Reds', 'Purples', 'YlOrBr',
    'BuGn', 'OrRd', 'PuRd', 'YlGn', 'GnBu', 'RdPu',
    'BuPu', 'YlGnBu', 'PuBuGn', 'YlOrRd', 'Greys', 'PuBu',
    'RdYlGn', 'RdYlBu', 'Spectral', 'coolwarm', 'copper', 'bone',
]

countries = sorted(df['Countries'].unique())
n_countries = len(countries)

# 给每个国家分配一个 colormap
country_cmaps = {}
for i, c in enumerate(countries):
    cmap_name = base_cmaps[i % len(base_cmaps)]
    country_cmaps[c] = plt.colormaps.get_cmap(cmap_name)

# ── 画图 ──
fig, ax = plt.subplots(figsize=(14, 10))

for _, row in df.iterrows():
    country = row['Countries']
    state   = row['Decoupling_State']
    year    = row['Years']
    x       = row['pct_GDP']
    y       = row['pct_CE']

    # 颜色：根据年份在 colormap 中取值（0.25~0.85 避免太浅/太深）
    t = year_to_idx[year] / max(n_years - 1, 1)
    color = country_cmaps[country](0.25 + 0.6 * t)

    marker = state_markers.get(state, 'o')

    ax.scatter(x, y, c=[color], marker=marker, s=80, edgecolors='white',
               linewidths=0.3, zorder=3)

# ── 坐标轴 ──
ax.axhline(0, color='black', linewidth=0.8, zorder=1)
ax.axvline(0, color='black', linewidth=0.8, zorder=1)
ax.set_xlabel('ΔGDP (Relative Change)', fontsize=13)
ax.set_ylabel('ΔCO₂ (Relative Change)', fontsize=13)
ax.set_title('Tapio Decoupling — All Countries', fontsize=15, fontweight='bold')
ax.grid(True, alpha=0.15)

# ── 图例1：脱钩状态（形状）──
legend_markers = []
for state, marker in state_markers.items():
    legend_markers.append(
        Line2D([0], [0], marker=marker, color='grey', linestyle='None',
               markersize=8, label=state)
    )
leg1 = ax.legend(handles=legend_markers, loc='upper right', title='Decoupling Type',
                 fontsize=8, title_fontsize=9, framealpha=0.9)
ax.add_artist(leg1)

# ── 图例2：国家颜色方块（类似参考图的色带）──
# 用 inset axes 画色带
n_cols = min(n_countries, 6)  # 每列最多6个国家
n_rows_legend = int(np.ceil(n_countries / 1))

# 在左上角放一个小的色带图例
cbar_width = 0.18
cbar_height = min(0.06 * n_countries, 0.55)
ax_inset = fig.add_axes([0.07, 0.98 - cbar_height - 0.02, cbar_width, cbar_height])
ax_inset.set_xlim(0, n_years)
ax_inset.set_ylim(0, n_countries)

for ci, country in enumerate(countries):
    cmap = country_cmaps[country]
    for yi in range(n_years):
        t = yi / max(n_years - 1, 1)
        color = cmap(0.25 + 0.6 * t)
        ax_inset.add_patch(plt.Rectangle((yi, n_countries - 1 - ci), 1, 1,
                                          facecolor=color, edgecolor='white',
                                          linewidth=0.3))

ax_inset.set_yticks(np.arange(n_countries) + 0.5)
ax_inset.set_yticklabels(list(reversed(countries)), fontsize=7)
ax_inset.set_xticks([])
ax_inset.set_title(f'light → dark ({years_sorted[0][:4]} → {years_sorted[-1][-4:]})',
                    fontsize=7, pad=3)
ax_inset.tick_params(axis='y', length=0, pad=2)
for spine in ax_inset.spines.values():
    spine.set_visible(False)

plt.subplots_adjust(left=0.28, right=0.95, top=0.93, bottom=0.08)
plt.savefig('/home/claude/decoupling_result_plot.png', dpi=200, bbox_inches='tight')
plt.close()
print("图已保存: decoupling_result_plot.png")
