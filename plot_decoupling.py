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
df_global = pd.read_csv('Data1/global_results/Global_Decoupling_5yr.csv', encoding='utf-8-sig')

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
                fontsize=9, color=str(1 - shade),
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7),
                zorder=6)

# ── 坐标轴 ──
ax.axhline(0, color='black', linewidth=0.8, zorder=1)
ax.axvline(0, color='black', linewidth=0.8, zorder=1)
ax.set_xlabel('ΔGDP (Relative Change)', fontsize=18)
ax.set_ylabel('ΔCO₂ (Relative Change)', fontsize=18)
ax.set_title('Tapio Decoupling — by Region', fontsize=20, fontweight='bold')
ax.tick_params(axis='both', labelsize=13)
ax.grid(True, alpha=0.15)

# ── 图例1：脱钩状态（形状）──
legend_markers = []
for state, marker in state_markers.items():
    legend_markers.append(
        Line2D([0], [0], marker=marker, color='grey', linestyle='None',
               markersize=10, label=state)
    )
leg1 = ax.legend(handles=legend_markers, loc='upper right', title='Decoupling Type',
                 fontsize=12, title_fontsize=14, framealpha=0.9)
ax.add_artist(leg1)

# ── 图例2：区域颜色色带（含 Global）──
all_labels = regions + ['Global']
n_rows = len(all_labels)
# 放在第二象限（主坐标轴的左上区域），缩小尺寸
ax_inset = ax.inset_axes([0.06, 0.68, 0.16, 0.28])
ax_inset.set_xlim(0, n_years)
ax_inset.set_ylim(0, n_rows)

for ri, label in enumerate(all_labels):
    for yi in range(n_years):
        t = yi / max(n_years - 1, 1)
        if label == 'Global':
            shade = 0.3 + 0.5 * t
            color = str(1 - shade)
        else:
            cmap = region_cmap_objs[label]
            color = cmap(0.3 + 0.55 * t)
        ax_inset.add_patch(plt.Rectangle((yi, n_rows - 1 - ri), 1, 1,
                                          facecolor=color, edgecolor='white',
                                          linewidth=0.3))

ax_inset.set_yticks(np.arange(n_rows) + 0.5)
ax_inset.set_yticklabels(list(reversed(all_labels)), fontsize=9)
ax_inset.set_xticks([])
ax_inset.set_title(f'light → dark ({years_sorted[0][:4]} → {years_sorted[-1][-4:]})',
                    fontsize=8, pad=2)
ax_inset.tick_params(axis='y', length=0, pad=2)
for spine in ax_inset.spines.values():
    spine.set_visible(False)

plt.subplots_adjust(left=0.18, right=0.95, top=0.93, bottom=0.08)
plt.savefig('GDP-CE decoupling_result_plot.png', dpi=200, bbox_inches='tight')
plt.close()
print("图已保存: GDP-CE decoupling_result_plot.png")

# ══════════════════════════════════════════════════════════════
# ── Facet 版本：每个区域一个子图，全球趋势叠加在每个子图上 ──
# ══════════════════════════════════════════════════════════════

facet_regions = regions  # 已排序的区域列表
n_facets = len(facet_regions)
ncols = 3
nrows = int(np.ceil(n_facets / ncols))

fig_f, axes_f = plt.subplots(nrows, ncols, figsize=(22, nrows * 7),
                              sharex=False, sharey=False)
axes_flat = axes_f.flatten()

for idx, region in enumerate(facet_regions):
    ax_f = axes_flat[idx]
    df_reg = df[df['Region'] == region]
    cmap_reg = region_cmap_objs[region]

    # ── 散点 ──
    for _, row in df_reg.iterrows():
        state = row['Decoupling_State']
        year  = row['Years']
        x     = row['pct_GDP']
        y     = row['pct_CE']

        t = year_to_idx[year] / max(n_years - 1, 1)
        color = cmap_reg(0.3 + 0.55 * t)
        marker = state_markers.get(state, 'o')

        ax_f.scatter(x, y, c=[color], marker=marker, s=120,
                     edgecolors='white', linewidths=0.4, zorder=3, alpha=0.8)

    # ── 全球趋势线（灰色箭头）──
    for i in range(len(global_x) - 1):
        shade = 0.3 + 0.5 * (i / max(len(global_x) - 2, 1))
        ax_f.annotate('', xy=(global_x[i+1], global_y[i+1]),
                      xytext=(global_x[i], global_y[i]),
                      arrowprops=dict(arrowstyle='->', color=str(1 - shade),
                                      lw=1.5, connectionstyle='arc3,rad=0.1'),
                      zorder=4)

    for i in range(len(global_x)):
        shade = 0.3 + 0.5 * (i / max(len(global_x) - 1, 1))
        ax_f.scatter(global_x[i], global_y[i], c=str(1 - shade), s=50,
                     edgecolors='white', linewidths=0.5, zorder=5)
        lbl = global_years[i].split(' - ')[-1]
        offset = (8, 8) if i % 2 == 0 else (-8, -12)
        ax_f.annotate(lbl, (global_x[i], global_y[i]),
                      textcoords='offset points', xytext=offset,
                      fontsize=10, color=str(1 - shade), fontweight='bold',
                      bbox=dict(boxstyle='round,pad=0.15', fc='white',
                                ec='none', alpha=0.7),
                      zorder=6)

    # ── 子图装饰 ──
    ax_f.axhline(0, color='black', linewidth=0.8, zorder=1)
    ax_f.axvline(0, color='black', linewidth=0.8, zorder=1)
    ax_f.set_title(region, fontsize=20, fontweight='bold')
    ax_f.set_xlabel('ΔGDP (Relative Change)', fontsize=15)
    ax_f.set_ylabel('ΔCO₂ (Relative Change)', fontsize=15)
    ax_f.tick_params(axis='both', labelsize=13)
    ax_f.grid(True, alpha=0.15)

# 隐藏多余的空白子图（保留一个用于颜色图例）
has_spare = len(axes_flat) > n_facets
if has_spare:
    # 用第一个空白子图放颜色色带图例
    ax_legend_color = axes_flat[n_facets]
    ax_legend_color.set_visible(True)
    facet_color_labels = facet_regions + ['Global']
    n_color_rows = len(facet_color_labels)
    ax_legend_color.set_xlim(0, n_years)
    ax_legend_color.set_ylim(0, n_color_rows)

    for ri, clabel in enumerate(facet_color_labels):
        for yi in range(n_years):
            t = yi / max(n_years - 1, 1)
            if clabel == 'Global':
                shade = 0.3 + 0.5 * t
                ccolor = str(1 - shade)
            else:
                ccolor = region_cmap_objs[clabel](0.3 + 0.55 * t)
            ax_legend_color.add_patch(plt.Rectangle(
                (yi, n_color_rows - 1 - ri), 1, 1,
                facecolor=ccolor, edgecolor='white', linewidth=0.4))

    ax_legend_color.set_yticks(np.arange(n_color_rows) + 0.5)
    ax_legend_color.set_yticklabels(list(reversed(facet_color_labels)), fontsize=14, fontweight='bold')
    ax_legend_color.set_xticks([0, n_years])
    ax_legend_color.set_xticklabels([years_sorted[0][:4], years_sorted[-1][-4:]], fontsize=12)
    ax_legend_color.set_title('Region Color (light → dark = early → late)', fontsize=14, fontweight='bold', pad=8)
    ax_legend_color.tick_params(axis='y', length=0, pad=4)
    ax_legend_color.tick_params(axis='x', length=3, pad=4)
    for spine in ax_legend_color.spines.values():
        spine.set_visible(False)

    # 隐藏剩余空白子图
    for idx in range(n_facets + 1, len(axes_flat)):
        axes_flat[idx].set_visible(False)
else:
    for idx in range(n_facets, len(axes_flat)):
        axes_flat[idx].set_visible(False)

# ── 统一图例（脱钩状态 + 时间色带）放在底部 ──
legend_handles_facet = []
for state, marker in state_markers.items():
    legend_handles_facet.append(
        Line2D([0], [0], marker=marker, color='grey', linestyle='None',
               markersize=11, label=state)
    )
legend_handles_facet.append(
    Line2D([0], [0], marker='o', color='grey', linestyle='-',
           markersize=8, label='Global Trend', linewidth=1.5)
)

fig_f.legend(handles=legend_handles_facet, loc='lower center',
             ncol=min(5, len(legend_handles_facet)),
             fontsize=16, title='Decoupling Type', title_fontsize=18,
             framealpha=0.9, borderpad=0.8)

# fig_f.suptitle('Tapio Decoupling — by Region', fontsize=24, fontweight='bold', y=0.99)
fig_f.subplots_adjust(hspace=0.35, wspace=0.28, bottom=0.18, top=0.93)
fig_f.savefig('GDP-CE decoupling_facet.png', dpi=200, bbox_inches='tight')
plt.close(fig_f)
print("图已保存: GDP-CE decoupling_facet.png")