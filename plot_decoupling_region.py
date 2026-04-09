"""
Region-level GDP–CE Decoupling Scatter Plot
- 5 regions × 9 intervals ≈ 45 points (far fewer than country-level)
- Each region has its own color; shade = year (light→dark = early→late)
- Shape = decoupling state
- Labels show time period on each point
- Global trend overlay with arrows
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# ── 1. 读取国家级数据，聚合到区域级别 ──
df = pd.read_csv('Data1/results_updated/GDP_CE_Decoupling_5yrs_interval.csv',
                  encoding='utf-8-sig')
df = df.dropna(subset=['Decoupling_State'])
df = df[df['Decoupling_State'] != 'N/A']
df = df[~df['Region'].isin(['Unknown'])]

# 需要原始 CE / GDP 数据来聚合（按区域×年份加总 CE_Start, CE_End, GDP_Start, GDP_End）
# 用 Start / End 聚合
agg = df.groupby(['Region', 'Years']).agg(
    CE_Start=('CE_Start', 'sum'),
    CE_End=('CE_End', 'sum'),
    GDP_Start=('GDP_Start', 'sum'),
    GDP_End=('GDP_End', 'sum'),
).reset_index()

agg['pct_CE']  = (agg['CE_End'] - agg['CE_Start']) / agg['CE_Start']
agg['pct_GDP'] = (agg['GDP_End'] - agg['GDP_Start']) / agg['GDP_Start']

# 脱钩弹性 & 分类
def classify(row):
    ce, gdp = row['pct_CE'], row['pct_GDP']
    if gdp == 0:
        return 'N/A'
    e = ce / gdp
    if gdp > 0 and ce < 0:
        return 'Strong Decoupling'
    elif gdp > 0 and ce >= 0:
        if e < 0.8:
            return 'Weak Decoupling'
        elif e <= 1.2:
            return 'Expansive Coupling'
        else:
            return 'Expansive Negative Decoupling'
    elif gdp < 0 and ce < 0:
        if e > 1.2:
            return 'Recessive Decoupling'
        elif e >= 0.8:
            return 'Recessive Coupling'
        else:
            return 'Weak Negative Decoupling'
    elif gdp < 0 and ce >= 0:
        return 'Strong Negative Decoupling'
    return 'N/A'

agg['Decoupling_State'] = agg.apply(classify, axis=1)
agg = agg[agg['Decoupling_State'] != 'N/A']

# 全球趋势
df_global = pd.read_csv('Data1/results_updated/Global_GDP_CE_Decoupling_5yrs_interval.csv',
                         encoding='utf-8-sig')
df_global = df_global.dropna(subset=['Decoupling_State'])
df_global = df_global[df_global['Decoupling_State'] != 'N/A']

# ── 2. 设置 ──
years_sorted = sorted(agg['Years'].unique())
year_to_idx = {y: i for i, y in enumerate(years_sorted)}
n_years = len(years_sorted)

state_markers = {
    'Strong Decoupling':              's',
    'Weak Decoupling':                'o',
    'Expansive Coupling':             'D',
    'Expansive Negative Decoupling':  '^',
    'Strong Negative Decoupling':     'v',
    'Weak Negative Decoupling':       'P',
    'Recessive Coupling':             'X',
    'Recessive Decoupling':           '*',
}

region_cmaps = {
    'Africa':   'Oranges',
    'Americas': 'Blues',
    'Asia':     'Reds',
    'Europe':   'Greens',
    'Oceania':  'Purples',
}

regions = sorted(agg['Region'].unique())
n_regions = len(regions)
region_cmap_objs = {r: plt.colormaps.get_cmap(region_cmaps.get(r, 'Greys'))
                    for r in regions}

# 区域固定颜色（中等深度，用于标签 / 连线）
region_base_color = {r: region_cmap_objs[r](0.6) for r in regions}

# ── 3. 画图 ──
fig, ax = plt.subplots(figsize=(14, 10))

# 画散点 + 年份标签
for _, row in agg.iterrows():
    region = row['Region']
    state  = row['Decoupling_State']
    year   = row['Years']
    x, y   = row['pct_GDP'], row['pct_CE']

    t = year_to_idx[year] / max(n_years - 1, 1)
    color = region_cmap_objs[region](0.3 + 0.55 * t)
    marker = state_markers.get(state, 'o')

    ax.scatter(x, y, c=[color], marker=marker, s=160, edgecolors='white',
               linewidths=0.6, zorder=5, alpha=0.9)

    # 年份标签（只显示末年）
    end_year = year.split(' - ')[-1]
    ax.annotate(end_year, (x, y), textcoords='offset points',
                xytext=(7, -4), fontsize=6.5, color=color,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.6),
                zorder=6)

# ── 全球趋势线 ──
df_global = df_global.sort_values('Years')
gx = df_global['pct_GDP'].values
gy = df_global['pct_CE'].values
g_years = df_global['Years'].values

for i in range(len(gx) - 1):
    shade = 0.3 + 0.5 * (i / max(len(gx) - 2, 1))
    ax.annotate('', xy=(gx[i+1], gy[i+1]), xytext=(gx[i], gy[i]),
                arrowprops=dict(arrowstyle='->', color=str(1 - shade),
                                lw=2.2, connectionstyle='arc3,rad=0.1'),
                zorder=7)

for i in range(len(gx)):
    shade = 0.3 + 0.5 * (i / max(len(gx) - 1, 1))
    ax.scatter(gx[i], gy[i], c=str(1 - shade), s=80,
               edgecolors='white', linewidths=0.8, zorder=8, marker='o')
    label = g_years[i].split(' - ')[-1]
    offset = (8, 8) if i % 2 == 0 else (-8, -12)
    ax.annotate(label, (gx[i], gy[i]),
                textcoords='offset points', xytext=offset,
                fontsize=7.5, color=str(1 - shade), fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7),
                zorder=9)

# ── 坐标轴 & 分区背景 ──
ax.axhline(0, color='black', linewidth=0.8, zorder=1)
ax.axvline(0, color='black', linewidth=0.8, zorder=1)

# 四象限标签
pad = 0.02
xlim = ax.get_xlim()
ylim = ax.get_ylim()
quad_labels = {
    'GDP↑ CO₂↓\n(Strong Decoupling)':      (xlim[1] - pad, ylim[0] + pad, 'right', 'bottom'),
    'GDP↑ CO₂↑':                             (xlim[1] - pad, ylim[1] - pad, 'right', 'top'),
    'GDP↓ CO₂↑\n(Strong Neg. Decoupling)':  (xlim[0] + pad, ylim[1] - pad, 'left',  'top'),
    'GDP↓ CO₂↓':                             (xlim[0] + pad, ylim[0] + pad, 'left',  'bottom'),
}
for txt, (qx, qy, ha, va) in quad_labels.items():
    ax.text(qx, qy, txt, fontsize=8, color='#888888', alpha=0.5,
            ha=ha, va=va, fontstyle='italic', zorder=1)

ax.set_xlabel('ΔGDP (Relative Change)', fontsize=13)
ax.set_ylabel('ΔCO₂ (Relative Change)', fontsize=13)
ax.set_title('GDP–CE Decoupling — by Region (Aggregated)', fontsize=15, fontweight='bold')
ax.grid(True, alpha=0.15)

# ── 图例1：脱钩状态 ──
present_states = set(agg['Decoupling_State'].unique())
legend_markers = []
for state, marker in state_markers.items():
    if state in present_states:
        legend_markers.append(
            Line2D([0], [0], marker=marker, color='grey', linestyle='None',
                   markersize=8, label=state))
leg1 = ax.legend(handles=legend_markers, loc='upper right', title='Decoupling Type',
                 fontsize=8, title_fontsize=9, framealpha=0.9)
ax.add_artist(leg1)

# ── 图例：全球趋势 ──
gl = Line2D([0], [0], color='grey', marker='o', markersize=5, linewidth=1.5,
            label='Global Trend →', markerfacecolor='grey', markeredgecolor='white')
leg_g = ax.legend(handles=[gl], loc='lower right', fontsize=9, framealpha=0.9)
ax.add_artist(leg_g)
ax.add_artist(leg1)

# ── 图例2：区域颜色色带 ──
cbar_width = 0.14
cbar_height = 0.06 * n_regions
ax_inset = fig.add_axes([0.02, 0.93 - cbar_height, cbar_width, cbar_height])
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
ax_inset.set_yticklabels(list(reversed(regions)), fontsize=9)
ax_inset.set_xticks([])
ax_inset.set_title(f'light → dark ({years_sorted[0][:4]} → {years_sorted[-1][-4:]})',
                    fontsize=8, pad=3)
ax_inset.tick_params(axis='y', length=0, pad=2)
for spine in ax_inset.spines.values():
    spine.set_visible(False)

plt.subplots_adjust(left=0.18, right=0.95, top=0.93, bottom=0.08)
plt.savefig('GDP-CE decoupling_region.png', dpi=200, bbox_inches='tight')
plt.close()

print(f"图已保存: GDP-CE decoupling_region.png")
print(f"区域级数据点共 {len(agg)} 个（vs 国家级数百个）")
for r in regions:
    n = len(agg[agg['Region'] == r])
    print(f"  {r}: {n} 个点")
