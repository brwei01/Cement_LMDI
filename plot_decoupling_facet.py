"""
方案2：分面图 (Facet)，每个区域一个子图 + 全球趋势叠加
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── 读取数据 ──
df = pd.read_csv('Data1/results_updated/GDP_CE_Decoupling_5yrs_interval.csv', encoding='utf-8-sig')
df_global = pd.read_csv('Data1/results_updated/Global_GDP_CE_Decoupling_5yrs_interval.csv', encoding='utf-8-sig')
df = df.dropna(subset=['Decoupling_State'])
df = df[df['Decoupling_State'] != 'N/A']
df = df[~df['Region'].isin(['Unknown'])]
df_global = df_global.dropna(subset=['Decoupling_State'])
df_global = df_global[df_global['Decoupling_State'] != 'N/A']
df_global = df_global.sort_values('Years')

years_sorted = sorted(df['Years'].unique())
year_to_idx = {y: i for i, y in enumerate(years_sorted)}
n_years = len(years_sorted)

state_markers = {
    'Strong Decoupling': 's', 'Weak Decoupling': 'o',
    'Expansive Coupling': 'D', 'Expansive Negative Decoupling': '^',
    'Strong Negative Decoupling': 'v', 'Weak Negative Decoupling': 'P',
    'Recessive Coupling': 'X', 'Recessive Decoupling': '*',
}

region_cmaps = {'Africa': 'Oranges', 'Americas': 'Blues', 'Asia': 'Reds',
                'Europe': 'Greens', 'Oceania': 'Purples'}
regions = sorted(df['Region'].unique())
n_regions = len(regions)
for i, r in enumerate(regions):
    if r not in region_cmaps:
        region_cmaps[r] = ['YlOrBr', 'BuGn', 'OrRd'][i % 3]
region_cmap_objs = {r: plt.colormaps.get_cmap(region_cmaps[r]) for r in regions}

# ── 分面布局 ──
n_cols = 3
n_rows = int(np.ceil(n_regions / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 5.5), sharex=False, sharey=False)
axes = axes.flatten()

# 全球趋势数据
gx = df_global['pct_GDP'].values
gy = df_global['pct_CE'].values
gyears = df_global['Years'].values

for idx, region in enumerate(regions):
    ax = axes[idx]
    rdf = df[df['Region'] == region]
    cmap = region_cmap_objs[region]

    # 画各国数据点
    for _, row in rdf.iterrows():
        state, year = row['Decoupling_State'], row['Years']
        x, y = row['pct_GDP'], row['pct_CE']
        t = year_to_idx[year] / max(n_years - 1, 1)
        color = cmap(0.3 + 0.55 * t)
        marker = state_markers.get(state, 'o')
        ax.scatter(x, y, c=[color], marker=marker, s=60, edgecolors='white',
                   linewidths=0.3, zorder=3, alpha=0.8)

    # 叠加全球趋势（灰色虚线箭头）
    for i in range(len(gx) - 1):
        shade = 0.4 + 0.4 * (i / max(len(gx) - 2, 1))
        ax.annotate('', xy=(gx[i+1], gy[i+1]), xytext=(gx[i], gy[i]),
                    arrowprops=dict(arrowstyle='->', color=str(1 - shade), lw=1.2,
                                    linestyle='--', connectionstyle='arc3,rad=0.1'),
                    zorder=2)
    for i in range(len(gx)):
        shade = 0.4 + 0.4 * (i / max(len(gx) - 1, 1))
        ax.scatter(gx[i], gy[i], c=str(1 - shade), s=20, edgecolors='white',
                   linewidths=0.4, zorder=2)

    ax.axhline(0, color='black', linewidth=0.6, zorder=1)
    ax.axvline(0, color='black', linewidth=0.6, zorder=1)
    ax.set_title(region, fontsize=14, fontweight='bold',
                 color=cmap(0.7))
    ax.set_xlabel('ΔGDP', fontsize=10)
    ax.set_ylabel('ΔCO₂', fontsize=10)
    ax.grid(True, alpha=0.12)

    # 标注国家数量
    n_countries = rdf['Countries'].nunique()
    ax.text(0.02, 0.98, f'{n_countries} countries', transform=ax.transAxes,
            fontsize=8, va='top', color='grey')

# 隐藏多余子图
for idx in range(n_regions, len(axes)):
    axes[idx].set_visible(False)

# ── 公共图例 ──
legend_markers = [Line2D([0], [0], marker=m, color='grey', linestyle='None', markersize=7, label=s)
                  for s, m in state_markers.items()]
global_line = Line2D([0], [0], color='grey', linestyle='--', marker='o', markersize=4,
                     linewidth=1.2, label='Global Trend')
legend_markers.append(global_line)

fig.legend(handles=legend_markers, loc='lower center', ncol=5,
           fontsize=9, framealpha=0.9, title='Decoupling Type',
           title_fontsize=10, bbox_to_anchor=(0.5, -0.02))

# 色带（放在右上角空白区域）
if n_regions < n_rows * n_cols:
    ax_cb = axes[n_regions]
    ax_cb.set_visible(True)
    ax_cb.set_xlim(0, n_years); ax_cb.set_ylim(0, n_regions)
    for ri, reg in enumerate(regions):
        cm = region_cmap_objs[reg]
        for yi in range(n_years):
            t = yi / max(n_years - 1, 1)
            ax_cb.add_patch(plt.Rectangle((yi, n_regions - 1 - ri), 1, 1,
                            facecolor=cm(0.3 + 0.55 * t), edgecolor='white', linewidth=0.3))
    ax_cb.set_yticks(np.arange(n_regions) + 0.5)
    ax_cb.set_yticklabels(list(reversed(regions)), fontsize=10)
    ax_cb.set_xticks([])
    ax_cb.set_title(f'light → dark\n({years_sorted[0][:4]} → {years_sorted[-1][-4:]})', fontsize=9, pad=5)
    ax_cb.tick_params(axis='y', length=0, pad=3)
    for sp in ax_cb.spines.values(): sp.set_visible(False)

fig.suptitle('Tapio Decoupling — by Region (Faceted)', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('GDP-CE decoupling_facet.png', dpi=200, bbox_inches='tight')
plt.close()
print("图已保存: GDP-CE decoupling_facet.png")
