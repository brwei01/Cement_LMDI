"""
方案3：对称对数坐标 (symlog)，拉开密集区间距，保留零附近细节
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

fig, ax = plt.subplots(figsize=(14, 10))

for _, row in df.iterrows():
    region, state, year = row['Region'], row['Decoupling_State'], row['Years']
    x, y = row['pct_GDP'], row['pct_CE']
    t = year_to_idx[year] / max(n_years - 1, 1)
    color = region_cmap_objs[region](0.3 + 0.55 * t)
    marker = state_markers.get(state, 'o')
    ax.scatter(x, y, c=[color], marker=marker, s=80, edgecolors='white',
               linewidths=0.3, zorder=3, alpha=0.75)

# ── 全球趋势 ──
df_global = df_global.sort_values('Years')
gx, gy, gyears = df_global['pct_GDP'].values, df_global['pct_CE'].values, df_global['Years'].values
for i in range(len(gx) - 1):
    shade = 0.3 + 0.5 * (i / max(len(gx) - 2, 1))
    ax.annotate('', xy=(gx[i+1], gy[i+1]), xytext=(gx[i], gy[i]),
                arrowprops=dict(arrowstyle='->', color=str(1 - shade), lw=1.8,
                                connectionstyle='arc3,rad=0.1'), zorder=4)
for i in range(len(gx)):
    shade = 0.3 + 0.5 * (i / max(len(gx) - 1, 1))
    ax.scatter(gx[i], gy[i], c=str(1 - shade), s=40, edgecolors='white', linewidths=0.6, zorder=5)
    label = gyears[i].split(' - ')[-1]
    offset = (8, 8) if i % 2 == 0 else (-8, -12)
    ax.annotate(label, (gx[i], gy[i]), textcoords='offset points', xytext=offset,
                fontsize=7, color=str(1 - shade), fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7), zorder=6)

# ── 对称对数坐标 ──
# linthresh: 线性区间阈值，在 ±0.1 内保持线性，外部用 log 拉伸
ax.set_xscale('symlog', linthresh=0.1)
ax.set_yscale('symlog', linthresh=0.1)

ax.axhline(0, color='black', linewidth=0.8, zorder=1)
ax.axvline(0, color='black', linewidth=0.8, zorder=1)
ax.set_xlabel('ΔGDP (Relative Change, symlog scale)', fontsize=13)
ax.set_ylabel('ΔCO₂ (Relative Change, symlog scale)', fontsize=13)
ax.set_title('Tapio Decoupling — Symmetric Log Scale', fontsize=15, fontweight='bold')
ax.grid(True, alpha=0.15, which='both')

# 图例
legend_markers = [Line2D([0], [0], marker=m, color='grey', linestyle='None', markersize=8, label=s)
                  for s, m in state_markers.items()]
leg1 = ax.legend(handles=legend_markers, loc='upper right', title='Decoupling Type',
                 fontsize=8, title_fontsize=9, framealpha=0.9)
ax.add_artist(leg1)

global_legend = Line2D([0], [0], color='grey', marker='o', markersize=5,
                       linewidth=1.5, label='Global Trend →',
                       markerfacecolor='grey', markeredgecolor='white')
leg_g = ax.legend(handles=[global_legend], loc='lower right', fontsize=9, framealpha=0.9)
ax.add_artist(leg_g)
ax.add_artist(leg1)

# 色带
cbar_w, cbar_h = 0.14, 0.06 * n_regions
ax_in = fig.add_axes([0.02, 0.93 - cbar_h, cbar_w, cbar_h])
ax_in.set_xlim(0, n_years); ax_in.set_ylim(0, n_regions)
for ri, reg in enumerate(regions):
    cm = region_cmap_objs[reg]
    for yi in range(n_years):
        t = yi / max(n_years - 1, 1)
        ax_in.add_patch(plt.Rectangle((yi, n_regions - 1 - ri), 1, 1,
                        facecolor=cm(0.3 + 0.55 * t), edgecolor='white', linewidth=0.3))
ax_in.set_yticks(np.arange(n_regions) + 0.5)
ax_in.set_yticklabels(list(reversed(regions)), fontsize=9)
ax_in.set_xticks([])
ax_in.set_title(f'light → dark ({years_sorted[0][:4]} → {years_sorted[-1][-4:]})', fontsize=8, pad=3)
ax_in.tick_params(axis='y', length=0, pad=2)
for sp in ax_in.spines.values(): sp.set_visible(False)

plt.subplots_adjust(left=0.18, right=0.95, top=0.93, bottom=0.08)
plt.savefig('GDP-CE decoupling_symlog.png', dpi=200, bbox_inches='tight')
plt.close()
print("图已保存: GDP-CE decoupling_symlog.png")
