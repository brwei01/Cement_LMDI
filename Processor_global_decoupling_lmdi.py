"""
global_decoupling_lmdi.py
==============================================================
直接在「全球单一时间序列」上计算：
  (1) GDP–CE 脱钩 (Tapio)
  (2) CFPc 的加性 LMDI 五因子分解
数据来自 Global__Raw_Factors_1975-2020.xlsx（已聚合到全球，含 CE/E/G/P/B/CS）。

CFPc = CE / CS，链式分解：
  CFPc = (CE/E)·(E/G)·(G/P)·(P/B)·(B/CS)
       =  ES  · EI · EG · PD · CL
  ES 能源碳强度, EI 能源强度, EG 经济增长(人均GDP),
  PD 人口需求(人口/建筑面积), CL 碳化负载(建筑面积/碳汇)

因为是同一条全球序列，加性 LMDI 恒等式精确成立：
  ΔCFPc = ΣΔ(ES,EI,EG,PD,CL)   —— 脚本末尾会打印校验值(≈0)
==============================================================
"""
import numpy as np
import pandas as pd

XLSX = 'Global__Raw_Factors_1975-2020.xlsx'
OUT_DIR = '.'
INTERVAL = 5            # 分解间隔（年）；改成 1 即逐年
BASE_YEAR, END_YEAR = 1975, 2020


# ---------- 读取全球序列 ----------
def load_global(path):
    raw = pd.read_excel(path, sheet_name='Sheet1')
    raw = raw.rename(columns={raw.columns[0]: 'var'})
    raw = raw[raw['var'].isin(['CE', 'E', 'G', 'P', 'B', 'CS'])].iloc[:6]  # 前6行是数据
    year_cols = [c for c in raw.columns if isinstance(c, (int, np.integer))]
    df = raw.set_index('var')[year_cols].T          # index=年份, 列=变量
    df.index = df.index.astype(int)
    df = df.astype(float).sort_index()
    df['CFPc'] = df['CE'] / df['CS']
    # 五因子
    df['ES'] = df['CE'] / df['E']
    df['EI'] = df['E'] / df['G']
    df['EG'] = df['G'] / df['P']
    df['PD'] = df['P'] / df['B']
    df['CL'] = df['B'] / df['CS']
    return df


# ---------- Tapio 脱钩分类 ----------
def tapio_state(elast, pct_gdp):
    if pd.isna(elast):
        return 'N/A'
    if pct_gdp > 0:
        if elast < 0:    return 'Strong Decoupling'
        if elast < 0.8:  return 'Weak Decoupling'
        if elast <= 1.2: return 'Expansive Coupling'
        return 'Expansive Negative Decoupling'
    else:
        if elast < 0:    return 'Strong Negative Decoupling'
        if elast < 0.8:  return 'Weak Negative Decoupling'
        if elast <= 1.2: return 'Recessive Coupling'
        return 'Recessive Decoupling'


def decoupling(df, nodes):
    rows = []
    for y0, y1 in zip(nodes[:-1], nodes[1:]):
        ce0, ce1 = df.loc[y0, 'CE'], df.loc[y1, 'CE']
        g0, g1 = df.loc[y0, 'G'], df.loc[y1, 'G']
        pct_ce = (ce1 - ce0) / ce0
        pct_gdp = (g1 - g0) / g0
        elast = pct_ce / pct_gdp if pct_gdp != 0 else np.nan
        rows.append({'Years': f'{y0}-{y1}', 'CE_start': ce0, 'CE_end': ce1,
                     'GDP_start': g0, 'GDP_end': g1,
                     'pct_CE': pct_ce, 'pct_GDP': pct_gdp,
                     'elasticity': elast, 'Decoupling_State': tapio_state(elast, pct_gdp)})
    return pd.DataFrame(rows)


# ---------- 加性 LMDI ----------
FACTORS = ['ES', 'EI', 'EG', 'PD', 'CL']

def _logmean(a, b):
    """对数平均 L(a,b)=(a-b)/(ln a-ln b)，a==b 时取 a"""
    if np.isclose(a, b):
        return a
    return (a - b) / (np.log(a) - np.log(b))

def lmdi_one(df, y0, y1):
    """单区间 y0→y1 的五因子分解"""
    cfp0, cfp1 = df.loc[y0, 'CFPc'], df.loc[y1, 'CFPc']
    L = _logmean(cfp1, cfp0)
    out = {'Years': f'{y0}-{y1}', 'CFPc_start': cfp0, 'CFPc_end': cfp1,
           'delta_CFPc': cfp1 - cfp0}
    for f in FACTORS:
        out[f'd_{f}'] = L * np.log(df.loc[y1, f] / df.loc[y0, f])
    out['sum_factors'] = sum(out[f'd_{f}'] for f in FACTORS)
    return out

def lmdi_periodwise(df, nodes):
    return pd.DataFrame([lmdi_one(df, y0, y1) for y0, y1 in zip(nodes[:-1], nodes[1:])])

def lmdi_whole(df, y0, y1):
    return pd.DataFrame([lmdi_one(df, y0, y1)])


# ---------- 主流程 ----------
def main():
    df = load_global(XLSX)
    nodes = list(range(BASE_YEAR, END_YEAR + 1, INTERVAL))
    if nodes[-1] != END_YEAR:
        nodes.append(END_YEAR)

    print('=' * 60)
    print(f'全球 CFPc: {df.loc[BASE_YEAR,"CFPc"]:.2f} ({BASE_YEAR}) '
          f'→ {df.loc[END_YEAR,"CFPc"]:.2f} ({END_YEAR})  '
          f'降幅 {(df.loc[END_YEAR,"CFPc"]/df.loc[BASE_YEAR,"CFPc"]-1)*100:.1f}%')
    print('=' * 60)

    # 1) 脱钩
    dec = decoupling(df, nodes)
    dec.to_csv(f'{OUT_DIR}/Global_Decoupling_{INTERVAL}yr.csv', index=False, encoding='utf-8-sig')
    print('\n【GDP–CE 脱钩】')
    print(dec[['Years', 'pct_CE', 'pct_GDP', 'elasticity', 'Decoupling_State']].round(4).to_string(index=False))

    # 2) LMDI 分期
    lp = lmdi_periodwise(df, nodes)
    lp.to_csv(f'{OUT_DIR}/Global_LMDI_periodwise_{INTERVAL}yr.csv', index=False, encoding='utf-8-sig')
    print('\n【LMDI 分期分解】(单位与 CFPc 同)')
    show = ['Years', 'delta_CFPc'] + [f'd_{f}' for f in FACTORS]
    print(lp[show].round(3).to_string(index=False))

    # 3) LMDI 全期 1975→2020
    lw = lmdi_whole(df, BASE_YEAR, END_YEAR)
    lw.to_csv(f'{OUT_DIR}/Global_LMDI_whole_period.csv', index=False, encoding='utf-8-sig')
    r = lw.iloc[0]
    print(f'\n【LMDI 全期 {BASE_YEAR}→{END_YEAR}】')
    print(f'  ΔCFPc = {r["delta_CFPc"]:.3f}')
    for f in FACTORS:
        print(f'  Δ{f} = {r[f"d_{f}"]:+.3f}')
    print(f'  五因子之和 = {r["sum_factors"]:.3f}')

    # 4) 自洽校验
    err_p = (lp['delta_CFPc'] - lp['sum_factors']).abs().max()
    err_w = abs(r['delta_CFPc'] - r['sum_factors'])
    chain = lp['delta_CFPc'].sum()
    print('\n【校验】')
    print(f'  分期: max|ΔCFPc − Σ因子| = {err_p:.2e}  (应≈0)')
    print(f'  全期: |ΔCFPc − Σ因子|    = {err_w:.2e}  (应≈0)')
    print(f'  分期ΔCFPc累加 = {chain:.3f}  vs  全期ΔCFPc = {r["delta_CFPc"]:.3f} '
          f'(两者都等于端点差，因子归因略有不同属正常)')

    print('\n输出: Global_Decoupling / Global_LMDI_periodwise / Global_LMDI_whole_period .csv')


if __name__ == '__main__':
    main()
