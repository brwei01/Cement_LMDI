"""
Processor_CFP_APC_fixed.py
修复版 LMDI 分解处理器

主要修改：
1. 不再用 SMALL_VALUE 填充 NaN，改为保留 NaN 并在后续计算中自然跳过
2. 新增 data_availability_report() 输出每个国家的有效数据起始年
3. LMDI 函数自动跳过含缺失数据的区间，并记录跳过原因
4. 零值（如阿联酋 cement_CS=0）同样标记为无效，不再被 >0 过滤器静默删除
"""

import pandas as pd
import numpy as np
import os

# ============================================================
# 1. 数据加载函数（与原版相同）
# ============================================================
def load_and_melt(file_path, value_name):
    # 尝试 utf-8，失败则用 latin-1
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        print(f"  ⚠️ {file_path}: utf-8 解码失败，改用 latin-1")
        df = pd.read_csv(file_path, encoding='latin-1')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    melted = df.melt(
        id_vars='Countries',
        var_name='Year',
        value_name=value_name
    )
    melted['Year_numeric'] = pd.to_numeric(melted['Year'], errors='coerce')
    invalid_mask = melted['Year_numeric'].isna()
    if invalid_mask.any():
        print(f"  ⚠️ {file_path}: 发现 {invalid_mask.sum()} 个无效年份值")
    melted = melted.dropna(subset=['Year_numeric'])
    melted['Year'] = melted['Year_numeric'].astype(int)
    melted = melted.drop(columns=['Year_numeric'])
    return melted


# ============================================================
# 2. 文件配置
# ============================================================
file_config = {
    'Data1/processed/CO2EmissionsFromEnergy.csv': 'total_CE',
    'Data1/processed/CarbonSequestration.csv': 'cement_CS',
    'Data1/processed/PrimaryEnergyConsumption.csv': 'energy_consumption',
    'Data1/processed/Population.csv': 'Population',
    'Data1/processed/GDP.csv': 'GDP',
    'Data1/processed/CementProduction.csv': 'cement_production',
    'Data1/processed/SurfaceArea.csv': 'built_surface',
}

numeric_cols = list(file_config.values())


# ============================================================
# 3. 合并数据（与原版相同）
# ============================================================
def merge_all_data(file_config):
    merged_df = None
    for file, col_name in file_config.items():
        print(f"处理文件: {file}")
        temp_df = load_and_melt(file, col_name)
        if merged_df is None:
            merged_df = temp_df
        else:
            before = set(merged_df['Countries'].unique())
            merged_df = pd.merge(merged_df, temp_df, on=['Countries', 'Year'], how='inner')
            lost = before - set(merged_df['Countries'].unique())
            if lost:
                print(f"  丢失国家: {list(lost)}")

    merged_df = merged_df.sort_values(['Countries', 'Year']).reset_index(drop=True)

    # 添加 region
    try:
        iso = pd.read_csv('Data1/processed/iso.csv', index_col=0, encoding='utf-8').reset_index()
    except UnicodeDecodeError:
        iso = pd.read_csv('Data1/processed/iso.csv', index_col=0, encoding='latin-1').reset_index()
    region_map = iso.dropna(subset=['region']).drop_duplicates('name')[['name', 'region']]
    merged_df = pd.merge(merged_df, region_map,
                         left_on='Countries', right_on='name', how='left').drop(columns='name')
    merged_df['region'] = merged_df['region'].fillna('Unknown')

    print(f"\n✅ 合并完成: {merged_df.shape}, {merged_df['Countries'].nunique()} 个国家")
    return merged_df


# ============================================================
# 4. 数据清洗（🔧 核心修改：不再填充 SMALL_VALUE）
# ============================================================
def clean_data(merged_df, start_year=1975):
    """
    清洗数据：
    - 筛选 >= start_year
    - 去重
    - 将数值列转为 float（无效值保留为 NaN，不填充）
    - 将零值替换为 NaN（零值在 LMDI 中同样无法取 log）
    - 记录每个国家的有效数据起始年
    """
    # 筛选年份
    df = merged_df[merged_df['Year'] >= start_year].copy()
    df = df.drop_duplicates(subset=['Countries', 'Year'], keep='first')

    # 转换数值列（保留 NaN，不填充！）
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            print(f"  ⚠️ 列 '{col}' 有 {nan_count} 个 NaN（保留，不填充）")

    # 🔧 关键修改：将零值也标记为 NaN
    #    零值在 LMDI 中无法取 log，与其让 >0 过滤器静默删除整行，
    #    不如标记为 NaN，这样只影响涉及该指标的区间
    for col in numeric_cols:
        zero_mask = df[col] == 0
        zero_count = zero_mask.sum()
        if zero_count > 0:
            print(f"  ⚠️ 列 '{col}' 有 {zero_count} 个零值 → 标记为 NaN")
            df.loc[zero_mask, col] = np.nan

    # 单位转换
    print("\n执行单位转换...")
    df['cement_production'] = df['cement_production'] / 1000  # kt → Mt
    df['GDP'] = df['GDP'] / 1e12                              # USD → trillion
    df['Population'] = df['Population'] / 1e8                 # → 亿人

    print(f"清洗后: {len(df)} 行, {df['Countries'].nunique()} 个国家")
    return df


# ============================================================
# 5. 计算因子（🔧 修改：不再预先过滤 >0 行）
# ============================================================
# APC 因子列表
factors = [
    'total_CE/energy_consumption',
    'energy_consumption/GDP',
    'GDP/Population',
    'built_surface/Population',
    'cement_CS/built_surface',
]

def compute_factors(df):
    """
    计算各分解因子。
    如果分母为 NaN 或 0，结果自然为 NaN，不会产生 SMALL_VALUE 伪影。
    """
    df = df.copy()

    df['total_CE/energy_consumption'] = df['total_CE'] / df['energy_consumption']
    df['energy_consumption/GDP'] = df['energy_consumption'] / df['GDP']
    df['GDP/Population'] = df['GDP'] / df['Population']
    df['built_surface/Population'] = df['built_surface'] / df['Population']
    df['cement_CS/built_surface'] = df['cement_CS'] / df['built_surface']
    df['CFP'] = df['total_CE'] / df['cement_CS']

    # 将 inf 替换为 NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    print(f"因子计算完成: {len(df)} 行")
    return df


# ============================================================
# 6. 数据可用性报告
# ============================================================
def data_availability_report(df):
    """
    输出每个国家各指标的有效数据起始年，
    帮助判断哪些国家的 LMDI 基准年需要推迟。
    """
    report = []
    all_cols = numeric_cols + factors + ['CFP']

    for country in sorted(df['Countries'].unique()):
        cdf = df[df['Countries'] == country].sort_values('Year')
        row = {'Countries': country, 'region': cdf['region'].iloc[0]}

        # 找到第一个所有关键列都有效的年份
        valid_mask = cdf[all_cols].notna().all(axis=1) & (cdf[all_cols] > 0).all(axis=1)
        valid_years = cdf.loc[valid_mask, 'Year']
        row['first_valid_year'] = valid_years.iloc[0] if len(valid_years) > 0 else None
        row['last_valid_year'] = valid_years.iloc[-1] if len(valid_years) > 0 else None
        row['n_valid_years'] = len(valid_years)
        row['total_years'] = len(cdf)

        # 标记有问题的指标
        problem_cols = []
        for col in numeric_cols:
            nan_years = cdf[cdf[col].isna()]['Year'].tolist()
            if nan_years:
                problem_cols.append(f"{col}: NaN@{nan_years}")
        row['issues'] = '; '.join(problem_cols) if problem_cols else ''

        report.append(row)

    report_df = pd.DataFrame(report)
    
    # 输出摘要
    problem_countries = report_df[report_df['issues'] != '']
    print(f"\n{'='*70}")
    print(f"数据可用性报告: {len(report_df)} 个国家")
    print(f"有数据问题的国家: {len(problem_countries)} 个")
    print(f"{'='*70}")

    if len(problem_countries) > 0:
        for _, r in problem_countries.iterrows():
            print(f"\n  🔍 {r['Countries']} ({r['region']})")
            print(f"     有效起始年: {r['first_valid_year']}, 有效数据点: {r['n_valid_years']}/{r['total_years']}")
            print(f"     问题: {r['issues']}")

    return report_df


# ============================================================
# 7. LMDI 分解（🔧 核心修改：自动跳过无效区间）
# ============================================================
def lmdi_decomposition(df, base_year, target_ratio='total_CE/cement_CS'):
    """
    执行 LMDI 分解。
    🔧 修改：不再依赖预过滤，而是在每个区间内检查数据有效性。
    如果某个区间的任何因子含 NaN，跳过该区间并记录。
    """
    df = df.copy()
    df[target_ratio] = df['total_CE'] / df['cement_CS']

    results = []
    skipped = []

    for country, country_df in df.groupby('Countries'):
        country_df = country_df.sort_values('Year')
        country_df = country_df[country_df['Year'] >= base_year]
        if len(country_df) < 2:
            continue

        region = country_df['region'].dropna().iloc[0] if not country_df['region'].isnull().all() else 'Unknown'

        for idx in range(1, len(country_df)):
            t_row = country_df.iloc[idx]
            b_row = country_df.iloc[idx - 1]

            cfp_t = t_row[target_ratio]
            cfp_b = b_row[target_ratio]

            # 🔧 检查 CFP 有效性
            if pd.isna(cfp_t) or pd.isna(cfp_b) or cfp_t <= 0 or cfp_b <= 0:
                skipped.append({
                    'Countries': country, 'Period': f"{int(b_row['Year'])}-{int(t_row['Year'])}",
                    'Reason': f"CFP无效 (base={cfp_b}, target={cfp_t})"
                })
                continue

            # 计算 L(CFP^t, CFP^b)
            if np.isclose(cfp_t, cfp_b, rtol=1e-6):
                L = cfp_t
            else:
                log_diff = np.log(cfp_t) - np.log(cfp_b)
                L = (cfp_t - cfp_b) / log_diff

            # 计算各因子贡献
            contributions = {}
            interval_valid = True
            for factor in factors:
                ratio_t = t_row[factor]
                ratio_b = b_row[factor]

                # 🔧 检查因子有效性
                if pd.isna(ratio_t) or pd.isna(ratio_b) or ratio_t <= 0 or ratio_b <= 0:
                    skipped.append({
                        'Countries': country,
                        'Period': f"{int(b_row['Year'])}-{int(t_row['Year'])}",
                        'Reason': f"因子 {factor} 无效 (base={ratio_b}, target={ratio_t})"
                    })
                    interval_valid = False
                    break

                if factor in ['built_surface/Population', 'cement_CS/built_surface']:
                    contributions[f'Δ{factor}'] = -L * np.log(ratio_t / ratio_b)
                else:
                    contributions[f'Δ{factor}'] = L * np.log(ratio_t / ratio_b)

            if not interval_valid:
                continue

            results.append({
                'Countries': country,
                'Region': region,
                'Start_Year': int(b_row['Year']),
                'End_Year': int(t_row['Year']),
                'ΔTotal': cfp_t - cfp_b,
                **contributions
            })

    results_df = pd.DataFrame(results)
    skipped_df = pd.DataFrame(skipped)

    # 输出跳过摘要
    if len(skipped_df) > 0:
        print(f"\n⚠️ 共跳过 {len(skipped_df)} 个无效区间:")
        skip_summary = skipped_df.groupby('Countries').size().sort_values(ascending=False)
        for country, count in skip_summary.items():
            reasons = skipped_df[skipped_df['Countries'] == country]
            periods = reasons['Period'].tolist()
            print(f"   {country}: 跳过 {count} 个区间 ({', '.join(periods)})")

    print(f"\n✅ LMDI 分解完成: {len(results_df)} 条有效结果")
    return results_df, skipped_df


# ============================================================
# 8. 单期 LMDI 分解（首尾年对比）
# ============================================================
def lmdi_decomposition_single_period(df, base_year, target_ratio='total_CE/cement_CS'):
    """
    每个国家只计算首尾年对比。
    🔧 修改：自动使用每个国家第一个所有数据都有效的年份作为基期。
    """
    df = df.copy()
    df[target_ratio] = df['total_CE'] / df['cement_CS']

    results = []
    all_check_cols = factors + [target_ratio]

    for country, country_df in df.groupby('Countries'):
        country_df = country_df.sort_values('Year')
        country_df = country_df[country_df['Year'] >= base_year]

        # 🔧 找到第一个所有数据都有效的年份
        valid_mask = country_df[all_check_cols].notna().all(axis=1) & \
                     (country_df[all_check_cols] > 0).all(axis=1)
        valid_df = country_df[valid_mask]

        if len(valid_df) < 2:
            continue

        region = country_df['region'].dropna().iloc[0] if not country_df['region'].isnull().all() else 'Unknown'

        first_row = valid_df.iloc[0]
        last_row = valid_df.iloc[-1]

        cfp_first = first_row[target_ratio]
        cfp_last = last_row[target_ratio]

        if cfp_first <= 0 or cfp_last <= 0:
            continue

        if np.isclose(cfp_last, cfp_first, rtol=1e-6):
            L = cfp_last
        else:
            log_diff = np.log(cfp_last) - np.log(cfp_first)
            L = (cfp_last - cfp_first) / log_diff if not np.isclose(log_diff, 0) else cfp_last

        contributions = {}
        skip = False
        for factor in factors:
            ratio_first = first_row[factor]
            ratio_last = last_row[factor]
            if pd.isna(ratio_first) or pd.isna(ratio_last) or ratio_first <= 0 or ratio_last <= 0:
                skip = True
                break
            if factor in ['built_surface/Population', 'cement_CS/built_surface']:
                contributions[f'Δ{factor}'] = -L * np.log(ratio_last / ratio_first)
            else:
                contributions[f'Δ{factor}'] = L * np.log(ratio_last / ratio_first)

        if skip:
            continue

        results.append({
            'Countries': country,
            'Region': region,
            'Start_Year': int(first_row['Year']),
            'End_Year': int(last_row['Year']),
            'Period_Length': int(last_row['Year'] - first_row['Year']),
            'ΔTotal': cfp_last - cfp_first,
            **contributions
        })

    return pd.DataFrame(results)


# ============================================================
# 9. 梯度计算与 Tapio 弹性（与原版逻辑相同）
# ============================================================
def calc_grad(attr, df, group_col='Countries'):
    res = []
    for group, group_df in df.groupby(group_col):
        group_df = group_df.sort_values('Year')
        if len(group_df) < 2:
            continue
        for idx in range(1, len(group_df)):
            current_val = group_df.iloc[idx][attr]
            previous_val = group_df.iloc[idx - 1][attr]
            if pd.isna(current_val) or pd.isna(previous_val):
                grad = np.nan
            elif previous_val == 0:
                grad = np.inf if current_val > 0 else (-np.inf if current_val < 0 else 0)
            else:
                grad = (current_val - previous_val) / previous_val
            res.append({
                group_col: group,
                'Years': f"{group_df.iloc[idx-1]['Year']} - {group_df.iloc[idx]['Year']}",
                f'grad({attr})': grad
            })
    return pd.DataFrame(res)


# ============================================================
# 10. Tapio 脱钩弹性（C5/CS, 5年间隔）
# ============================================================
def calc_tapio_5yrs(df):
    """计算各国 cement_CS/built_surface 与 cement_CS 的 5 年间隔 Tapio 弹性"""
    C5_grad = calc_grad('cement_CS/built_surface', df, group_col='Countries')
    CS_grad = calc_grad('cement_CS', df, group_col='Countries')
    merged = pd.merge(C5_grad, CS_grad, on=['Countries', 'Years'], how='inner')
    merged = merged[['Countries', 'Years', 'grad(cement_CS/built_surface)', 'grad(cement_CS)']]
    merged['CS_tapio_elasticity'] = merged['grad(cement_CS/built_surface)'] / merged['grad(cement_CS)']
    return merged


# ============================================================
# 11. Tapio 脱钩弹性（C5/CS, 整体时段，基于均值）
# ============================================================
def calc_avg_based_total_tapio(df, group_col='Countries',
                               carbon_col='cement_CS', activity_col='built_surface'):
    """
    基于均值的总 Tapio 弹性系数。
    group_col='Countries' → 国家级, group_col='region' → 区域级
    """
    results = []
    for group, gdf in df.groupby(group_col):
        gdf = gdf.sort_values('Year')
        if len(gdf) < 2:
            continue

        F5_vals = gdf[carbon_col] / gdf[activity_col]
        avg_F5 = np.nanmean(F5_vals)
        avg_CS = np.nanmean(gdf[carbon_col])

        start_row, end_row = gdf.iloc[0], gdf.iloc[-1]
        F5_start = start_row[carbon_col] / start_row[activity_col] if pd.notna(start_row[activity_col]) and start_row[activity_col] != 0 else np.nan
        F5_end = end_row[carbon_col] / end_row[activity_col] if pd.notna(end_row[activity_col]) and end_row[activity_col] != 0 else np.nan
        CS_start = start_row[carbon_col]
        CS_end = end_row[carbon_col]

        delta_F5 = F5_end - F5_start if pd.notna(F5_end) and pd.notna(F5_start) else np.nan
        delta_CS = CS_end - CS_start if pd.notna(CS_end) and pd.notna(CS_start) else np.nan

        pct_delta_F5 = delta_F5 / avg_F5 if pd.notna(delta_F5) and avg_F5 != 0 else np.nan
        pct_delta_CS = delta_CS / avg_CS if pd.notna(delta_CS) and avg_CS != 0 else np.nan

        if pd.notna(pct_delta_F5) and pd.notna(pct_delta_CS) and pct_delta_CS != 0:
            tapio_elasticity = pct_delta_F5 / pct_delta_CS
        else:
            tapio_elasticity = np.nan

        results.append({
            group_col: group,
            'Start_Year': int(start_row['Year']),
            'End_Year': int(end_row['Year']),
            'CS_End': CS_end,
            'Delta_F5': delta_F5,
            'Delta_CS': delta_CS,
            'Pct_Delta_F5': pct_delta_F5,
            'Pct_Delta_CS': pct_delta_CS,
            'Total_Tapio_Elasticity': tapio_elasticity,
        })
    return pd.DataFrame(results)


# ============================================================
# 12. GDP-CE 脱钩弹性（5年间隔）
# ============================================================
def _classify_decoupling(elasticity, pct_driver):
    """根据弹性值和驱动力变化率判断脱钩状态"""
    if pd.isna(elasticity):
        return 'N/A'
    if pct_driver > 0:  # GDP 增长
        if elasticity < 0:
            return 'Strong Decoupling'
        elif elasticity < 0.8:
            return 'Weak Decoupling'
        elif elasticity <= 1.2:
            return 'Expansive Coupling'
        else:
            return 'Expansive Negative Decoupling'
    else:  # GDP 下降
        if elasticity < 0:
            return 'Strong Negative Decoupling'
        elif elasticity < 0.8:
            return 'Weak Negative Decoupling'
        elif elasticity <= 1.2:
            return 'Recessive Coupling'
        else:
            return 'Recessive Decoupling'


def calc_gdp_decoupling_5yrs(df):
    """各国 5 年间隔 GDP-CE Tapio 脱钩弹性"""
    results = []
    for country, cdf in df.groupby('Countries'):
        cdf = cdf.sort_values('Year')
        region = cdf['region'].dropna().iloc[0] if not cdf['region'].isnull().all() else 'Unknown'
        if len(cdf) < 2:
            continue
        for idx in range(1, len(cdf)):
            curr, prev = cdf.iloc[idx], cdf.iloc[idx - 1]
            CE_curr, CE_prev = curr['total_CE'], prev['total_CE']
            GDP_curr, GDP_prev = curr['GDP'], prev['GDP']

            if pd.isna(CE_prev) or pd.isna(CE_curr) or pd.isna(GDP_prev) or pd.isna(GDP_curr) \
                    or CE_prev == 0 or GDP_prev == 0:
                pct_CE = pct_GDP = elasticity = np.nan
            else:
                pct_CE = (CE_curr - CE_prev) / CE_prev
                pct_GDP = (GDP_curr - GDP_prev) / GDP_prev
                elasticity = pct_CE / pct_GDP if pct_GDP != 0 else np.nan

            results.append({
                'Countries': country, 'Region': region,
                'Years': f"{int(prev['Year'])} - {int(curr['Year'])}",
                'CE_Start': CE_prev, 'CE_End': CE_curr,
                'GDP_Start': GDP_prev, 'GDP_End': GDP_curr,
                'pct_CE': pct_CE, 'pct_GDP': pct_GDP,
                'GDP_Decoupling_Elasticity': elasticity,
                'Decoupling_State': _classify_decoupling(elasticity, pct_GDP if not pd.isna(pct_GDP) else 0),
            })
    return pd.DataFrame(results)


# ============================================================
# 13. GDP-CE 脱钩弹性（整体时段，基于均值）
# ============================================================
def calc_avg_based_gdp_decoupling_total(df, group_col='Countries',
                                         carbon_col='total_CE', gdp_col='GDP'):
    """
    基于均值的整体 GDP-CE 脱钩弹性。
    group_col='Countries' → 国家级, group_col='region' → 区域级
    """
    results = []
    for group, gdf in df.groupby(group_col):
        gdf = gdf.sort_values('Year')
        region = gdf['region'].dropna().iloc[0] if 'region' in gdf.columns and not gdf['region'].isnull().all() else 'Unknown'
        if len(gdf) < 2:
            continue

        avg_CE = np.nanmean(gdf[carbon_col])
        avg_GDP = np.nanmean(gdf[gdp_col])
        start_row, end_row = gdf.iloc[0], gdf.iloc[-1]

        CE_start, CE_end = start_row[carbon_col], end_row[carbon_col]
        GDP_start, GDP_end = start_row[gdp_col], end_row[gdp_col]

        delta_CE = CE_end - CE_start if pd.notna(CE_end) and pd.notna(CE_start) else np.nan
        delta_GDP = GDP_end - GDP_start if pd.notna(GDP_end) and pd.notna(GDP_start) else np.nan

        pct_delta_CE = delta_CE / avg_CE if pd.notna(delta_CE) and avg_CE != 0 else np.nan
        pct_delta_GDP = delta_GDP / avg_GDP if pd.notna(delta_GDP) and avg_GDP != 0 else np.nan

        if pd.notna(pct_delta_CE) and pd.notna(pct_delta_GDP) and pct_delta_GDP != 0:
            elasticity = pct_delta_CE / pct_delta_GDP
        else:
            elasticity = np.nan

        row = {
            group_col: group,
            'Start_Year': int(start_row['Year']),
            'End_Year': int(end_row['Year']),
            'CE_End': CE_end, 'GDP_End': GDP_end,
            'Delta_CE': delta_CE, 'Delta_GDP': delta_GDP,
            'Pct_Delta_CE': pct_delta_CE, 'Pct_Delta_GDP': pct_delta_GDP,
            'GDP_Decoupling_Elasticity': elasticity,
            'Decoupling_State': _classify_decoupling(elasticity, delta_GDP if pd.notna(delta_GDP) else 0),
        }
        if group_col == 'Countries':
            row['Region'] = region
        results.append(row)
    return pd.DataFrame(results)


# ============================================================
# 14. 全球级 GDP-CE 脱钩（5年间隔 / 整体时段）
# ============================================================
def calc_region_gdp_decoupling_5yrs(df, carbon_col='total_CE', gdp_col='GDP'):
    """
    区域级 5 年间隔 GDP-CE 脱钩弹性。
    输入: 已按 region×Year 聚合的 DataFrame（含 region, Year, total_CE, GDP 列）
    """
    results = []
    for region, rdf in df.groupby('region'):
        rdf = rdf.sort_values('Year')
        if len(rdf) < 2:
            continue
        for idx in range(1, len(rdf)):
            curr, prev = rdf.iloc[idx], rdf.iloc[idx - 1]
            CE_curr, CE_prev = curr[carbon_col], prev[carbon_col]
            GDP_curr, GDP_prev = curr[gdp_col], prev[gdp_col]

            if CE_prev == 0 or GDP_prev == 0:
                pct_CE = pct_GDP = elasticity = np.nan
            else:
                pct_CE = (CE_curr - CE_prev) / CE_prev
                pct_GDP = (GDP_curr - GDP_prev) / GDP_prev
                elasticity = pct_CE / pct_GDP if pct_GDP != 0 else np.nan

            results.append({
                'region': region,
                'Years': f"{int(prev['Year'])} - {int(curr['Year'])}",
                'CE_Start': CE_prev, 'CE_End': CE_curr,
                'GDP_Start': GDP_prev, 'GDP_End': GDP_curr,
                'pct_CE': pct_CE, 'pct_GDP': pct_GDP,
                'GDP_Decoupling_Elasticity': elasticity,
                'Decoupling_State': _classify_decoupling(elasticity, pct_GDP if not pd.isna(pct_GDP) else 0),
            })
    return pd.DataFrame(results)


def calc_global_gdp_decoupling_5yrs(df, carbon_col='total_CE', gdp_col='GDP'):
    """全球 5 年间隔 GDP-CE 脱钩弹性（基于按年聚合的全球数据）"""
    df = df.sort_values('Year')
    results = []
    for idx in range(1, len(df)):
        curr, prev = df.iloc[idx], df.iloc[idx - 1]
        CE_curr, CE_prev = curr[carbon_col], prev[carbon_col]
        GDP_curr, GDP_prev = curr[gdp_col], prev[gdp_col]

        if CE_prev == 0 or GDP_prev == 0:
            pct_CE = pct_GDP = elasticity = np.nan
        else:
            pct_CE = (CE_curr - CE_prev) / CE_prev
            pct_GDP = (GDP_curr - GDP_prev) / GDP_prev
            elasticity = pct_CE / pct_GDP if pct_GDP != 0 else np.nan

        results.append({
            'Region': 'Global',
            'Years': f"{int(prev['Year'])} - {int(curr['Year'])}",
            'CE_Start': CE_prev, 'CE_End': CE_curr,
            'GDP_Start': GDP_prev, 'GDP_End': GDP_curr,
            'pct_CE': pct_CE, 'pct_GDP': pct_GDP,
            'GDP_Decoupling_Elasticity': elasticity,
            'Decoupling_State': _classify_decoupling(elasticity, pct_GDP if not pd.isna(pct_GDP) else 0),
        })
    return pd.DataFrame(results)


def calc_global_gdp_decoupling_total(df, carbon_col='total_CE', gdp_col='GDP'):
    """全球整体时段 GDP-CE 脱钩弹性（基于均值）"""
    df = df.sort_values('Year')
    avg_CE = np.mean(df[carbon_col])
    avg_GDP = np.mean(df[gdp_col])
    start_row, end_row = df.iloc[0], df.iloc[-1]

    delta_CE = end_row[carbon_col] - start_row[carbon_col]
    delta_GDP = end_row[gdp_col] - start_row[gdp_col]

    pct_delta_CE = delta_CE / avg_CE if avg_CE != 0 else np.nan
    pct_delta_GDP = delta_GDP / avg_GDP if avg_GDP != 0 else np.nan

    if pct_delta_GDP != 0 and pd.notna(pct_delta_CE) and pd.notna(pct_delta_GDP):
        elasticity = pct_delta_CE / pct_delta_GDP
    else:
        elasticity = np.nan

    return pd.DataFrame([{
        'Region': 'Global',
        'Start_Year': int(start_row['Year']),
        'End_Year': int(end_row['Year']),
        'CE_Start': start_row[carbon_col], 'CE_End': end_row[carbon_col],
        'GDP_Start': start_row[gdp_col], 'GDP_End': end_row[gdp_col],
        'Delta_CE': delta_CE, 'Delta_GDP': delta_GDP,
        'Pct_Delta_CE': pct_delta_CE, 'Pct_Delta_GDP': pct_delta_GDP,
        'GDP_Decoupling_Elasticity': elasticity,
        'Decoupling_State': _classify_decoupling(elasticity, delta_GDP),
    }])


# ============================================================
# 主流程
# ============================================================
if __name__ == '__main__':

    OUT = 'Data1/results_updated'
    os.makedirs(OUT, exist_ok=True)

    # --- 合并 ---
    merged_df = merge_all_data(file_config)

    # --- 清洗（🔧 不再填充 SMALL_VALUE）---
    merged_df = clean_data(merged_df, start_year=1975)

    # --- 计算因子（🔧 不再预过滤 >0 行）---
    merged_df = compute_factors(merged_df)

    # --- 数据可用性报告 ---
    report_df = data_availability_report(merged_df)
    report_df.to_csv(f'{OUT}/data_availability_report.csv', index=False)

    # --- 保存因子 ---
    merged_df.to_csv(f'{OUT}/factors.csv')

    # =====================================================
    # A. LMDI 分解
    # =====================================================
    print("\n" + "=" * 70)
    print("LMDI 分解（5年间隔）")
    print("=" * 70)
    lmdi_results_5, skipped_5 = lmdi_decomposition(merged_df, base_year=1975)

    # 验证
    lmdi_results_5['ΔSum'] = lmdi_results_5[[f'Δ{f}' for f in factors]].sum(axis=1)
    max_err = (lmdi_results_5['ΔTotal'] - lmdi_results_5['ΔSum']).abs().max()
    print(f"验证: ΔTotal vs ΔSum 最大误差 = {max_err:.2e}")
    lmdi_results_5 = lmdi_results_5.drop(columns='ΔSum')

    lmdi_results_5.to_csv(f'{OUT}/LMDI_Contributions_5yrs_interval.csv', index=False, encoding='utf-8-sig')
    skipped_5.to_csv(f'{OUT}/LMDI_skipped_intervals.csv', index=False, encoding='utf-8-sig')

    # --- LMDI 分解（单期：首尾年对比）---
    print("\n" + "=" * 70)
    print("LMDI 分解（首尾年对比）")
    print("=" * 70)
    lmdi_results_ttl = lmdi_decomposition_single_period(merged_df, base_year=1975)
    print(f"✅ 完成: {len(lmdi_results_ttl)} 个国家")

    late_start = lmdi_results_ttl[lmdi_results_ttl['Start_Year'] > 1975]
    if len(late_start) > 0:
        print(f"\n📌 以下国家因早期数据缺失，基期晚于 1975:")
        for _, r in late_start.iterrows():
            print(f"   {r['Countries']}: {int(r['Start_Year'])} → {int(r['End_Year'])} ({int(r['Period_Length'])}年)")

    lmdi_results_ttl.to_csv(f'{OUT}/LMDI_Contributions_total_yrs_interval.csv', index=False, encoding='utf-8-sig')

    # --- 全球 LMDI 汇总 ---
    delta_cols = [f'Δ{f}' for f in factors] + ['ΔTotal']

    global_lmdi_5yrs = lmdi_results_5.groupby(['Start_Year', 'End_Year'])[delta_cols].sum().reset_index()
    global_lmdi_5yrs.insert(0, 'Region', 'Global')
    global_lmdi_5yrs.to_csv(f'{OUT}/Global_LMDI_Contributions_5yrs_interval.csv', index=False, encoding='utf-8-sig')

    global_lmdi_total = lmdi_results_ttl[delta_cols].sum().to_frame().T
    global_lmdi_total.insert(0, 'Region', 'Global')
    global_lmdi_total.insert(1, 'Start_Year', lmdi_results_ttl['Start_Year'].min())
    global_lmdi_total.insert(2, 'End_Year', lmdi_results_ttl['End_Year'].max())
    global_lmdi_total.to_csv(f'{OUT}/Global_LMDI_Contributions_total_interval.csv', index=False, encoding='utf-8-sig')

    # =====================================================
    # B. C5/CS Tapio 脱钩弹性
    # =====================================================
    print("\n" + "=" * 70)
    print("C5/CS Tapio 脱钩弹性")
    print("=" * 70)

    # 国家级 5 年间隔
    tapio_5yrs = calc_tapio_5yrs(merged_df)
    tapio_5yrs.to_csv(f'{OUT}/C5_CS_Tapio_results_5_yrs_interval.csv', index=False, encoding='utf-8-sig')
    print(f"  国家级 5 年 Tapio: {len(tapio_5yrs)} 条")

    # 国家级整体时段
    tapio_total = calc_avg_based_total_tapio(merged_df, group_col='Countries',
                                             carbon_col='cement_CS', activity_col='built_surface')
    tapio_total.to_csv(f'{OUT}/C5_CS_Tapio_results_35yrs_interval.csv', index=False, encoding='utf-8-sig')
    print(f"  国家级总 Tapio: {len(tapio_total)} 条")

    # 区域级（从 merged_df 自行聚合，保持数据来源一致）
    region_agg_cols = ['total_CE', 'cement_CS', 'energy_consumption', 'Population',
                       'GDP', 'cement_production', 'built_surface']
    region_factors_df = merged_df.groupby(['region', 'Year'])[region_agg_cols].sum().reset_index()
    # 重新计算区域级因子（聚合后的比率）
    region_factors_df['cement_CS/built_surface'] = region_factors_df['cement_CS'] / region_factors_df['built_surface']
    region_factors_df.to_csv(f'{OUT}/Factors_by_Region_and_Year_5yrs_interval.csv', index=False, encoding='utf-8-sig')
    print(f"\n  从 merged_df 聚合区域数据: {len(region_factors_df)} 行, "
          f"{region_factors_df['region'].nunique()} 个区域")

    # 区域级 5 年间隔 Tapio
    region_C5_grad = calc_grad('cement_CS/built_surface', region_factors_df, group_col='region')
    region_CS_grad = calc_grad('cement_CS', region_factors_df, group_col='region')
    region_tapio_5yrs = pd.merge(region_C5_grad, region_CS_grad, on=['region', 'Years'], how='inner')
    region_tapio_5yrs = region_tapio_5yrs[['region', 'Years', 'grad(cement_CS/built_surface)', 'grad(cement_CS)']]
    region_tapio_5yrs['tapio_elasticity'] = region_tapio_5yrs['grad(cement_CS/built_surface)'] / region_tapio_5yrs['grad(cement_CS)']
    region_tapio_5yrs.to_csv(f'{OUT}/Region_C5_CS_Tapio_results_5_yrs_interval.csv', index=False, encoding='utf-8-sig')
    print(f"  区域级 5 年 Tapio: {len(region_tapio_5yrs)} 条")

    # 区域级整体时段 Tapio
    region_tapio_total = calc_avg_based_total_tapio(region_factors_df, group_col='region',
                                                    carbon_col='total_CE', activity_col='built_surface')
    region_tapio_total.to_csv(f'{OUT}/Region_total_C5_CS_Tapio_results_35yrs_interval.csv', index=False, encoding='utf-8-sig')
    print(f"  区域级总 Tapio: {len(region_tapio_total)} 条")

    # =====================================================
    # C. GDP-CE 脱钩弹性
    # =====================================================
    print("\n" + "=" * 70)
    print("GDP-CE 脱钩弹性")
    print("=" * 70)

    # 国家级 5 年间隔
    gdp_decoupling_5yrs = calc_gdp_decoupling_5yrs(merged_df)
    gdp_decoupling_5yrs.to_csv(f'{OUT}/GDP_CE_Decoupling_5yrs_interval.csv', index=False, encoding='utf-8-sig')
    print(f"  国家级 5 年 GDP 脱钩: {len(gdp_decoupling_5yrs)} 条")

    # 国家级整体时段
    gdp_decoupling_total = calc_avg_based_gdp_decoupling_total(merged_df, group_col='Countries')
    gdp_decoupling_total.to_csv(f'{OUT}/GDP_CE_Decoupling_total_interval.csv', index=False, encoding='utf-8-sig')
    print(f"  国家级总 GDP 脱钩: {len(gdp_decoupling_total)} 条")

    # 区域级 5 年间隔（🆕 新增）
    region_gdp_5yrs = calc_region_gdp_decoupling_5yrs(region_factors_df)
    region_gdp_5yrs.to_csv(f'{OUT}/Region_GDP_CE_Decoupling_5yrs_interval.csv', index=False, encoding='utf-8-sig')
    print(f"  区域级 5 年 GDP 脱钩: {len(region_gdp_5yrs)} 条")

    # 区域级整体时段
    region_gdp_decoupling = calc_avg_based_gdp_decoupling_total(
        region_factors_df, group_col='region', carbon_col='total_CE', gdp_col='GDP')
    region_gdp_decoupling.to_csv(f'{OUT}/Region_GDP_CE_Decoupling_total.csv', index=False, encoding='utf-8-sig')
    print(f"  区域级总 GDP 脱钩: {len(region_gdp_decoupling)} 条")

    # 全球级（使用与区域相同的 region_factors_df 再聚合到全球，保持一致）
    global_factors_df = region_factors_df.groupby('Year')[region_agg_cols].sum().reset_index()

    # 全球 5 年间隔
    global_gdp_5yrs = calc_global_gdp_decoupling_5yrs(global_factors_df)
    global_gdp_5yrs.to_csv(f'{OUT}/Global_GDP_CE_Decoupling_5yrs_interval.csv', index=False, encoding='utf-8-sig')
    print(f"  全球 5 年 GDP 脱钩: {len(global_gdp_5yrs)} 条")

    # 全球整体时段
    global_gdp_total = calc_global_gdp_decoupling_total(global_factors_df)
    global_gdp_total.to_csv(f'{OUT}/Global_GDP_CE_Decoupling_total.csv', index=False, encoding='utf-8-sig')
    print(f"  全球总 GDP 脱钩: {len(global_gdp_total)} 条")

    # 合并区域 + 全球 GDP 脱钩
    combined_gdp = pd.concat([region_gdp_decoupling, global_gdp_total], ignore_index=True)
    combined_gdp.to_csv(f'{OUT}/Combined_GDP_CE_Decoupling_total.csv', index=False, encoding='utf-8-sig')

    # =====================================================
    # 汇总
    # =====================================================
    print("\n" + "=" * 70)
    print("✅ 全部完成！输出目录:", OUT)
    print("=" * 70)
    print(f"   LMDI 5年间隔:  {len(lmdi_results_5)} 条")
    print(f"   LMDI 首尾年:   {len(lmdi_results_ttl)} 条")
    print(f"   跳过无效区间:  {len(skipped_5)} 个")
    print(f"   Tapio 5年:     {len(tapio_5yrs)} 条")
    print(f"   GDP脱钩 5年:   {len(gdp_decoupling_5yrs)} 条")
