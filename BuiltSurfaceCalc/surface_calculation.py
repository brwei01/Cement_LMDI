import pandas as pd
import os
import glob
from pathlib import Path

def calculate_building_surface_area():
    """
    计算每一年的建筑表面积
    通过合并体积数据(total_vol)和占地面积数据(total_fp)来计算建筑高度
    建筑高度 = total_vol / total_fp
    """
    
    # 定义输入和输出目录
    vol_dir = "vol_results"
    fp_dir = "fp_results"
    output_dir = "surface_results"
    
    # 创建输出目录
    Path(output_dir).mkdir(exist_ok=True)
    
    # 获取所有年份的体积数据文件
    vol_files = glob.glob(os.path.join(vol_dir, "volume_statistics_*.csv"))
    
    print(f"找到 {len(vol_files)} 个体积数据文件")
    
    for vol_file in vol_files:
        # 提取年份
        year = vol_file.split("_")[-1].replace(".csv", "")
        print(f"处理年份: {year}")
        
        # 对应的占地面积文件
        fp_file = os.path.join(fp_dir, f"footprint_statistics_{year}.csv")
        
        # 检查占地面积文件是否存在
        if not os.path.exists(fp_file):
            print(f"警告: 找不到对应的占地面积文件 {fp_file}")
            continue
        
        try:
            # 读取体积数据
            vol_df = pd.read_csv(vol_file)
            print(f"  体积数据: {len(vol_df)} 条记录")
            
            # 读取占地面积数据
            fp_df = pd.read_csv(fp_file)
            print(f"  占地面积数据: {len(fp_df)} 条记录")
            
            # 按ISO_A3合并数据
            merged_df = pd.merge(
                vol_df[['ISO_A3', 'ISO_A2', 'WB_A3', 'HASC_0', 'GAUL_0', 'WB_REGION', 'WB_STATUS', 'SOVEREIGN', 'NAM_0', 'total_vol']], 
                fp_df[['ISO_A3', 'total_fp']], 
                on='ISO_A3', 
                how='inner'
            )
            
            print(f"  合并后数据: {len(merged_df)} 条记录")
            
            # 计算建筑高度 (total_vol / total_fp)
            # 避免除零错误
            merged_df['building_height'] = merged_df.apply(
                lambda row: row['total_vol'] / row['total_fp'] if row['total_fp'] > 0 else 0, 
                axis=1
            )
            
            # 计算建筑总表面积
            # 使用公式: S_total = C * V / sqrt(Af) + Af
            # 其中 C = 4 (紧凑矩形建筑)
            C = 4  # 紧凑度系数
            
            merged_df['total_surface_area'] = merged_df.apply(
                lambda row: (C * row['total_vol'] / (row['total_fp'] ** 0.5) + row['total_fp']) 
                if row['total_fp'] > 0 else 0, 
                axis=1
            )
            
            # 添加年份列
            merged_df['year'] = year
            
            # 只保留需要的字段
            columns_order = [
                'ISO_A3', 'NAM_0', 'year', 'total_vol', 'total_fp', 'building_height', 'total_surface_area'
            ]
            merged_df = merged_df[columns_order]
            
            # 按国家代码和年份排序
            merged_df = merged_df.sort_values(['ISO_A3', 'year'])
            
            # 保存结果
            output_file = os.path.join(output_dir, f"surface_statistics_{year}.csv")
            merged_df.to_csv(output_file, index=False)
            print(f"  结果已保存到: {output_file}")
            
            # 显示一些统计信息
            valid_heights = merged_df[merged_df['building_height'] > 0]
            if len(valid_heights) > 0:
                print(f"  有效建筑高度统计:")
                print(f"    平均高度: {valid_heights['building_height'].mean():.2f}")
                print(f"    最大高度: {valid_heights['building_height'].max():.2f}")
                print(f"    最小高度: {valid_heights['building_height'].min():.2f}")
            
            # 显示表面积统计信息
            valid_surfaces = merged_df[merged_df['total_surface_area'] > 0]
            if len(valid_surfaces) > 0:
                print(f"  有效建筑表面积统计:")
                print(f"    平均表面积: {valid_surfaces['total_surface_area'].mean():.2f}")
                print(f"    最大表面积: {valid_surfaces['total_surface_area'].max():.2f}")
                print(f"    最小表面积: {valid_surfaces['total_surface_area'].min():.2f}")
            
        except Exception as e:
            print(f"处理年份 {year} 时出错: {str(e)}")
            continue
    
    print("\n所有年份处理完成！")
    
    # 创建汇总文件
    create_summary_file(output_dir)
    
    # 创建按ISO_A3汇总的文件
    create_iso_summary_file(output_dir)

def create_summary_file(output_dir):
    """
    创建所有年份的汇总文件
    """
    print("\n创建汇总文件...")
    
    # 获取所有表面统计数据文件
    surface_files = glob.glob(os.path.join(output_dir, "surface_statistics_*.csv"))
    
    if not surface_files:
        print("没有找到表面统计数据文件")
        return
    
    # 读取并合并所有文件
    all_data = []
    for file in surface_files:
        df = pd.read_csv(file)
        all_data.append(df)
    
    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 按国家代码和年份排序
    combined_df = combined_df.sort_values(['ISO_A3', 'year'])
    
    # 保存汇总文件
    summary_file = os.path.join(output_dir, "surface_statistics_all_years.csv")
    combined_df.to_csv(summary_file, index=False)
    print(f"汇总文件已保存到: {summary_file}")
    print(f"总计 {len(combined_df)} 条记录，涵盖 {combined_df['year'].nunique()} 个年份")

def create_iso_summary_file(output_dir):
    """
    创建按ISO_A3汇总的文件，将相同ISO_A3和年份的条目进行加和
    """
    print("\n创建按ISO_A3汇总的文件...")
    
    # 读取汇总文件
    summary_file = os.path.join(output_dir, "surface_statistics_all_years.csv")
    if not os.path.exists(summary_file):
        print("找不到汇总文件，无法创建ISO汇总")
        return
    
    # 读取数据
    df = pd.read_csv(summary_file)
    print(f"原始数据: {len(df)} 条记录")
    
    # 按ISO_A3和年份分组，对数值字段进行加和
    grouped_df = df.groupby(['ISO_A3', 'year']).agg({
        'total_vol': 'sum',
        'total_fp': 'sum'
    }).reset_index()
    
    # 重新计算建筑高度
    grouped_df['building_height'] = grouped_df.apply(
        lambda row: row['total_vol'] / row['total_fp'] if row['total_fp'] > 0 else 0, 
        axis=1
    )
    
    # 重新计算建筑总表面积
    # 使用公式: S_total = C * V / sqrt(Af) + Af
    # 其中 C = 4 (紧凑矩形建筑)
    C = 4  # 紧凑度系数
    
    grouped_df['total_surface_area'] = grouped_df.apply(
        lambda row: (C * row['total_vol'] / (row['total_fp'] ** 0.5) + row['total_fp']) 
        if row['total_fp'] > 0 else 0, 
        axis=1
    )
    
    # 添加国家名称（取第一个出现的名称）
    country_names = df.groupby('ISO_A3')['NAM_0'].first().reset_index()
    grouped_df = pd.merge(grouped_df, country_names, on='ISO_A3', how='left')
    
    # 重新排列列顺序
    columns_order = [
        'ISO_A3', 'year', 'total_vol', 'total_fp', 'building_height', 'total_surface_area'
    ]
    grouped_df = grouped_df[columns_order]
    
    # 按国家代码和年份排序
    grouped_df = grouped_df.sort_values(['ISO_A3', 'year'])
    
    # 保存汇总文件
    iso_summary_file = os.path.join(output_dir, "surface_statistics_iso_summary.csv")
    grouped_df.to_csv(iso_summary_file, index=False)
    
    print(f"ISO汇总文件已保存到: {iso_summary_file}")
    print(f"汇总后数据: {len(grouped_df)} 条记录")
    print(f"涉及 {grouped_df['ISO_A3'].nunique()} 个不同的国家代码")
    
    # 显示一些统计信息
    valid_heights = grouped_df[grouped_df['building_height'] > 0]
    if len(valid_heights) > 0:
        print(f"汇总后建筑高度统计:")
        print(f"  平均高度: {valid_heights['building_height'].mean():.2f}")
        print(f"  最大高度: {valid_heights['building_height'].max():.2f}")
        print(f"  最小高度: {valid_heights['building_height'].min():.2f}")
    
    # 显示表面积统计信息
    valid_surfaces = grouped_df[grouped_df['total_surface_area'] > 0]
    if len(valid_surfaces) > 0:
        print(f"汇总后建筑表面积统计:")
        print(f"  平均表面积: {valid_surfaces['total_surface_area'].mean():.2f}")
        print(f"  最大表面积: {valid_surfaces['total_surface_area'].max():.2f}")
        print(f"  最小表面积: {valid_surfaces['total_surface_area'].min():.2f}")

if __name__ == "__main__":
    calculate_building_surface_area()
