import os
import time
import numpy as np
import rasterio
import geopandas as gpd
from rasterstats import zonal_stats

# 生成表面积栅格
def calculate_raster_surface(volume_tif, footprint_tif, output_tif, C=4):
    os.makedirs(os.path.dirname(output_tif), exist_ok=True)
    with rasterio.open(volume_tif) as vol_src, rasterio.open(footprint_tif) as fp_src:
        vol = vol_src.read(1)
        fp = fp_src.read(1)
        profile = vol_src.profile
        vol_nodata = vol_src.nodata if vol_src.nodata is not None else -9999
        fp_nodata = fp_src.nodata if fp_src.nodata is not None else -9999
        mask = (fp > 0) & (vol > 0) & (fp != fp_nodata) & (vol != vol_nodata)
        surface = np.full(vol.shape, profile['nodata'] if profile['nodata'] is not None else -9999, dtype=np.float32)
        surface[mask] = C * vol[mask] / np.sqrt(fp[mask]) + fp[mask]
        profile.update(dtype=rasterio.float32, nodata=profile['nodata'] if profile['nodata'] is not None else -9999)
        with rasterio.open(output_tif, 'w', **profile) as dst:
            dst.write(surface, 1)
    print(f"表面积栅格已保存到: {output_tif}")

# zonal stats 聚合国家表面积
def zonal_surface_stats(shapefile_path, surface_tif, output_csv, year=None):
    countries = gpd.read_file(shapefile_path)
    with rasterio.open(surface_tif) as src:
        nodata_value = src.nodata if src.nodata is not None else -9999
    stats = zonal_stats(
        vectors=countries,
        raster=surface_tif,
        stats="sum",
        nodata=nodata_value,
        all_touched=False,
        geojson_out=False,
        raster_out=False,
        prefix="",
        add_stats=None
    )
    countries["total_surface"] = [s['sum'] if s and 'sum' in s else np.nan for s in stats]
    if year is not None:
        countries['year'] = year
    non_geom_cols = [col for col in countries.columns if col != 'geometry']
    countries[non_geom_cols].to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"国家表面积属性表已保存至: {output_csv}")

def create_summary_file(output_dir):
    """
    创建所有年份的汇总文件
    """
    import glob
    import pandas as pd
    print("\n创建汇总文件...")
    surface_files = glob.glob(os.path.join(output_dir, "surface_statistics_*.csv"))
    if not surface_files:
        print("没有找到表面统计数据文件")
        return
    all_data = []
    for file in surface_files:
        df = pd.read_csv(file)
        all_data.append(df)
    combined_df = pd.concat(all_data, ignore_index=True)
    if 'ISO_A3' in combined_df.columns and 'year' in combined_df.columns:
        combined_df = combined_df.sort_values(['ISO_A3', 'year'])
    summary_file = os.path.join(output_dir, "surface_statistics_all_years.csv")
    combined_df.to_csv(summary_file, index=False)
    print(f"汇总文件已保存到: {summary_file}")
    if 'year' in combined_df.columns:
        print(f"总计 {len(combined_df)} 条记录，涵盖 {combined_df['year'].nunique()} 个年份")


def create_iso_summary_file(output_dir):
    """
    创建按ISO_A3汇总的文件，将相同ISO_A3和年份的条目进行加和
    """
    import pandas as pd
    print("\n创建按ISO_A3汇总的文件...")
    summary_file = os.path.join(output_dir, "surface_statistics_all_years.csv")
    if not os.path.exists(summary_file):
        print("找不到汇总文件，无法创建ISO汇总")
        return
    df = pd.read_csv(summary_file)
    print(f"原始数据: {len(df)} 条记录")
    # 只聚合有的字段
    agg_dict = {}
    for col in ['total_vol', 'total_fp', 'total_surface']:
        if col in df.columns:
            agg_dict[col] = 'sum'
    grouped_df = df.groupby(['ISO_A3', 'year']).agg(agg_dict).reset_index()
    # 重新计算建筑高度
    if 'total_vol' in grouped_df.columns and 'total_fp' in grouped_df.columns:
        grouped_df['building_height'] = grouped_df.apply(
            lambda row: row['total_vol'] / row['total_fp'] if row['total_fp'] > 0 else 0, 
            axis=1
        )
    # 重新计算建筑总表面积
    if 'total_vol' in grouped_df.columns and 'total_fp' in grouped_df.columns:
        C = 4
        grouped_df['total_surface_area'] = grouped_df.apply(
            lambda row: (C * row['total_vol'] / (row['total_fp'] ** 0.5) + row['total_fp']) 
            if row['total_fp'] > 0 else 0, 
            axis=1
        )
    # 添加国家名称（取第一个出现的名称）
    if 'ISO_A3' in df.columns and 'NAM_0' in df.columns:
        country_names = df.groupby('ISO_A3')['NAM_0'].first().reset_index()
        grouped_df = pd.merge(grouped_df, country_names, on='ISO_A3', how='left')
    # 重新排列列顺序
    columns_order = [col for col in ['ISO_A3', 'NAM_0', 'year', 'total_vol', 'total_fp', 'total_surface', 'building_height', 'total_surface_area'] if col in grouped_df.columns]
    grouped_df = grouped_df[columns_order]
    grouped_df = grouped_df.sort_values(['ISO_A3', 'year'])
    iso_summary_file = os.path.join(output_dir, "surface_statistics_iso_summary.csv")
    grouped_df.to_csv(iso_summary_file, index=False)
    print(f"ISO汇总文件已保存到: {iso_summary_file}")
    print(f"汇总后数据: {len(grouped_df)} 条记录")
    if 'ISO_A3' in grouped_df.columns:
        print(f"涉及 {grouped_df['ISO_A3'].nunique()} 个不同的国家代码")
    # 显示一些统计信息
    if 'building_height' in grouped_df.columns:
        valid_heights = grouped_df[grouped_df['building_height'] > 0]
        if len(valid_heights) > 0:
            print(f"汇总后建筑高度统计:")
            print(f"  平均高度: {valid_heights['building_height'].mean():.2f}")
            print(f"  最大高度: {valid_heights['building_height'].max():.2f}")
            print(f"  最小高度: {valid_heights['building_height'].min():.2f}")
    if 'total_surface_area' in grouped_df.columns:
        valid_surfaces = grouped_df[grouped_df['total_surface_area'] > 0]
        if len(valid_surfaces) > 0:
            print(f"汇总后建筑表面积统计:")
            print(f"  平均表面积: {valid_surfaces['total_surface_area'].mean():.2f}")
            print(f"  最大表面积: {valid_surfaces['total_surface_area'].max():.2f}")
            print(f"  最小表面积: {valid_surfaces['total_surface_area'].min():.2f}")


def main():
    print("开始批量处理1975-2030年建筑表面积数据...")
    start_time = time.time()
    base_vol_dir = "D:/data/Built_Volume/"
    base_fp_dir = "D:/data/Built_Footprint/"
    base_surface_dir = "D:/data/Built_Surface/"
    shapefile_path = "D:/data/World Bank Official Boundaries - Admin 0_all_layers/WB_GAD_ADM0_complete.shp"
    output_dir = "surface_results_rastercalc_first"
    os.makedirs(output_dir, exist_ok=True)
    years = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025, 2030]
    for year in years:
        print(f"\n{'='*60}\n处理 {year} 年")
        volume_tif = os.path.join(base_vol_dir, f"GHS_BUILT_V_E{year}_GLOBE_R2023A_4326_30ss_V1_0/GHS_BUILT_V_E{year}_GLOBE_R2023A_4326_30ss_V1_0.tif")
        footprint_tif = os.path.join(base_fp_dir, f"GHS_BUILT_S_E{year}_GLOBE_R2023A_4326_30ss_V1_0/GHS_BUILT_S_E{year}_GLOBE_R2023A_4326_30ss_V1_0.tif")
        surface_tif = os.path.join(base_surface_dir, f"surface_{year}.tif")
        output_csv = os.path.join(output_dir, f"surface_statistics_{year}.csv")
        if not (os.path.exists(volume_tif) and os.path.exists(footprint_tif)):
            print(f"缺少体积或占地面积栅格，跳过 {year}")
            continue
        try:
            calculate_raster_surface(volume_tif, footprint_tif, surface_tif)
            zonal_surface_stats(shapefile_path, surface_tif, output_csv, year=year)
        except Exception as e:
            print(f"{year}年处理出错: {e}")
            continue
        print(f"{year}年处理完成，休息3秒...")
        time.sleep(3)
    print("全部年份处理完成！")
    total_duration = time.time() - start_time
    print(f"总耗时: {total_duration:.2f} 秒 ({total_duration/60:.2f} 分钟)")
    create_summary_file(output_dir)
    create_iso_summary_file(output_dir)

if __name__ == "__main__":
    main() 