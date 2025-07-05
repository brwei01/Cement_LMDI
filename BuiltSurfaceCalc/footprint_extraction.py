import rasterio
import geopandas as gpd
import numpy as np
from rasterstats import zonal_stats
import os
import time
import warnings
import math
from rasterio.warp import calculate_default_transform, reproject, Resampling
import gc

# 禁用不必要的警告
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def process_countries_in_batches(countries, raster_path, nodata_value, batch_size=10):
    """
    分批处理国家数据，避免内存溢出
    """
    all_stats = []
    total_countries = len(countries)
    
    for i in range(0, total_countries, batch_size):
        end_idx = min(i + batch_size, total_countries)
        batch_countries = countries.iloc[i:end_idx]
        
        print(f"处理批次 {i//batch_size + 1}/{(total_countries + batch_size - 1)//batch_size} "
              f"(国家 {i+1}-{end_idx}/{total_countries})")
        
        # 逐个处理国家，避免单个国家失败影响整个批次
        batch_stats = []
        for j, (idx, country) in enumerate(batch_countries.iterrows()):
            country_name = country['NAM_0'] if 'NAM_0' in country else f"国家_{idx+1}"
            print(f"  处理: {country_name}")
            
            try:
                # 单个国家处理
                single_stats = zonal_stats(
                    vectors=[country.geometry],
                    raster=raster_path,
                    stats="sum",
                    nodata=nodata_value,
                    all_touched=False,  # 减少计算量
                    geojson_out=False,
                    raster_out=False,   # 不输出栅格数据
                    prefix="",
                    add_stats=None
                )
                
                if single_stats and len(single_stats) > 0:
                    batch_stats.append(single_stats[0])
                    print(f"    ✓ 成功: {single_stats[0].get('sum', 'N/A')}")
                else:
                    batch_stats.append({'sum': None})
                    print(f"    ⚠ 无结果")
                
            except Exception as e:
                print(f"    ✗ 失败: {e}")
                batch_stats.append({'sum': None})
            
            # 每处理几个国家就进行一次垃圾回收
            if (j + 1) % 3 == 0:
                gc.collect()
        
        all_stats.extend(batch_stats)
        
        # 批次完成后强制垃圾回收
        gc.collect()
        print(f"  批次 {i//batch_size + 1} 完成，成功处理 {len([s for s in batch_stats if s.get('sum') is not None])}/{len(batch_stats)} 个国家")
    
    return all_stats

def process_single_year(year, base_raster_dir, shapefile_path):
    """
    处理单个年份的数据
    """
    print(f"\n{'='*60}")
    print(f"开始处理 {year} 年数据")
    print(f"{'='*60}")
    
    # 构建年份对应的栅格文件路径
    raster_path = os.path.join(base_raster_dir, f"GHS_BUILT_S_E{year}_GLOBE_R2023A_4326_30ss_V1_0/GHS_BUILT_S_E{year}_GLOBE_R2023A_4326_30ss_V1_0.tif")
    
    # 检查文件是否存在
    if not os.path.exists(raster_path):
        print(f"错误: {year}年栅格文件不存在: {raster_path}")
        return False
    
    if not os.path.exists(shapefile_path):
        print(f"错误: Shapefile文件不存在: {shapefile_path}")
        return False
    
    # 1. 加载国家边界数据（包含所有属性）
    try:
        countries = gpd.read_file(shapefile_path)
        print(f"加载了 {len(countries)} 个国家边界 | 属性字段: {list(countries.columns)}")
        print(f"坐标系: {countries.crs}")
        print(f"边界范围: {countries.total_bounds}")
    except Exception as e:
        print(f"加载国家边界数据失败: {e}")
        return False
    
    # 2. 检查栅格数据
    print("检查栅格数据...")
    try:
        with rasterio.open(raster_path) as src:
            print(f"栅格坐标系: {src.crs}")
            print(f"栅格范围: {src.bounds}")
            print(f"NODATA值: {src.nodata}")
            print(f"栅格尺寸: {src.width} x {src.height}")
            print(f"数据类型: {src.dtypes[0]}")
            print(f"块大小: {src.block_shapes[0]}")
            nodata_value = src.nodata if src.nodata is not None else -9999
            
            # 检查坐标系是否匹配
            if countries.crs != src.crs:
                print(f"坐标系不匹配，正在转换国家边界到栅格坐标系...")
                try:
                    countries = countries.to_crs(src.crs)
                    print(f"转换后坐标系: {countries.crs}")
                    print(f"转换后边界范围: {countries.total_bounds}")
                except Exception as e:
                    print(f"坐标系转换失败: {e}")
                    return False
    except Exception as e:
        print(f"读取栅格数据失败: {e}")
        return False
    
    # 3. 检查数据重叠
    print("检查数据重叠...")
    try:
        with rasterio.open(raster_path) as src:
            raster_bounds = src.bounds
            countries_bounds = countries.total_bounds
            
            # 检查是否有重叠
            if (countries_bounds[0] > raster_bounds[2] or  # 左边界 > 右边界
                countries_bounds[2] < raster_bounds[0] or  # 右边界 < 左边界
                countries_bounds[1] > raster_bounds[3] or  # 下边界 > 上边界
                countries_bounds[3] < raster_bounds[1]):   # 上边界 < 下边界
                print("警告: 国家边界与栅格数据没有重叠!")
                print(f"国家边界范围: {countries_bounds}")
                print(f"栅格数据范围: {raster_bounds}")
            else:
                print("✓ 数据重叠检查通过")
    except Exception as e:
        print(f"重叠检查失败: {e}")
    
    # 4. 分批处理分区统计
    print("开始分批计算建筑足迹面积...")
    
    # 根据栅格大小调整批次大小 - 使用更保守的设置
    with rasterio.open(raster_path) as src:
        total_pixels = src.width * src.height
        if total_pixels > 1000000000:  # 超过10亿像素
            batch_size = 2
        elif total_pixels > 500000000:  # 超过5亿像素
            batch_size = 3
        elif total_pixels > 100000000:  # 超过1亿像素
            batch_size = 5
        else:
            batch_size = 10
    
    print(f"使用批次大小: {batch_size}")
    
    # 如果仍然遇到问题，可以手动设置更小的批次大小
    # batch_size = 1  # 取消注释这行来逐个处理国家
    
    try:
        stats = process_countries_in_batches(countries, raster_path, nodata_value, batch_size)
        print(f"统计计算完成，获得 {len(stats)} 个结果")
        
    except Exception as e:
        print(f"分区统计完全失败: {e}")
        return False
    
    # 5. 处理统计结果
    footprints = []
    valid_count = 0
    
    for i, stat in enumerate(stats):
        country_name = countries.iloc[i]['NAM_0'] if 'NAM_0' in countries.columns else f"国家_{i+1}"
        
        # 处理可能的None结果
        if stat is None:
            footprints.append(np.nan)
            print(f"警告: {country_name} 计算结果为None")
            continue
            
        if 'sum' not in stat:
            footprints.append(np.nan)
            print(f"警告: {country_name} 缺少sum字段，可用字段: {list(stat.keys())}")
            continue
            
        footprint = stat['sum']
        
        # 处理可能的异常值
        if footprint is None:
            footprints.append(np.nan)
            print(f"警告: {country_name} 足迹面积值为None")
        elif math.isnan(footprint):
            footprints.append(np.nan)
            print(f"警告: {country_name} 足迹面积值为NaN")
        else:
            # 转换为整数以避免Shapefile字段限制问题
            try:
                footprints.append(int(footprint))
                valid_count += 1
                if valid_count <= 5:  # 只打印前5个有效结果作为示例
                    print(f"✓ {country_name}: {footprint:,.0f}")
            except (OverflowError, ValueError) as e:
                footprints.append(np.nan)
                print(f"警告: {country_name} 足迹面积值转换失败: {footprint} - {e}")
    
    # 6. 添加结果到原始数据
    countries["total_fp"] = footprints
    
    # 7. 输出结果
    output_dir = "fp_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为GeoPackage格式（避免Shapefile限制）
    # output_gpkg = os.path.join(output_dir, "countries_with_footprint.gpkg")
    # countries.to_file(output_gpkg, driver='GPKG', encoding='utf-8')
    # print(f"GeoPackage结果已保存至: {output_gpkg}")
    
    # 保存CSV属性表 - 按年份命名
    output_csv = os.path.join(output_dir, f"footprint_statistics_{year}.csv")
    # 选择所有非几何列
    non_geom_cols = [col for col in countries.columns if col != 'geometry']
    countries[non_geom_cols].to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"{year}年属性表已保存至: {output_csv}")
    
    # 8. 性能报告和统计摘要
    if valid_count > 0:
        # 过滤掉NaN值
        valid_footprints = countries["total_fp"].dropna()
        
        print(f"\n{year}年足迹面积统计摘要:")
        print(f"有效国家数量: {valid_count}/{len(countries)}")
        print(f"总和: {valid_footprints.sum():,.0f}")
        print(f"平均值: {valid_footprints.mean():,.0f}")
        print(f"最大值: {valid_footprints.max():,.0f}")
        print(f"最小值: {valid_footprints.min():,.0f}")
        
        # 找到最大值和最小值对应的国家
        max_idx = valid_footprints.idxmax()
        min_idx = valid_footprints.idxmin()
        
        print(f"建筑足迹面积最大的国家: {countries.loc[max_idx, 'NAM_0']} ({valid_footprints.max():,.0f})")
        print(f"建筑足迹面积最小的国家: {countries.loc[min_idx, 'NAM_0']} ({valid_footprints.min():,.0f})")
    else:
        print(f"警告: {year}年没有有效的足迹面积计算结果")
        print("可能的原因:")
        print("1. 栅格数据与矢量数据没有重叠")
        print("2. 栅格数据中没有有效值")
        print("3. 坐标系不匹配")
        print("4. 栅格数据路径错误")
    
    return True

def main():
    """
    主函数：循环处理1975-2030年每5年的数据
    """
    print("开始批量处理1975-2030年建筑足迹面积数据...")
    start_time = time.time()
    
    # 配置路径
    base_raster_dir = "D:/data/Built_Footprint/"  # 栅格数据根目录
    shapefile_path = "D:/data/World Bank Official Boundaries - Admin 0_all_layers/WB_GAD_ADM0_complete.shp"
    
    # 定义年份列表（1975-2030年，每5年）
    years = [1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025, 2030]
    
    # 统计处理结果
    success_count = 0
    failed_years = []
    
    for year in years:
        try:
            success = process_single_year(year, base_raster_dir, shapefile_path)
            if success:
                success_count += 1
            else:
                failed_years.append(year)
        except Exception as e:
            print(f"处理{year}年数据时发生错误: {e}")
            failed_years.append(year)
        
        # 年份间休息一下，避免内存累积
        print(f"\n{year}年处理完成，休息3秒...")
        time.sleep(3)
    
    # 最终统计
    total_duration = time.time() - start_time
    print(f"\n{'='*60}")
    print("批量处理完成!")
    print(f"{'='*60}")
    print(f"总耗时: {total_duration:.2f} 秒 ({total_duration/60:.2f} 分钟)")
    print(f"成功处理: {success_count}/{len(years)} 年")
    print(f"失败年份: {failed_years}")
    
    if failed_years:
        print(f"\n失败年份的文件路径:")
        for year in failed_years:
            expected_path = os.path.join(base_raster_dir, f"GHS_BUILT_S_E{year}_GLOBE_R2023A_4326_30ss_V1_0/GHS_BUILT_S_E{year}_GLOBE_R2023A_4326_30ss_V1_0.tif")
            print(f"  {year}年: {expected_path}")

if __name__ == "__main__":
    main()
