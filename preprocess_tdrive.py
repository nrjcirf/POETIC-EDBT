# src/preprocess_tdrive.py
import pandas as pd
import numpy as np
import glob
import os
import pickle
from tqdm import tqdm


class PreprocessingConfig:
    # --- 路径修改 ---
    # 定位到你桌面的原始数据文件夹
    INPUT_FOLDER = r"C:\Users\jdkso\Desktop\taxi_log_2008_by_id"

    # 输出路径依然保持在项目根目录下的 data 文件夹，方便 main.py 读取
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data')
    OUTPUT_DATA_FILE = os.path.join(OUTPUT_DIR, 'processed_tdrive.parquet')
    OUTPUT_CONTEXT_FILE = os.path.join(OUTPUT_DIR, 'context_maps.pkl')

    BEIJING_BOUNDS = {
        'min_lon': 116.0, 'max_lon': 116.8,
        'min_lat': 39.7, 'max_lat': 40.2
    }
    GRID_CELL_SIZE_METERS = 500


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def gps_to_grid_id(lat, lon, bounds, cell_size_m):
    lat_degree_per_cell = (cell_size_m / 1000) / 111.0
    lon_degree_per_cell = (cell_size_m / 1000) / (
            111.0 * np.cos(np.radians(bounds['min_lat'])))
    row = int((lat - bounds['min_lat']) / lat_degree_per_cell)
    col = int((lon - bounds['min_lon']) / lon_degree_per_cell)
    return row, col


def preprocess_tdrive(config):
    print(f"--- 启动 T-Drive 预处理 ---")
    print(f"输入路径: {config.INPUT_FOLDER}")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Step 1: 合并
    all_files = glob.glob(os.path.join(config.INPUT_FOLDER, '*.txt'))
    if not all_files:
        print(f"错误：在路径 {config.INPUT_FOLDER} 下未找到任何 .txt 文件！")
        return

    df_list = []
    for filename in tqdm(all_files, desc="Merging files"):
        try:
            df_single = pd.read_csv(
                filename, header=None,
                names=['worker_id', 'timestamp', 'longitude', 'latitude'],
                dtype={'worker_id': str}
            )
            df_list.append(df_single)
        except Exception as e:
            print(f"读取文件 {filename} 出错: {e}")

    df = pd.concat(df_list, ignore_index=True)

    # Step 2: 清洗
    print("[Step 2/5] 正在清洗数据...")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    bounds = config.BEIJING_BOUNDS
    df = df[(df['latitude'].between(bounds['min_lat'], bounds['max_lat'])) & (
        df['longitude'].between(bounds['min_lon'], bounds['max_lon']))]
    df = df.sort_values(by=['worker_id', 'timestamp']).reset_index(drop=True)

    # Step 3: 特征工程
    print("[Step 3/5] 正在特征工程...")
    df['lat_prev'] = df.groupby('worker_id')['latitude'].shift(1)
    df['lon_prev'] = df.groupby('worker_id')['longitude'].shift(1)
    df['time_prev'] = df.groupby('worker_id')['timestamp'].shift(1)
    same_worker = df['worker_id'] == df['worker_id'].shift(1)

    df.loc[same_worker, 'distance_km'] = haversine_distance(
        df.loc[same_worker, 'lat_prev'], df.loc[same_worker, 'lon_prev'],
        df.loc[same_worker, 'latitude'], df.loc[same_worker, 'longitude']
    )
    df['distance_km'] = df['distance_km'].fillna(0)
    df.loc[same_worker, 'time_diff_s'] = (
                df.loc[same_worker, 'timestamp'] - df.loc[same_worker, 'time_prev']).dt.total_seconds()
    df['time_diff_s'] = df['time_diff_s'].fillna(0)
    df['speed'] = np.where(df['time_diff_s'] > 0, df['distance_km'] / (df['time_diff_s'] / 3600), 0)
    df['speed'] = np.clip(df['speed'], 0, 150)

    # 计算 round_id
    df['round_id'] = (df['timestamp'].astype(np.int64) // 10 ** 9 // 300).astype(int)

    # 坐标映射与 ID 碰撞风险处理
    grid_coords = df.apply(
        lambda row: gps_to_grid_id(row['latitude'], row['longitude'], bounds, config.GRID_CELL_SIZE_METERS), axis=1)
    df['grid_row'] = grid_coords.apply(lambda x: x[0])
    df['grid_col'] = grid_coords.apply(lambda x: x[1])
    # 使用 10000 倍率防止 ID 碰撞
    df['grid_cell_id'] = df['grid_row'] * 10000 + df['grid_col']

    # Step 4: 上下文地图 (gid, hr, type) 格式化
    print("[Step 4/5] 正在生成上下文地图 (三元组格式)...")
    df['hour'] = df['timestamp'].dt.hour
    importance_series = df.groupby(['grid_cell_id', 'hour']).size()
    density_series = df.groupby(['grid_cell_id', 'hour'])['worker_id'].nunique()

    context_maps = {}
    for (gid, hr), val in importance_series.items():
        # 显式使用 int 确保 pickle 兼容性
        context_maps[(int(gid), int(hr), 'importance')] = float(val)

    for (gid, hr), val in density_series.items():
        context_maps[(int(gid), int(hr), 'density')] = float(val)

    # Step 5: 保存
    print(f"[Step 5/5] 正在保存到 {config.OUTPUT_DIR}...")
    final_df = df[['round_id', 'worker_id', 'timestamp', 'latitude', 'longitude', 'speed', 'grid_cell_id']].copy()
    final_df.to_parquet(config.OUTPUT_DATA_FILE, index=False)
    with open(config.OUTPUT_CONTEXT_FILE, 'wb') as f:
        pickle.dump(context_maps, f)

    print(f"预处理成功！输出文件: {config.OUTPUT_DATA_FILE}")


if __name__ == '__main__':
    config = PreprocessingConfig()
    preprocess_tdrive(config)