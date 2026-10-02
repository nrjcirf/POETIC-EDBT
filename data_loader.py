# src/data_loader.py
import pandas as pd
import os
import pickle
import logging


class SpatialDataLoader:
    def __init__(self, config, dataset_type='TDrive'):
        self.config = config
        self.dataset_type = dataset_type
        self.df = None
        self.context_maps = {}
        self.rounds_data_map = {}
        self.available_rounds = []
        self._load_data()
        logging.info(f"Successfully loaded {len(self.available_rounds)} rounds from {dataset_type} dataset.")

    def _load_data(self):
        # 根据数据集类型选择对应的文件名
        if self.dataset_type.upper() == 'NYC':
            data_file = self.config.NYC_DATA_FILENAME
            maps_file = self.config.NYC_CONTEXT_FILENAME
        else:  # 默认为 TDrive
            data_file = self.config.PROCESSED_DATA_FILENAME
            maps_file = self.config.CONTEXT_MAPS_FILENAME

        processed_file_path = os.path.join(self.config.DATA_DIR, data_file)
        context_maps_path = os.path.join(self.config.DATA_DIR, maps_file)

        if os.path.exists(processed_file_path) and os.path.exists(context_maps_path):
            logging.info(f"Loading {self.dataset_type} data from {processed_file_path}")
            self.df = pd.read_parquet(processed_file_path)
            with open(context_maps_path, 'rb') as f:
                self.context_maps = pickle.load(f)
            # 预分组以提高 run_round 时的访问速度
            self.rounds_data_map = {round_id: group for round_id, group in self.df.groupby('round_id')}
            self.available_rounds = sorted(self.rounds_data_map.keys())
        else:
            logging.error(f"Missing files for {self.dataset_type} at {self.config.DATA_DIR}")
            raise FileNotFoundError(f"Required files for {self.dataset_type} not found.")

    def get_round_data(self, round_id):
        """获取特定轮次的所有行程数据"""
        return self.rounds_data_map.get(round_id, pd.DataFrame())

    def get_context_info(self, grid_cell_id, timestamp):
        """
        获取重要性和密度信息。
        逻辑保持与原版 T-Drive 逻辑一致，确保接口兼容。
        """
        if hasattr(timestamp, 'hour'):
            hour = timestamp.hour
        elif hasattr(timestamp, 'dt'):
            hour = timestamp.dt.hour.iloc[0] if not timestamp.empty else 0
        else:
            try:
                dt_obj = pd.to_datetime(timestamp)
                hour = dt_obj.hour
            except:
                hour = 0

        if hasattr(grid_cell_id, 'iloc'):
            grid_cell_id = grid_cell_id.iloc[0] if not grid_cell_id.empty else -1

        grid_cell_id = int(grid_cell_id)
        hour = int(hour)

        # 从预计算的地图中提取上下文
        importance = self.context_maps.get((grid_cell_id, hour, 'importance'), 1.0)
        density = self.context_maps.get((grid_cell_id, hour, 'density'), 1.0)

        return importance, density


# --- 整合进一个工厂函数 ---
def get_dataloader(config, dataset_type='TDrive'):
    """
    一键切换数据集的入口函数
    :param dataset_type: 字符串 'TDrive' 或 'NYC'
    """
    return SpatialDataLoader(config, dataset_type=dataset_type)