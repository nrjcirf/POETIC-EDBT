# src/config.py
import os
import numpy as np


class Config:
    # --- 运行模式 ---
    QUICK_TEST_MODE = False  # True: 小数据集快速测试, False: 完整实验

    # --- 模拟参数 ---
    TOTAL_ROUNDS = 5000
    DEFAULT_NUM_WORKERS = 2000
    DEFAULT_NUM_WINNERS = 20

    # --- POETIC 及学习参数 ---
    LEARNING_RATE = 0.15
    WEIGHT_FLOOR = 0.0
    WEIGHT_CAP = 1000
    FEEDBACK_MAX_CLIP = 5.0
    # --- 最终探索机制参数 ---
    EXPLORATION_EPSILON = 0.05
    ELITE_POOL_FACTOR = 3

    # --- 新增：信誉机制参数 ---
    REPUTATION_LEARNING_RATE = 0.2# η_R: 信誉学习率
    REPUTATION_BONUS_FACTOR = 0.5  # β: 信誉奖励系数 (最高10%奖励)

    # --- 工作者与任务属性 ---
    BASE_QUALITY_RANGE = (0.0,1.2)
    UNIT_COST_FACTOR_RANGE = (0.1, 0.4)
    NON_STATIONARY_SWITCH_PROB = 0.05
    TASK_DIFFICULTY_RANGE = (1.0, 5.0)

    # --- 隐私参数 ---
    DEFAULT_EPSILON = 1.0
    LAPLACE_SENSITIVITY =200.0

    # --- 协议参数 (新的多阶段选择) ---
    DEFAULT_GROUP_SIZE = 16  # g: 分组大小，用于第一阶段
    WINNERS_PER_GROUP = 4  # c: 每组晋级人数，用于第一阶段 (这里修复了缺失的属性)

    # --- SMPC 模拟参数 ---
    SMPC_COMM_OVERHEAD_PER_COMPARE_MB = 0.0001
    SMPC_RUNTIME_PER_COMPARE_MS = 0.1
    NETWORK_LATENCY_MS = 40

    # --- 数据文件路径 ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # 强制将 DATA_DIR 指向 src 目录下的 data 文件夹
    # 结果通过: C:\Users\jdkso\Desktop\POETIC\src\data
    DATA_DIR = os.path.join(BASE_DIR, 'data')

    PROCESSED_DATA_FILENAME = 'processed_tdrive.parquet'
    CONTEXT_MAPS_FILENAME = 'context_maps.pkl'
    NYC_DATA_FILENAME = 'processed_nyc_with_grid.parquet'  # 请确保文件名与你本地处理好的一致
    NYC_CONTEXT_FILENAME = 'context_maps_nyc.pkl'

    # --- 可视化与存储 ---
    RESULTS_DIR = os.path.join('results', 'experiment_data')
    FIGURES_DIR = os.path.join('results', 'figures')
    LOGS_DIR = 'logs'
    FIG_DPI = 300
    FIG_SIZE = (10, 6)
    COLOR_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                     '#17becf']
    MODEL_STYLES = {
        "POETIC": {'color': '#d62728', 'linestyle': '-', 'marker': 'o', 'zorder': 10},
        "POETIC-slow": {'color': '#ff7f0e', 'linestyle': '--', 'marker': 's', 'zorder': 9},
        "POETIC-No-Adaptive": {'color': '#2ca02c', 'linestyle': ':', 'marker': 'p', 'zorder': 8},  # 新增用于消融实验的标识
        "PrivCO": {'color': '#1f77b4', 'linestyle': '-.', 'marker': '^', 'zorder': 7},
        "OPPS-adapted": {'color': '#9467bd', 'linestyle': '-', 'marker': 'x', 'zorder': 6},
        "Incentivized FL-SC": {'color': '#8c564b', 'linestyle': '--', 'marker': 'd', 'zorder': 5},
        "Truthful-DP-MAB": {'color': '#e377c2', 'linestyle': ':', 'marker': '*', 'zorder': 4},
        "Non-Private Optimal": {'color': 'black', 'linestyle': '-', 'marker': '', 'zorder': 11},
        "Random": {'color': 'grey', 'linestyle': ':', 'marker': '.', 'zorder': 3},
        "Exp 6.4 (Ours)": {'color': '#d62728', 'linestyle': '-', 'marker': 'o', 'zorder': 10},

        # 假设 6.5 是消融版本1 (想要绿色，保持一致)
        "Exp 6.5 (Ablation 1)": {'color': '#2ca02c', 'linestyle': '--', 'marker': 's', 'zorder': 9},

        # 假设 6.6 是消融版本2 (想要蓝色，保持一致)
        "Exp 6.6 (Ablation 2)": {'color': '#1f77b4', 'linestyle': '-.', 'marker': '^', 'zorder': 8},
    }

    # --- 实验参数范围 ---
    N_RANGE = [ 100, 200, 500, 1000,2000,5000]
    EFFICIENCY_N_RANGE = [ 100, 200, 500, 1000, 2000,5000]
    K_RANGE = [10, 20, 50, 100]
    EPSILON_RANGE = [0.1, 0.5, 1.0, 5.0, 10.0, np.inf]


    TIME_HORIZON_RANGE = [100, 200, 500, 1000, 2000,5000]
    # --- 基准模型参数 ---
    FL_SC_CONTRIBUTION_LAMBDA = 0.05
    FL_SC_BASE_REWARD = 10.0
    FL_SC_DP_EPSILON_FACTOR = 0.5
    MAB_UCB_C = 0.2
    MAB_MIN_PULLS = 5
    MAB_BASE_PAYMENT = 10.0

    # --- 快速测试模式 ---
    NUM_ROUNDS_FOR_QUICK_TEST = 50
    NUM_WORKERS_FOR_QUICK_TEST = 100

    SENSITIVITY_LR_RANGE = [0.001, 0.005, 0.01, 0.05, 0.1]  # LEARNING_RATE
    SENSITIVITY_EPSILON_RANGE = [0.01, 0.05, 0.1, 0.2, 0.5]  # EXPLORATION_EPSILON
    SENSITIVITY_FACTOR_RANGE = [1.5, 2, 3, 5, 10]  # ELITE_POOL_FACTOR
    SENSITIVITY_REP_LR_RANGE = [0.001, 0.005, 0.01, 0.05, 0.1]  # REPUTATION_LEARNING_RATE
    SENSITIVITY_REP_BETA_RANGE = [0.0, 0.05, 0.1, 0.2, 0.5]

    # --- ICDE: adaptive worker behavior ---
    WORKER_HISTORY_WINDOW = 10   # sliding window length for bid/utility history
    REJECTION_THRESHOLD = 3      # consecutive rejections before underbid pressure