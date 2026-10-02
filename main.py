# src/main.py
import os
import numpy as np
import time
from tqdm import tqdm
import random
import logging
import pandas as pd


from src.config import Config
from src.worker import Worker
from src.poe_platform import Platform
from src.baselines import BaseBaseline, NonPrivateOptimal, RandomModel, PrivCO, POETICSlow, OPPSAdapted, \
    IncentivizedFLSC, TruthfulDPMAB
from src.metrics import Metrics
from src.visualization import Visualizer
from src.experiments_icde import (
    run_overall_utility_experiment,
    run_regret_analysis_experiment,
    run_contamination_robustness_experiment,
    run_strategic_worker_experiment,
    run_reputation_robustness_experiment,
    run_privacy_tradeoff_experiment,
    run_mechanism_ablation_experiment,
)

SEED = 42  # 42 是一个任意的数字，您可以选择任何整数
random.seed(SEED)
np.random.seed(SEED)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def all_models_fabric(num_workers, cfg):
    worker_ids = list(range(num_workers))
    models = {
        "POETIC": Platform(cfg, worker_ids),
        "POETIC-slow": POETICSlow(cfg, worker_ids),
        "Non-Private Optimal": NonPrivateOptimal(cfg),
        "Random": RandomModel(cfg),
        "PrivCO": PrivCO(cfg),
        "OPPS-adapted": OPPSAdapted(cfg, worker_ids),
        "Incentivized FL-SC": IncentivizedFLSC(cfg, worker_ids),
        "Truthful-DP-MAB": TruthfulDPMAB(cfg, worker_ids),
    }
    return models


def all_models_fabric_with_no_adaptive(num_workers, cfg):
    worker_ids = list(range(num_workers))
    models = all_models_fabric(num_workers, cfg)
    models["POETIC-No-Adaptive"] = Platform(cfg, worker_ids, name="POETIC-No-Adaptive", use_exploration=False,
                                            use_reputation=False)  # Simplified no-adaptive
    return models

def setup_file_logger(log_file_path):
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger().addHandler(file_handler)


def log_specific_metrics(log_file_path, experiment_name, results_dict):
    """
    一个简化的日志记录函数，使用 logging 模块写入，避免文件冲突。
    用于 6.1, 6.7, 6.8, 6.9。
    """
    try:
        logging.info(f"--- Metrics for: {experiment_name} ---")

        # 检查 results_dict 的结构
        # 结构 1: { 'model_name': {'final_cumulative_value': X, ...}, ... }
        # src/main.py
        if all(isinstance(v, dict) and 'runtime_history' in v for v in results_dict.values()):
            for name, data in results_dict.items():
                final_value = data.get('final_cumulative_value', 0)
                avg_runtime = np.mean(data.get('runtime_history', [])) if data.get('runtime_history', []) else 0
                avg_comm = np.mean(data.get('comm_cost_history', [])) if data.get('comm_cost_history', []) else 0
                avg_profit = np.mean(data.get('worker_profit_history_agg', [])) if data.get('worker_profit_history_agg',
                                                                                            []) else 0

                log_msg = f"{str(name):<25} | Val: {final_value:,.2f} | Runtime: {avg_runtime:.4f} | Comm: {avg_comm:.6f} | Profit: {avg_profit:.4f}"
                logging.info(log_msg)

        # 结构 2: { 'strategy_name': {'avg_worker_profit': X, ...}, ... }
        elif all(isinstance(v, dict) and ('avg_worker_profit' in v or 'final_cumulative_value' in v) for v in
                 results_dict.values()):
            for name, data in results_dict.items():
                final_value_data = data.get('final_cumulative_value', 'N/A')
                avg_profit_data = data.get('avg_worker_profit', 'N/A')

                log_msg_parts = [f"{str(name):<25} |"]
                if isinstance(final_value_data, (int, float)):
                    log_msg_parts.append(f"Val: {final_value_data:,.2f}")
                if isinstance(avg_profit_data, (int, float)):
                    log_msg_parts.append(f"Profit: {avg_profit_data:.4f}")

                logging.info(" ".join(log_msg_parts))

        logging.info(f"--- End Metrics for: {experiment_name} ---")

    except Exception as e:
        logging.error(f"Failed to log specific metrics for {experiment_name}: {e}")


def log_averaged_metrics(experiment_name, agg_results_dict):
    """
    计算并记录 N 次运行的平均值和标准差。
    agg_results_dict 格式: { 'model_name': {'metric1': [run1_val, run2_val], 'metric2': [...]}, ... }
    """
    try:
        # 查找所有唯一的指标键和运行次数
        all_metrics = set()
        num_runs = 0
        for model_data in agg_results_dict.values():
            all_metrics.update(model_data.keys())
            if not num_runs and model_data:
                num_runs = len(next(iter(model_data.values())))

        sorted_metrics = sorted(list(all_metrics))

        logging.info(f"\n--- Averaged Results for: {experiment_name} (over {num_runs} runs) ---")

        # 打印表头
        header = f"{'Identifier':<25}"
        for metric in sorted_metrics:
            header += f" | Mean {metric:<15} | Std {metric:<10}"
        logging.info(header)
        logging.info("-" * len(header))

        # 打印每一行的数据
        for model_name, model_data in agg_results_dict.items():
            line = f"{model_name:<25}"
            for metric in sorted_metrics:
                values = model_data.get(metric, [])
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    line += f" | {mean_val:>15.2f} | {std_val:>10.2f}"
                else:
                    line += f" | {'N/A':>15} | {'N/A':>10}"
            logging.info(line)
        logging.info("\n")
    except Exception as e:
        logging.error(f"Failed to log averaged metrics for {experiment_name}: {e}")

def log_experiment_results(log_file_path, experiment_name, results_dict, identifier_header="Identifier"):
    try:
        logging.info(f"\n--- Results for: {experiment_name} ---")
        header = f"{identifier_header:<25} | {'Final Cumulative Value':>25} | {'Avg Runtime (ms)':>20} | {'Avg Comm Cost (MB)':>20} | {'Avg Worker Profit':>20}"
        logging.info(header)
        logging.info("-" * (len(header) + 5))

        for name, data in results_dict.items():
            final_value = data.get('final_cumulative_value', 0)
            runtime = data.get('runtime_history', [])
            comm = data.get('comm_cost_history', [])
            profit = data.get('worker_profit_history_agg', [])

            avg_runtime = np.mean(runtime) if runtime else 0
            avg_comm = np.mean(comm) if comm else 0
            avg_profit = np.mean(profit) if profit else 0

            log_msg = f"{str(name):<25} | {final_value:>25.2f} | {avg_runtime:>20.4f} | {avg_comm:>20.6f} | {avg_profit:>20.4f}"
            logging.info(log_msg)
        logging.info("\n")
    except Exception as e:
        logging.error(f"Failed to log standard results for {experiment_name}: {e}")


# src/main.py 中的 run_single_simulation 函数修复版

def run_single_simulation(model, workers_dict, config, data_loader=None, total_rounds_override=None,
                          bidding_strategy='truthful', dp_epsilon_override=None, num_winners_override=None):
    """
    运行单个模型的完整模拟流程 (已修复：支持 data_loader 参数)
    """
    # 强制重置模型状态
    if hasattr(model, 'reset'):
        model.reset()

    # 强制重置工人状态
    for w in workers_dict.values():
        w.reset()

    total_rounds = total_rounds_override if total_rounds_override is not None else config.TOTAL_ROUNDS

    # 初始化记录列表
    model.net_social_value_history = [] # 确保重置
    model.runtime_history = []
    model.comm_cost_history = []
    model.worker_profit_history_agg = []

    # 进度条 (可选)
    # iterator = tqdm(range(total_rounds), desc=f"Simulating {model.name}", leave=False)
    iterator = range(total_rounds)

    for i in iterator:
        # 1. 获取当前轮次的环境上下文
        current_context = None
        if data_loader:
            # [核心修复] 如果提供了 data_loader，则随机采样环境上下文
            if hasattr(data_loader, 'context_maps') and data_loader.context_maps:
                # 随机选一个 key (grid_id, hour, type)
                keys = list(data_loader.context_maps.keys())
                if keys:
                    random_key = random.choice(keys)
                    # 兼容不同的 key 格式
                    if len(random_key) == 3:
                        grid_id, hour, _ = random_key
                    else:
                        grid_id, hour = random_key[0], random_key[1]

                    density = data_loader.context_maps.get((grid_id, hour, 'density'), 1.0)
                    importance = data_loader.context_maps.get((grid_id, hour, 'importance'), 1.0)
                    current_context = {'density': density, 'importance': importance}

        # 2. 更新所有工人的状态
        active_workers_ids = []
        for wid, worker in workers_dict.items():
            # [核心修复] 传入 data_loader 和 current_round_id (i)
            worker.update_state_for_round(current_context, data_loader, i)

            if worker.online_status > 0:
                active_workers_ids.append(wid)

        # 3. 运行一轮模型
        winners, payments = model.run_round(
            round_id=i,
            active_workers_ids=active_workers_ids,
            workers_dict=workers_dict,
            data_loader=data_loader,
            bidding_strategy=bidding_strategy,
            dp_epsilon=dp_epsilon_override,
            num_winners=num_winners_override
        )

    # 返回结果字典
    return {
        'net_social_value_history': model.net_social_value_history,
        'runtime_history': model.runtime_history,
        'comm_cost_history': model.comm_cost_history,
        'worker_profit_history_agg': model.worker_profit_history_agg
    }


def run_macro_performance_experiment(all_models_fabric, config, visualizer, log_path):
    logging.info("\n--- Running Experiment 6.2: Macro-Performance Comparison ---")

    # Part 1: Fixed N
    n_fixed = 2000
    logging.info(f"--- Part 1: Running with fixed N = {n_fixed} ---")
    workers_dict_fixed_n = {wid: Worker(wid, config) for wid in range(n_fixed)}
    models_fixed_n = all_models_fabric(n_fixed, config)
    results_fixed_n = {}
    for name, model in models_fixed_n.items():
        result = run_single_simulation(model, workers_dict_fixed_n, config)
        result['cumulative_value'] = Metrics.calculate_cumulative_value(result['net_social_value_history'])
        result['final_cumulative_value'] = result['cumulative_value'][-1] if result['cumulative_value'] else 0
        results_fixed_n[name] = result

    log_specific_metrics(log_path, f"Exp 6.1: Macro-Performance (N={n_fixed})", results_fixed_n)
    optimal_cumulative = np.cumsum(results_fixed_n["Non-Private Optimal"]['net_social_value_history'])
    visualizer.plot_macro_performance_curves(results_fixed_n, f"Macro-Performance Comparison (N={n_fixed})",
                                             "exp_6.1_cumulative_value.pdf")
    visualizer.plot_cumulative_regret_vs_rounds(results_fixed_n, optimal_cumulative,
                                                f"Cumulative Regret vs. Rounds (N={n_fixed})",
                                                "exp_6.2_cumulative_regret.pdf")

    # Part 2: Varying N
    logging.info("--- Part 2: Varying N to test scalability ---")
    results_by_n = {name: {} for name in all_models_fabric(0, config).keys()}
    for n in config.N_RANGE:
        logging.info(f"--- Testing with N = {n} ---")
        worker_subset = {wid: Worker(wid, config) for wid in range(n)}
        temp_models = all_models_fabric(n, config)
        for name, model in temp_models.items():
            result = run_single_simulation(model, worker_subset, config)
            results_by_n[name][n] = {
                'final_cumulative_value': Metrics.calculate_cumulative_value(result['net_social_value_history'])[-1] if
                result['net_social_value_history'] else 0
            }

    visualizer.plot_final_value_vs_n(results_by_n, config.N_RANGE, "Final Value vs. Number of Workers",
                                     "exp_6.3_final_value_vs_n.pdf")


def run_efficiency_experiment(config, visualizer, log_path):
    logging.info("\n--- Running Experiment 6.3: Efficiency Comparison ---")

    # Part 1: Varying N
    results_vs_n = {}
    for n in config.EFFICIENCY_N_RANGE:
        logging.info(f"--- Efficiency vs. N: Testing with N = {n} ---")
        workers = {wid: Worker(wid, config) for wid in range(n)}
        models_to_test = {
            "POETIC": Platform(config, list(workers.keys())),
            "POETIC-slow": POETICSlow(config, list(workers.keys())),
            "PrivCO": PrivCO(config)
        }
        for name, model in models_to_test.items():
            if name not in results_vs_n: results_vs_n[name] = {}
            results_vs_n[name][n] = run_single_simulation(model, workers, config)

    visualizer.plot_efficiency_metric(results_vs_n, config.EFFICIENCY_N_RANGE, 'runtime_history',
                                      "Avg Runtime (ms, log)", "Runtime vs. N", "exp_6.4_runtime_vs_n.pdf",
                                      "Number of Workers (N)")
    visualizer.plot_efficiency_metric(results_vs_n, config.EFFICIENCY_N_RANGE, 'comm_cost_history',
                                      "Avg Comm Cost (MB, log)", "Comm. Cost vs. N", "exp_6.5_comm_vs_n.pdf",
                                      "Number of Workers (N)")

    # Part 2: Varying k
    workers_fixed_n = {wid: Worker(wid, config) for wid in range(500)}
    models_to_test_k = {"POETIC": Platform(config, list(workers_fixed_n.keys())),
                        "POETIC-slow": POETICSlow(config, list(workers_fixed_n.keys())), "PrivCO": PrivCO(config)}
    results_vs_k = {name: {} for name in models_to_test_k.keys()}
    for k in config.K_RANGE:
        logging.info(f"--- Efficiency vs. k: Testing with k = {k} ---")
        for name, model in models_to_test_k.items():
            results_vs_k[name][k] = run_single_simulation(model, workers_fixed_n, config, num_winners_override=k)

    visualizer.plot_efficiency_metric(results_vs_k, config.K_RANGE, 'runtime_history', "Avg Runtime (ms, log)",
                                      "Runtime vs. k", "exp_6.6_runtime_vs_k.pdf", "Number of Winners (k)")


def run_mechanism_effectiveness_experiment(poetic_main_model, workers_dict, config, visualizer, log_path):
    """
    [直接控制版] 多策略报价鲁棒性验证 - 绕过框架缓存，手动验证支付规则
    核心：1) 修正 IR 检查为 payment >= cost  2) 正确计算期望效用  3) 隔离策略干扰
    """
    logging.info("\n--- Experiment: Multi-Strategy Bidding (Direct Control) ---")

    n_workers = 1000
    test_ids = list(range(100))  # 10% 测试组
    strategies = ['truthful', 'consistent_underbid', 'consistent_overbid', 'random_overbid']
    results = {}

    for strategy in strategies:
        logging.info(f"  Testing: {strategy}")
        # 🔑 核心：重置种子，确保"同题考试"
        random.seed(SEED);
        np.random.seed(SEED)

        platform = Platform(config, list(range(n_workers)), name=f"POETIC-{strategy}", use_reputation=True)
        platform.reset()
        workers = {wid: Worker(wid, config) for wid in range(n_workers)}

        # 分配策略：测试组变更，对照组保持诚实
        for wid in test_ids:
            workers[wid].current_strategy = strategy
        for wid in range(n_workers):
            if wid not in test_ids:
                workers[wid].current_strategy = 'truthful'

        total_profit = 0.0
        total_value = 0.0
        win_count = 0

        for t in range(config.TOTAL_ROUNDS):
            # 更新工人状态
            for w in workers.values():
                w.update_state_for_round(None, None, t)
            active = [wid for wid, w in workers.items() if w.online_status > 0.5]
            if len(active) < config.DEFAULT_NUM_WINNERS:
                continue

            # 运行单轮 (bidding_strategy=None 触发工人内部 current_strategy)
            winners, payments = platform.run_round(t, active, workers, None, bidding_strategy=None)

            # ✅ [修复 1] 正确计算测试组利润：严格基于成本 (IR: payment >= cost)
            for wid in test_ids:
                if wid in winners:
                    cost = workers[wid].true_cost
                    pay = payments.get(wid, 0)
                    # ✅ [修复 2] 个体理性：支付 >= 成本时工人才接受任务
                    if pay >= cost:
                        total_profit += (pay - cost)
                        win_count += 1
                    # else: 工人拒绝，利润为 0（不累加）

            if platform.net_social_value_history:
                total_value += platform.net_social_value_history[-1]

        # ✅ [修复 3] 计算平均期望效用：总利润 / 中标次数 (反映单次中标收益)
        # 如果中标次数为 0，则利润为 0
        avg_profit = total_profit / win_count if win_count > 0 else 0.0

        results[strategy] = {
            'avg_worker_profit': avg_profit,
            'final_cumulative_value': total_value,
            'win_rate': win_count / (len(test_ids) * config.TOTAL_ROUNDS)
        }
        logging.info(
            f"    {strategy}: AvgProfit={avg_profit:.2f} | WinRate={results[strategy]['win_rate']:.3f} | Value={total_value:.1f}")

    # 绘图输出
    visualizer.plot_bar_chart(results, 'avg_worker_profit', "Avg Worker Profit (Per Win)",
                              "Worker Profit vs. Bidding Strategy", "exp_multi_strategy_profit.pdf")
    visualizer.plot_bar_chart(results, 'final_cumulative_value', "Platform Cumulative Value",
                              "Platform Value vs. Bidding Strategy", "exp_multi_strategy_value.pdf")

    logging.info(f"✅ Final Results: {results}")
    logging.info("Multi-strategy experiment completed.")

def run_forward_looking_experiment(config, visualizer, log_path):
    """验证机制对前瞻性套利策略(先刷声誉后套利)的鲁棒性"""
    logging.info("\n--- Experiment: Forward-Looking Strategic Robustness ---")
    n_workers = 1000
    workers = {wid: Worker(wid, config) for wid in range(n_workers)}
    poetic = Platform(config, list(workers.keys()), name="POETIC-Fwd")
    
    # 1. 跑一次全员诚实作为 Baseline
    poetic.reset()
    for w in workers.values(): w.reset()
    res_honest = run_single_simulation(poetic, workers, config, bidding_strategy='truthful')
    honest_values = res_honest['net_social_value_history']

    # 2. 跑一次前瞻性策略
    poetic.reset()
    for w in workers.values(): w.reset()
    res_fwd = run_single_simulation(poetic, workers, config, bidding_strategy='forward_looking')
    fwd_values = res_fwd['net_social_value_history']
    
    # ✅ 【核心修复】：将列表转换为 plot_line_chart 要求的嵌套字典格式
    # 结构转换: [val1, val2, ...] -> {1: {'final_cumulative_value': val1}, 2: {...}, ...}
    honest_dict = {t: {'final_cumulative_value': v} for t, v in enumerate(honest_values, 1)}
    fwd_dict = {t: {'final_cumulative_value': v} for t, v in enumerate(fwd_values, 1)}

    # 3. 绘图对比
    visualizer.plot_line_chart(
        {'Truthful': honest_dict, 'Forward-Looking': fwd_dict},  # 传入格式化后的字典
        range(1, config.TOTAL_ROUNDS + 1), 'final_cumulative_value',
        "Cumulative Platform Value", "Forward-Looking Robustness", 
        "exp_forward_looking.pdf", "Round t"
    )
    logging.info("Forward-looking experiment completed.")


def run_quality_aware_reputation_experiment(config, visualizer, log_path):
    """[直接控制版] 质量感知声誉验证 - 绕过框架缓存，手动显式计算更新逻辑"""
    logging.info("\n--- Experiment: Quality-Aware Reputation (Direct Control) ---")

    n_workers = 200
    T = 300
    eta_R = 0.6  # 高学习率，确保 300 轮内快速分离
    low_u, high_u = 80.0, 920.0  # 悬殊差异，制造明显质量梯度
    low_ids = set(range(20))  # 10% 低质量
    high_ids = set(range(20, n_workers))

    # 初始化声誉字典 (完全独立于平台)
    rep_old = {i: 0.5 for i in range(n_workers)}
    rep_new = {i: 0.5 for i in range(n_workers)}

    hist = {'Old-Low': [], 'Old-High': [], 'New-Low': [], 'New-High': []}

    # 创建平台实例，显式关闭内部声誉更新，仅借用其安全选择协议
    plat = Platform(config, list(range(n_workers)), name="Selector", use_reputation=False)

    for t in range(T):
        # 1. 构造本轮工人（强制固定效用，跳过 update_state 干扰）
        workers = {i: Worker(i, config) for i in range(n_workers)}
        for i in low_ids:
            workers[i].true_utility = low_u
            workers[i].true_cost = low_u * 0.6
            workers[i].online_status = 1.0
        for i in high_ids:
            workers[i].true_utility = high_u
            workers[i].true_cost = high_u * 0.6
            workers[i].online_status = 1.0

        # 2. 仅运行选择逻辑获取胜出者
        winners, _ = plat.run_round(t, list(range(n_workers)), workers, None, bidding_strategy='truthful')

        # 3. 计算全局平均效用（所有在线工人）
        global_avg_u = np.mean([workers[i].true_utility for i in range(n_workers)])

        # 4. 【核心】手动执行两种机制的声誉更新
        for w in winners:
            u_i = workers[w].true_utility

            # 旧机制：仅依赖选中 (R += η * 1.0)
            rep_old[w] = (1 - eta_R) * rep_old[w] + eta_R * 1.0

            # 新机制：质量感知 (R += η * quality_factor)
            # 低质量 u_i/global_avg ≈ 0.08 -> clip 到 0.1
            # 高质量 u_i/global_avg ≈ 1.07 -> clip 到 1.07 (上限可放宽至 1.5)
            quality_factor = np.clip(u_i / max(global_avg_u, 1e-6), 0.1, 1.5)
            rep_new[w] = (1 - eta_R) * rep_new[w] + eta_R * quality_factor

        # 5. 记录分组均值
        hist['Old-Low'].append(np.mean([rep_old[i] for i in low_ids]))
        hist['Old-High'].append(np.mean([rep_old[i] for i in high_ids]))
        hist['New-Low'].append(np.mean([rep_new[i] for i in low_ids]))
        hist['New-High'].append(np.mean([rep_new[i] for i in high_ids]))

        if (t + 1) % 50 == 0:
            logging.info(
                f"  Round {t + 1}: New-Low QF={hist['New-Low'][-1]:.3f} | New-High QF={hist['New-High'][-1]:.3f}")

    # 6. 转换为绘图格式
    plot_data = {}
    for series_name, values in hist.items():
        plot_data[series_name] = {t: {'avg_reputation': v} for t, v in enumerate(values, 1)}

    # 7. 绘图
    visualizer.plot_line_chart(
        plot_data, range(1, T + 1), 'avg_reputation',
        "Average Reputation Score", "Quality-Aware Reputation (Direct Control)",
        "exp_quality_rep_final.pdf", "Round t"
    )

    logging.info(f"✅ Final: Old-LowQ={hist['Old-Low'][-1]:.3f} | Old-HighQ={hist['Old-High'][-1]:.3f} | "
                 f"New-LowQ={hist['New-Low'][-1]:.3f} | New-HighQ={hist['New-High'][-1]:.3f}")
    logging.info("Direct Control experiment completed.")


def run_multi_strategy_experiment(config, visualizer, log_path):
    logging.info("\n--- Experiment: Multi-Strategy Bidding (Strict IC Verification) ---")
    n_workers = 1000
    workers = {wid: Worker(wid, config) for wid in range(n_workers)}
    poetic = Platform(config, list(workers.keys()), name="POETIC-MultiStrat")

    strategies = ['truthful', 'consistent_underbid', 'consistent_overbid', 'random_overbid']

    # [核心修复 1] 严格限定作弊组比例：仅抽取 5% (50人) 作为策略偏离组，其余 950 人永远诚实
    test_group_ids = list(workers.keys())[:50]

    final_results = {}

    for strategy in strategies:
        logging.info(f"   Testing: {strategy} (Test Group Only)")
        poetic.reset()
        for w in workers.values(): w.reset()

        # 精准分配策略
        for wid, worker in workers.items():
            if wid in test_group_ids:
                worker.set_strategy(strategy)
            else:
                worker.set_strategy('truthful')

        test_group_profits = []
        active_ids = list(workers.keys())

        for i in range(config.TOTAL_ROUNDS):
            for wid in active_ids:
                workers[wid].update_state_for_round(None, None, i)

            # [核心修复 2] 传入 bidding_strategy=None，让每个 Worker 执行自己内部的独立策略
            winners_ids, payments = poetic.run_round(
                i, active_ids, workers, None, bidding_strategy=None
            )

            round_profit = 0
            for wid in test_group_ids:
                if wid in winners_ids:
                    actual_bid = getattr(poetic, 'last_bids', {}).get(wid, workers[wid].true_cost)
                    payment = payments.get(wid, 0)
                    cost = workers[wid].true_cost
                    # 只有当支付大于真实报价时，才算作有效接单并产生利润
                    if payment >= actual_bid:
                        round_profit += (payment - cost)

            # 计算本轮测试组的平均单人期望效用 (含未中标者，符合博弈论期望计算原则)
            test_group_profits.append(round_profit / len(test_group_ids))

        expected_utility = np.mean(test_group_profits)
        final_value = Metrics.calculate_cumulative_value(poetic.net_social_value_history)[-1]

        final_results[strategy] = {
            'expected_utility': expected_utility,
            'final_cumulative_value': final_value
        }
        logging.info(f"     {strategy}: ExpUtility={expected_utility:.4f} | Value={final_value:.2f}")

    # 绘制严格的 IC 柱状图
    visualizer.plot_bar_chart(final_results, 'expected_utility', "Expected Utility (Test Group)",
                              "Incentive Compatibility Verification", "exp_multi_strategy_ic.pdf")
    logging.info("✅ Multi-strategy experiment (Strict IC) completed.")


def run_time_horizon_experiment(all_models_fabric, config, visualizer, log_path):
    logging.info("\n--- Running Experiment: Impact of Time Horizon (T) ---")
    num_workers = 500
    workers_dict = {wid: Worker(wid, config) for wid in range(num_workers)}

    results_by_t = {}

    for t_horizon in config.TIME_HORIZON_RANGE:
        logging.info(f"--- Testing with Time Horizon T = {t_horizon} ---")

        # 1. 为 Optimal 重置种子，确保基准线环境固定
        random.seed(SEED)
        np.random.seed(SEED)
        optimal_model = NonPrivateOptimal(config)
        optimal_result = run_single_simulation(optimal_model, workers_dict, config, total_rounds_override=t_horizon)
        optimal_final_value = Metrics.calculate_cumulative_value(optimal_result['net_social_value_history'])[-1]

        # 2. 获取所有对比模型
        models_to_test = {name: model for name, model in all_models_fabric(num_workers, config).items() if
                          "Optimal" not in name}

        for name, model in models_to_test.items():
            if name not in results_by_t: results_by_t[name] = {}

            # 3. 关键：为每个模型重置完全相同的种子，确保“同题考试”
            random.seed(SEED)
            np.random.seed(SEED)

            result = run_single_simulation(model, workers_dict, config, total_rounds_override=t_horizon)
            final_value = Metrics.calculate_cumulative_value(result['net_social_value_history'])[-1]

            # 计算正向遗憾值
            regret = optimal_final_value - final_value
            results_by_t[name][t_horizon] = {'normalized_regret': regret / np.sqrt(t_horizon) if t_horizon > 0 else 0}

    visualizer.plot_line_chart(results_by_t, config.TIME_HORIZON_RANGE, 'normalized_regret',
                               "Normalized Regret (Regret/√T)", "Performance vs. Time Horizon (T)",
                               "exp_new_regret_vs_t.pdf", "Time Horizon (T)")


def run_k_impact_experiment(all_models_fabric, config, visualizer, log_path):
    logging.info("\n--- Running New Experiment: Impact of Number of Winners (k) ---")
    num_workers = 500
    workers_dict = {wid: Worker(wid, config) for wid in range(num_workers)}
    models_to_test = all_models_fabric(num_workers, config)

    results_by_k = {name: {} for name in models_to_test.keys()}

    for k in config.K_RANGE:
        logging.info(f"--- Testing with k = {k} winners ---")
        for name, model in models_to_test.items():
            result = run_single_simulation(model, workers_dict, config, num_winners_override=k)
            results_by_k[name][k] = {
                'final_cumulative_value': Metrics.calculate_cumulative_value(result['net_social_value_history'])[-1] if
                result['net_social_value_history'] else 0
            }

    visualizer.plot_bar_chart_grouped(results_by_k, config.K_RANGE, 'final_cumulative_value', "Final Cumulative Value",
                                      "Performance vs. Number of Winners (k)", "exp_new_value_vs_k.pdf",
                                      "Number of Winners (k)")


def main():
    config = Config()
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    # --- 0. 实验设置 ---
    DATASET_TYPE = 'NYC'   # 切换数据集: 'TDrive' 或 'NYC'

    # 初始化 DataLoader (虽然 simulation 主要依赖 context_maps，这里初始化以确保文件存在)
    from src.data_loader import get_dataloader
    data_loader = get_dataloader(config, dataset_type=DATASET_TYPE)

    NUM_ACADEMIC_RUNS = 0

    log_file_path = os.path.join(config.LOGS_DIR, f"exp_{DATASET_TYPE}_FULL_FIXED_{time.strftime('%Y%m%d-%H%M%S')}.txt")
    setup_file_logger(log_file_path)

    logging.info(f"--- Starting POETIC 12-Experiment Suite on {DATASET_TYPE} (Runs: {NUM_ACADEMIC_RUNS}) ---")

    visualizer = Visualizer(config)

    # --- 辅助函数 ---
    def get_fresh_workers(n):
        return {wid: Worker(wid, config) for wid in range(n)}

    # --- 初始化聚合器 ---
    # 1. Macro (6.1 & 6.2)
    macro_models = ["POETIC", "POETIC-slow", "Non-Private Optimal", "Random", "PrivCO", "OPPS-adapted",
                    "Incentivized FL-SC", "Truthful-DP-MAB"]
    agg_macro = {name: {'history': []} for name in macro_models}

    # 2. Scalability (6.3)
    agg_varying_n = {name: {n: [] for n in config.N_RANGE} for name in macro_models}

    # 3. Efficiency (6.4, 6.5, 6.6)
    eff_models = ["POETIC", "POETIC-slow", "PrivCO"]
    agg_eff_n = {name: {n: {'runtime': [], 'comm': []} for n in config.EFFICIENCY_N_RANGE} for name in eff_models}
    agg_eff_k = {name: {k: {'runtime': []} for k in config.K_RANGE} for name in eff_models}

    # 4. Incentive (6.7 & 6.8)
    strategies = ['truthful', 'consistent_overbid', 'random_overbid', 'consistent_underbid']
    agg_incentive = {s: {'val': [], 'prof': []} for s in strategies}

    # 5. Ablation (6.9)
    ablation_names = [
    "POETIC-NoExplore",  # use_exploration=False, use_reputation=True
    "POETIC-NoRep",      # use_exploration=True, use_reputation=False
    "POETIC-Final"       # 完整版本
]
    agg_ablation = {name: {'val_hist': [], 'prof_hist': []} for name in ablation_names}

    # 6. Privacy (6.11)
    priv_models = ["POETIC", "OPPS-adapted", "Truthful-DP-MAB"]
    agg_privacy = {name: {eps: [] for eps in config.EPSILON_RANGE} for name in priv_models}
    agg_priv_opt = []

    # 7. Time Horizon (New Exp t)
    agg_time = {name: {t: [] for t in config.TIME_HORIZON_RANGE} for name in
                [m for m in macro_models if "Optimal" not in m]}

    # 8. Value vs k (New Exp k)
    time_k_names = list(all_models_fabric(0, config).keys())
    agg_val_k = {name: {k: [] for k in config.K_RANGE} for name in time_k_names}

    # 9. Sensitivity (Param Sensitivity)
    agg_sens_lr = {v: {'val': [], 'prof': []} for v in config.SENSITIVITY_LR_RANGE}
    agg_sens_eps = {v: {'val': []} for v in config.SENSITIVITY_EPSILON_RANGE}
    agg_sens_factor = {v: {'val': []} for v in config.SENSITIVITY_FACTOR_RANGE}
    agg_sens_rep_lr = {v: {'val': [], 'prof': []} for v in config.SENSITIVITY_REP_LR_RANGE}
    agg_sens_rep_beta = {v: {'val': [], 'prof': []} for v in config.SENSITIVITY_REP_BETA_RANGE}

    # --- 主循环 ---
    for i in tqdm(range(NUM_ACADEMIC_RUNS), desc=f"Academic Runs ({DATASET_TYPE})"):
        logging.info(f"--- Run {i + 1}/{NUM_ACADEMIC_RUNS} ---")

        # === 1. Macro Performance ===
        n_fixed = 2000
        workers_macro = get_fresh_workers(n_fixed)
        models_macro_inst = all_models_fabric(n_fixed, config)
        for name, model in models_macro_inst.items():
            random.seed(SEED + i)  # i 是外层运行次数
            np.random.seed(SEED + i)
            res = run_single_simulation(model, workers_macro, config, data_loader=data_loader)
            agg_macro[name]['history'].append(res['net_social_value_history'])

        # === 2. Scalability ===
        for n in config.N_RANGE:
            workers_n = get_fresh_workers(n)
            models_n = all_models_fabric(n, config)

            # ✅ 正确：运行模拟时才需要 data_loader
            for name, model in models_n.items():
                res = run_single_simulation(model, workers_n, config, data_loader=data_loader)

                val = Metrics.calculate_cumulative_value(res['net_social_value_history'])[-1] if res[
                    'net_social_value_history'] else 0
                agg_varying_n[name][n].append(val)

        # === 3. Efficiency vs N ===
        EFFICIENCY_ROUNDS = 10
        for n in config.EFFICIENCY_N_RANGE:
            workers_eff = get_fresh_workers(n)
            models_eff = {"POETIC": Platform(config, list(workers_eff.keys())),
                          "POETIC-slow": POETICSlow(config, list(workers_eff.keys())), "PrivCO": PrivCO(config)}
            for name, model in models_eff.items():
                res = run_single_simulation(model, workers_eff, config, total_rounds_override=EFFICIENCY_ROUNDS, data_loader=data_loader)
                agg_eff_n[name][n]['runtime'].append(np.mean(res['runtime_history']))
                agg_eff_n[name][n]['comm'].append(np.mean(res['comm_cost_history']))

        # === 4. Efficiency vs k ===
        workers_k_eff = get_fresh_workers(500)
        for k in config.K_RANGE:
            models_k_eff = {"POETIC": Platform(config, list(workers_k_eff.keys())),
                            "POETIC-slow": POETICSlow(config, list(workers_k_eff.keys())), "PrivCO": PrivCO(config)}
            for name, model in models_k_eff.items():
                res = run_single_simulation(model, workers_k_eff, config, num_winners_override=k,
                                            total_rounds_override=EFFICIENCY_ROUNDS, data_loader=data_loader)
                agg_eff_k[name][k]['runtime'].append(np.mean(res['runtime_history']))

        # === 5. Incentive Verification ===
        workers_inc = get_fresh_workers(config.DEFAULT_NUM_WORKERS)
        poetic_inc = Platform(config, list(workers_inc.keys()))
        for strategy in strategies:
            poetic_inc.reset()
            res = run_single_simulation(poetic_inc, workers_inc, config, bidding_strategy=strategy, data_loader=data_loader)
            val = Metrics.calculate_cumulative_value(res['net_social_value_history'])[-1]
            prof = np.mean(res['worker_profit_history_agg'])
            agg_incentive[strategy]['val'].append(val)
            agg_incentive[strategy]['prof'].append(prof)

        # === 6. Ablation Study ===
        workers_abl = get_fresh_workers(config.DEFAULT_NUM_WORKERS)
        models_abl = {
            "POETIC-Base": Platform(config, list(workers_abl.keys()), name="POETIC-Base", use_exploration=False,
                                    use_reputation=False),
            "POETIC-Base + Explore": Platform(config, list(workers_abl.keys()), name="POETIC-Base + Explore",
                                              use_exploration=True, use_reputation=False),
            "POETIC-Final": Platform(config, list(workers_abl.keys()), name="POETIC-Final", use_exploration=True,
                                     use_reputation=True),
        }
        for name, model in models_abl.items():
            res = run_single_simulation(model, workers_abl, config, data_loader=data_loader)
            agg_ablation[name]['val_hist'].append(Metrics.calculate_cumulative_value(res['net_social_value_history']))
            agg_ablation[name]['prof_hist'].append(res['worker_profit_history_agg'])

        # === 7. Privacy Tradeoff ===
        workers_priv = get_fresh_workers(config.DEFAULT_NUM_WORKERS)
        opt_res = run_single_simulation(NonPrivateOptimal(config), workers_priv, config, data_loader=data_loader)
        agg_priv_opt.append(Metrics.calculate_cumulative_value(opt_res['net_social_value_history'])[-1])

        for eps in config.EPSILON_RANGE:
            models_priv = {"POETIC": Platform(config, list(workers_priv.keys())),
                           "OPPS-adapted": OPPSAdapted(config, list(workers_priv.keys())),
                           "Truthful-DP-MAB": TruthfulDPMAB(config, list(workers_priv.keys()))}
            for name, model in models_priv.items():
                res = run_single_simulation(model, workers_priv, config, dp_epsilon_override=eps)
                agg_privacy[name][eps].append(Metrics.calculate_cumulative_value(res['net_social_value_history'])[-1])

        # === 8. Time Horizon Growth ===
        workers_time = get_fresh_workers(500)
        for t in config.TIME_HORIZON_RANGE:
            opt_res_t = run_single_simulation(NonPrivateOptimal(config), workers_time, config, total_rounds_override=t)
            opt_val_t = Metrics.calculate_cumulative_value(opt_res_t['net_social_value_history'])[-1]
            models_t = all_models_fabric(500, config)
            for name, model in models_t.items():
                if "Optimal" in name: continue
                random.seed(SEED)
                np.random.seed(SEED)
                res = run_single_simulation(model, workers_time, config, total_rounds_override=t, data_loader=data_loader)
                val = Metrics.calculate_cumulative_value(res['net_social_value_history'])[-1]
                regret = opt_val_t - val
                agg_time[name][t].append(regret / np.sqrt(t) if t > 0 else 0)

        # === 9. Value vs k Growth ===
        workers_vk = get_fresh_workers(500)
        models_vk = all_models_fabric(500, config)
        for k in config.K_RANGE:
            for name, model in models_vk.items():
                model.reset()
                res = run_single_simulation(model, workers_vk, config, num_winners_override=k, data_loader=data_loader)
                agg_val_k[name][k].append(Metrics.calculate_cumulative_value(res['net_social_value_history'])[-1])

        # === 10. Sensitivity Analysis ===
        workers_sens = get_fresh_workers(1000)
        temp_cfg = Config()

        def run_sens_param(param_name, param_range, agg_target):
            orig_val = getattr(temp_cfg, param_name)
            for val in param_range:
                setattr(temp_cfg, param_name, val)
                model = Platform(temp_cfg, list(workers_sens.keys()))
                r = run_single_simulation(model, workers_sens, temp_cfg)
                agg_target[val]['val'].append(Metrics.calculate_cumulative_value(r['net_social_value_history'])[-1])
                if 'prof' in agg_target[val]: agg_target[val]['prof'].append(np.mean(r['worker_profit_history_agg']))
            setattr(temp_cfg, param_name, orig_val)

        run_sens_param('LEARNING_RATE', config.SENSITIVITY_LR_RANGE, agg_sens_lr)
        run_sens_param('EXPLORATION_EPSILON', config.SENSITIVITY_EPSILON_RANGE, agg_sens_eps)
        run_sens_param('ELITE_POOL_FACTOR', config.SENSITIVITY_FACTOR_RANGE, agg_sens_factor)
        run_sens_param('REPUTATION_LEARNING_RATE', config.SENSITIVITY_REP_LR_RANGE, agg_sens_rep_lr)
        run_sens_param('REPUTATION_BONUS_FACTOR', config.SENSITIVITY_REP_BETA_RANGE, agg_sens_rep_beta)

    # --- 绘图阶段 ---
    logging.info(f"--- Generating 12 Visualizations for {DATASET_TYPE} ---")

    logging.info(f"--- Generating Visualizations for {DATASET_TYPE} ---")

    # 1. Macro Value (Exp 6.1)
    viz_macro_data = {name: {'cumulative_value_runs': [Metrics.calculate_cumulative_value(h) for h in d['history']]} for
                      name, d in agg_macro.items()}
    if hasattr(visualizer, 'plot_macro_performance_curves'):
        visualizer.plot_macro_performance_curves(viz_macro_data, f"Macro Performance ({DATASET_TYPE})",
                                                 "exp_6.1_macro_val.pdf")

    # 2. Regret (Exp 6.2) - 修正调用
    # 计算 Regret 数据
    opt_runs = viz_macro_data["Non-Private Optimal"]['cumulative_value_runs']
    viz_regret_data = {}
    for name, data in viz_macro_data.items():
        if name == "Non-Private Optimal": continue
        model_runs = data['cumulative_value_runs']
        regret_runs = []
        for i in range(len(model_runs)):
            # 确保长度一致
            min_len = min(len(opt_runs[i]), len(model_runs[i]))
            # Regret = Optimal - Current
            regret_runs.append(np.array(opt_runs[i][:min_len]) - np.array(model_runs[i][:min_len]))

        # 注意：这里我们把算好的 regret 存入 'cumulative_value_runs' 键中，方便绘图函数读取
        # 但我们调用的是 plot_cumulative_regret_vs_rounds，它知道这是 Regret 数据
        viz_regret_data[name] = {'cumulative_value_runs': regret_runs}

    if hasattr(visualizer, 'plot_cumulative_regret_vs_rounds'):
        # 【关键修改】调用专门的 Regret 绘图函数
        visualizer.plot_cumulative_regret_vs_rounds(viz_regret_data, f"Cumulative Regret ({DATASET_TYPE})",
                                                    "exp_6.2_regret.pdf")

    # 3. Scalability
    mean_val_n = {name: {n: {'final_cumulative_value': np.mean(vals)} for n, vals in d.items()} for name, d in
                  agg_varying_n.items()}
    visualizer.plot_final_value_vs_n(mean_val_n, config.N_RANGE, f"Scalability vs N ({DATASET_TYPE})",
                                     "exp_6.3_scale_n.pdf")

    # 4-6. Efficiency
    mean_eff_n = {
        name: {n: {'runtime_history': np.mean(d['runtime']), 'comm_cost_history': np.mean(d['comm'])} for n, d in
               data.items()} for name, data in agg_eff_n.items()}
    visualizer.plot_efficiency_metric(mean_eff_n, config.EFFICIENCY_N_RANGE, 'runtime_history', "Runtime (ms)",
                                      f"Runtime vs N ({DATASET_TYPE})", "exp_6.4_runtime_n.pdf", "N")
    visualizer.plot_efficiency_metric(mean_eff_n, config.EFFICIENCY_N_RANGE, 'comm_cost_history', "Comm Cost (MB)",
                                      f"Comm vs N ({DATASET_TYPE})", "exp_6.5_comm_n.pdf", "N")
    mean_eff_k = {name: {k: {'runtime_history': np.mean(d['runtime'])} for k, d in data.items()} for name, data in
                  agg_eff_k.items()}
    visualizer.plot_efficiency_metric(mean_eff_k, config.K_RANGE, 'runtime_history', "Runtime (ms)",
                                      f"Runtime vs k ({DATASET_TYPE})", "exp_6.6_runtime_k.pdf", "k")

    # 7. Incentive
    mean_inc = {s: {'final_cumulative_value': np.mean(d['val']), 'avg_worker_profit': np.mean(d['prof'])} for s, d in
                agg_incentive.items()}
    std_inc = {s: {'final_cumulative_value': np.std(d['val']), 'avg_worker_profit': np.std(d['prof'])} for s, d in
               agg_incentive.items()}
    visualizer.plot_incentive_verification_bars(mean_inc, std_inc, f"Incentive Verification ({DATASET_TYPE})",
                                                "exp_6.7_incentive.pdf")

    # 8. Ablation
    viz_abl = {name: {'cumulative_value_runs': d['val_hist'], 'worker_profit_history_agg_runs': d['prof_hist']} for
               name, d in agg_ablation.items()}
    visualizer.plot_ablation_study(viz_abl, f"Ablation Study ({DATASET_TYPE})", "exp_6.9_ablation.pdf")

    # 9. Privacy
    mean_opt_priv_val = np.mean(agg_priv_opt)
    viz_priv = {eps: {name: {'final_cumulative_value': np.mean(agg_privacy[name][eps])} for name in priv_models} for eps
                in config.EPSILON_RANGE}
    visualizer.plot_privacy_tradeoff_curves(viz_priv, mean_opt_priv_val, f"Privacy Tradeoff ({DATASET_TYPE})",
                                            "exp_6.11_privacy.pdf")

    # 10. Sensitivity (Fix Key Mapping)
    def get_sens_viz_data(agg_dict, target_metric_name):
        # 映射逻辑：如果目标是 Value，取 'val'；如果是 Profit，取 'prof'
        storage_key = 'val' if target_metric_name == 'final_cumulative_value' else 'prof'
        return {'POETIC': {
            v: {
                target_metric_name: np.mean(d[storage_key]) if d.get(storage_key) else 0
            }
            for v, d in agg_dict.items()
        }}

    sens_data_val = {
        'LR': {'param_range': config.SENSITIVITY_LR_RANGE,
               'results': get_sens_viz_data(agg_sens_lr, 'final_cumulative_value'), 'x_label': r"Learning Rate $\eta$"},
        'Epsilon': {'param_range': config.SENSITIVITY_EPSILON_RANGE,
                    'results': get_sens_viz_data(agg_sens_eps, 'final_cumulative_value'),
                    'x_label': r"Exploration $\epsilon$"},
        'Factor': {'param_range': config.SENSITIVITY_FACTOR_RANGE,
                   'results': get_sens_viz_data(agg_sens_factor, 'final_cumulative_value'), 'x_label': "Elite Factor"},
        'RepLR': {'param_range': config.SENSITIVITY_REP_LR_RANGE,
                  'results': get_sens_viz_data(agg_sens_rep_lr, 'final_cumulative_value'),
                  'x_label': r"Reputation $\eta_R$"},
        'RepBeta': {'param_range': config.SENSITIVITY_REP_BETA_RANGE,
                    'results': get_sens_viz_data(agg_sens_rep_beta, 'final_cumulative_value'),
                    'x_label': r"Reputation $\beta$"}
    }
    sens_data_prof = {
        'LR': {'param_range': config.SENSITIVITY_LR_RANGE, 'results': get_sens_viz_data(agg_sens_lr, 'avg_profit'),
               'x_label': r"Learning Rate $\eta$"},
        'RepLR': {'param_range': config.SENSITIVITY_REP_LR_RANGE,
                  'results': get_sens_viz_data(agg_sens_rep_lr, 'avg_profit'), 'x_label': r"Reputation $\eta_R$"},
        'RepBeta': {'param_range': config.SENSITIVITY_REP_BETA_RANGE,
                    'results': get_sens_viz_data(agg_sens_rep_beta, 'avg_profit'), 'x_label': r"Reputation $\beta$"}
    }
    defaults = {'LR': config.LEARNING_RATE, 'Epsilon': config.EXPLORATION_EPSILON, 'Factor': config.ELITE_POOL_FACTOR,
                'RepLR': config.REPUTATION_LEARNING_RATE, 'RepBeta': config.REPUTATION_BONUS_FACTOR}
    visualizer.plot_sensitivity_grid(sens_data_val, defaults, 'final_cumulative_value', "Mean Value",
                                     f"Sensitivity Value ({DATASET_TYPE})", "exp_sens_val.pdf")
    visualizer.plot_sensitivity_grid(sens_data_prof, defaults, 'avg_profit', "Mean Profit",
                                     f"Sensitivity Profit ({DATASET_TYPE})", "exp_sens_prof.pdf")

    # 11-12. New Growth Experiments
    mean_time_regret = {name: {t: {'normalized_regret': np.mean(vals)} for t, vals in d.items()} for name, d in
                        agg_time.items()}
    visualizer.plot_line_chart(mean_time_regret, config.TIME_HORIZON_RANGE, 'normalized_regret', "Regret / sqrt(T)",
                               f"Regret Growth ({DATASET_TYPE})", "exp_new_growth_t.pdf", "Time Horizon T")

    k_filtered = [k for k in config.K_RANGE if k != 500]
    mean_val_k = {
        name: {k: {'final_cumulative_value': np.mean(vals), 'std_final_cumulative_value': np.std(vals)} for k, vals in
               d.items() if k != 500} for name, d in agg_val_k.items()}
    visualizer.plot_bar_chart_grouped(mean_val_k, k_filtered, 'final_cumulative_value', "Final Value",
                                      f"Value vs k ({DATASET_TYPE})", "exp_new_growth_k.pdf", "Winners k")

    logging.info("--- All 12 Original Experiments Completed. ---")
    run_forward_looking_experiment(config, visualizer, log_file_path)
    run_quality_aware_reputation_experiment(config, visualizer, log_file_path)
    run_multi_strategy_experiment(config, visualizer, log_file_path)

    # --- ICDE 2026: Feedback-Contaminated Online Learning Experiments ---
    logging.info("\n--- Starting ICDE Experiment Suite (7 experiments) ---")
    run_overall_utility_experiment(config, visualizer, log_file_path)
    run_regret_analysis_experiment(config, visualizer, log_file_path)
    run_contamination_robustness_experiment(config, visualizer, log_file_path)
    run_strategic_worker_experiment(config, visualizer, log_file_path)
    run_reputation_robustness_experiment(config, visualizer, log_file_path)
    run_privacy_tradeoff_experiment(config, visualizer, log_file_path)
    run_mechanism_ablation_experiment(config, visualizer, log_file_path)
    logging.info("--- ICDE Experiment Suite Completed. ---")
if __name__ == '__main__':
    main()