"""
run_official_table3.py
======================
Official master execution script to compute full Table III metrics across NYC and TDrive datasets.
Runs T=5000, N=2000, num_runs=3 (or CLI flags).
"""

import os
import sys
import random
import logging
import json
import time
import numpy as np

from config import Config
from data_loader import get_dataloader
from worker import Worker
from poe_platform import POEPlatform
from baselines import NonPrivateOptimal, OPPSAdapted, TruthfulDPMAB
from metrics import Metrics

SEED = 42
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def build_models(config, N):
    ids = list(range(N))
    return {
        'POETIC (30% Adv)': POEPlatform(
            config, ids, name='POETIC',
            use_reputation=True, use_exploration=True,
            use_decoupled_learning=True, use_stability_reputation=True),
        'POETIC (Clean r=0%)': POEPlatform(
            config, ids, name='POETIC Clean',
            use_reputation=True, use_exploration=True,
            use_decoupled_learning=True, use_stability_reputation=True),
        'w/o Phi (30% Adv)': POEPlatform(
            config, ids, name='w/o Phi',
            use_reputation=True, use_exploration=True,
            use_decoupled_learning=False, use_stability_reputation=True),
        'w/o StabRep (30% Adv)': POEPlatform(
            config, ids, name='w/o StabRep',
            use_reputation=True, use_exploration=True,
            use_decoupled_learning=True, use_stability_reputation=False),
        'w/o Both (30% Adv)': POEPlatform(
            config, ids, name='w/o Both',
            use_reputation=True, use_exploration=True,
            use_decoupled_learning=False, use_stability_reputation=False),
        'Truthful-DP-MAB (30% Adv)': TruthfulDPMAB(config, ids),
        'OPPS-adapted (30% Adv)': OPPSAdapted(config, ids),
        'Greedy Selection (30% Adv)': POEPlatform(
            config, ids, name='Greedy Selection',
            use_reputation=False, use_exploration=False,
            use_decoupled_learning=False, use_stability_reputation=False),
    }

def run_one_sim(model, workers, config, data_loader, T):
    if hasattr(model, 'reset'):
        model.reset()
    for w in workers.values():
        w.reset()

    dl_keys = []
    if data_loader and hasattr(data_loader, 'context_maps') and data_loader.context_maps:
        dl_keys = list(data_loader.context_maps.keys())

    for i in range(T):
        current_context = None
        if dl_keys:
            random_key = random.choice(dl_keys)
            grid_id, hour = random_key[0], random_key[1]
            density = data_loader.context_maps.get((grid_id, hour, 'density'), 1.0)
            importance = data_loader.context_maps.get((grid_id, hour, 'importance'), 1.0)
            current_context = {'density': density, 'importance': importance}

        for w in workers.values():
            w.update_state_for_round(current_context, data_loader, i)

        active = [wid for wid, w in workers.items() if w.online_status > 0.5]
        if len(active) < config.DEFAULT_NUM_WINNERS:
            if hasattr(model, '_record_empty_round'):
                model._record_empty_round()
            continue

        model.run_round(round_id=i, active_workers_ids=active,
                        workers_dict=workers, data_loader=data_loader)

        if (i + 1) % (T // 5) == 0 or (i + 1) == T:
            tmp_nsv = getattr(model, 'net_social_value_history', [])
            tmp_val = Metrics.calculate_cumulative_value(tmp_nsv)[-1] if tmp_nsv else 0.0
            tmp_ee = getattr(model, 'estimation_error_history', [])
            tmp_mean_ee = Metrics.calculate_mean_estimation_error(tmp_ee) if tmp_ee else 0.0
            logging.info(f"    [Round {i+1}] {model.name}: Val={tmp_val:>13,.1f}  EstErr={tmp_mean_ee:>7.2f}")

    nsv = getattr(model, 'net_social_value_history', [])
    cv  = Metrics.calculate_cumulative_value(nsv)
    rt  = getattr(model, 'runtime_history', [0])
    cc  = getattr(model, 'comm_cost_history', [0])
    return {
        'net_social_value_history': nsv,
        'estimation_error_history': getattr(model, 'estimation_error_history', []),
        'cumulative_value':         cv,
        'final_cumulative_value':   cv[-1] if cv else 0.0,
        'mean_runtime':             float(np.mean(rt)) if rt else 0.0,
        'mean_comm_cost':           float(np.mean(cc)) if cc else 0.0,
    }

def run_table3_dataset(dataset_name, config, T, N, num_runs, adv_ratio=0.3):
    logging.info(f"\n{'='*75}")
    logging.info(f"  Dataset: {dataset_name}  |  T={T}  N={N}  runs={num_runs}  adv_ratio={adv_ratio}")
    logging.info(f"{'='*75}")

    data_loader = get_dataloader(config, dataset_type=dataset_name)

    # First run Non-Private Optimal baseline for Regret calculation
    opt_runs = []
    for ri in range(num_runs):
        random.seed(SEED + ri)
        np.random.seed(SEED + ri)
        opt_workers = {wid: Worker(wid, config) for wid in range(N)}
        opt_model = NonPrivateOptimal(config)
        opt_runs.append(run_one_sim(opt_model, opt_workers, config, data_loader, T))
    opt_cv_mean = float(np.mean([r['final_cumulative_value'] for r in opt_runs]))
    logging.info(f"  Non-Private Optimal Val = {opt_cv_mean:15,.1f}")

    models_dict = build_models(config, N)
    dataset_results = {}

    for model_label, model in models_dict.items():
        runs = []
        is_clean = 'Clean' in model_label
        current_adv_ratio = 0.0 if is_clean else adv_ratio

        for ri in range(num_runs):
            random.seed(SEED + ri)
            np.random.seed(SEED + ri)
            workers = {wid: Worker(wid, config) for wid in range(N)}
            if current_adv_ratio > 0:
                n_adv = int(N * current_adv_ratio)
                for wid in range(n_adv // 2):
                    workers[wid].set_strategy('utility_spoofer')
                for wid in range(n_adv // 2, n_adv):
                    workers[wid].set_strategy('oscillatory')

            res = run_one_sim(model, workers, config, data_loader, T)
            runs.append(res)

        cv_mean     = float(np.mean([r['final_cumulative_value'] for r in runs]))
        ee_mean     = float(np.mean([Metrics.calculate_mean_estimation_error(r['estimation_error_history']) for r in runs]))
        regret_mean = opt_cv_mean - cv_mean
        rt_mean     = float(np.mean([r['mean_runtime'] for r in runs]))
        cc_mean     = float(np.mean([r['mean_comm_cost'] for r in runs]))

        dataset_results[model_label] = {
            'final_utility': cv_mean,
            'regret':        regret_mean,
            'mean_esterr':   ee_mean,
            'runtime_ms':    rt_mean,
            'comm_mb':       cc_mean,
        }
        logging.info(f"  {model_label:30s}  Val={cv_mean:>13,.1f}  EstErr={ee_mean:>7.2f}  Regret={regret_mean:>13,.1f}  RT={rt_mean:>6.1f}ms")

    return dataset_results

def main():
    quick  = '--quick' in sys.argv
    config = Config()

    if quick:
        T, N, num_runs = 500, 500, 1
        logging.info("*** QUICK TEST MODE (T=500, N=500, 1 run) ***")
    else:
        T, N, num_runs = config.TOTAL_ROUNDS, config.DEFAULT_NUM_WORKERS, 3
        logging.info(f"*** FULL OFFICIAL MODE (T={T}, N={N}, {num_runs} runs) ***")

    os.makedirs('results', exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    log_path = f'results/official_table3_{timestamp}.log'
    fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(fh)

    output = {'settings': {'T': T, 'N': N, 'num_runs': num_runs, 'quick': quick}}

    for dataset in ('NYC', 'TDrive'):
        res = run_table3_dataset(dataset, config, T, N, num_runs)
        output[dataset] = res

    json_path = f'results/official_table3_{timestamp}.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logging.info("\n" + "="*85)
    logging.info("  FINAL OFFICIAL TABLE III SUMMARY")
    logging.info("="*85)
    for dataset in ('NYC', 'TDrive'):
        logging.info(f"\n  Dataset: {dataset}")
        logging.info(f"  {'Model':30s}  {'Utility (Val)':>15}  {'Regret':>15}  {'Mean EstErr':>12}  {'Runtime(ms)':>12}")
        logging.info(f"  {'-'*30}  {'-'*15}  {'-'*15}  {'-'*12}  {'-'*12}")
        for model_label, data in output[dataset].items():
            logging.info(f"  {model_label:30s}  {data['final_utility']:>15,.1f}  "
                         f"{data['regret']:>15,.1f}  {data['mean_esterr']:>12.2f}  "
                         f"{data['runtime_ms']:>12.1f}")

    logging.info(f"\n✅ Official Table III JSON saved to: {json_path}")
    logging.info(f"✅ Official Log saved to: {log_path}")

if __name__ == '__main__':
    main()
