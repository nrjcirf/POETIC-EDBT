# run_stability_10runs.py
"""
Audited 10-Run Stability & Concept Drift Benchmark for ICDE Revision
===================================================================
Runs 10 independent repetitions (seeds 42..51) for all 8 baselines:
  - POETIC
  - POETIC-slow
  - Non-Private Optimal
  - Truthful-DP-MAB
  - OPPS-adapted
  - PrivCO
  - Greedy Selection
  - Random Selection

Computes:
  - (a) Utility Convergence trajectory (every 500 rounds)
  - (b) Dynamic Regret trajectory (every 500 rounds)
  - (c) True Rolling Variance trajectory (sliding window W=50, every 500 rounds)

Outputs:
  - results/stability_10runs_official.json
  - paper/fig4_stability.pdf
  - conference_latex_template/fig4_stability.pdf
"""

import os
import sys
import time
import json
import random
import pickle
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from config import Config
from worker import Worker
from poe_platform import POEPlatform
from baselines import (NonPrivateOptimal, POETICSlow, PrivCO,
                       OPPSAdapted, TruthfulDPMAB, RandomModel)
from metrics import Metrics

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9.0,
    'figure.dpi': 300,
    'lines.linewidth': 2.0,
    'axes.grid': True,
    'grid.linestyle': ':',
    'grid.alpha': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

def pre_sample_contexts(context_maps_path, T=5000, seed=42):
    with open(context_maps_path, 'rb') as f:
        cm = pickle.load(f)
    keys = list(set((k[0], k[1]) for k in cm.keys()))
    rng = random.Random(seed)
    contexts = []
    for _ in range(T):
        k = rng.choice(keys)
        d = cm.get((k[0], k[1], 'density'), 1.0)
        imp = cm.get((k[0], k[1], 'importance'), 1.0)
        contexts.append((d, imp))
    return contexts

def create_model(model_name, config, N):
    ids = list(range(N))
    if model_name == 'POETIC':
        return POEPlatform(config, ids, name='POETIC',
                           use_reputation=True, use_exploration=True,
                           use_decoupled_learning=True, use_stability_reputation=True)
    elif model_name == 'POETIC-slow':
        return POETICSlow(config, ids)
    elif model_name == 'Non-Private Optimal':
        return NonPrivateOptimal(config)
    elif model_name == 'PrivCO':
        return PrivCO(config)
    elif model_name == 'OPPS-adapted':
        return OPPSAdapted(config, ids)
    elif model_name == 'Truthful-DP-MAB':
        return TruthfulDPMAB(config, ids)
    elif model_name == 'Greedy Selection':
        return POEPlatform(config, ids, name='Greedy Selection',
                           use_reputation=False, use_exploration=False,
                           use_decoupled_learning=False, use_stability_reputation=False)
    elif model_name == 'Random Selection':
        return RandomModel(config)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def run_single_stability_run(model_name, seed, T, N, adv_ratio, contexts, config):
    random.seed(seed)
    np.random.seed(seed)

    workers = {wid: Worker(wid, config) for wid in range(N)}
    if adv_ratio > 0:
        n_adv = int(N * adv_ratio)
        for wid in range(n_adv // 2):
            workers[wid].set_strategy('utility_spoofer')
        for wid in range(n_adv // 2, n_adv):
            workers[wid].set_strategy('oscillatory')

    model = create_model(model_name, config, N)

    step = 500
    checkpoints = list(range(step, T + 1, step))
    cp_vals = []

    for r in range(T):
        d, imp = contexts[r]
        current_context = {'density': d, 'importance': imp}

        for w in workers.values():
            w.update_state_for_round(current_context, None, r)

        active = [wid for wid, w in workers.items() if w.online_status > 0.5]
        if len(active) < config.DEFAULT_NUM_WINNERS:
            if hasattr(model, '_record_empty_round'):
                model._record_empty_round()
            continue

        model.run_round(r, active, workers, data_loader=None)

        if (r + 1) in checkpoints:
            nsv_so_far = model.net_social_value_history
            cval = float(np.sum(nsv_so_far)) if nsv_so_far else 0.0
            cp_vals.append(cval)

    nsv = np.array(model.net_social_value_history)
    cum_val = np.cumsum(nsv)

    W = 50
    r_var = Metrics.calculate_rolling_variance(list(nsv), window=W)

    cp_vars = [float(r_var[i - 1]) for i in checkpoints]

    return {
        'final_val': float(cum_val[-1]) if len(cum_val) else 0.0,
        'cp_vals': cp_vals,
        'cp_vars': cp_vars,
    }

def _worker_task(args):
    model_name, seed, T, N, adv_ratio, contexts, config = args
    res = run_single_stability_run(model_name, seed, T, N, adv_ratio, contexts, config)
    return model_name, seed, res

def plot_figure4(summary, T, out_dirs=['paper', 'conference_latex_template']):
    checkpoints = list(range(500, T + 1, 500))

    MODELS_ORDER = ['POETIC', 'POETIC-slow', 'Non-Private Optimal', 'Truthful-DP-MAB', 'OPPS-adapted', 'PrivCO', 'Greedy Selection', 'Random Selection']
    STYLES = {
        'POETIC':              {'color': '#003366', 'ls': '-',  'marker': 'o', 'label': 'POETIC (Ours)'},
        'POETIC-slow':         {'color': '#0055A4', 'ls': '--', 'marker': 's', 'label': 'POETIC-slow'},
        'Non-Private Optimal': {'color': 'black',   'ls': '--', 'marker': '',  'label': 'Oracle (Opt)'},
        'Truthful-DP-MAB':     {'color': '#8e44ad', 'ls': ':',  'marker': 'v', 'label': 'Truthful-DP-MAB'},
        'OPPS-adapted':        {'color': '#27ae60', 'ls': '-.', 'marker': '^', 'label': 'OPPS-adapted'},
        'PrivCO':              {'color': '#c0392b', 'ls': '-.', 'marker': 'x', 'label': 'PrivCO'},
        'Greedy Selection':    {'color': '#16a085', 'ls': '--', 'marker': 'd', 'label': 'Greedy'},
        'Random Selection':    {'color': '#7f8c8d', 'ls': ':',  'marker': '.', 'label': 'Random'},
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # (a) Utility
    ax = axes[0]
    for m in MODELS_ORDER:
        if m not in summary: continue
        st = STYLES.get(m, {'color': 'gray', 'ls': '-', 'marker': '', 'label': m})
        means = np.array(summary[m]['cp_vals_mean']) / 1e6
        ax.plot(checkpoints, means, color=st['color'], ls=st['ls'], marker=st['marker'],
                label=st['label'], markevery=2, lw=2.0)
    ax.set_title('(a) Utility Convergence')
    ax.set_xlabel('Round $t$')
    ax.set_ylabel('Cumulative Utility ($10^6$)')
    ax.axvline(2500, color='gray', ls=':', alpha=0.7)
    ax.text(2550, ax.get_ylim()[0] + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.1, 'Concept Drift',
            fontsize=9, color='gray', style='italic')
    ax.legend(loc='upper left', frameon=True, fontsize=8.5)

    # (b) Regret = Oracle - Model
    ax = axes[1]
    oracle_vals = np.array(summary['Non-Private Optimal']['cp_vals_mean']) / 1e6
    for m in MODELS_ORDER:
        if m == 'Non-Private Optimal' or m not in summary: continue
        st = STYLES.get(m, {'color': 'gray', 'ls': '-', 'marker': '', 'label': m})
        regrets = oracle_vals - (np.array(summary[m]['cp_vals_mean']) / 1e6)
        ax.plot(checkpoints, regrets, color=st['color'], ls=st['ls'], marker=st['marker'],
                label=st['label'], markevery=2, lw=2.0)
    ax.set_title('(b) Dynamic Regret')
    ax.set_xlabel('Round $t$')
    ax.set_ylabel('Cumulative Regret ($10^6$)')
    ax.axvline(2500, color='gray', ls=':', alpha=0.7)
    ax.text(2550, ax.get_ylim()[0] + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.1, 'Concept Drift',
            fontsize=9, color='gray', style='italic')
    ax.legend(loc='upper left', frameon=True, fontsize=8.5)

    # (c) True Rolling Variance
    ax = axes[2]
    for m in MODELS_ORDER:
        if m not in summary: continue
        st = STYLES.get(m, {'color': 'gray', 'ls': '-', 'marker': '', 'label': m})
        r_var = np.array(summary[m]['cp_vars_mean']) / 1e6
        ax.plot(checkpoints, r_var, color=st['color'], ls=st['ls'], marker=st['marker'],
                label=st['label'], markevery=2, lw=2.0)
    ax.set_title('(c) Learning Variance Over Time')
    ax.set_xlabel('Round $t$')
    ax.set_ylabel(r'Rolling Variance ($10^6, W=50$)')
    ax.axvline(2500, color='gray', ls=':', alpha=0.7)
    ax.text(2550, ax.get_ylim()[0] + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.1, 'Concept Drift',
            fontsize=9, color='gray', style='italic')
    ax.legend(loc='upper left', frameon=True, fontsize=8.5)

    plt.tight_layout()
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        p_pdf = os.path.join(d, 'fig4_stability.pdf')
        p_png = os.path.join(d, 'fig4_stability.png')
        plt.savefig(p_pdf, bbox_inches='tight')
        plt.savefig(p_png, bbox_inches='tight', dpi=300)
        print(f"Saved Figure 4 to: {p_pdf}")
    plt.close()

def main():
    config = Config()
    T = 5000
    N = 1000
    adv_ratio = 0.3
    num_runs = 10
    seeds = [42 + i for i in range(num_runs)]

    maps_file = os.path.join(config.DATA_DIR, config.NYC_CONTEXT_FILENAME)
    print(f"Pre-sampling context maps from {maps_file}...")
    t0_pre = time.time()
    seed_contexts = {s: pre_sample_contexts(maps_file, T=T, seed=s) for s in seeds}
    print(f"Pre-sampled contexts for {num_runs} seeds in {time.time()-t0_pre:.2f} s")

    models = ['Non-Private Optimal', 'POETIC', 'POETIC-slow', 'PrivCO',
              'Truthful-DP-MAB', 'OPPS-adapted', 'Greedy Selection', 'Random Selection']
    results = {m: {'final_val': [], 'cp_vals': [], 'cp_vars': []} for m in models}

    print("\n" + "="*80)
    print(f"  STARTING 10-RUN STABILITY STUDY (T={T}, N={N}, Adv={adv_ratio:.0%})")
    print("="*80)

    tasks = []
    for m in models:
        for seed in seeds:
            tasks.append((m, seed, T, N, adv_ratio, seed_contexts[seed], config))

    total_runs = len(tasks)
    run_idx = 0
    start_time = time.time()
    num_workers = min(8, mp.cpu_count())
    print(f"Executing {total_runs} tasks in parallel with {num_workers} worker processes...", flush=True)

    with mp.Pool(processes=num_workers) as pool:
        for m, seed, res in pool.imap_unordered(_worker_task, tasks):
            run_idx += 1
            for k in ['final_val', 'cp_vals', 'cp_vars']:
                results[m][k].append(res[k])
            print(f"  [{run_idx:2d}/{total_runs}] {m:20s} | seed={seed} | "
                  f"Val={res['final_val']/1e6:6.2f}M | {time.time()-start_time:.1f}s elapsed", flush=True)

    total_time = time.time() - start_time
    print(f"\n=======================================================")
    print(f"  All {total_runs} stability runs finished in {total_time:.1f} s ({total_time/60:.2f} min)")
    print(f"=======================================================")

    summary = {}
    print("\n" + "-"*65)
    print(f"{'Model':25s} | {'Final Val (10^6)':>18} | {'Final Regret (10^6)':>18}")
    print("-"*65)

    oracle_val_mean = float(np.mean(results['Non-Private Optimal']['final_val']))
    for m in models:
        v_m = np.mean(results[m]['final_val']) / 1e6
        v_s = np.std(results[m]['final_val']) / 1e6
        regret_m = (oracle_val_mean - np.mean(results[m]['final_val'])) / 1e6

        summary[m] = {
            'val_mean': float(np.mean(results[m]['final_val'])),
            'val_std': float(np.std(results[m]['final_val'])),
            'regret_mean': float(regret_m * 1e6),
            'cp_vals_mean': np.mean(results[m]['cp_vals'], axis=0).tolist(),
            'cp_vals_std': np.std(results[m]['cp_vals'], axis=0).tolist(),
            'cp_vars_mean': np.mean(results[m]['cp_vars'], axis=0).tolist(),
            'cp_vars_std': np.std(results[m]['cp_vars'], axis=0).tolist(),
        }
        print(f"{m:25s} | {v_m:7.2f} +/- {v_s:5.2f}  | {regret_m:7.2f}")
    print("-"*65)

    os.makedirs('results', exist_ok=True)
    out_json = 'results/stability_10runs_official.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"\nJSON summary saved to: {out_json}")

    plot_figure4(summary, T)
    print("\nFigure 4 successfully updated!")

if __name__ == '__main__':
    main()
