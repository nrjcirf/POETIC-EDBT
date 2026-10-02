# run_paper_experiments_10runs.py
"""
Official 10-Run Master Execution Script for ICDE Revision
=========================================================
Runs 10 independent repetitions (seeds 42..51) for:
  1. Ablation Study (Figure 7 & Table III):
     POETIC (Full), w/o Phi, w/o StabRep, w/o Both
  2. Stability & Concept Drift Study (Figure 4):
     POETIC, POETIC-slow, Oracle (Opt), PrivCO, Truthful-DP-MAB,
     OPPS-adapted, Greedy Selection, Random Selection
  3. Computes true Rolling Variance (window=50), Cumulative Utility, Regret, and EstErr.
  4. Automatically renders publication-quality vector PDFs:
     - paper/fig7_mechanism_ablation.pdf
     - paper/fig4_stability.pdf
"""

import os
import sys
import time
import json
import random
import logging
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib as mpl

from config import Config
from data_loader import get_dataloader
from worker import Worker
from poe_platform import POEPlatform
from baselines import (NonPrivateOptimal, POETICSlow, PrivCO,
                       OPPSAdapted, TruthfulDPMAB, RandomModel)
from metrics import Metrics

# Global styles for ICDE publication
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9.5,
    'figure.dpi': 300,
    'lines.linewidth': 2.0,
    'axes.grid': True,
    'grid.linestyle': ':',
    'grid.alpha': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

def create_model(model_name, config, N):
    ids = list(range(N))
    if model_name == 'POETIC (Full)' or model_name == 'POETIC':
        return POEPlatform(config, ids, name='POETIC',
                           use_reputation=True, use_exploration=True,
                           use_decoupled_learning=True, use_stability_reputation=True)
    elif model_name == 'w/o Phi':
        return POEPlatform(config, ids, name='w/o Phi',
                           use_reputation=True, use_exploration=True,
                           use_decoupled_learning=False, use_stability_reputation=True)
    elif model_name == 'w/o StabRep':
        return POEPlatform(config, ids, name='w/o StabRep',
                           use_reputation=True, use_exploration=True,
                           use_decoupled_learning=True, use_stability_reputation=False)
    elif model_name == 'w/o Both':
        return POEPlatform(config, ids, name='w/o Both',
                           use_reputation=True, use_exploration=True,
                           use_decoupled_learning=False, use_stability_reputation=False)
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

def _run_single_sim(args):
    model_name, seed, dataset_name, T, N, adv_ratio = args
    random.seed(seed)
    np.random.seed(seed)

    config = Config()
    config.TOTAL_ROUNDS = T
    config.DEFAULT_NUM_WORKERS = N
    import pickle
    class FastContextLoader:
        def __init__(self, dname, cfg):
            mfile = cfg.NYC_CONTEXT_FILENAME if dname.upper() == 'NYC' else cfg.CONTEXT_MAPS_FILENAME
            mpath = os.path.join(cfg.DATA_DIR, mfile)
            with open(mpath, 'rb') as f:
                self.context_maps = pickle.load(f)
    data_loader = FastContextLoader(dataset_name, config)

    workers = {wid: Worker(wid, config) for wid in range(N)}
    if adv_ratio > 0:
        n_adv = int(N * adv_ratio)
        for wid in range(n_adv // 2):
            workers[wid].set_strategy('utility_spoofer')
        for wid in range(n_adv // 2, n_adv):
            workers[wid].set_strategy('oscillatory')

    model = create_model(model_name, config, N)

    dl_keys = list(data_loader.context_maps.keys()) if (data_loader and hasattr(data_loader, 'context_maps') and data_loader.context_maps) else []

    step = 500
    checkpoints = list(range(step, T + 1, step))
    cp_vals = []
    cp_errs = []

    for r in range(T):
        current_context = None
        if dl_keys:
            rk = random.choice(dl_keys)
            density = data_loader.context_maps.get((rk[0], rk[1], 'density'), 1.0)
            importance = data_loader.context_maps.get((rk[0], rk[1], 'importance'), 1.0)
            current_context = {'density': density, 'importance': importance}

        for w in workers.values():
            w.update_state_for_round(current_context, data_loader, r)

        active = [wid for wid, w in workers.items() if w.online_status > 0.5]
        if len(active) < config.DEFAULT_NUM_WINNERS:
            if hasattr(model, '_record_empty_round'):
                model._record_empty_round()
            continue

        model.run_round(r, active, workers, data_loader=data_loader)

        if (r + 1) in checkpoints:
            nsv_so_far = model.net_social_value_history
            cval = float(np.sum(nsv_so_far)) if nsv_so_far else 0.0
            cp_vals.append(cval)

            ee_so_far = model.estimation_error_history
            c_err = float(np.mean(ee_so_far)) if ee_so_far else 0.0
            cp_errs.append(c_err)

    nsv = np.array(model.net_social_value_history)
    cum_val = np.cumsum(nsv)

    W = 50
    r_var = Metrics.calculate_rolling_variance(list(nsv), window=W)
    mean_var = float(np.mean(r_var))

    ee = model.estimation_error_history
    mean_ee = float(np.mean(ee)) if ee else 0.0

    return {
        'model': model_name,
        'seed': seed,
        'final_val': float(cum_val[-1]) if len(cum_val) else 0.0,
        'mean_esterr': mean_ee,
        'mean_var': mean_var,
        'cp_vals': cp_vals,
        'cp_errs': cp_errs,
        'cum_val_subsampled': [float(cum_val[i]) for i in range(step - 1, T, step)],
        'rolling_var_subsampled': [float(r_var[i]) for i in range(step - 1, T, step)],
    }

def run_suite(suite_name, models, dataset_name, T, N, adv_ratio, num_runs=10):
    print(f"\n=======================================================", flush=True)
    print(f"  Starting Suite: {suite_name} (10 Runs per model)", flush=True)
    print(f"  Dataset: {dataset_name} | T={T} | N={N} | Adv={adv_ratio:.0%}", flush=True)
    print(f"=======================================================", flush=True)

    tasks = []
    for model_name in models:
        for run_id in range(num_runs):
            seed = 42 + run_id
            tasks.append((model_name, seed, dataset_name, T, N, adv_ratio))

    print(f"Total tasks to execute in parallel: {len(tasks)}", flush=True)
    t0 = time.time()

    num_workers = min(mp.cpu_count(), 8)
    with mp.Pool(processes=num_workers) as pool:
        all_results = pool.map(_run_single_sim, tasks)

    elapsed = time.time() - t0
    print(f"All {len(tasks)} tasks finished in {elapsed:.1f} s ({elapsed/60:.2f} min)", flush=True)

    agg = {m: {'final_val': [], 'mean_esterr': [], 'mean_var': [],
               'cp_vals': [], 'cp_errs': [], 'cum_val_sub': [], 'rolling_var_sub': []}
           for m in models}

    for r in all_results:
        m = r['model']
        agg[m]['final_val'].append(r['final_val'])
        agg[m]['mean_esterr'].append(r['mean_esterr'])
        agg[m]['mean_var'].append(r['mean_var'])
        agg[m]['cp_vals'].append(r['cp_vals'])
        agg[m]['cp_errs'].append(r['cp_errs'])
        agg[m]['cum_val_sub'].append(r['cum_val_subsampled'])
        agg[m]['rolling_var_sub'].append(r['rolling_var_subsampled'])

    summary = {}
    print("\n" + "-"*75)
    print(f"{'Model':25s} | {'Final Value (10^6)':>18} | {'EstErr':>14} | {'Rolling Var':>12}")
    print("-"*75)
    for m in models:
        v_m = np.mean(agg[m]['final_val']) / 1e6
        v_s = np.std(agg[m]['final_val']) / 1e6
        e_m = np.mean(agg[m]['mean_esterr'])
        e_s = np.std(agg[m]['mean_esterr'])
        var_m = np.mean(agg[m]['mean_var'])

        summary[m] = {
            'val_mean': float(np.mean(agg[m]['final_val'])),
            'val_std': float(np.std(agg[m]['final_val'])),
            'esterr_mean': float(e_m),
            'esterr_std': float(e_s),
            'var_mean': float(var_m),
            'cp_vals_mean': np.mean(agg[m]['cp_vals'], axis=0).tolist(),
            'cp_vals_std': np.std(agg[m]['cp_vals'], axis=0).tolist(),
            'cp_errs_mean': np.mean(agg[m]['cp_errs'], axis=0).tolist(),
            'cp_errs_std': np.std(agg[m]['cp_errs'], axis=0).tolist(),
            'cum_val_mean': np.mean(agg[m]['cum_val_sub'], axis=0).tolist(),
            'rolling_var_mean': np.mean(agg[m]['rolling_var_sub'], axis=0).tolist(),
        }
        print(f"{m:25s} | {v_m:7.2f} +/- {v_s:5.2f}  | {e_m:6.2f} +/- {e_s:4.2f} | {var_m:12.1f}")
    print("-"*75)
    return summary

def plot_figure7_ablation(summary, T, out_dir="paper"):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    checkpoints = list(range(500, T + 1, 500))

    STYLES = {
        'POETIC (Full)': {'color': '#003366', 'ls': '-',  'marker': 'o', 'label': 'POETIC (Full)'},
        'w/o StabRep':   {'color': '#d35400', 'ls': '-.', 'marker': '^', 'label': 'w/o StabRep'},
        'w/o Phi':       {'color': '#c0392b', 'ls': '--', 'marker': 's', 'label': r'w/o $\Phi$'},
        'w/o Both':      {'color': '#7f8c8d', 'ls': ':',  'marker': 'v', 'label': 'w/o Both'},
    }

    # (a) Utility
    ax = axes[0]
    for m in ['POETIC (Full)', 'w/o StabRep', 'w/o Phi', 'w/o Both']:
        st = STYLES[m]
        means = np.array(summary[m]['cp_vals_mean']) / 1e6
        stds = np.array(summary[m]['cp_vals_std']) / 1e6
        ax.plot(checkpoints, means, color=st['color'], ls=st['ls'], marker=st['marker'],
                label=st['label'], markevery=2, lw=2.2)
        ax.fill_between(checkpoints, means - stds, means + stds, color=st['color'], alpha=0.12)
    ax.set_title('(a) Cumulative Utility ($10^6$)')
    ax.set_xlabel('Round $t$')
    ax.set_ylabel('Cumulative Utility ($10^6$)')
    ax.axvline(2500, color='gray', ls=':', alpha=0.7)
    ax.text(2550, ax.get_ylim()[0] + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.1, 'Concept Drift',
            fontsize=9, color='gray', style='italic')
    ax.legend(loc='upper left', frameon=True)

    # (b) EstErr
    ax = axes[1]
    for m in ['POETIC (Full)', 'w/o StabRep', 'w/o Phi', 'w/o Both']:
        st = STYLES[m]
        means = np.array(summary[m]['cp_errs_mean'])
        stds = np.array(summary[m]['cp_errs_std'])
        ax.plot(checkpoints, means, color=st['color'], ls=st['ls'], marker=st['marker'],
                label=st['label'], markevery=2, lw=2.2)
        ax.fill_between(checkpoints, means - stds, means + stds, color=st['color'], alpha=0.12)
    ax.set_title('(b) Utility Estimation Error')
    ax.set_xlabel('Round $t$')
    ax.set_ylabel(r'Mean Estimation Error $|W - q|$')
    ax.axvline(2500, color='gray', ls=':', alpha=0.7)
    ax.text(2550, ax.get_ylim()[0] + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.1, 'Concept Drift',
            fontsize=9, color='gray', style='italic')
    ax.legend(loc='upper left', frameon=True)

    # (c) Variance Bar
    ax = axes[2]
    models = ['POETIC (Full)', 'w/o StabRep', 'w/o Phi', 'w/o Both']
    vars_mean = [summary[m]['var_mean'] / 1e6 for m in models]
    colors = [STYLES[m]['color'] for m in models]
    bars = ax.bar([m.replace(' (Full)', '') for m in models], vars_mean, color=colors, width=0.5, edgecolor='black', lw=1.2)
    ax.set_title(r'(c) Decision Variance ($10^6$)')
    ax.set_ylabel(r'Rolling Variance ($10^6$)')
    ax.tick_params(axis='x', rotation=15)
    for b, v in zip(bars, vars_mean):
        ax.text(b.get_x() + b.get_width()/2, v * 1.02, f"{v:.1f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    pdf_path = os.path.join(out_dir, 'fig7_mechanism_ablation.pdf')
    png_path = os.path.join(out_dir, 'fig7_mechanism_ablation.png')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Figure 7 saved: {pdf_path}")

def plot_figure4_stability(summary, T, out_dir="paper"):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
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

    # (a) Utility
    ax = axes[0]
    for m in MODELS_ORDER:
        if m not in summary: continue
        st = STYLES.get(m, {'color': 'gray', 'ls': '-', 'marker': '', 'label': m})
        means = np.array(summary[m]['cum_val_mean']) / 1e6
        ax.plot(checkpoints, means, color=st['color'], ls=st['ls'], marker=st['marker'],
                label=st['label'], markevery=2, lw=2.0)
    ax.set_title('(a) Utility Convergence')
    ax.set_xlabel('Round $t$')
    ax.set_ylabel('Cumulative Utility ($10^6$)')
    ax.axvline(2500, color='gray', ls=':', alpha=0.7)
    ax.legend(loc='upper left', frameon=True, fontsize=8.5)

    # (b) Regret = Oracle - Model
    ax = axes[1]
    oracle_vals = np.array(summary['Non-Private Optimal']['cum_val_mean']) / 1e6
    for m in MODELS_ORDER:
        if m == 'Non-Private Optimal' or m not in summary: continue
        st = STYLES.get(m, {'color': 'gray', 'ls': '-', 'marker': '', 'label': m})
        regrets = oracle_vals - (np.array(summary[m]['cum_val_mean']) / 1e6)
        ax.plot(checkpoints, regrets, color=st['color'], ls=st['ls'], marker=st['marker'],
                label=st['label'], markevery=2, lw=2.0)
    ax.set_title('(b) Dynamic Regret')
    ax.set_xlabel('Round $t$')
    ax.set_ylabel('Cumulative Regret ($10^6$)')
    ax.axvline(2500, color='gray', ls=':', alpha=0.7)
    ax.legend(loc='upper left', frameon=True, fontsize=8.5)

    # (c) True Rolling Variance
    ax = axes[2]
    for m in MODELS_ORDER:
        if m not in summary: continue
        st = STYLES.get(m, {'color': 'gray', 'ls': '-', 'marker': '', 'label': m})
        r_var = np.array(summary[m]['rolling_var_mean']) / 1e6
        ax.plot(checkpoints, r_var, color=st['color'], ls=st['ls'], marker=st['marker'],
                label=st['label'], markevery=2, lw=2.0)
    ax.set_title('(c) Learning Variance Over Time')
    ax.set_xlabel('Round $t$')
    ax.set_ylabel(r'Rolling Variance ($10^6, W=50$)')
    ax.axvline(2500, color='gray', ls=':', alpha=0.7)
    ax.legend(loc='upper left', frameon=True, fontsize=8.5)

    plt.tight_layout()
    pdf_path = os.path.join(out_dir, 'fig4_stability.pdf')
    png_path = os.path.join(out_dir, 'fig4_stability.png')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Figure 4 saved: {pdf_path}")

if __name__ == '__main__':
    mp.freeze_support()

    N = 2000
    T = 5000
    num_runs = 10
    dataset = 'NYC'

    print(f"Running formal 10-run experiments on {dataset} Taxi (N={N}, T={T}, runs={num_runs})...")

    # 1. Ablation suite
    ablation_models = ['POETIC (Full)', 'w/o Phi', 'w/o StabRep', 'w/o Both']
    ablation_summary = run_suite("Ablation Study", ablation_models, dataset, T, N, adv_ratio=0.3, num_runs=num_runs)

    os.makedirs('results', exist_ok=True)
    with open('results/ablation_10runs_results.json', 'w', encoding='utf-8') as f:
        json.dump(ablation_summary, f, indent=2)

    plot_figure7_ablation(ablation_summary, T, out_dir="paper")
    plot_figure7_ablation(ablation_summary, T, out_dir="conference_latex_template")

    # 2. Stability & Drift suite
    stability_models = ['POETIC', 'POETIC-slow', 'Non-Private Optimal', 'Truthful-DP-MAB', 'OPPS-adapted', 'PrivCO', 'Greedy Selection', 'Random Selection']
    stability_summary = run_suite("Stability & Drift Study", stability_models, dataset, T, N, adv_ratio=0.3, num_runs=num_runs)

    with open('results/stability_10runs_results.json', 'w', encoding='utf-8') as f:
        json.dump(stability_summary, f, indent=2)

    plot_figure4_stability(stability_summary, T, out_dir="paper")
    plot_figure4_stability(stability_summary, T, out_dir="conference_latex_template")

    print("\nALL 10-RUN EXPERIMENTS AND FIGURES SUCCESSFULLY GENERATED!")
