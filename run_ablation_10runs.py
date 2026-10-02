# run_ablation_10runs.py
"""
Audited 10-Run Ablation Benchmark for ICDE Revision
===================================================
Runs 10 independent repetitions (seeds 42..51) for:
  - POETIC (Full)
  - w/o Phi
  - w/o StabRep
  - w/o Both

Computes:
  - Cumulative Utility trajectory (every 500 rounds)
  - Mean Estimation Error trajectory (every 500 rounds)
  - Rolling Variance of net social value (window=50)

Outputs:
  - results/ablation_10runs_official.json
  - paper/fig7_mechanism_ablation.pdf
  - conference_latex_template/fig7_mechanism_ablation.pdf
"""

import os
import sys
import time
import json
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from config import Config
from worker import Worker
from poe_platform import POEPlatform
from metrics import Metrics

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

def run_single_ablation_run(model_name, seed, T, N, adv_ratio, contexts, config):
    random.seed(seed)
    np.random.seed(seed)

    workers = {wid: Worker(wid, config) for wid in range(N)}
    n_adv = int(N * adv_ratio)
    for wid in range(n_adv // 2):
        workers[wid].set_strategy('utility_spoofer')
    for wid in range(n_adv // 2, n_adv):
        workers[wid].set_strategy('oscillatory')

    ids = list(range(N))
    if model_name == 'POETIC (Full)':
        model = POEPlatform(config, ids, name='POETIC',
                            use_reputation=True, use_exploration=True,
                            use_decoupled_learning=True, use_stability_reputation=True)
    elif model_name == 'w/o Phi':
        model = POEPlatform(config, ids, name='w/o Phi',
                            use_reputation=True, use_exploration=True,
                            use_decoupled_learning=False, use_stability_reputation=True)
    elif model_name == 'w/o StabRep':
        model = POEPlatform(config, ids, name='w/o StabRep',
                            use_reputation=True, use_exploration=True,
                            use_decoupled_learning=True, use_stability_reputation=False)
    elif model_name == 'w/o Both':
        model = POEPlatform(config, ids, name='w/o Both',
                            use_reputation=True, use_exploration=True,
                            use_decoupled_learning=False, use_stability_reputation=False)
    else:
        raise ValueError(f"Unknown ablation model: {model_name}")

    step = 500
    checkpoints = list(range(step, T + 1, step))
    cp_vals = []
    cp_errs = []

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

            ee_so_far = model.estimation_error_history
            c_err = float(np.mean(ee_so_far)) if ee_so_far else 0.0
            cp_errs.append(c_err)

    nsv = np.array(model.net_social_value_history)
    cum_val = np.cumsum(nsv)

    W = 50
    r_var = Metrics.calculate_rolling_variance(list(nsv), window=W)
    mean_var = float(np.mean(r_var))
    mean_ee = float(np.mean(model.estimation_error_history))

    return {
        'final_val': float(cum_val[-1]) if len(cum_val) else 0.0,
        'mean_esterr': mean_ee,
        'mean_var': mean_var,
        'cp_vals': cp_vals,
        'cp_errs': cp_errs,
    }

def plot_figure7(summary, T, out_dirs=['paper', 'conference_latex_template']):
    checkpoints = list(range(500, T + 1, 500))

    STYLES = {
        'POETIC (Full)': {'color': '#003366', 'ls': '-',  'marker': 'o', 'label': 'POETIC (Full)'},
        'w/o StabRep':   {'color': '#d35400', 'ls': '-.', 'marker': '^', 'label': 'w/o StabRep'},
        'w/o Phi':       {'color': '#c0392b', 'ls': '--', 'marker': 's', 'label': r'w/o $\Phi$'},
        'w/o Both':      {'color': '#7f8c8d', 'ls': ':',  'marker': 'v', 'label': 'w/o Both'},
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

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
    vars_std = [summary[m]['var_std'] / 1e6 for m in models]
    colors = [STYLES[m]['color'] for m in models]
    bars = ax.bar([m.replace(' (Full)', '') for m in models], vars_mean,
                  yerr=vars_std, capsize=4, color=colors, width=0.5, edgecolor='black', lw=1.2)
    ax.set_title(r'(c) Decision Variance ($10^6$)')
    ax.set_ylabel(r'Rolling Variance ($10^6$)')
    ax.tick_params(axis='x', rotation=15)
    for b, v in zip(bars, vars_mean):
        ax.text(b.get_x() + b.get_width()/2, v * 1.05, f"{v:.1f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        p_pdf = os.path.join(d, 'fig7_mechanism_ablation.pdf')
        p_png = os.path.join(d, 'fig7_mechanism_ablation.png')
        plt.savefig(p_pdf, bbox_inches='tight')
        plt.savefig(p_png, bbox_inches='tight', dpi=300)
        print(f"Saved Figure 7 to: {p_pdf}")
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
    # Pre-sample context sequences for each seed
    seed_contexts = {s: pre_sample_contexts(maps_file, T=T, seed=s) for s in seeds}
    print(f"Pre-sampled contexts for {num_runs} seeds in {time.time()-t0_pre:.2f} s")

    models = ['POETIC (Full)', 'w/o Phi', 'w/o StabRep', 'w/o Both']
    results = {m: {'final_val': [], 'mean_esterr': [], 'mean_var': [],
                  'cp_vals': [], 'cp_errs': []} for m in models}

    print("\n" + "="*80)
    print(f"  STARTING 10-RUN ABLATION STUDY (T={T}, N={N}, Adv={adv_ratio:.0%})")
    print("="*80)

    total_runs = len(models) * num_runs
    run_idx = 0
    start_time = time.time()

    for m in models:
        print(f"\n--- Evaluating Model: {m} ---", flush=True)
        m_t0 = time.time()
        for ri, seed in enumerate(seeds):
            run_idx += 1
            t_run = time.time()
            res = run_single_ablation_run(m, seed, T, N, adv_ratio, seed_contexts[seed], config)
            for k in ['final_val', 'mean_esterr', 'mean_var', 'cp_vals', 'cp_errs']:
                results[m][k].append(res[k])

            print(f"  [{run_idx:2d}/{total_runs}] {m:15s} | Run {ri+1:2d} (seed={seed}) | "
                  f"Val={res['final_val']/1e6:6.2f}M | EstErr={res['mean_esterr']:6.2f} | "
                  f"Var={res['mean_var']/1e6:6.2f}M | time={time.time()-t_run:.1f}s", flush=True)

        print(f"  Completed {m} (10 runs) in {time.time()-m_t0:.1f} s")

    total_time = time.time() - start_time
    print(f"\n=======================================================")
    print(f"  All {total_runs} runs finished in {total_time:.1f} s ({total_time/60:.2f} min)")
    print(f"=======================================================")

    summary = {}
    print("\n" + "-"*85)
    print(f"{'Model':20s} | {'Final Val (10^6)':>20} | {'Mean EstErr':>18} | {'Rolling Var (10^6)':>18}")
    print("-"*85)
    for m in models:
        v_m = np.mean(results[m]['final_val']) / 1e6
        v_s = np.std(results[m]['final_val']) / 1e6
        e_m = np.mean(results[m]['mean_esterr'])
        e_s = np.std(results[m]['mean_esterr'])
        var_m = np.mean(results[m]['mean_var']) / 1e6
        var_s = np.std(results[m]['mean_var']) / 1e6

        summary[m] = {
            'val_mean': float(np.mean(results[m]['final_val'])),
            'val_std': float(np.std(results[m]['final_val'])),
            'esterr_mean': float(e_m),
            'esterr_std': float(e_s),
            'var_mean': float(np.mean(results[m]['mean_var'])),
            'var_std': float(np.std(results[m]['mean_var'])),
            'cp_vals_mean': np.mean(results[m]['cp_vals'], axis=0).tolist(),
            'cp_vals_std': np.std(results[m]['cp_vals'], axis=0).tolist(),
            'cp_errs_mean': np.mean(results[m]['cp_errs'], axis=0).tolist(),
            'cp_errs_std': np.std(results[m]['cp_errs'], axis=0).tolist(),
        }
        print(f"{m:20s} | {v_m:7.2f} +/- {v_s:5.2f}     | {e_m:6.2f} +/- {e_s:4.2f}   | {var_m:7.2f} +/- {var_s:5.2f}")
    print("-"*85)

    os.makedirs('results', exist_ok=True)
    out_json = 'results/ablation_10runs_official.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"\nJSON summary saved to: {out_json}")

    plot_figure7(summary, T)
    print("\nFigure 7 successfully updated!")

if __name__ == '__main__':
    main()
