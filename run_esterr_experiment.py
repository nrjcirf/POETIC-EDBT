"""
run_esterr_experiment.py
========================
Runs POETIC, OPPS-adapted, Truthful-DP-MAB and Greedy Selection
on NYC and TDrive datasets and records EstErr (mean estimation error)
over time.  CLI and parameters match experiments_icde.py conventions.

Usage:
    python run_esterr_experiment.py              # full run (num_runs=3)
    python run_esterr_experiment.py --quick      # quick run (num_runs=1)
"""
import os, sys, random, logging, json, time
import numpy as np

from config import Config
from data_loader import get_dataloader
from worker import Worker
from poe_platform import POEPlatform
from baselines import OPPSAdapted, TruthfulDPMAB
from metrics import Metrics
from experiments_icde import _fw, _sim, _avg, _mix, _adv_ids

SEED = 42
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ── Model factory ─────────────────────────────────────────────────────────
def _build_models(config, N):
    ids = list(range(N))
    return {
        'POETIC': POEPlatform(
            config, ids, name='POETIC',
            use_reputation=True, use_exploration=True,
            use_decoupled_learning=True, use_stability_reputation=True),
        'OPPS-adapted':     OPPSAdapted(config, ids),
        'Truthful-DP-MAB':  TruthfulDPMAB(config, ids),
        'Greedy Selection': POEPlatform(
            config, ids, name='Greedy Selection',
            use_reputation=False, use_exploration=False,
            use_decoupled_learning=False, use_stability_reputation=False),
    }

# ── Single dataset run ─────────────────────────────────────────────────────
def run_one_dataset(dataset_name, config, T, N, num_runs, adv_ratio=0.3):
    logging.info(f"\n{'='*68}")
    logging.info(f"  Dataset: {dataset_name}  |  T={T}  N={N}  runs={num_runs}")
    logging.info(f"{'='*68}")

    data_loader = get_dataloader(config, dataset_type=dataset_name)

    results = {}
    for model_name, model in _build_models(config, N).items():
        runs = []
        for ri in range(num_runs):
            random.seed(SEED + ri)
            np.random.seed(SEED + ri)
            workers = _fw(N, config)
            adv = _adv_ids(N, adv_ratio)
            _mix(workers, adv, strategy='adaptive')
            run_res = _sim(model, workers, config, data_loader=data_loader, T=T)
            runs.append(run_res)
        avg = _avg(runs)

        ee_hist  = avg['estimation_error_history']
        mean_ee  = Metrics.calculate_mean_estimation_error(ee_hist)
        final_cv = avg['final_cumulative_value']

        # Build per-checkpoint EstErr curve (every 10% of T)
        ee_curve = []
        step = max(1, T // 10)
        for end in range(step, T + 1, step):
            sub = ee_hist[:end]
            ee_curve.append(Metrics.calculate_mean_estimation_error(sub) if sub else 0.0)

        results[model_name] = {
            'final_cumulative_value': final_cv,
            'mean_estimation_error':  mean_ee,
            'esterr_curve':           ee_curve,           # 10 checkpoints
            'esterr_full':            ee_hist,            # per-round
        }
        logging.info(f"  {model_name:20s}  Val={final_cv:>13,.1f}  EstErr={mean_ee:>7.2f}")

    return results

# ── Main ──────────────────────────────────────────────────────────────────
def main():
    quick    = '--quick' in sys.argv
    config   = Config()

    if quick:
        T, N, num_runs = 500, 500, 1
        logging.info("*** QUICK TEST MODE (T=500, N=500, 1 run) ***")
    else:
        T, N, num_runs = config.TOTAL_ROUNDS, config.DEFAULT_NUM_WORKERS, 3
        logging.info(f"*** FULL MODE (T={T}, N={N}, {num_runs} runs) ***")

    os.makedirs('results', exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    output = {'settings': {'T': T, 'N': N, 'num_runs': num_runs, 'quick': quick}}

    for dataset in ('NYC', 'TDrive'):
        res = run_one_dataset(dataset, config, T, N, num_runs)
        # Strip full per-round history from JSON to keep file small
        output[dataset] = {
            model: {k: v for k, v in data.items() if k != 'esterr_full'}
            for model, data in res.items()
        }

    json_path = f'results/esterr_results_{timestamp}.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logging.info(f"\n✅ Results saved to: {json_path}")

    # ── Summary table ──────────────────────────────────────────────────
    for dataset in ('NYC', 'TDrive'):
        logging.info(f"\n  EstErr Summary — {dataset}")
        logging.info(f"  {'Model':20s}  {'MeanEstErr':>12}  {'FinalVal':>15}")
        logging.info(f"  {'-'*20}  {'-'*12}  {'-'*15}")
        for model, data in output[dataset].items():
            logging.info(f"  {model:20s}  {data['mean_estimation_error']:>12.2f}  "
                         f"{data['final_cumulative_value']:>15,.1f}")

    # ── Auto-generate EstErr figure ────────────────────────────────────
    try:
        _plot_esterr(output, timestamp)
    except Exception as e:
        logging.warning(f"Plotting skipped: {e}")


def _plot_esterr(output, timestamp):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams.update({'font.family': 'serif', 'font.size': 11,
                                'axes.labelsize': 12, 'legend.fontsize': 9,
                                'grid.alpha': 0.3, 'grid.linestyle': '--'})

    PALETTE = {
        'POETIC':           '#1f77b4',
        'OPPS-adapted':     '#2ca02c',
        'Truthful-DP-MAB':  '#8c564b',
        'Greedy Selection': '#d62728',
    }
    MARKERS = {'POETIC': 'o', 'OPPS-adapted': '^',
               'Truthful-DP-MAB': 'v', 'Greedy Selection': 'd'}

    T = output['settings']['T']
    step = max(1, T // 10)
    x_axis = list(range(step, T + 1, step))

    datasets = [d for d in ('NYC', 'TDrive') if d in output]
    fig, axs = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4.5))
    if len(datasets) == 1:
        axs = [axs]

    for ax, dataset in zip(axs, datasets):
        for model, data in output[dataset].items():
            curve = data.get('esterr_curve', [])
            if not any(v > 0 for v in curve):
                continue           # skip models that don't track EstErr
            color  = PALETTE.get(model, '#333333')
            marker = MARKERS.get(model, 's')
            lw     = 2.2 if model == 'POETIC' else 1.6
            ax.plot(x_axis[:len(curve)], curve,
                    color=color, marker=marker, linewidth=lw, markersize=5,
                    label=model)

        sub = '(a)' if dataset == 'NYC' else '(b)'
        ax.set_title(f'{sub} Estimation Error — {dataset}')
        ax.set_xlabel('Round $t$')
        ax.set_ylabel('Mean Estimation Error')
        ax.grid(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper right', frameon=True)

    plt.tight_layout()
    os.makedirs('paper', exist_ok=True)
    path = f'paper/fig_esterr_{timestamp}'
    plt.savefig(f'{path}.pdf', dpi=300)
    plt.savefig(f'{path}.png', dpi=300)
    logging.info(f"✅ EstErr figure saved to {path}.{{pdf,png}}")
    plt.close()


if __name__ == '__main__':
    main()
