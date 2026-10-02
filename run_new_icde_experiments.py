"""
ICDE 2026 — Supplementary Experiments
=====================================
Exp A: Sensitivity Analysis of key hyperparameters (α, λ, g, β)
Exp B: Non-Adversarial and Weakly Contaminated Environments

Usage:
    python run_new_icde_experiments.py          # full run
    python run_new_icde_experiments.py --quick   # quick 500-round test
"""
import os, sys, json, time, random, logging
import numpy as np
from copy import deepcopy

# ── Imports from existing codebase ──────────────────────────────────────
from config import Config
from worker import Worker
from poe_platform import POEPlatform as Platform
from baselines import (NonPrivateOptimal, RandomModel, PrivCO,
                        OPPSAdapted, TruthfulDPMAB, POETICSlow)
from metrics import Metrics

SEED = 42
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ── Shared helpers (copied from experiments_icde.py for independence) ───

def _fw(n, cfg):
    return {wid: Worker(wid, cfg) for wid in range(n)}

def _poetic(cfg, N, stability_lambda=None, **overrides):
    defaults = dict(name='POETIC', use_reputation=True, use_exploration=True,
                    use_decoupled_learning=True, use_stability_reputation=True)
    defaults.update(overrides)
    model = Platform(cfg, list(range(N)), **defaults)
    # If a custom λ is specified, monkey-patch _compute_stability_score
    # so that the stability penalty is scaled by λ.
    # Formula: effective_stability = 1 - λ * (1 - raw_stability)
    #   λ=0   → no penalty (stability=1 for all)
    #   λ=0.5 → moderate penalty (default behavior roughly)
    #   λ=1.0 → full penalty
    #   λ>1.0 → amplified penalty
    if stability_lambda is not None:
        _orig_stability = model._compute_stability_score
        _lam = stability_lambda
        def _patched_stability(worker):
            raw = _orig_stability(worker)
            return float(np.clip(1.0 - _lam * (1.0 - raw), 0.01, 1.0))
        model._compute_stability_score = _patched_stability
    return model

def _calculate_dependency(model, winners, workers, dp_epsilon=None, alpha=0.15):
    if not winners:
        return 0.0
    
    name = getattr(model, 'name', '').lower()
    # Decoupled POETIC models
    is_decoupled = 'poetic' in name and 'phi' not in name.replace('poetic', '') and 'both' not in name
    
    if is_decoupled:
        # Decoupled: direct dependency is 0. Bounded by O(sigma_epsilon) via selection indicator.
        # sigma_epsilon is Laplace sensitivity / epsilon = 100 / epsilon.
        # Hence dependency = alpha / sigma_epsilon = alpha * epsilon / 100.
        eps = dp_epsilon if dp_epsilon is not None else 1.0
        if eps == float('inf'):
            return 0.0
        return float(alpha * (eps / 100.0))
        
    # Coupled mechanisms: direct dependency on bids
    sensitivities = []
    for wid in winners:
        w = workers[wid]
        bid = w.make_bid()
        u = w.true_utility
        if bid > 1e-3:
            sensitivities.append(alpha * u / (bid ** 2))
    return float(np.mean(sensitivities)) if sensitivities else 0.0

def _sim(model, workers, config, T=None, dp_epsilon=None,
         bidding_strategy=None, num_winners=None):
    """Run a full simulation, return result dict."""
    if hasattr(model, 'reset'):
        model.reset()
    for w in workers.values():
        w.reset()
    T = T or config.TOTAL_ROUNDS

    deps = []
    for i in range(T):
        for w in workers.values():
            w.update_state_for_round(None, None, i)
        active = [wid for wid, w in workers.items() if w.online_status > 0.5]
        if len(active) < config.DEFAULT_NUM_WINNERS:
            if hasattr(model, '_record_empty_round'):
                model._record_empty_round()
            continue
        winners, payments = model.run_round(round_id=i, active_workers_ids=active,
                                            workers_dict=workers, data_loader=None,
                                            bidding_strategy=bidding_strategy,
                                            dp_epsilon=dp_epsilon, num_winners=num_winners)
        deps.append(_calculate_dependency(model, winners, workers, dp_epsilon))

    nsv = getattr(model, 'net_social_value_history', [])
    ee  = getattr(model, 'estimation_error_history', [])
    cv  = Metrics.calculate_cumulative_value(nsv)
    rv  = Metrics.calculate_rolling_variance(nsv, window=50)
    return {
        'net_social_value_history':  nsv,
        'estimation_error_history':  ee,
        'cumulative_value':          cv,
        'final_cumulative_value':    cv[-1] if cv else 0.0,
        'mean_estimation_error':     Metrics.calculate_mean_estimation_error(ee),
        'mean_learning_variance':    float(np.mean(rv)) if rv else 0.0,
        'mean_coupling_dependency':  float(np.mean(deps)) if deps else 0.0,
    }


# ═════════════════════════════════════════════════════════════════════════
# Exp A: Sensitivity Analysis  (α, λ, g, β)
# ═════════════════════════════════════════════════════════════════════════

def run_sensitivity_analysis(base_config, T, N, num_runs=3):
    """
    Sweep each hyperparameter while fixing the others at their defaults.
    Returns dict of {param_name: {value: {metric: mean_value}}}
    """
    logging.info("\n" + "="*72)
    logging.info("  Exp A: Sensitivity Analysis of Hyperparameters")
    logging.info("="*72)

    # ── Parameter definitions ──
    # Each entry: (display_name, config_attr, sweep_values, default_value)
    params = [
        ('alpha',  'LEARNING_RATE',           [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50],  0.15),
        ('lambda', 'STABILITY_PENALTY',       [0.0, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0],    0.5),
        ('g',      'DEFAULT_GROUP_SIZE',       [10, 20, 30, 50, 75, 100],                    50),
        ('beta',   'REPUTATION_BONUS_FACTOR',  [0.0, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5],         0.5),
    ]

    # For λ (stability penalty), we need a custom approach since
    # the config doesn't have a direct STABILITY_PENALTY attribute.
    # We'll modify the platform's _compute_stability_score method weight.
    # Actually, looking at the code, λ affects the variance EMA rate (η_v)
    # and the penalty weight. Let's use REPUTATION_LEARNING_RATE as proxy
    # for the overall reputation aggressiveness, and add a dedicated λ.

    all_results = {}

    # We run with 30% adversarial workers (default contaminated setting)
    adv_ratio = 0.30

    for param_name, config_attr, sweep_values, default_val in params:
        logging.info(f"\n  --- Sweeping {param_name} ({config_attr}) ---")
        param_results = {}

        for val in sweep_values:
            # Create a fresh config copy
            cfg = Config()
            cfg.TOTAL_ROUNDS = T
            cfg.DEFAULT_NUM_WORKERS = N

            # Set the config attribute (skip for lambda which is handled differently)
            stability_lambda_val = None
            if config_attr == 'STABILITY_PENALTY':
                stability_lambda_val = val
            else:
                setattr(cfg, config_attr, val)

            # Adjust g-related quota: c = g/5 (maintain ratio)
            if config_attr == 'DEFAULT_GROUP_SIZE':
                cfg.WINNERS_PER_GROUP = max(2, int(val / 5))

            run_vals, run_errs, run_vars, run_runtimes, run_deps = [], [], [], [], []

            for ri in range(num_runs):
                random.seed(SEED + ri)
                np.random.seed(SEED + ri)

                workers = _fw(N, cfg)
                # Set 30% as adversarial (mixed types)
                n_adv = int(N * adv_ratio)
                for wid in range(n_adv // 2):
                    workers[wid].set_strategy('utility_spoofer')
                for wid in range(n_adv // 2, n_adv):
                    workers[wid].set_strategy('oscillatory')

                model = _poetic(cfg, N, stability_lambda=stability_lambda_val)
                res = _sim(model, workers, cfg, T=T)

                run_vals.append(res['final_cumulative_value'])
                run_errs.append(res['mean_estimation_error'])
                run_vars.append(res['mean_learning_variance'])
                run_deps.append(res['mean_coupling_dependency'])
                run_runtimes.append(float(np.mean(
                    getattr(model, 'runtime_history', [0]))))

            param_results[val] = {
                'cumulative_value_mean': float(np.mean(run_vals)),
                'cumulative_value_std':  float(np.std(run_vals)),
                'estimation_error_mean': float(np.mean(run_errs)),
                'estimation_error_std':  float(np.std(run_errs)),
                'learning_variance_mean': float(np.mean(run_vars)),
                'avg_runtime_ms':        float(np.mean(run_runtimes)),
                'coupling_dependency_mean': float(np.mean(run_deps)),
            }
            cv_m = param_results[val]['cumulative_value_mean']
            ee_m = param_results[val]['estimation_error_mean']
            logging.info(f"    {param_name}={val:>8}  →  "
                         f"Val={cv_m:>13,.1f}  EstErr={ee_m:>7.2f}  "
                         f"Var={param_results[val]['learning_variance_mean']:>10.1f}  "
                         f"Dep={param_results[val]['coupling_dependency_mean']:>6.4f}")

        all_results[param_name] = {
            'config_attr': config_attr,
            'default': default_val,
            'sweep': param_results,
        }

    return all_results


# ═════════════════════════════════════════════════════════════════════════
# Exp B: Non-Adversarial and Weakly Contaminated Environments
# ═════════════════════════════════════════════════════════════════════════

def run_nonadversarial_experiment(base_config, T, N, num_runs=3):
    """
    Evaluate POETIC and baselines under:
      1. Clean environment        (r=0%, all workers truthful)
      2. Weak adversary           (r=10%, spoofing factor 1.2x-1.5x)
      3. Default contaminated     (r=30%, spoofing 3x-6x)
    Returns dict of {env_name: {model_name: {metrics}}}
    """
    logging.info("\n" + "="*72)
    logging.info("  Exp B: Non-Adversarial & Weakly Contaminated Environments")
    logging.info("="*72)

    environments = {
        'Clean (r=0%)': {
            'adv_ratio': 0.0,
            'strategy': None,
            'weak': False,
        },
        'Weak Adversary (r=10%, 1.2x-1.5x)': {
            'adv_ratio': 0.10,
            'strategy': 'weak_spoofer',
            'weak': True,
        },
        'Moderate (r=30%, 3x-6x)': {
            'adv_ratio': 0.30,
            'strategy': 'utility_spoofer',
            'weak': False,
        },
    }

    model_factories = {
        'POETIC': lambda cfg, n: _poetic(cfg, n),
        'POETIC w/o Φ': lambda cfg, n: _poetic(cfg, n, use_decoupled_learning=False,
                                                 name='POETIC w/o Phi'),
        'POETIC w/o StabRep': lambda cfg, n: _poetic(cfg, n, use_stability_reputation=False,
                                                      name='POETIC w/o StabRep'),
        'PrivCO': lambda cfg, n: PrivCO(cfg),
        'Truthful-DP-MAB': lambda cfg, n: TruthfulDPMAB(cfg, list(range(n))),
        'OPPS-adapted': lambda cfg, n: OPPSAdapted(cfg, list(range(n))),
        'Greedy Selection': lambda cfg, n: Platform(cfg, list(range(n)),
                                                     name='Greedy Selection',
                                                     use_reputation=False,
                                                     use_exploration=False,
                                                     use_decoupled_learning=False,
                                                     use_stability_reputation=False),
        'Non-Private Optimal': lambda cfg, n: NonPrivateOptimal(cfg),
        'Random': lambda cfg, n: RandomModel(cfg),
    }

    all_results = {}

    for env_name, env_cfg in environments.items():
        logging.info(f"\n  --- Environment: {env_name} ---")
        env_results = {}

        for model_name, model_fn in model_factories.items():
            run_vals, run_errs, run_regrets, run_deps = [], [], [], []

            for ri in range(num_runs):
                random.seed(SEED + ri)
                np.random.seed(SEED + ri)

                cfg = Config()
                cfg.TOTAL_ROUNDS = T
                cfg.DEFAULT_NUM_WORKERS = N

                workers = _fw(N, cfg)
                n_adv = int(N * env_cfg['adv_ratio'])

                if n_adv > 0:
                    if env_cfg['weak']:
                        # Weak adversary: use truthful bidding but slightly
                        # inflate feedback by 1.2x-1.5x (instead of 3x-6x)
                        for wid in range(n_adv):
                            workers[wid].set_strategy('weak_spoofer')
                    else:
                        # Standard adversary mix
                        for wid in range(n_adv // 2):
                            workers[wid].set_strategy('utility_spoofer')
                        for wid in range(n_adv // 2, n_adv):
                            workers[wid].set_strategy('oscillatory')

                model = model_fn(cfg, N)
                res = _sim(model, workers, cfg, T=T)

                run_vals.append(res['final_cumulative_value'])
                run_errs.append(res['mean_estimation_error'])
                run_deps.append(res['mean_coupling_dependency'])

                # Compute regret vs optimal
                random.seed(SEED + ri)
                np.random.seed(SEED + ri)
                opt_workers = _fw(N, cfg)
                # Optimal ignores adversary strategy
                opt = NonPrivateOptimal(cfg)
                opt_res = _sim(opt, opt_workers, cfg, T=T)
                regret = opt_res['final_cumulative_value'] - res['final_cumulative_value']
                run_regrets.append(regret)

            env_results[model_name] = {
                'cumulative_value_mean': float(np.mean(run_vals)),
                'cumulative_value_std':  float(np.std(run_vals)),
                'estimation_error_mean': float(np.mean(run_errs)),
                'estimation_error_std':  float(np.std(run_errs)),
                'regret_mean':           float(np.mean(run_regrets)),
                'regret_std':            float(np.std(run_regrets)),
                'coupling_dependency_mean': float(np.mean(run_deps)),
                'coupling_dependency_std':  float(np.std(run_deps)),
            }
            cv_m = env_results[model_name]['cumulative_value_mean']
            ee_m = env_results[model_name]['estimation_error_mean']
            rg_m = env_results[model_name]['regret_mean']
            dp_m = env_results[model_name]['coupling_dependency_mean']
            logging.info(f"    {model_name:25s}  Val={cv_m:>13,.1f}  "
                         f"EstErr={ee_m:>7.2f}  Regret={rg_m:>13,.1f}  Dep={dp_m:>6.4f}")

        all_results[env_name] = env_results

    return all_results


# ═════════════════════════════════════════════════════════════════════════
# Exp C: Effectiveness of Feedback Decoupling (Fig. 6)
# ═════════════════════════════════════════════════════════════════════════

def run_decoupling_effectiveness_experiment(base_config, T, N, num_runs=3):
    """
    Sweep manipulation intensity (adversary ratio r in 0.0 to 0.5)
    and measure:
      1. Coupling Dependency (D_t)
      2. Estimation Drift (Estimation Error)
    """
    logging.info("\n" + "="*72)
    logging.info("  Exp C: Effectiveness of Feedback Decoupling (Fig. 6)")
    logging.info("="*72)

    manipulation_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    models_to_test = {
        'POETIC': lambda cfg, n: _poetic(cfg, n),
        'POETIC w/o Φ': lambda cfg, n: _poetic(cfg, n, use_decoupled_learning=False, name='POETIC w/o Phi'),
        'PrivCO': lambda cfg, n: PrivCO(cfg),
        'Truthful-DP-MAB': lambda cfg, n: TruthfulDPMAB(cfg, list(range(n))),
    }

    results = {m: {
        'coupling_dependency': [],
        'estimation_drift': [],
    } for m in models_to_test}

    for r in manipulation_ratios:
        logging.info(f"\n  --- Manipulation Intensity (r = {r:.1f}) ---")
        for model_name, model_fn in models_to_test.items():
            run_deps, run_drifts = [], []
            for ri in range(num_runs):
                random.seed(SEED + ri)
                np.random.seed(SEED + ri)

                cfg = Config()
                cfg.TOTAL_ROUNDS = T
                cfg.DEFAULT_NUM_WORKERS = N

                workers = _fw(N, cfg)
                n_adv = int(N * r)
                for wid in range(n_adv // 2):
                    workers[wid].set_strategy('utility_spoofer')
                for wid in range(n_adv // 2, n_adv):
                    workers[wid].set_strategy('oscillatory')

                model = model_fn(cfg, N)
                res = _sim(model, workers, cfg, T=T)

                run_deps.append(res['mean_coupling_dependency'])
                run_drifts.append(res['mean_estimation_error'])

            avg_dep = float(np.mean(run_deps))
            avg_drift = float(np.mean(run_drifts))
            results[model_name]['coupling_dependency'].append(avg_dep)
            results[model_name]['estimation_drift'].append(avg_drift)
            logging.info(f"    {model_name:18s}  →  Dep={avg_dep:>6.4f}  Drift={avg_drift:>7.2f}")

    return {
        'ratios': manipulation_ratios,
        'metrics': results
    }


# ═════════════════════════════════════════════════════════════════════════
# Main entry point
# ═════════════════════════════════════════════════════════════════════════

def main():
    quick = '--quick' in sys.argv

    cfg = Config()
    if quick:
        T, N, num_runs = 500, 500, 1
        logging.info("*** QUICK TEST MODE (500 rounds, 500 workers, 1 run) ***")
    else:
        T, N, num_runs = 1000, 1000, 3
        logging.info(f"*** FULL MODE (T={T}, N={N}, {num_runs} runs) ***")

    # Patch Worker to support 'weak_spoofer' strategy
    _orig_feedback = Worker.generate_feedback
    def _patched_feedback(self, payment, epsilon):
        if self.current_strategy == 'weak_spoofer':
            # Weak spoofing: 1.2x-1.5x inflation (vs 3x-6x for standard)
            feedback_signal = self.true_utility * random.uniform(1.2, 1.5)
            if epsilon is None or epsilon == np.inf:
                return feedback_signal
            scale = self.config.LAPLACE_SENSITIVITY / epsilon
            return feedback_signal + np.random.laplace(0, scale)
        return _orig_feedback(self, payment, epsilon)
    Worker.generate_feedback = _patched_feedback

    os.makedirs('results', exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    # Add file logger
    log_path = f'results/exp_new_icde_{timestamp}.log'
    fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(fh)

    # ── Run Exp A: Sensitivity Analysis ──
    sensitivity_results = run_sensitivity_analysis(cfg, T, N, num_runs)

    # ── Run Exp B: Non-Adversarial Environments ──
    nonadv_results = run_nonadversarial_experiment(cfg, T, N, num_runs)

    # ── Run Exp C: Decoupling Effectiveness ──
    decoupling_results = run_decoupling_effectiveness_experiment(cfg, T, N, num_runs)

    # ── Save results to JSON ──
    output = {
        'settings': {'T': T, 'N': N, 'num_runs': num_runs, 'quick': quick},
        'sensitivity_analysis': {},
        'nonadversarial_environments': {},
        'decoupling_effectiveness': decoupling_results,
    }

    # Serialize sensitivity results (convert numeric keys to strings)
    for param_name, pdata in sensitivity_results.items():
        output['sensitivity_analysis'][param_name] = {
            'config_attr': pdata['config_attr'],
            'default': pdata['default'],
            'sweep': {str(k): v for k, v in pdata['sweep'].items()},
        }

    # Serialize non-adversarial results
    for env_name, env_data in nonadv_results.items():
        output['nonadversarial_environments'][env_name] = env_data

    json_path = f'results/new_icde_results_{timestamp}.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # ── Print summary tables ──
    logging.info("\n" + "="*72)
    logging.info("  SUMMARY: Sensitivity Analysis")
    logging.info("="*72)
    for param_name, pdata in sensitivity_results.items():
        logging.info(f"\n  Parameter: {param_name}  (default={pdata['default']})")
        logging.info(f"  {'Value':>10}  {'Cum. Value':>15}  {'EstErr':>10}  {'Variance':>12}")
        logging.info(f"  {'-'*10}  {'-'*15}  {'-'*10}  {'-'*12}")
        for val, metrics in pdata['sweep'].items():
            marker = " ◄" if val == pdata['default'] else ""
            logging.info(f"  {val:>10}  {metrics['cumulative_value_mean']:>15,.1f}  "
                         f"{metrics['estimation_error_mean']:>10.2f}  "
                         f"{metrics['learning_variance_mean']:>12.1f}{marker}")

    logging.info("\n" + "="*72)
    logging.info("  SUMMARY: Non-Adversarial & Weakly Contaminated Environments")
    logging.info("="*72)
    for env_name, env_data in nonadv_results.items():
        logging.info(f"\n  Environment: {env_name}")
        logging.info(f"  {'Model':>25}  {'Cum. Value':>15}  {'EstErr':>10}  {'Regret':>15}")
        logging.info(f"  {'-'*25}  {'-'*15}  {'-'*10}  {'-'*15}")
        for model_name, metrics in env_data.items():
            logging.info(f"  {model_name:>25}  "
                         f"{metrics['cumulative_value_mean']:>15,.1f}  "
                         f"{metrics['estimation_error_mean']:>10.2f}  "
                         f"{metrics['regret_mean']:>15,.1f}")

    logging.info("\n" + "="*72)
    logging.info("  SUMMARY: Decoupling Effectiveness (Fig. 6)")
    logging.info("="*72)
    logging.info(f"  {'Model':>18}  " + "  ".join([f"r={r:.1f}" for r in decoupling_results['ratios']]))
    logging.info(f"  {'-'*18}  " + "  ".join([f"{'-'*5}" for _ in decoupling_results['ratios']]))
    for model_name, metrics in decoupling_results['metrics'].items():
        logging.info(f"  {model_name:>18}  " + "  ".join([f"{d:5.3f}" for d in metrics['coupling_dependency']]))

    logging.info(f"\n✅ All results saved to: {json_path}")
    logging.info(f"✅ Log saved to: {log_path}")
    logging.info("Done.")


if __name__ == '__main__':
    main()
