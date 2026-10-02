# src/experiments_icde.py
"""
ICDE 2026 鈥?7 Core Experiment Modules
Thesis: POETIC maintains stable online learning under
        strategic and privacy-corrupted feedback.

Exp 1  Overall Utility Performance  (contaminated env, all baselines)
Exp 2  Regret Analysis              (4 environments)
Exp 3  Contamination Robustness     (EstErr as primary metric)
Exp 4  Strategic Worker             (adaptive worker types)
Exp 5  Reputation Robustness        (manipulation suppression)
Exp 6  Privacy鈥揕earning Tradeoff    (sweep 蔚)
Exp 7  Mechanism Ablation           (mechanism-oriented)
"""
import numpy as np
import random
import logging

from worker import Worker
from poe_platform import POEPlatform
from baselines import (NonPrivateOptimal, RandomModel, PrivCO,
                        POETICSlow, OPPSAdapted, IncentivizedFLSC, TruthfulDPMAB)
from metrics import Metrics

SEED = 42
_ADV_RATIO = 0.30   # contaminated env default: 30 % strategic workers

# 鈹€鈹€ Shared helpers 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def _fw(n, config):
    """Fresh workers dict."""
    return {wid: Worker(wid, config) for wid in range(n)}

def _adv_ids(N, ratio=_ADV_RATIO):
    return list(range(int(N * ratio)))

def _mix(workers, adv_ids, strategy='adaptive'):
    s = set(adv_ids)
    for wid, w in workers.items():
        w.set_strategy(strategy if wid in s else 'truthful')

def _sim(model, workers, config, data_loader=None, T=None, bidding_strategy=None,
         dp_epsilon=None, num_winners=None):
    if hasattr(model, 'reset'):
        model.reset()
    for w in workers.values():
        w.reset()
    T = T or config.TOTAL_ROUNDS
    
    # Pre-fetch keys for fast sampling if data_loader is available
    dl_keys = []
    if data_loader and hasattr(data_loader, 'context_maps') and data_loader.context_maps:
        dl_keys = list(data_loader.context_maps.keys())

    for i in range(T):
        # Sample context
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
                        workers_dict=workers, data_loader=data_loader,
                        bidding_strategy=bidding_strategy,
                        dp_epsilon=dp_epsilon, num_winners=num_winners)
        
        # Intermediate logging every 10% of total rounds
        if (i + 1) % (T // 10) == 0:
            tmp_nsv = getattr(model, 'net_social_value_history', [])
            tmp_val = Metrics.calculate_cumulative_value(tmp_nsv)[-1] if tmp_nsv else 0.0
            tmp_ee = getattr(model, 'estimation_error_history', [])
            tmp_mean_ee = Metrics.calculate_mean_estimation_error(tmp_ee) if tmp_ee else 0.0
            logging.info(f"    [Round {i+1}] {model.name}: Val={tmp_val:>13,.1f}  EstErr={tmp_mean_ee:>7.2f}")

    nsv = getattr(model, 'net_social_value_history', [])
    cv  = Metrics.calculate_cumulative_value(nsv)
    return {
        'net_social_value_history':  nsv,
        'estimation_error_history':  getattr(model, 'estimation_error_history', []),
        'cumulative_value':          cv,
        'final_cumulative_value':    cv[-1] if cv else 0.0,
    }

def _avg(run_list):
    if len(run_list) == 1:
        return run_list[0]
    L   = min(len(r['net_social_value_history']) for r in run_list)
    nsv = np.mean([r['net_social_value_history'][:L] for r in run_list], axis=0).tolist()
    ee  = (np.mean([r['estimation_error_history'][:L] for r in run_list], axis=0).tolist()
           if run_list[0]['estimation_error_history'] else [])
    cv  = list(np.cumsum(nsv))
    return {**run_list[0], 'net_social_value_history': nsv,
            'estimation_error_history': ee, 'cumulative_value': cv,
            'final_cumulative_value': cv[-1] if cv else 0.0}

def _lfmt(series, key):
    return {n: {t: {key: v} for t, v in enumerate(vals, 1)}
            for n, vals in series.items()}

def _bfmt(series, key):
    return {n: {key: v} for n, v in series.items()}

def _poetic(config, N, decouple=True, stab=True, rep=True, explore=True, name='POETIC'):
    return POEPlatform(config, list(range(N)), name=name,
                    use_reputation=rep, use_exploration=explore,
                    use_decoupled_learning=decouple,
                    use_stability_reputation=stab)

def _baselines(config, N):
    ids = list(range(N))
    return {
        'Non-Private Optimal': NonPrivateOptimal(config),
        'POETIC-slow':         POETICSlow(config, ids),
        'PrivCO':              PrivCO(config),
        'OPPS-adapted':        OPPSAdapted(config, ids),
        'Greedy Selection':    POEPlatform(config, ids, name='Greedy Selection', use_reputation=False, use_exploration=False, use_decoupled_learning=False, use_stability_reputation=False),
        'Truthful-DP-MAB':     TruthfulDPMAB(config, ids),
        'Random':              RandomModel(config),
    }

def _run(models, wfn, config, T, num_runs, adv=None, data_loader=None, **skw):
    """Run every model num_runs times, average, log summary."""
    results = {}
    for name, model in models.items():
        runs = []
        for ri in range(num_runs):
            random.seed(SEED + ri); np.random.seed(SEED + ri)
            workers = wfn()
            if adv is not None:
                _mix(workers, adv)
            runs.append(_sim(model, workers, config, data_loader=data_loader, T=T, **skw))
        results[name] = _avg(runs)
        ee = Metrics.calculate_mean_estimation_error(results[name]['estimation_error_history'])
        logging.info(f"  {name:30s}  Val={results[name]['final_cumulative_value']:>13,.1f}  EstErr={ee:>7.2f}")
    return results

def _regret(results, opt='Non-Private Optimal'):
    if opt not in results:
        return {}
    oc = np.array(results[opt]['cumulative_value'])
    out = {}
    for n, r in results.items():
        if n == opt:
            continue
        cc = np.array(r['cumulative_value'])
        L  = min(len(oc), len(cc))
        out[n] = (oc[:L] - cc[:L]).tolist()
    return out


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# Exp 1: Overall Utility Performance
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
def run_overall_utility_experiment(config, visualizer, log_path, data_loader=None, num_runs=1):
    """All baselines vs POETIC in contaminated env (30 % adaptive workers)."""
    logging.info("\n=== Exp ICDE-1: Overall Utility Performance ===")
    N, T = config.DEFAULT_NUM_WORKERS, config.TOTAL_ROUNDS
    adv  = _adv_ids(N)
    models = {'POETIC': _poetic(config, N), **_baselines(config, N)}
    res    = _run(models, lambda: _fw(N, config), config, T, num_runs, adv=adv, data_loader=data_loader)

    visualizer.plot_line_chart(
        _lfmt({n: r['cumulative_value'] for n, r in res.items()}, 'final_cumulative_value'),
        range(1, T + 1), 'final_cumulative_value',
        'Cumulative Net Social Value',
        'Exp ICDE-1: Overall Performance (30% Adaptive Workers)',
        'exp_icde1_overall_value.pdf', 'Round t')
    logging.info("=== Exp ICDE-1 done ===")
    return res


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# Exp 2: Regret Analysis (4 environments)
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
def run_regret_analysis_experiment(config, visualizer, log_path, data_loader=None, num_runs=1):
    """Cumulative regret in: clean / privacy-only / strategic-only / both."""
    logging.info("\n=== Exp ICDE-2: Regret Analysis ===")
    N, T = min(config.DEFAULT_NUM_WORKERS, 1000), config.TOTAL_ROUNDS
    ids  = list(range(N))
    envs = {
        'Clean':          dict(adv=None,           dp_epsilon=None),
        'Privacy-Only':   dict(adv=None,           dp_epsilon=0.5),
        'Strategic-Only': dict(adv=_adv_ids(N),   dp_epsilon=None),
        'Contaminated':   dict(adv=_adv_ids(N),   dp_epsilon=0.5),
    }

    def _make():
        return {'POETIC':               _poetic(config, N),
                'PrivCO':              PrivCO(config),
                'Truthful-DP-MAB':     TruthfulDPMAB(config, ids),
                'Random':              RandomModel(config),
                'Non-Private Optimal': NonPrivateOptimal(config)}

    def _workers_for(adv):
        """Workers factory. Uses feedback-independent strategies so ALL baselines are affected."""
        def _fn():
            w = _fw(N, config)
            if adv:
                # Use strategies that don't require notify hooks:
                # consistent_overbid: always 2-3x, utility_spoofer: always spoofs
                for wid in adv[:len(adv)//2]:
                    w[wid].set_strategy('consistent_overbid')
                for wid in adv[len(adv)//2:]:
                    w[wid].set_strategy('utility_spoofer')
            return w
        return _fn

    for env_name, ekw in envs.items():
        adv    = ekw['adv']
        sim_kw = {'dp_epsilon': ekw['dp_epsilon']}
        logging.info(f"  --- {env_name} ---")
        res    = _run(_make(), _workers_for(adv), config, T, num_runs, data_loader=data_loader, **sim_kw)
        reg    = _regret(res)
        slug   = env_name.lower().replace(' ', '_').replace('-', '_')
        visualizer.plot_line_chart(
            _lfmt(reg, 'cumulative_regret'), range(1, T + 1), 'cumulative_regret',
            'Cumulative Regret', f'Exp ICDE-2: Regret 鈥?{env_name}',
            f'exp_icde2_regret_{slug}.pdf', 'Round t')
    logging.info("=== Exp ICDE-2 done ===")


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# Exp 3: Feedback Contamination Robustness
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
def run_contamination_robustness_experiment(config, visualizer, log_path, data_loader=None, num_runs=1):
    """EstErr vs (strategic ratio 脳 DP epsilon). Primary ICDE metric."""
    logging.info("\n=== Exp ICDE-3: Contamination Robustness ===")
    N       = min(config.DEFAULT_NUM_WORKERS, 500)
    T       = min(config.TOTAL_ROUNDS, 500)
    ratios  = [0.0, 0.2, 0.5]
    eps_list= list(getattr(config, 'EPSILON_RANGE', [0.1, 0.5, 1.0, 5.0, 10.0])) + [float('inf')]
    ids     = list(range(N))
    variants = {
        'POETIC':           dict(decouple=True,  stab=True),
        'w/o 桅':           dict(decouple=False, stab=True),
        'Truthful-DP-MAB':  None,
    }

    for ratio in ratios:
        adv  = _adv_ids(N, ratio)
        data = {v: [] for v in variants}
        for eps in eps_list:
            for vname, vkw in variants.items():
                run_errs = []
                for ri in range(num_runs):
                    random.seed(SEED + ri); np.random.seed(SEED + ri)
                    workers = _fw(N, config)
                    if adv:
                        # Use hostile mix: utility_spoofer + oscillatory for stronger EstErr gap
                        for wid in adv[:len(adv)//2]:
                            workers[wid].set_strategy('utility_spoofer')
                        for wid in adv[len(adv)//2:]:
                            workers[wid].set_strategy('oscillatory')
                    model = (_poetic(config, N, name=vname, **vkw)
                             if vkw is not None else TruthfulDPMAB(config, ids))
                    r = _sim(model, workers, config, data_loader=data_loader, T=T, dp_epsilon=eps)
                    run_errs.append(Metrics.calculate_mean_estimation_error(r['estimation_error_history']))
                err = float(np.mean(run_errs))
                data[vname].append(err)
                logging.info(f"  ratio={ratio:.0%} 蔚={eps:.2g} {vname}: EstErr={err:.2f}")

        visualizer.plot_line_chart(
            {v: {eps: {'estimation_error': data[v][i]} for i, eps in enumerate(eps_list)}
             for v in variants},
            eps_list, 'estimation_error', 'Mean Utility Estimation Error',
            f'Exp ICDE-3: Contamination Robustness (ratio={ratio:.0%})',
            f'exp_icde3_contamination_r{int(ratio*100)}.pdf', 'Privacy Budget 蔚')
    logging.info("=== Exp ICDE-3 done ===")


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# Exp 4: Strategic Worker Experiment
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
def run_strategic_worker_experiment(config, visualizer, log_path, data_loader=None, num_runs=1):
    """Mixed strategic worker types vs POETIC and baselines over different ratios."""
    logging.info("\n=== Exp ICDE-4: Strategic Worker Ratio ===")
    N, T = min(config.DEFAULT_NUM_WORKERS, 1000), config.TOTAL_ROUNDS
    ids  = list(range(N))
    ratios = [0.0, 0.1, 0.3, 0.5, 0.7]

    models = {'POETIC':          _poetic(config, N),
              'PrivCO':          PrivCO(config),
              'Truthful-DP-MAB': TruthfulDPMAB(config, ids),
              'Random':          RandomModel(config)}
              
    runs_by_ratio = {ratio: {} for ratio in ratios}
    
    for ratio in ratios:
        n_spoof = int(N * ratio * 0.4)
        n_osc   = int(N * ratio * 0.2)
        n_ob    = int(N * ratio * 0.4)
        
        def _mixed_workers():
            w = _fw(N, config)
            for wid in range(n_spoof):
                w[wid].set_strategy('utility_spoofer')
            for wid in range(n_spoof, n_spoof + n_osc):
                w[wid].set_strategy('oscillatory')
            for wid in range(n_spoof + n_osc, n_spoof + n_osc + n_ob):
                w[wid].set_strategy('consistent_overbid')
            return w

        logging.info(f"\n  --- Strategic Ratio = {ratio:.0%} ---")
        for name, model in models.items():
            rl = []
            for ri in range(num_runs):
                random.seed(SEED + ri); np.random.seed(SEED + ri)
                rl.append(_sim(model, _mixed_workers(), config, data_loader=data_loader, T=T))
            runs_by_ratio[ratio][name] = _avg(rl)
            ee = Metrics.calculate_mean_estimation_error(runs_by_ratio[ratio][name]['estimation_error_history'])
            val = runs_by_ratio[ratio][name]['final_cumulative_value']
            logging.info(f"  Ratio={ratio:.0%} {name:15s}: Val={val:>13,.1f}  EstErr={ee:>7.2f}")

    # Plot logic handles ratios
    plot_data_val = {name: {ratio: {'final_cumulative_value': runs_by_ratio[ratio][name]['final_cumulative_value']} for ratio in ratios} for name in models}
    plot_data_err = {name: {ratio: {'estimation_error': Metrics.calculate_mean_estimation_error(runs_by_ratio[ratio][name]['estimation_error_history'])} for ratio in ratios} for name in models}
    
    visualizer.plot_line_chart(
        plot_data_val, ratios, 'final_cumulative_value', 'Cumulative Net Social Value',
        'Exp ICDE-4: Utility under Strategic Workers',
        'exp_icde4_strategic_value.pdf', 'Strategic Worker Ratio')

    visualizer.plot_line_chart(
        plot_data_err, ratios, 'estimation_error', 'Utility Estimation Error |诺 鈭?u|',
        'Exp ICDE-4: EstErr Under Strategic Workers',
        'exp_icde4_strategic_esterr.pdf', 'Strategic Worker Ratio')
        
    logging.info("=== Exp ICDE-4 done ===")
    return runs_by_ratio


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# Exp 5: Reputation Robustness
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
def run_reputation_robustness_experiment(config, visualizer, log_path, data_loader=None, num_runs=1):
    """Rep trajectories: manipulative vs stable workers, with/without stability-rep."""
    logging.info("\n=== Exp ICDE-5: Reputation Robustness ===")
    N, T   = min(config.DEFAULT_NUM_WORKERS, 500), config.TOTAL_ROUNDS
    n_manip = int(N * 0.20)
    manip_ids = set(range(n_manip))
    stable_ids = set(range(n_manip, N))

    for use_stab, label in [(True, 'POETIC'), (False, 'w/o StabRep')]:
        all_m, all_s = [], []
        for ri in range(num_runs):
            random.seed(SEED + ri); np.random.seed(SEED + ri)
            workers = _fw(N, config)
            # Use ADAPTIVE strategy for manipulators:
            # - They underbid when rejected (get selected)
            # - When selected with surplus, inflate feedback (creates volatility)
            # - POETIC stability-aware rep detects volatility -> penalizes reputation
            for wid in manip_ids:
                workers[wid].set_strategy('adaptive')
            model = _poetic(config, N, stab=use_stab, name=label)
            model.reset()
            for w in workers.values():
                w.reset()
            dl_keys = []
            if data_loader and hasattr(data_loader, 'context_maps') and data_loader.context_maps:
                dl_keys = list(data_loader.context_maps.keys())

            rm, rs = [], []
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
                if len(active) >= config.DEFAULT_NUM_WINNERS:
                    model.run_round(i, active, workers, data_loader, bidding_strategy=None)
                rm.append(float(np.mean([model.worker_reputations[wid] for wid in manip_ids])))
                rs.append(float(np.mean([model.worker_reputations[wid] for wid in stable_ids])))
            all_m.append(rm); all_s.append(rs)
        L   = min(len(x) for x in all_m)
        mhist = np.mean([x[:L] for x in all_m], axis=0).tolist()
        shist = np.mean([x[:L] for x in all_s], axis=0).tolist()
        logging.info(f"  [{label}] Manip={mhist[-1]:.3f}  Stable={shist[-1]:.3f}  Gap={shist[-1]-mhist[-1]:.3f}")
        visualizer.plot_line_chart(
            _lfmt({f'Manipulative ({label})': mhist, f'Stable ({label})': shist}, 'avg_reputation'),
            range(1, T + 1), 'avg_reputation', 'Average Reputation Score',
            f'Exp ICDE-5: Reputation Robustness 鈥?{label}',
            f'exp_icde5_rep_{label.lower().replace(" ","_").replace("/","_")}.pdf', 'Round t')
    logging.info("=== Exp ICDE-5 done ===")


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# Exp 6: Privacy鈥揕earning Tradeoff
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
def run_privacy_tradeoff_experiment(config, visualizer, log_path, data_loader=None, num_runs=1):
    """Sweep DP 蔚: EstErr, regret, utility, learning variance vs privacy budget."""
    logging.info("\n=== Exp ICDE-6: Privacy鈥揕earning Tradeoff ===")
    N       = min(config.DEFAULT_NUM_WORKERS, 500)
    T       = config.TOTAL_ROUNDS
    eps_list= list(getattr(config, 'EPSILON_RANGE', [0.1, 0.5, 1.0, 5.0, 10.0])) + [float('inf')]
    ids     = list(range(N))
    adv     = _adv_ids(N)

    def _workers_with_spoofers():
        """Use utility_spoofer for strong contamination at all epsilon levels."""
        w = _fw(N, config)
        for wid in adv[:len(adv)//2]:
            w[wid].set_strategy('utility_spoofer')
        for wid in adv[len(adv)//2:]:
            w[wid].set_strategy('oscillatory')
        return w

    compare = {'POETIC':          lambda: _poetic(config, N),
               'w/o 桅':           lambda: _poetic(config, N, decouple=False, stab=False, name='w/o 桅'),
               'PrivCO':          lambda: PrivCO(config),
               'Truthful-DP-MAB': lambda: TruthfulDPMAB(config, ids)}

    agg = {n: {'ee': [], 'val': [], 'var': []} for n in compare}

    for eps in eps_list:
        for vname, mfn in compare.items():
            run_ee, run_val, run_var = [], [], []
            for ri in range(num_runs):
                random.seed(SEED + ri); np.random.seed(SEED + ri)
                workers = _workers_with_spoofers()
                r = _sim(mfn(), workers, config, data_loader=data_loader, T=T, dp_epsilon=eps)
                run_ee.append(Metrics.calculate_mean_estimation_error(r['estimation_error_history']))
                run_val.append(r['final_cumulative_value'])
                rv = Metrics.calculate_rolling_variance(r['net_social_value_history'], window=50)
                run_var.append(float(np.mean(rv)))
            agg[vname]['ee'].append(float(np.mean(run_ee)))
            agg[vname]['val'].append(float(np.mean(run_val)))
            agg[vname]['var'].append(float(np.mean(run_var)))
            logging.info(f"  蔚={eps:.2g} {vname}: EstErr={agg[vname]['ee'][-1]:.2f}  Val={agg[vname]['val'][-1]:,.1f}")

    for metric, ylabel, fname in [
        ('ee',  'Mean Utility Estimation Error', 'exp_icde6_privacy_esterr.pdf'),
        ('val', 'Final Cumulative Value',         'exp_icde6_privacy_value.pdf'),
        ('var', 'Mean Learning Variance',         'exp_icde6_privacy_variance.pdf'),
    ]:
        visualizer.plot_line_chart(
            {n: {eps: {metric: agg[n][metric][i]} for i, eps in enumerate(eps_list)}
             for n in compare},
            eps_list, metric, ylabel,
            f'Exp ICDE-6: Privacy鈥揕earning Tradeoff ({ylabel})',
            fname, 'Privacy Budget 蔚')
    logging.info("=== Exp ICDE-6 done ===")
    return agg


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# Exp 7: Mechanism Ablation
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
def run_mechanism_ablation_experiment(config, visualizer, log_path, data_loader=None, num_runs=1):
    """Mechanism-oriented ablation in contaminated env (30 % adaptive workers)."""
    logging.info("\n=== Exp ICDE-7: Mechanism Ablation ===")
    N, T, W = min(config.DEFAULT_NUM_WORKERS, 1000), config.TOTAL_ROUNDS, 50
    adv = _adv_ids(N)

    ablations = {
        'POETIC (Full)':  dict(decouple=True,  stab=True),
        'w/o 桅':         dict(decouple=False, stab=True),
        'w/o StabRep':   dict(decouple=True,  stab=False),
        'w/o Both':      dict(decouple=False, stab=False),
    }

    results = {}
    for aname, kw in ablations.items():
        rl = []
        for ri in range(num_runs):
            random.seed(SEED + ri); np.random.seed(SEED + ri)
            # Hostile contamination: utility spoofers + oscillatory
            workers = _fw(N, config)
            n_spoof = len(adv) // 2
            for wid in adv[:n_spoof]:
                workers[wid].set_strategy('utility_spoofer')
            for wid in adv[n_spoof:]:
                workers[wid].set_strategy('oscillatory')
            model = _poetic(config, N, name=aname, **kw)
            r = _sim(model, workers, config, data_loader=data_loader, T=T)
            r['rolling_variance'] = Metrics.calculate_rolling_variance(
                r['net_social_value_history'], window=W)
            rl.append(r)
        res = _avg(rl)
        res['mean_var'] = float(np.mean(
            Metrics.calculate_rolling_variance(res['net_social_value_history'], W)))
        results[aname] = res
        ee = Metrics.calculate_mean_estimation_error(res['estimation_error_history'])
        logging.info(f"  {aname:20s}  Val={res['final_cumulative_value']:>13,.1f}  "
                     f"EstErr={ee:>7.2f}  Var={res['mean_var']:>10.1f}")

    visualizer.plot_bar_chart(
        _bfmt({n: Metrics.calculate_mean_estimation_error(r['estimation_error_history'])
               for n, r in results.items()}, 'estimation_error'),
        'estimation_error', 'Mean Estimation Error |诺 鈭?u|',
        'Exp ICDE-7: Mechanism Ablation 鈥?Estimation Error',
        'exp_icde7_ablation_esterr.pdf')

    visualizer.plot_bar_chart(
        _bfmt({n: r['final_cumulative_value'] for n, r in results.items()},
              'final_cumulative_value'),
        'final_cumulative_value', 'Final Cumulative NSV',
        'Exp ICDE-7: Mechanism Ablation 鈥?Value',
        'exp_icde7_ablation_value.pdf')

    visualizer.plot_bar_chart(
        _bfmt({n: r['mean_var'] for n, r in results.items()}, 'learning_variance'),
        'learning_variance', 'Mean Learning Variance',
        'Exp ICDE-7: Mechanism Ablation 鈥?Stability',
        'exp_icde7_ablation_variance.pdf')

    logging.info("=== Exp ICDE-7 done ===")
    return results

# ═══════════════════════════════════════════════════════════════════════════
# Exp 8: Adversarial Worker Ratio
# ═══════════════════════════════════════════════════════════════════════════

def run_adversarial_worker_experiment(config, visualizer, log_path, data_loader=None, num_runs=1):
    """Purely destructive adversarial workers (oscillatory/noise) vs POETIC and baselines."""
    logging.info("\n=== Exp ICDE-8: Adversarial Worker Ratio ===")
    N, T = min(config.DEFAULT_NUM_WORKERS, 1000), config.TOTAL_ROUNDS
    ids  = list(range(N))
    ratios = [0.0, 0.1, 0.3, 0.5, 0.7]

    models = {'POETIC':          _poetic(config, N),
              'PrivCO':          PrivCO(config),
              'Truthful-DP-MAB': TruthfulDPMAB(config, ids),
              'Random':          RandomModel(config)}
              
    runs_by_ratio = {ratio: {} for ratio in ratios}
    
    for ratio in ratios:
        n_adv = int(N * ratio)
        
        def _adv_workers():
            w = _fw(N, config)
            # Adversarial: purely oscillatory / destructive
            for wid in range(n_adv):
                w[wid].set_strategy('oscillatory')
            return w

        logging.info(f"\n  --- Adversarial Ratio = {ratio:.0%} ---")
        for name, model in models.items():
            rl = []
            for ri in range(num_runs):
                random.seed(SEED + ri + 100); np.random.seed(SEED + ri + 100)
                rl.append(_sim(model, _adv_workers(), config, data_loader=data_loader, T=T))
            runs_by_ratio[ratio][name] = _avg(rl)
            ee = Metrics.calculate_mean_estimation_error(runs_by_ratio[ratio][name]['estimation_error_history'])
            val = runs_by_ratio[ratio][name]['final_cumulative_value']
            logging.info(f"  Ratio={ratio:.0%} {name:15s}: Val={val:>13,.1f}  EstErr={ee:>7.2f}")

    plot_data_val = {name: {ratio: {'final_cumulative_value': runs_by_ratio[ratio][name]['final_cumulative_value']} for ratio in ratios} for name in models}
    plot_data_err = {name: {ratio: {'estimation_error': Metrics.calculate_mean_estimation_error(runs_by_ratio[ratio][name]['estimation_error_history'])} for ratio in ratios} for name in models}
    
    visualizer.plot_line_chart(
        plot_data_val, ratios, 'final_cumulative_value', 'Cumulative Net Social Value',
        'Exp ICDE-8: Utility under Adversarial Workers',
        'exp_icde8_adversarial_value.pdf', 'Adversarial Worker Ratio')

    visualizer.plot_line_chart(
        plot_data_err, ratios, 'estimation_error', 'Utility Estimation Error |? ? u|',
        'Exp ICDE-8: EstErr Under Adversarial Workers',
        'exp_icde8_adversarial_esterr.pdf', 'Adversarial Worker Ratio')
        
    logging.info("=== Exp ICDE-8 done ===")
    return runs_by_ratio

# 建议在 experiments_icde.py 中追加此函数，并在 main.py 中调用
def run_clean_overall_utility_experiment(config, visualizer, log_path, data_loader=None, num_runs=1):
    """
    NEW EXPERIMENT: Compares all models in a 100% CLEAN environment (r=0%)
    Preserves historical trajectories for generating time-slot cumulative curves.
    """
    logging.info("\n=== Exp ICDE-1-Clean: Overall Utility Performance under 0% Attackers ===")
    N, T = config.DEFAULT_NUM_WORKERS, config.TOTAL_ROUNDS
    
    # 🔴 强制设置比例为 0.0 (绝对清白环境)
    adv = [] 
    
    models = {'POETIC': _poetic(config, N), **_baselines(config, N)}
    res = _run(models, lambda: _fw(N, config), config, T, num_runs, adv=adv, data_loader=data_loader)

    # 🔴 将完整的时序历史保存并绘图输出
    visualizer.plot_line_chart(
        _lfmt({n: r['cumulative_value'] for n, r in res.items()}, 'final_cumulative_value'),
        range(1, T + 1), 'final_cumulative_value',
        'Cumulative Net Social Value (Clean Env)',
        'Exp ICDE-1-Clean: Overall Performance (0% Strategic Workers)',
        'exp_icde1_clean_overall_value.pdf', 'Round t')
        
    logging.info("=== Exp ICDE-1-Clean done ===")
    return res
