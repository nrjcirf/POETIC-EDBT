# src/platform.py
import numpy as np
import random
from protocols import Protocols
import logging


class POEPlatform:
    def __init__(self, config, all_worker_ids, name="POETIC",
                 use_reputation=True, use_exploration=True,
                 use_decoupled_learning=True, use_stability_reputation=True):
        self.config = config
        self.name = name
        self.protocols = Protocols(config)
        self.all_worker_ids = all_worker_ids
        self.use_reputation = use_reputation
        self.use_exploration = use_exploration
        self.use_quality_aware = True
        # ICDE: mechanism ablation flags
        self.use_decoupled_learning = use_decoupled_learning      # enables Φ()
        self.use_stability_reputation = use_stability_reputation  # enables stability × quality
        self.reset()

    def reset(self):
        self.net_social_value_history = []
        self.runtime_history = []
        self.comm_cost_history = []
        self.worker_profit_history_agg = []
        self.estimation_error_history = []          # NEW: |ŵ_{i,t} − u_{i,t}|
        self.worker_weights = {wid: 300 for wid in self.all_worker_ids}
        self.worker_reputations = {wid: 0.5 for wid in self.all_worker_ids}

    # ------------------------------------------------------------------
    # Φ(ũ_{i,t}, r_{i,t}) — Contamination-aware processed feedback
    # ------------------------------------------------------------------

    def _compute_processed_feedback(self, raw_feedback: float,
                                    reputation: float, worker) -> float:
        """
        Φ(ũ_{i,t}, r_{i,t}): decouples raw (noisy/strategic) feedback
        from the weight-learning update.

        Steps:
          1. Hard cap  — reject extreme outlier signals (>5x mean)
          2. Reputation-weighted scaling  — trust high-rep workers more.
          3. Fluctuation penalty  — aggressively attenuate z-score outliers.
        """
        rep_weight = np.clip(0.5 + reputation, 0.5, 1.5)

        if len(worker.utility_history) >= 2:
            hist_mean = np.mean(worker.utility_history)
            hist_std  = float(np.std(worker.utility_history)) + 1e-6

            # Hard cap: if feedback is >5x historical mean, clip aggressively
            if hist_mean > 0 and raw_feedback > 5.0 * hist_mean:
                raw_feedback = hist_mean * 1.2  # cap to slightly above mean

            z_score = abs(raw_feedback - hist_mean) / hist_std
            # Stronger penalty: z/3 instead of z/10, max 0.85
            fluctuation_penalty = np.clip(z_score / 3.0, 0.0, 0.85)
        else:
            # No history yet: apply moderate penalty to any large signal
            baseline = getattr(worker.config, 'WEIGHT_FLOOR', 0) or 100.0
            if raw_feedback > 10 * baseline:
                raw_feedback = baseline * 1.5
            fluctuation_penalty = 0.0

        return float(raw_feedback * (1.0 - fluctuation_penalty))

    # ------------------------------------------------------------------
    # Behavioral stability score
    # ------------------------------------------------------------------

    def _compute_stability_score(self, worker) -> float:
        """
        Aggregate stability score in [0, 1].
        Low score → erratic / manipulative worker → dampened reputation.
        """
        bid_vol = worker.compute_bid_volatility()
        bid_stability = 1.0 - np.clip(bid_vol, 0.0, 1.0)
        utility_stability = worker.compute_utility_stability()
        return float(np.clip(0.5 * bid_stability + 0.5 * utility_stability,
                             0.0, 1.0))

    # ------------------------------------------------------------------
    # Main per-round logic
    # ------------------------------------------------------------------

    def run_round(self, round_id, active_workers_ids, workers_dict,
                  data_loader, bidding_strategy='truthful',
                  dp_epsilon=None, num_winners=None):
        k = num_winners if num_winners is not None else self.config.DEFAULT_NUM_WINNERS

        if not active_workers_ids or len(active_workers_ids) < k:
            self._record_empty_round()
            return [], {}

        current_epsilon = max(0.01, 1.0 * (0.999 ** round_id))

        # --- 1. Bid collection ---
        bids = {}
        for wid in active_workers_ids:
            strategy_to_use = bidding_strategy if bidding_strategy is not None else None
            bids[wid] = workers_dict[wid].make_bid(strategy=strategy_to_use)
        self.last_bids = bids

        # --- 2. Net social value ---
        net_social_values = {wid: self.worker_weights[wid] - bids[wid]
                             for wid in active_workers_ids}

        # --- 3. Winner selection (ε-greedy) ---
        if self.use_exploration and random.random() < current_epsilon:
            pool_size = min(len(active_workers_ids),
                            int(k * self.config.ELITE_POOL_FACTOR))
            elite_ids, _, c_sel, t_sel = self.protocols.secure_multistage_selection(
                net_social_values, pool_size)
            winners_ids = random.sample(elite_ids, k) if len(elite_ids) >= k else elite_ids
            sorted_vals = sorted([net_social_values[w] for w in elite_ids], reverse=True)
            v_k_plus_1 = sorted_vals[k] if len(sorted_vals) > k else 0.0
        else:
            winners_ids, v_k_plus_1, c_sel, t_sel = \
                self.protocols.secure_multistage_selection(net_social_values, k)

        # --- 4. Payment ---
        payments, c_pay, t_pay = {}, 0, 0
        if winners_ids:
            reps = ({wid: self.worker_reputations[wid] for wid in winners_ids}
                    if self.use_reputation
                    else {wid: 0.0 for wid in winners_ids})
            payments, c_pay, t_pay = self.protocols.secure_reputation_payment(
                {wid: bids[wid] for wid in winners_ids},
                {wid: self.worker_weights[wid] for wid in winners_ids},
                reps, v_k_plus_1
            )

        # --- 5. Notify workers (enables adaptive δ_{i,t} next round) ---
        winners_set = set(winners_ids)
        for wid in active_workers_ids:
            if wid in winners_set:
                workers_dict[wid].notify_selected(payments.get(wid, 0.0))
            else:
                workers_dict[wid].notify_rejected()

        # --- 6. Weight update ---
        lr = self.config.LEARNING_RATE
        for wid in winners_ids:
            raw_feedback = workers_dict[wid].generate_feedback(
                workers_dict[wid].true_utility, dp_epsilon)

            if self.use_decoupled_learning:
                # ICDE core: Φ(ũ_{i,t}, r_{i,t})
                signal = self._compute_processed_feedback(
                    raw_feedback, self.worker_reputations[wid], workers_dict[wid])
            else:
                # Ablation baseline: raw reward → direct update
                signal = raw_feedback

            new_w = (1 - lr) * self.worker_weights[wid] + lr * signal
            self.worker_weights[wid] = np.clip(
                new_w, self.config.WEIGHT_FLOOR, self.config.WEIGHT_CAP)

        # --- 7. Reputation update ---
        if self.use_reputation:
            eta_R = self.config.REPUTATION_LEARNING_RATE
            all_utils = [workers_dict[w].true_utility
                         for w in self.all_worker_ids
                         if workers_dict[w].online_status > 0.5]
            global_avg_u = np.mean(all_utils) if all_utils else 1.0

            for wid in winners_ids:
                u_i = workers_dict[wid].true_utility
                if u_i < 100:
                    quality_factor = 0.2
                elif u_i > 500:
                    quality_factor = 1.5
                else:
                    quality_factor = np.clip(u_i / max(global_avg_u, 1e-6), 0.2, 1.5)

                if self.use_stability_reputation:
                    # ICDE: reputation = quality × behavioral stability
                    stability = self._compute_stability_score(workers_dict[wid])
                    reputation_signal = quality_factor * stability
                else:
                    # Ablation: quality only
                    reputation_signal = quality_factor

                if wid in [0, 10, 50, 100]:
                    logging.debug(
                        f"RepUpdate wid={wid} u={u_i:.1f} q={quality_factor:.3f} "
                        f"rep_sig={reputation_signal:.3f}")

                self.worker_reputations[wid] = (
                    (1 - eta_R) * self.worker_reputations[wid] + eta_R * reputation_signal)

        # --- 8. Utility estimation error: |ŵ_{i,t} − u_{i,t}| ---
        errors = [abs(self.worker_weights[wid] - workers_dict[wid].true_utility)
                  for wid in active_workers_ids
                  if workers_dict[wid].true_utility > 0]
        self.estimation_error_history.append(float(np.mean(errors)) if errors else 0.0)

        # --- 9. Record histories ---
        round_val = sum(workers_dict[wid].true_utility - workers_dict[wid].true_cost
                        for wid in winners_ids)
        self.net_social_value_history.append(round_val)
        self.runtime_history.append(t_sel + t_pay)
        self.comm_cost_history.append(c_sel + c_pay)
        avg_profit = (float(np.mean(
            [payments.get(wid, 0) - workers_dict[wid].true_cost for wid in winners_ids]))
                      if winners_ids else 0.0)
        self.worker_profit_history_agg.append(avg_profit)

        return winners_ids, payments

    def _record_empty_round(self):
        self.net_social_value_history.append(0)
        self.runtime_history.append(0)
        self.comm_cost_history.append(0)
        self.worker_profit_history_agg.append(0)
        self.estimation_error_history.append(0)