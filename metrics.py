# src/metrics.py
import numpy as np


class Metrics:
    # ------------------------------------------------------------------
    # Original metrics (unchanged)
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_cumulative_value(history_list):
        if not isinstance(history_list, list) or not history_list:
            return []
        return np.cumsum(history_list).tolist()

    @staticmethod
    def calculate_net_social_value(winners_ids, true_utilities, true_costs):
        if not winners_ids:
            return 0
        return sum(true_utilities.get(wid, 0) - true_costs.get(wid, 0)
                   for wid in winners_ids)

    @staticmethod
    def calculate_average_value_per_round(history):
        if not history:
            return 0
        return float(np.mean(history))

    @staticmethod
    def calculate_cumulative_regret(optimal_history_list, current_history_list):
        if (not optimal_history_list or not current_history_list
                or len(optimal_history_list) != len(current_history_list)):
            return []
        optimal_cumulative = np.cumsum(optimal_history_list)
        current_cumulative = np.cumsum(current_history_list)
        return (optimal_cumulative - current_cumulative).tolist()

    # ------------------------------------------------------------------
    # ICDE new metrics
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_rolling_variance(history, window=50):
        """
        Per-round rolling variance of net social value.
        Measures learning stability: low variance = stable convergence.
        Returns a list of the same length as `history`.
        """
        out = []
        for i in range(len(history)):
            start = max(0, i - window + 1)
            out.append(float(np.var(history[start: i + 1])))
        return out

    @staticmethod
    def calculate_selection_stability(winner_history, window=20):
        """
        Mean Jaccard similarity between consecutive winner sets in the
        last `window` rounds.  Range [0, 1]. Higher = more stable selection.
        """
        recent = list(winner_history)[-window:]
        if len(recent) < 2:
            return 1.0
        sims = []
        for i in range(1, len(recent)):
            a, b = set(recent[i - 1]), set(recent[i])
            if not a and not b:
                sims.append(1.0)
            elif not a or not b:
                sims.append(0.0)
            else:
                sims.append(len(a & b) / len(a | b))
        return float(np.mean(sims)) if sims else 1.0

    @staticmethod
    def calculate_mean_estimation_error(estimation_error_history):
        """
        Mean |ŵ_{i,t} − u_{i,t}| over all rounds.
        Lower = platform's weight estimate tracks true utility more closely.
        """
        if not estimation_error_history:
            return 0.0
        return float(np.mean(estimation_error_history))

    @staticmethod
    def calculate_regret_growth_rate(regret_history, window=100):
        """
        Average per-round increase in cumulative regret over the last
        `window` rounds.  Low rate = regret has flattened (converged).
        """
        if len(regret_history) < 2:
            return 0.0
        tail = regret_history[-window:]
        if len(tail) < 2:
            return 0.0
        deltas = [tail[i] - tail[i - 1] for i in range(1, len(tail))]
        return float(np.mean(deltas))

    @staticmethod
    def calculate_normalized_regret(regret_history):
        """
        Regret / sqrt(T) at each round t — matches theoretical O(√T) bound.
        """
        result = []
        for t, r in enumerate(regret_history, 1):
            result.append(r / np.sqrt(t) if t > 0 else 0.0)
        return result