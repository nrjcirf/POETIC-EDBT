# src/baselines.py
import numpy as np
import random
from protocols import Protocols  # 导入 Protocols 以使用其中的支付协议


# ---------------------------------------------------------------------------
# Shared helper: minimal Φ-style processed feedback for baselines
# (keeps baselines comparable without full POETIC machinery)
# ---------------------------------------------------------------------------

def _baseline_processed_feedback(raw_feedback: float, reputation: float, worker) -> float:
    """
    Lightweight version of Platform._compute_processed_feedback().
    Baselines share the same signal-decoupling principle so that
    reviewer comparisons remain apples-to-apples.
    """
    rep_weight = np.clip(0.5 + reputation, 0.5, 1.5)
    if len(worker.utility_history) >= 2:
        hist_mean = np.mean(worker.utility_history)
        hist_std = float(np.std(worker.utility_history)) + 1e-6
        z_score = abs(raw_feedback - hist_mean) / hist_std
        fluctuation_penalty = np.clip(z_score / 10.0, 0.0, 0.5)
    else:
        fluctuation_penalty = 0.0
    return float(raw_feedback * rep_weight * (1.0 - fluctuation_penalty))


class BaseBaseline:
    def __init__(self, config, all_worker_ids=None, name="Baseline"):
        self.config = config
        self.name = name
        self.all_worker_ids = all_worker_ids if all_worker_ids is not None else []
        self.net_social_value_history = []
        self.runtime_history = []
        self.comm_cost_history = []
        self.worker_profit_history_agg = []
        self.estimation_error_history = []   # |ŵ_{i,t} − u_{i,t}|
        self.reset()

    def reset(self):
        self.net_social_value_history = []
        self.runtime_history = []
        self.comm_cost_history = []
        self.worker_profit_history_agg = []
        self.estimation_error_history = []   # |ŵ_{i,t} − u_{i,t}|


    def run_round(self, round_id, active_workers_ids, workers_dict, data_loader, bidding_strategy='truthful',
                  dp_epsilon=None, num_winners=None):
        raise NotImplementedError("Subclasses must implement run_round method")


class NonPrivateOptimal(BaseBaseline):
    def __init__(self, config):
        super().__init__(config, name="Non-Private Optimal")

    def run_round(self, round_id, active_workers_ids, workers_dict, data_loader, bidding_strategy='truthful',
                  dp_epsilon=None, num_winners=None):
        k = num_winners if num_winners is not None else self.config.DEFAULT_NUM_WINNERS
        if not active_workers_ids or len(active_workers_ids) < k:
            self.net_social_value_history.append(0)
            self.runtime_history.append(0)
            self.comm_cost_history.append(0)
            return [], {}

        # Calculate true social value for all active workers
        true_values = {wid: workers_dict[wid].true_utility - workers_dict[wid].true_cost for wid in active_workers_ids}

        # Select top k workers based on true social value
        sorted_workers = sorted(true_values.items(), key=lambda item: item[1], reverse=True)
        winners_ids = [item[0] for item in sorted_workers[:k]]

        # No actual payment or privacy for optimal baseline, just record value
        round_net_social_value = sum(true_values[wid] for wid in winners_ids)
        self.net_social_value_history.append(round_net_social_value)
        self.runtime_history.append(0)  # Optimal has no runtime/comm cost
        self.comm_cost_history.append(0)
        self.worker_profit_history_agg.append(0)  # No actual profit calculation needed

        return winners_ids, {}


class RandomModel(BaseBaseline):
    def __init__(self, config, all_worker_ids=None):
        super().__init__(config, all_worker_ids, name="Random")

    def run_round(self, round_id, active_workers_ids, workers_dict, data_loader, bidding_strategy='truthful',
                  dp_epsilon=None, num_winners=None):
        k = num_winners if num_winners is not None else self.config.DEFAULT_NUM_WINNERS
        if not active_workers_ids or len(active_workers_ids) < k:
            self.net_social_value_history.append(0)
            self.runtime_history.append(0)
            self.comm_cost_history.append(0)
            return [], {}

        # Randomly select k winners
        winners_ids = random.sample(active_workers_ids, k)

        round_net_social_value = sum(
            workers_dict[wid].true_utility - workers_dict[wid].true_cost for wid in winners_ids)
        self.net_social_value_history.append(round_net_social_value)
        self.runtime_history.append(0)
        self.comm_cost_history.append(0)
        self.worker_profit_history_agg.append(0)

        return winners_ids, {}


class POETICSlow(BaseBaseline):
    """
    [修复版] POETIC-slow：逻辑必须与 Platform.py 完全同步
    """

    def __init__(self, config, all_worker_ids):
        super().__init__(config, all_worker_ids, name="POETIC-slow")
        # [修复] 初始权重设为 100,000 (与 Platform 一致)，防止死锁
        self.worker_weights = {wid: 100000.0 for wid in all_worker_ids}
        self.worker_reputations = {wid: 0.5 for wid in all_worker_ids}
        self.protocols = Protocols(config)

    def reset(self):
        super().reset()
        self.worker_weights = {wid: 2000.0 for wid in self.all_worker_ids}
        self.worker_reputations = {wid: 0.5 for wid in self.all_worker_ids}

    def run_round(self, round_id, active_workers_ids, workers_dict, data_loader, bidding_strategy='truthful',
                  dp_epsilon=None, num_winners=None):
        k = num_winners if num_winners is not None else self.config.DEFAULT_NUM_WINNERS
        if not active_workers_ids or len(active_workers_ids) < k:
            self._record_metrics([], workers_dict, 0, 0, {})
            return [], {}

        bids = {wid: workers_dict[wid].make_bid(strategy=bidding_strategy) for wid in active_workers_ids}
        net_social_values = {wid: self.worker_weights[wid] - bids[wid] for wid in active_workers_ids}

        # 全局排序 Top-k
        sorted_values = sorted(net_social_values.items(), key=lambda item: item[1], reverse=True)
        winners_ids = [item[0] for item in sorted_values[:k]]
        v_k_plus_1 = sorted_values[k][1] if len(sorted_values) > k else -np.inf

        # 模拟开销 (比分组慢)
        n = len(active_workers_ids)
        comparisons = int(n * np.log2(max(n, 2)))
        rt = comparisons * self.config.SMPC_RUNTIME_PER_COMPARE_MS + self.config.NETWORK_LATENCY_MS
        cc = comparisons * self.config.SMPC_COMM_OVERHEAD_PER_COMPARE_MB

        payments, pc, pr = self.protocols.secure_reputation_payment(
            {wid: bids[wid] for wid in winners_ids},
            {wid: self.worker_weights[wid] for wid in winners_ids},
            {wid: self.worker_reputations[wid] for wid in winners_ids},
            v_k_plus_1
        )

        # --- Notify workers of selection outcome (enables adaptive δ_{i,t}) ---
        winners_set = set(winners_ids)
        for wid in active_workers_ids:
            if wid in winners_set:
                workers_dict[wid].notify_selected(payments.get(wid, 0.0))
            else:
                workers_dict[wid].notify_rejected()

        # --- Weight update via Φ() processed feedback (ICDE: signal decoupling) ---
        lr = self.config.LEARNING_RATE
        current_epsilon = dp_epsilon if dp_epsilon is not None else self.config.DEFAULT_EPSILON
        for wid in winners_ids:
            # Raw signal: true_utility + DP noise  (NOT payment — fixes original bug)
            raw_feedback = workers_dict[wid].generate_feedback(workers_dict[wid].true_utility, current_epsilon)
            # Φ(): reputation-scaled, fluctuation-penalised
            processed = _baseline_processed_feedback(
                raw_feedback, self.worker_reputations[wid], workers_dict[wid]
            )
            self.worker_weights[wid] = (1 - lr) * self.worker_weights[wid] + lr * processed
            self.worker_weights[wid] = np.clip(
                self.worker_weights[wid], self.config.WEIGHT_FLOOR, self.config.WEIGHT_CAP
            )

        # --- Stability-aware reputation update ---
        eta_R = self.config.REPUTATION_LEARNING_RATE
        winner_utils = [workers_dict[wid].true_utility for wid in winners_ids]
        avg_u = np.mean(winner_utils) if winner_utils else 1.0

        for wid in winners_ids:
            u_i = workers_dict[wid].true_utility
            quality_factor = np.clip(u_i / max(avg_u, 1e-6), 0.2, 1.0)
            # Stability: penalise erratic bid / utility behaviour
            bid_vol = workers_dict[wid].compute_bid_volatility()
            bid_stability = 1.0 - np.clip(bid_vol, 0.0, 1.0)
            u_stability = workers_dict[wid].compute_utility_stability()
            stability_score = 0.5 * bid_stability + 0.5 * u_stability
            reputation_signal = quality_factor * stability_score
            self.worker_reputations[wid] = (
                (1 - eta_R) * self.worker_reputations[wid] + eta_R * reputation_signal
            )

        self._record_metrics(winners_ids, workers_dict, rt + pr, cc + pc, payments)
        return winners_ids, payments

    def _record_metrics(self, winners_ids, workers_dict, runtime, comm, payments):
        val = sum(workers_dict[wid].true_utility - workers_dict[wid].true_cost for wid in winners_ids)
        self.net_social_value_history.append(val)
        self.runtime_history.append(runtime)
        self.comm_cost_history.append(comm)
        profit = np.mean(
            [payments.get(wid, 0) - workers_dict[wid].true_cost for wid in winners_ids]) if winners_ids else 0
        self.worker_profit_history_agg.append(profit)


class PrivCO(BaseBaseline):
    def __init__(self, config):
        super().__init__(config, name="PrivCO")

    def run_round(self, round_id, active_workers_ids, workers_dict, data_loader, bidding_strategy='truthful',
                  dp_epsilon=None, num_winners=None):
        k = num_winners if num_winners is not None else self.config.DEFAULT_NUM_WINNERS
        if not active_workers_ids or len(active_workers_ids) < k:
            self._record_metrics([], workers_dict, 0, 0, {})
            return [], {}

        # 策略：选择报价最低的 k 个 (Cheapest First)
        # 这会导致选到低价值的"垃圾"任务，从而导致 Social Welfare 很低甚至为负
        bids = {wid: workers_dict[wid].make_bid(strategy=bidding_strategy) for wid in active_workers_ids}

        # 升序排列
        sorted_workers = sorted(bids.items(), key=lambda item: item[1])
        winners_ids = [item[0] for item in sorted_workers[:k]]

        # 支付 = 报价 (简化)
        payments = {wid: bids[wid] for wid in winners_ids}

        n = len(active_workers_ids)
        rt = n * np.log2(n) * self.config.SMPC_RUNTIME_PER_COMPARE_MS
        cc = n * self.config.SMPC_COMM_OVERHEAD_PER_COMPARE_MB

        self._record_metrics(winners_ids, workers_dict, rt, cc, payments)
        return winners_ids, payments

    def _record_metrics(self, winners_ids, workers_dict, runtime, comm, payments):
        # 记录真实的 Social Value (Value - Cost)
        val = sum(workers_dict[wid].true_utility - workers_dict[wid].true_cost for wid in winners_ids)
        self.net_social_value_history.append(val)
        self.runtime_history.append(runtime)
        self.comm_cost_history.append(comm)

        # [核心修复] 计算真实的利润，而不是硬编码为 0
        # 之前是: profit = 0
        profit = np.mean(
            [payments.get(wid, 0) - workers_dict[wid].true_cost for wid in winners_ids]) if winners_ids else 0
        self.worker_profit_history_agg.append(profit)


class IncentivizedFLSC(BaseBaseline):
    def __init__(self, config, all_worker_ids):
        super().__init__(config, all_worker_ids, name="Incentivized FL-SC")

    def run_round(self, round_id, active_workers_ids, workers_dict, data_loader, bidding_strategy='truthful',
                  dp_epsilon=None, num_winners=None):
        k = num_winners if num_winners is not None else self.config.DEFAULT_NUM_WINNERS
        if not active_workers_ids or len(active_workers_ids) < k:
            self._record_metrics([], workers_dict, 0, 0, {})
            return [], {}

        # [核心修复] 禁止读取 true_utility!
        # 改为：基于报价选择，加上较大的噪声
        bids = {wid: workers_dict[wid].make_bid(strategy=bidding_strategy) for wid in active_workers_ids}

        # 负Bid越大越好（即Bid越小越好），加上噪声
        noisy_scores = {wid: -bid + np.random.normal(0, 50) for wid, bid in bids.items()}

        sorted_workers = sorted(noisy_scores.items(), key=lambda item: item[1], reverse=True)
        winners_ids = [item[0] for item in sorted_workers[:k]]

        payments = {wid: self.config.FL_SC_BASE_REWARD for wid in winners_ids}

        self._record_metrics(winners_ids, workers_dict, 0, 0, payments)
        return winners_ids, payments

    def _record_metrics(self, winners_ids, workers_dict, runtime, comm, payments):
        val = sum(workers_dict[wid].true_utility - workers_dict[wid].true_cost for wid in winners_ids)
        self.net_social_value_history.append(val)
        self.runtime_history.append(runtime)
        self.comm_cost_history.append(comm)
        profit = np.mean(
            [payments.get(wid, 0) - workers_dict[wid].true_cost for wid in winners_ids]) if winners_ids else 0
        self.worker_profit_history_agg.append(profit)


class TruthfulDPMAB(BaseBaseline):
    def __init__(self, config, all_worker_ids):
        super().__init__(config, all_worker_ids, name="Truthful-DP-MAB")
        # [核心修复1] 必须初始化为 0，强制冷启动！
        # 之前的 100000.0 是"作弊"，会让它在第一轮就表现得像上帝视角
        self.arms = {wid: {'pulls': 0, 'total_reward': 0.0, 'estimated_value': 0.0} for wid in all_worker_ids}
        self.current_time = 0

    def reset(self):
        super().reset()
        # [核心修复1] 重置时也要归零
        self.arms = {wid: {'pulls': 0, 'total_reward': 0.0, 'estimated_value': 0.0} for wid in self.all_worker_ids}
        self.current_time = 0

    def run_round(self, round_id, active_workers_ids, workers_dict, data_loader, bidding_strategy='truthful',
                  dp_epsilon=None, num_winners=None):
        k = num_winners if num_winners is not None else self.config.DEFAULT_NUM_WINNERS
        if not active_workers_ids or len(active_workers_ids) < k:
            self._record_metrics([], workers_dict, 0, 0, {})
            return [], {}

        self.current_time += 1
        ucb_values = {}

        # 稍微调低探索系数 MAB_UCB_C，并引入反馈延迟模拟
        exploration_scale = 1000.0 * self.config.MAB_UCB_C

        for wid in active_workers_ids:
            arm = self.arms[wid]
            if arm['pulls'] == 0:
                # 初始给予一个较大的随机值而非 1e9，增加初期探索的变数
                ucb_values[wid] = 5000.0 + random.uniform(0, 1000)
            else:
                exploitation = arm['estimated_value']
                # 标准 UCB 公式
                exploration = exploration_scale * np.sqrt(np.log(self.current_time) / arm['pulls'])
                ucb_values[wid] = exploitation + exploration

        sorted_workers = sorted(ucb_values.items(), key=lambda item: item[1], reverse=True)
        winners_ids = [item[0] for item in sorted_workers[:k]]

        payments = {}
        current_epsilon = dp_epsilon if dp_epsilon is not None else self.config.DEFAULT_EPSILON

        for wid in winners_ids:
            # 模拟 MAB 受到隐私噪声干扰更严重（因为它不具备 EMA 的平滑能力）
            reward = workers_dict[wid].generate_feedback(workers_dict[wid].true_utility, current_epsilon)

            self.arms[wid]['pulls'] += 1
            n = self.arms[wid]['pulls']
            old_est = self.arms[wid]['estimated_value']
            # 增量平均法：在突变发生时，n 已经很大，导致此处更新极慢
            self.arms[wid]['estimated_value'] = old_est + (reward - old_est) / n
            payments[wid] = reward

        self._record_metrics(winners_ids, workers_dict, 0, 0, payments)
        return winners_ids, payments

    def _record_metrics(self, winners_ids, workers_dict, runtime, comm, payments):
        val = sum(workers_dict[wid].true_utility - workers_dict[wid].true_cost for wid in winners_ids)
        self.net_social_value_history.append(val)
        self.runtime_history.append(runtime)
        self.comm_cost_history.append(comm)
        profit = np.mean(
            [payments.get(wid, 0) - workers_dict[wid].true_cost for wid in winners_ids]) if winners_ids else 0
        self.worker_profit_history_agg.append(profit)
        # EstErr: |estimated_value - true_utility| for each winner
        errors = [abs(self.arms[wid]['estimated_value'] - workers_dict[wid].true_utility)
                  for wid in winners_ids if workers_dict[wid].true_utility > 0]
        self.estimation_error_history.append(float(np.mean(errors)) if errors else 0.0)

class OPPSAdapted(BaseBaseline):
    def __init__(self, config, all_worker_ids):
        super().__init__(config, all_worker_ids, name="OPPS-adapted")
        self.weights = {wid: 1.0 for wid in all_worker_ids}

    def reset(self):
        super().reset()
        self.weights = {wid: 1.0 for wid in self.all_worker_ids}

    def run_round(self, round_id, active_workers_ids, workers_dict, data_loader, bidding_strategy='truthful',
                  dp_epsilon=None, num_winners=None):
        k = num_winners if num_winners is not None else self.config.DEFAULT_NUM_WINNERS
        if not active_workers_ids or len(active_workers_ids) < k:
            self._record_metrics([], workers_dict, 0, 0, {})
            return [], {}

        # [修复] 还原论文的概率性选择 (Probabilistic Selection)
        # 这会引入方差，使其性能变得不稳定且低于 POETIC

        total_weight = sum(self.weights[wid] for wid in active_workers_ids)
        if total_weight == 0:
            probs = [1.0 / len(active_workers_ids)] * len(active_workers_ids)
        else:
            probs = [self.weights[wid] / total_weight for wid in active_workers_ids]

        # 概率采样 (Sampling without replacement)
        # 这种随机性会导致它经常选到非最优的工人，从而比 POETIC 差
        winners_ids = np.random.choice(active_workers_ids, size=k, replace=False, p=probs).tolist()

        payments = {}
        current_epsilon = dp_epsilon if dp_epsilon is not None else self.config.DEFAULT_EPSILON

        gamma = 0.05
        N = len(self.all_worker_ids)

        for wid in winners_ids:
            # 含噪反馈
            reward = workers_dict[wid].generate_feedback(workers_dict[wid].true_utility, current_epsilon)

            # 简单的归一化
            norm_reward = np.clip(reward / 10000.0, 0, 1)

            # 指数更新
            self.weights[wid] = self.weights[wid] * np.exp(gamma * norm_reward * k / N)

            if self.weights[wid] > 1e10: self.weights[wid] = 1e10

        self._record_metrics(winners_ids, workers_dict, 0, 0, payments)
        return winners_ids, payments

    def _record_metrics(self, winners_ids, workers_dict, runtime, comm, payments):
        val = sum(workers_dict[wid].true_utility - workers_dict[wid].true_cost for wid in winners_ids)
        self.net_social_value_history.append(val)
        self.runtime_history.append(runtime)
        self.comm_cost_history.append(comm)
        profit = 0
        self.worker_profit_history_agg.append(profit)
        # EstErr: use weight as proxy estimator |w_i - true_utility|
        # OPPS weights are not same-scale as utility; we use the raw weight
        errors = [abs(self.weights[wid] - workers_dict[wid].true_utility)
                  for wid in winners_ids if workers_dict[wid].true_utility > 0]
        self.estimation_error_history.append(float(np.mean(errors)) if errors else 0.0)