# src/protocols.py
import numpy as np
import math
import random


class Protocols:
    def __init__(self, config):
        self.config = config

    def secure_multistage_selection(self, net_social_values, k):
        n = len(net_social_values)
        if n <= k:
            candidates = sorted(net_social_values.items(), key=lambda x: x[1], reverse=True)
            winner_ids = [c[0] for c in candidates]
            k_plus_1_value = -np.inf
            runtime = n * self.config.SMPC_RUNTIME_PER_COMPARE_MS + self.config.NETWORK_LATENCY_MS
            comm_cost = n * self.config.SMPC_COMM_OVERHEAD_PER_COMPARE_MB
            return winner_ids, k_plus_1_value, comm_cost, runtime

        g = self.config.DEFAULT_GROUP_SIZE
        c = self.config.WINNERS_PER_GROUP

        candidates = list(net_social_values.items())
        random.shuffle(candidates)

        num_groups = math.ceil(n / g)
        finalist_pool = []

        stage1_comparisons = num_groups * (g * np.log2(g + 1e-9))

        for i in range(num_groups):
            group = candidates[i * g: (i + 1) * g]
            if not group: continue
            group.sort(key=lambda x: x[1], reverse=True)
            finalist_pool.extend(group[:c])

        stage2_comparisons = len(finalist_pool) * np.log2(len(finalist_pool) + 1e-9)
        finalist_pool.sort(key=lambda x: x[1], reverse=True)

        winner_ids = [c[0] for c in finalist_pool[:k]]
        k_plus_1_value = finalist_pool[k][1] if len(finalist_pool) > k else -np.inf

        total_comparisons = stage1_comparisons + stage2_comparisons
        runtime = total_comparisons * self.config.SMPC_RUNTIME_PER_COMPARE_MS + 2 * self.config.NETWORK_LATENCY_MS
        comm_cost = total_comparisons * self.config.SMPC_COMM_OVERHEAD_PER_COMPARE_MB

        return winner_ids, k_plus_1_value, comm_cost, runtime

    def secure_reputation_payment(self, winning_bids, winning_weights, reputations, v_k_plus_1):
        num_winners = len(winning_bids)
        runtime = (num_winners * 4) * self.config.SMPC_RUNTIME_PER_COMPARE_MS + self.config.NETWORK_LATENCY_MS
        comm_cost = (num_winners * 4) * self.config.SMPC_COMM_OVERHEAD_PER_COMPARE_MB

        payments = {}
        beta = self.config.REPUTATION_BONUS_FACTOR

        for worker_id in winning_bids:
            weight = winning_weights[worker_id]
            reputation = reputations[worker_id]

            if v_k_plus_1 == -np.inf:
                base_payment = weight
            else:
                # 【核心修复】彻底删除 min()，严格遵循临界值支付，与工人自身报价解耦！
                base_payment = max(0, weight - v_k_plus_1)

            final_payment = base_payment * (1 + beta * reputation)
            payments[worker_id] = final_payment

        return payments, comm_cost, runtime