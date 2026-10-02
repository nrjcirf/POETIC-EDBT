# src/worker.py
import numpy as np
import random


class Worker:
    def __init__(self, worker_id, config):
        self.id = worker_id
        self.config = config
        self.base_quality = random.uniform(*self.config.BASE_QUALITY_RANGE)
        self.unit_cost_factor = random.uniform(*self.config.UNIT_COST_FACTOR_RANGE)
        self.current_strategy = 'truthful'
        self.force_utility = None
        self.force_cost = None
        self.reset()

    def reset(self):
        self.online_status = 1.0
        self.true_utility = 0.0
        self.true_cost = 0.0
        self.reputation = 0.5
        # NOTE: current_strategy is intentionally NOT reset here.
        # Use set_strategy() before _sim() to assign a strategy that persists.
        # self.current_strategy is set in __init__ and via set_strategy().

        # --- Adaptive behavior state (ICDE: strategic δ_{i,t}) ---
        self.consecutive_rejections = 0   # rounds since last selection
        self.last_payment = None          # most recent payment received
        self._delta = 0.0                 # current strategic distortion δ_{i,t}

        # Sliding-window history for Φ() and stability scoring
        self._window = None               # resolved lazily from config
        self.bid_history = []             # recent submitted bids
        self.utility_history = []         # recent true_utility values

        # Current round tracker (fixes forward_looking strategy bug)
        self.current_round = 0

    # ------------------------------------------------------------------
    # Platform → Worker feedback callbacks
    # ------------------------------------------------------------------

    def notify_selected(self, payment: float):
        """
        Called by the platform after this worker wins a round.
        Resets rejection pressure and records received payment.
        """
        self.last_payment = payment
        self.consecutive_rejections = 0

    def notify_rejected(self):
        """
        Called by the platform when this worker was active but NOT selected.
        Increments rejection pressure, which raises the underbid incentive δ.
        """
        self.consecutive_rejections += 1

    def set_strategy(self, strategy: str):
        """Explicitly set the bidding strategy (used by experiments and main.py)."""
        self.current_strategy = strategy
    # ------------------------------------------------------------------
    # Strategic distortion δ_{i,t}
    # ------------------------------------------------------------------

    def _compute_delta(self) -> float:
        """
        Compute the strategic bid distortion δ_{i,t}.
        Aggressive adaptive behavior to create meaningful contamination.
        """
        cost = self.true_cost
        threshold = getattr(self.config, 'REJECTION_THRESHOLD', 3)

        # Rule 1: Sustained rejection pressure → aggressive underbidding
        if self.consecutive_rejections >= threshold:
            # Grows sharply with rejections, capped at 60% cost reduction
            pressure = min(self.consecutive_rejections / 5.0, 0.60)
            return -cost * pressure

        if self.last_payment is not None:
            surplus_ratio = (self.last_payment - cost) / max(cost, 1e-6)
            # Thin margin → moderate underbid to stay competitive
            if 0.0 < surplus_ratio < 0.15:
                return -cost * 0.20
            # Comfortable surplus → aggressive markup to extract rent
            if surplus_ratio > 0.20:
                return cost * random.uniform(0.30, 0.60)

        return 0.0

    # ------------------------------------------------------------------
    # State update (called once per round before bidding)
    # ------------------------------------------------------------------

    def update_state_for_round(self, task_context, data_loader, current_round_id):
        # Track round number (fixes forward_looking strategy)
        self.current_round = current_round_id

        # Resolve history window from config (lazy, once)
        if self._window is None:
            self._window = getattr(self.config, 'WORKER_HISTORY_WINDOW', 10)

        # Force-override mode (used in quality-aware reputation experiments)
        if self.force_utility is not None:
            self.true_utility = self.force_utility
            self.true_cost = self.force_cost if self.force_cost is not None else self.true_utility * 0.8
            self.online_status = 1.0
            return

        # 1. Concept drift at round 2500
        is_post_drift = current_round_id > 2500
        if is_post_drift:
            effective_quality = max(0.1, 2.0 - self.base_quality)
            effective_cost_factor = max(0.05, 0.5 - self.unit_cost_factor)
        else:
            effective_quality = self.base_quality
            effective_cost_factor = self.unit_cost_factor

        # 2. Online status stochastic switch
        if random.random() < self.config.NON_STATIONARY_SWITCH_PROB:
            self.online_status = 0.3 if self.online_status == 1.0 else 1.0

        worker_perf = effective_quality * self.online_status

        # 3. Compute true utility and cost
        if task_context is not None:
            density = task_context.get('density', 1.0)
            importance = task_context.get('importance', 1.0)
            dens_factor = np.log1p(density) + 1.0
            imp_factor = np.log1p(importance) + 1.0
            self.true_utility = worker_perf * 500.0 * dens_factor * random.uniform(0.8, 1.2)
            self.true_cost = effective_cost_factor * 100.0 * imp_factor * random.uniform(0.8, 1.2)
            if self.true_utility < self.true_cost and self.online_status > 0.8:
                self.true_utility = self.true_cost * random.uniform(1.05, 1.15)
        else:
            self.true_utility = worker_perf * random.uniform(100, 500)
            self.true_cost = effective_cost_factor * random.uniform(50, 400)

        # 4. Slide utility history
        self.utility_history.append(self.true_utility)
        if len(self.utility_history) > self._window:
            self.utility_history.pop(0)

        # 5. Recompute strategic distortion δ_{i,t} for this round
        self._delta = self._compute_delta()

    # ------------------------------------------------------------------
    # Bidding
    # ------------------------------------------------------------------

    def make_bid(self, strategy=None) -> float:
        """
        Return the worker's submitted bid.

        For the 'adaptive' strategy, the bid includes the strategic
        distortion δ_{i,t} computed from interaction history.
        All other strategies remain unchanged for IC verification.
        """
        use_strat = strategy if strategy is not None else self.current_strategy
        cost = self.true_cost

        if use_strat == 'truthful':
            bid = cost

        elif use_strat == 'adaptive':
            bid = cost + self._delta

        elif use_strat == 'consistent_overbid':
            # Aggressive: 2x–3x overbid to extract maximum rent
            bid = cost * random.uniform(2.0, 3.0)

        elif use_strat == 'consistent_underbid':
            bid = cost * 0.5

        elif use_strat == 'random_overbid':
            bid = cost * random.uniform(1.5, 3.5)

        elif use_strat == 'forward_looking':
            # Strong reputation farming: deep underbid first half, heavy extraction second half
            if self.current_round < (self.config.TOTAL_ROUNDS / 2):
                bid = cost * 0.3    # build reputation cheaply
            else:
                bid = cost * 2.5    # extract rent aggressively

        elif use_strat == 'utility_spoofer':
            # Reports truthful bid but will spoof feedback — bid is truthful
            bid = cost

        elif use_strat == 'oscillatory':
            # Alternating overbid/underbid to game selection pattern
            if (self.current_round % 20) < 10:
                bid = cost * 0.4
            else:
                bid = cost * 2.8

        else:
            bid = cost

        # Slide bid history (used by platform for stability scoring)
        if self._window is not None:
            self.bid_history.append(bid)
            if len(self.bid_history) > self._window:
                self.bid_history.pop(0)

        return max(0.0, bid)

    # ------------------------------------------------------------------
    # Feedback generation (with differential privacy)
    # ------------------------------------------------------------------

    def generate_feedback(self, payment, epsilon):
        """
        Generate the noisy utility signal ũ_{i,t} sent to the platform.
        Strategic workers (utility_spoofer, consistent_overbid, oscillatory)
        INFLATE feedback to poison the platform's weight estimates.
        This is the core contamination source for EstErr.
        """
        feedback_signal = self.true_utility

        # --- Strategic utility spoofing (feedback contamination) ---
        if self.current_strategy in ('utility_spoofer', 'consistent_overbid', 'oscillatory'):
            # Inflate reported utility 3-6x to poison platform weight updates
            spoof_multiplier = random.uniform(3.0, 6.0)
            feedback_signal = self.true_utility * spoof_multiplier
        elif self.current_strategy in ('adaptive', 'forward_looking'):
            # Moderate inflation during exploitation phase
            if self.last_payment is not None:
                surplus_ratio = ((self.last_payment - self.true_cost) /
                                 max(self.true_cost, 1e-6))
                if surplus_ratio > 0.20:
                    feedback_signal = self.true_utility * random.uniform(1.5, 2.5)

        # --- DP noise ---
        if epsilon is None or epsilon == np.inf:
            return feedback_signal
        scale = self.config.LAPLACE_SENSITIVITY / epsilon
        return feedback_signal + np.random.laplace(0, scale)

    # ------------------------------------------------------------------
    # Stability metrics (queried by platform for Φ and reputation)
    # ------------------------------------------------------------------

    def compute_bid_volatility(self) -> float:
        """
        Coefficient of Variation (CV) of recent bids.
        High value → worker is behaving erratically / strategically.
        Returns 0 if insufficient history.
        """
        if len(self.bid_history) < 2:
            return 0.0
        mean_bid = np.mean(self.bid_history)
        if mean_bid < 1e-9:
            return 0.0
        return float(np.std(self.bid_history) / mean_bid)

    def compute_utility_stability(self) -> float:
        """
        Utility stability score in [0, 1].
        1.0 = perfectly stable utility; 0.0 = wildly fluctuating.
        Returns 1.0 if insufficient history (benefit of the doubt).
        """
        if len(self.utility_history) < 2:
            return 1.0
        mean_u = np.mean(self.utility_history)
        if mean_u < 1e-9:
            return 1.0
        cv = np.std(self.utility_history) / mean_u
        return float(1.0 - np.clip(cv, 0.0, 1.0))