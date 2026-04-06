from __future__ import annotations

import math
from typing import List

from src.fl.privacy.budget import PrivacyBudget
from src.utils.logging import get_logger

log = get_logger(__name__)

class RDPAccountant:

    def __init__(
        self,
        noise_multiplier: float,
        max_grad_norm: float,
        delta: float,
        sample_rate: float,
        target_epsilon: float,
        total_rounds: int,
        warning_threshold: float = 0.8,
    ) -> None:
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm    = max_grad_norm
        self.delta            = delta
        self.sample_rate      = sample_rate
        self.total_steps      = 0

        self.budget = PrivacyBudget(
            target_epsilon=target_epsilon,
            delta=delta,
            total_rounds=total_rounds,
            warning_threshold=warning_threshold,
        )

        self._opacus_available = self._try_import_opacus()

    def step(self, num_steps: int) -> float:

        self.total_steps += num_steps
        epsilon = self._compute_epsilon(self.total_steps)
        self.budget.update(epsilon)

        if self.budget.is_exhausted:
            log.warning(
                "Privacy budget EXHAUSTED: ε=%.4f >= target=%.4f "
                "after %d rounds. Training should stop.",
                self.budget.current_epsilon,
                self.budget.target_epsilon,
                self.budget.rounds_spent,
            )
        elif self.budget.is_warning:
            log.warning(
                "Privacy budget WARNING: ε=%.4f (%.1f%% of %.4f) "
                "after %d rounds. %d rounds remaining.",
                self.budget.current_epsilon,
                self.budget.budget_fraction_used * 100,
                self.budget.target_epsilon,
                self.budget.rounds_spent,
                self.budget.rounds_remaining,
            )
        else:
            log.debug(
                "Privacy: ε=%.4f (%.1f%% used)  round=%d/%d",
                self.budget.current_epsilon,
                self.budget.budget_fraction_used * 100,
                self.budget.rounds_spent,
                self.budget.total_rounds,
            )

        return epsilon

    def get_epsilon(self) -> float:
        return self.budget.current_epsilon

    def get_budget(self) -> PrivacyBudget:
        return self.budget

    def _compute_epsilon(self, total_steps: int) -> float:
        if self._opacus_available:
            return self._opacus_epsilon(total_steps)
        return self._analytic_rdp_epsilon(total_steps)

    def _opacus_epsilon(self, total_steps: int) -> float:
        try:
            from opacus.accountants import RDPAccountant as OpacusRDP
            accountant = OpacusRDP()
            accountant.history = [(self.noise_multiplier, self.sample_rate, total_steps)]
            epsilon = accountant.get_epsilon(delta=self.delta)
            return float(epsilon)
        except Exception as e:
            log.debug("Opacus accounting failed (%s), falling back to analytic RDP.", e)
            return self._analytic_rdp_epsilon(total_steps)

    def _analytic_rdp_epsilon(self, total_steps: int) -> float:

        sigma = self.noise_multiplier
        q     = self.sample_rate
        T     = total_steps
        best_eps = float("inf")
        for alpha in [2, 4, 8, 16, 32, 64, 128]:
            rdp = (alpha * q**2 * T) / (2 * sigma**2)
            eps = rdp + math.log(1.0 / self.delta) / max(alpha - 1, 1e-9)
            best_eps = min(best_eps, eps)

        return max(0.0, best_eps)

    @staticmethod
    def _try_import_opacus() -> bool:
        try:
            import opacus  
            return True
        except ImportError:
            log.debug("Opacus not available — using analytic RDP fallback.")
            return False

    @property
    def epsilon_history(self) -> List[float]:
        cumulative = []
        running = 0.0
        for delta_eps in self.budget.epsilon_per_round:
            running += delta_eps
            cumulative.append(round(running, 8))
        return cumulative

    def summary(self) -> str:
        return (
            f"RDPAccountant("
            f"σ={self.noise_multiplier}, C={self.max_grad_norm}, "
            f"q={self.sample_rate:.4f}, steps={self.total_steps}) | "
            f"{self.budget.summary()}"
        )

    def __repr__(self) -> str:
        return self.summary()