from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

@dataclass
class PrivacyBudget:

    target_epsilon:    float
    delta:             float
    total_rounds:      int
    current_epsilon:   float          = 0.0
    rounds_spent:      int            = 0
    warning_threshold: float          = 0.8
    epsilon_per_round: List[float]    = field(default_factory=list)

    @property
    def rounds_remaining(self) -> int:
        return max(0, self.total_rounds - self.rounds_spent)

    @property
    def budget_fraction_used(self) -> float:
        if self.target_epsilon <= 0:
            return 1.0
        return self.current_epsilon / self.target_epsilon

    @property
    def is_exhausted(self) -> bool:
        return self.current_epsilon >= self.target_epsilon

    @property
    def is_warning(self) -> bool:
        return self.budget_fraction_used >= self.warning_threshold

    @property
    def projected_final_epsilon(self) -> float:

        if self.rounds_spent == 0:
            return 0.0
        avg_per_round = self.current_epsilon / self.rounds_spent
        return avg_per_round * self.total_rounds

    

    def update(self, new_epsilon: float) -> None:

        delta_epsilon = max(0.0, new_epsilon - self.current_epsilon)
        self.epsilon_per_round.append(round(delta_epsilon, 8))
        self.current_epsilon = new_epsilon
        self.rounds_spent += 1

    def as_dict(self, round_num: int | None = None) -> dict:
        return {
            "round":              round_num if round_num is not None else self.rounds_spent,
            "epsilon":            round(self.current_epsilon, 6),
            "delta":              self.delta,
            "target_epsilon":     self.target_epsilon,
            "budget_used_pct":    round(self.budget_fraction_used * 100, 2),
            "rounds_remaining":   self.rounds_remaining,
            "projected_epsilon":  round(self.projected_final_epsilon, 6),
        }

    def summary(self) -> str:
        return (
            f"PrivacyBudget("
            f"ε={self.current_epsilon:.4f}/{self.target_epsilon}, "
            f"δ={self.delta:.1e}, "
            f"used={self.budget_fraction_used*100:.1f}%, "
            f"rounds={self.rounds_spent}/{self.total_rounds})"
        )

    def __repr__(self) -> str:
        return self.summary()