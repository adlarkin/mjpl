import numpy as np

from .constraint_interface import Constraint


def obeys_constraints(q: np.ndarray, constraints: list[Constraint]) -> bool:
    for c in constraints:
        if not c.valid_config(q):
            return False
    return True


def apply_constraints(
    q: np.ndarray, constraints: list[Constraint]
) -> np.ndarray | None:
    q_constrained = q.copy()
    for c in constraints:
        q_constrained = c.apply(q_constrained)
        if q_constrained is None:
            return None
    return q_constrained
