import numpy as np

from .constraint_interface import Constraint


def obeys_constraints(q: np.ndarray, constraints: list[Constraint]) -> bool:
    """Check if a configuration obeys constraints.

    Args:
        q: The configuration.
        constraints: The constraints.

    Returns:
        True if `q` obeys each constraint in `constraints`. False otherwise.
    """
    for c in constraints:
        if not c.valid_config(q):
            return False
    return True


def apply_constraints(
    q_old: np.ndarray, q: np.ndarray, constraints: list[Constraint]
) -> np.ndarray | None:
    """Apply constraints to a configuration.

    Args:
        q_old: An older configuration w.r.t `q`.
        q: The configuration to apply `constraints` to.
        constraints: The constraints to apply. The order of this list is the order in
            which the constraints are applied.

    Returns:
        A configuration derived from `q` that adheres to all constraints, or None if
        applying `constraints` to `q` is not possible.
    """
    q_constrained = q
    for c in constraints:
        q_constrained = c.apply(q_old, q_constrained)
        if q_constrained is None:
            return None
    # Make sure that applying a constraint does not invalidate previously applied constraints.
    return q_constrained if obeys_constraints(q_constrained, constraints) else None
