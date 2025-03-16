import numpy as np


class Node:
    def __init__(self, q: np.ndarray, parent) -> None:
        self.q = q
        self.parent = parent

    def __hash__(self) -> int:
        return hash(self.q.tobytes())

    # Define node equality as having the same jont config value.
    # TODO: update this? Might want to check for the same parent.
    # There is also the chance that two nodes can have the same config value,
    # but belong to different trees (maybe enforcing a parent check will help resolve this).
    def __eq__(self, other) -> bool:
        return np.array_equal(self.q, other.q)
