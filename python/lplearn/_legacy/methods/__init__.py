from .learner import learner
from .solver import solver
try:  # PyTorch is optional; only the NumPy/PyTorch comparison path needs it.
    from .numpy_torch import numpy_vs_torch
except ImportError:  # pragma: no cover - depends on the local environment
    numpy_vs_torch = None
