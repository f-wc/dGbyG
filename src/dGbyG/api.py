"""
dGbyG API – public entry point.

This module re-exports the core classes and functions so that existing code
like ``from dGbyG.api import Compound, Reaction`` continues to work unchanged.

The actual implementations live in:
    - compound.py   → Compound
    - reaction.py   → Reaction
    - gem.py        → predict_transformed_dG_prime_for_GEM
    - _globals.py   → shared module-level state (pKa_source, model_cache, …)
    - config.py     → path configuration (infer_model_path, database paths, …)
"""

# ---- Re-export classes ----------------------------------------------------
from .compound import Compound
from .reaction import Reaction
from .gem import predict_transformed_dG_prime_for_GEM

# ---- Re-export shared globals (read/write accessible via dGbyG.api) ------
from ._globals import pKa_source, model_cache
from .config import config

# Backward-compatible alias
infer_model_path = config.infer_model_path

# ---- Convenience: allow ``from dGbyG.api import *`` -----------------------
__all__ = [
    "Compound",
    "Reaction",
    "predict_transformed_dG_prime_for_GEM",
    "pKa_source",
    "infer_model_path",
    "model_cache",
]

