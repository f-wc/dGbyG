"""
Global shared state for dGbyG.

Centralizes module-level variables (pKa source, model cache, model path)
so that compound.py / reaction.py / gem.py can share them without
circular imports.
"""
# ---------------------------------------------------------------------------
# pKa source: can be set globally so every Compound inherits it
# ---------------------------------------------------------------------------
pKa_source = None

# ---------------------------------------------------------------------------
# Model cache (runtime state – the path itself lives in config.py)
# ---------------------------------------------------------------------------
model_cache = {}
