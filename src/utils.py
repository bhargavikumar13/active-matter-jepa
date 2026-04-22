"""
src/utils.py — Shared utilities for active_matter scripts

Provides:
  - DotDict       : dict with attribute access
  - load_config   : load YAML config with env var expansion + CLI overrides
  - resolve_paths : recursively expand $USER, $HOME etc. in config values
"""

import os
import yaml


class DotDict(dict):
    """dict with attribute access: cfg.training.lr"""
    def __getattr__(self, k):
        v = self[k]
        return DotDict(v) if isinstance(v, dict) else v
    def __setattr__(self, k, v): self[k] = v


def resolve_paths(obj):
    """
    Recursively expand environment variables in all string values of a
    nested dict/list structure.

    Handles:
      $USER   -> current username
      $HOME   -> home directory
      ${VAR}  -> any environment variable

    This allows configs to use $USER in paths (e.g. /scratch/$USER/...)
    and work correctly regardless of who runs the script.
    """
    if isinstance(obj, dict):
        return {k: resolve_paths(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [resolve_paths(v) for v in obj]
    elif isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj


def load_config(path: str, overrides: list = None) -> DotDict:
    """
    Load a YAML config file, expand environment variables, and apply
    optional CLI overrides of the form key.subkey=value.

    Parameters
    ----------
    path      : path to YAML config file
    overrides : list of strings like ['training.lr=1e-4', 'training.epochs=50']

    Returns
    -------
    DotDict with attribute access and all $USER/$HOME expanded
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Expand environment variables ($USER, $HOME, etc.) in all string values
    cfg = resolve_paths(cfg)

    # Apply command-line overrides: key.subkey=value
    if overrides:
        for ov in overrides:
            key_path, val = ov.split('=', 1)
            keys = key_path.split('.')
            d = cfg
            for k in keys[:-1]:
                d = d[k]
            try:    val = int(val)
            except ValueError:
                try: val = float(val)
                except ValueError: pass
            d[keys[-1]] = val

    return DotDict(cfg)
