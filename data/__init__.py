"""Deterministic data utilities.

MNIST loader objects live in :mod:`data.mnist`. They are intentionally not
imported here so ``python -m data.mnist`` has a clean, single module import.
"""

from .splits import create_stratified_split, validate_split_indices

__all__ = ["create_stratified_split", "validate_split_indices"]
