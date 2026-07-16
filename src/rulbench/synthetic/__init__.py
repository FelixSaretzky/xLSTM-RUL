"""Synthetic run-to-failure / RUL data generation.

Reuses the dotime CausalTimePrior SCM generator (vendored as a git submodule
under ``third_party/``) and adds a degradation + failure + response-surface
layer to turn observational SCM trajectories into labelled RUL data.
"""
from .rul_mechanism import (RulConfig, RulSample, Mechanism, FamilyCalibration,
                            calibrate_family, induce_run_to_failure)
from .rul_prior import RULPrior, generate_rul_dataset

__all__ = [
    "RulConfig", "RulSample", "Mechanism", "FamilyCalibration",
    "calibrate_family", "induce_run_to_failure",
    "RULPrior", "generate_rul_dataset",
]
