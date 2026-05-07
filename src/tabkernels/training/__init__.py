"""Training loops for tabular foundation models (Chapter 19 onwards).

PFNTrainer    — prior-data fitted networks (Chapter 19).
ICLTrainer    — ICL training with explicit dynamics tracking (Chapter 21).
"""
from tabkernels.training.pfn_trainer import (
    PFNTrainer,
    PFNTrainState,
    regression_loss,
    classification_loss,
)
from tabkernels.training.icl_trainer import ICLTrainer, ICLDiagnostics

__all__ = [
    "PFNTrainer",
    "PFNTrainState",
    "ICLTrainer",
    "ICLDiagnostics",
    "regression_loss",
    "classification_loss",
]
