"""Classical kernel-based predictors (Chapter 2).

The pre-attention kernel toolkit that grounds the rest of the book:
$k$-nearest-neighbours, Nadaraya--Watson smoothing, kernel ridge regression.
"""
from tabkernels.classical.knn import KNNRegressor, KNNClassifier
from tabkernels.classical.nadaraya_watson import NadarayaWatson, rbf_kernel
from tabkernels.classical.krr import KernelRidgeRegression

__all__ = [
    "KNNRegressor",
    "KNNClassifier",
    "NadarayaWatson",
    "KernelRidgeRegression",
    "rbf_kernel",
]
