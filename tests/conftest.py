"""Shared pytest fixtures."""
import pytest
import torch


@pytest.fixture(autouse=True)
def set_seeds():
    """Set deterministic seeds for every test."""
    torch.manual_seed(42)


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def random_features():
    """A small random feature matrix for testing kernel APIs."""
    torch.manual_seed(0)
    return torch.randn(20, 8)
