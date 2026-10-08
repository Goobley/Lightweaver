"""
Tests for small helpers in `lightweaver.utils`.
"""

import numpy as np
import pytest

from lightweaver.utils import view_flatten


def test_view_flatten_returns_view():
    x = np.arange(12.0).reshape(3, 4)
    flat = view_flatten(x)
    assert flat.shape == (12,)
    flat[0] = -1.0
    assert x[0, 0] == -1.0


def test_view_flatten_raises_without_view():
    x = np.arange(12.0).reshape(3, 4).T
    with pytest.raises(ValueError):
        view_flatten(x)


def test_view_flatten_empty():
    assert view_flatten(np.zeros((0, 3))).shape == (0,)
