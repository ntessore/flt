import numpy as np
import pytest

import flt


@pytest.mark.parametrize("n", [1, 2, 5, 10, 11, 100, 101, 1000, 1001])
def test_dlt(n):
    t = flt.theta(n)
    a = np.random.uniform(0, 1, size=n)
    f = np.polynomial.legendre.legval(np.cos(t), a)

    np.testing.assert_allclose(flt.dlt(f), a)


@pytest.mark.parametrize("n", [1, 2, 5, 10, 11, 100, 101, 1000, 1001])
def test_idlt(n):
    t = flt.theta(n)
    a = np.random.uniform(0, 1, size=n)
    f = np.polynomial.legendre.legval(np.cos(t), a)

    np.testing.assert_allclose(flt.idlt(a), f)
