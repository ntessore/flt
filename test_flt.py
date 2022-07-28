import numpy as np
import pytest

from flt import dlt, idlt, theta


@pytest.mark.parametrize('n', [2, 5, 10, 11, 100, 101, 1000, 1001])
@pytest.mark.parametrize('closed', [False, True])
def test_dlt(n, closed):
    t = theta(n, closed=closed)
    for i in range(n):
        a = np.zeros(n)
        a[i] = 1

        f = np.polynomial.legendre.legval(np.cos(t), a)

        np.testing.assert_allclose(dlt(f, closed=closed), a, rtol=0, atol=1e-12)


@pytest.mark.parametrize('n', [2, 5, 10, 11, 100, 101, 1000, 1001])
@pytest.mark.parametrize('closed', [False, True])
def test_idlt(n, closed):
    t = theta(n, closed=closed)
    for i in range(n):
        a = np.zeros(n)
        a[i] = 1

        f = np.polynomial.legendre.legval(np.cos(t), a)

        np.testing.assert_allclose(idlt(a, closed=closed), f, rtol=0, atol=1e-10)
