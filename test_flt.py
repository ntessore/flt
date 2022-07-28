import numpy as np
import pytest

from flt import dlt, idlt, theta


@pytest.mark.parametrize('n', [1, 2, 5, 10, 11, 100, 101, 1000, 1001])
@pytest.mark.parametrize('closed', [False, True])
def test_dlt(n, closed):
    if n == 1 and closed:
        pytest.skip('closed FFT requires n > 1')

    t = theta(n, closed=closed)
    a = np.random.uniform(0, 1, size=n)
    f = np.polynomial.legendre.legval(np.cos(t), a)

    np.testing.assert_allclose(dlt(f, closed=closed), a)


@pytest.mark.parametrize('n', [1, 2, 5, 10, 11, 100, 101, 1000, 1001])
@pytest.mark.parametrize('closed', [False, True])
def test_idlt(n, closed):
    if n == 1 and closed:
        pytest.skip('closed FFT requires n > 1')

    t = theta(n, closed=closed)
    a = np.random.uniform(0, 1, size=n)
    f = np.polynomial.legendre.legval(np.cos(t), a)

    np.testing.assert_allclose(idlt(a, closed=closed), f)
