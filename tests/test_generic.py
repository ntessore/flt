import numpy as np
import pytest
import flt


def test_dlt(xp, n, closed, rng):
    t = flt.theta(n, closed)
    a = xp.asarray(rng.uniform(0.0, 1.0, size=n))
    f = xp.asarray(np.polynomial.legendre.legval(np.cos(t), a))

    try:
        out = flt.dlt(f, closed)
    except NotImplementedError as e:
        pytest.skip(str(e))

    assert isinstance(out, type(f))
    np.testing.assert_allclose(out, a)


def test_idlt(xp, n, closed, rng):
    t = flt.theta(n, closed)
    a = xp.asarray(rng.uniform(0.0, 1.0, size=n))
    f = xp.asarray(np.polynomial.legendre.legval(np.cos(t), a))

    try:
        out = flt.idlt(a, closed)
    except NotImplementedError as e:
        pytest.skip(str(e))

    assert isinstance(out, type(a))
    np.testing.assert_allclose(out, f)
