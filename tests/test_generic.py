import numpy as np
import flt


def test_dlt(xp, n, rng):
    t = flt.theta(n)
    a = rng.uniform(0.0, 1.0, size=n)
    f = np.polynomial.legendre.legval(np.cos(t), a)

    arg = xp.asarray(f)
    out = flt.dlt(arg)

    assert isinstance(out, type(arg))

    np.testing.assert_allclose(out, xp.asarray(a, dtype=out.dtype))


def test_idlt(xp, n, rng):
    t = flt.theta(n)
    a = xp.asarray(rng.uniform(0.0, 1.0, size=n))
    f = xp.asarray(np.polynomial.legendre.legval(np.cos(t), a))

    out = flt.idlt(a)

    assert isinstance(out, type(a))
    np.testing.assert_allclose(out, f)
