import jax
import jax.test_util
import numpy as np
import pytest

import flt

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("n", [1, 2, 5, 10, 11, 100, 101, 1000, 1001])
def test_dlt(n, rng):
    t = flt.theta(n)
    a = rng.uniform(0, 1, size=n)
    f = np.polynomial.legendre.legval(np.cos(t), a)

    out = flt.dlt(jax.numpy.asarray(f))
    assert isinstance(out, jax.Array)
    np.testing.assert_allclose(out, a)


@pytest.mark.parametrize("n", [1, 10, 11, 100, 101, 1000, 1001])
def test_dlt_grads(n, rng):
    a = rng.uniform(0, 1, size=n)
    jax.test_util.check_grads(flt.dlt, (a,), 2)


@pytest.mark.parametrize("n", [1, 2, 5, 10, 11, 100, 101, 1000, 1001])
def test_idlt(n, rng):
    t = flt.theta(n)
    a = rng.uniform(0, 1, size=n)
    f = np.polynomial.legendre.legval(np.cos(t), a)

    out = flt.idlt(jax.numpy.asarray(a))
    assert isinstance(out, jax.Array)
    np.testing.assert_allclose(out, f)


@pytest.mark.parametrize("n", [1, 10, 11, 100, 101, 1000, 1001])
def test_idlt_grads(n, rng):
    a = rng.uniform(0, 1, size=n)
    jax.test_util.check_grads(flt.idlt, (a,), 2)


def test_jit():
    dlt = jax.jit(flt.dlt)
    dlt(jax.numpy.zeros(10))

    idlt = jax.jit(flt.idlt)
    idlt(jax.numpy.zeros(10))
