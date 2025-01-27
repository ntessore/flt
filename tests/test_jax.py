import jax
import jax.numpy as jnp
import jax.test_util
import flt

jax.config.update("jax_enable_x64", True)


def test_dlt_grads(n, rng):
    a = jnp.asarray(rng.uniform(0.0, 1.0, size=n))
    jax.test_util.check_grads(flt.dlt, (a,), 2)


def test_idlt_grads(n, rng):
    a = jnp.asarray(rng.uniform(0.0, 1.0, size=n))
    jax.test_util.check_grads(flt.idlt, (a,), 2)


def test_jit():
    dlt = jax.jit(flt.dlt)
    dlt(jnp.zeros(10))

    idlt = jax.jit(flt.idlt)
    idlt(jnp.zeros(10))
