import jax
import jax.numpy as jnp
import jax.scipy.fft

import flt.generic


def _dct2dlt_row(i, n):
    """Return row *i* of the dct2dlt matrix of size *n*"""
    j = jnp.arange(i + 2, n, 2)
    if i == 0:
        x = 2 * jnp.multiply.accumulate(1 - 4 / (j + 1))
    else:
        x = jnp.multiply.accumulate(
            (1 + 2 / (j - 2)) * (1 - 3 / (j + i + 1)) * (1 - 3 / (j - i))
        )
    return jnp.concatenate([jnp.asarray([1.0]), x])


@jax.jit
def dct2dlt(a):
    """JAX implementation of dct2dlt"""

    n = a.size
    out = []
    z = 0.5 / n
    for i in range(n):
        y = _dct2dlt_row(i, n)
        out.append(z * jnp.dot(y, a[i::2]))
        z = z / (1 - 0.5 / (i + 1))

    return jnp.asarray(out)


def _dlt2dct_row(i, n):
    """Return row *i* of the dlt2dct matrix of size *n*"""
    j = jnp.arange(i + 2, n, 2)
    if i == 0:
        x = jnp.multiply.accumulate((1 - 1 / j) * (1 - 1 / j))
    else:
        x = jnp.multiply.accumulate((1 - 1 / (j - i)) * (1 - 1 / (j + i)))
    return jnp.concatenate([jnp.asarray([1.0]), x])


@jax.jit
def dlt2dct(b):
    """JAX implementation of dlt2dct"""

    n = b.size
    out = []
    z = 2.0 * n
    for i in range(n):
        y = _dlt2dct_row(i, n)
        out.append(z * jnp.dot(y, b[i::2]))
        z = z * (1 - 0.5 / (i + 1))

    return jnp.asarray(out)


flt.generic.dct.register(jax.Array, jax.scipy.fft.dct)
flt.generic.idct.register(jax.Array, jax.scipy.fft.idct)
flt.generic.dct2dlt.register(jax.Array, dct2dlt)
flt.generic.dlt2dct.register(jax.Array, dlt2dct)
