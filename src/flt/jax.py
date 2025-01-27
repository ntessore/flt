import jax
import jax.numpy as jnp
import jax.scipy.fft

import flt.generic


@jax.jit
def _dct2dlt_init(x, j):
    """compute initial row of L matrix"""
    even = j % 2 == 0
    y = jax.lax.select(even, x * (1 - 4 / (j + 3)), x)
    y = jax.lax.select(j == 0, 2 * y, y)
    z = jax.lax.select(even, x, 0.0)
    return y, z


@jax.jit
def _dct2dlt_iter(jxb, i):
    """compute next row of L matrix

    uses the recurrence
    L[i+1, j] = L[i, j-1] * (1 + 2/(2i+1)) * (1 + 1/(j-1)) * (1 - 3/(i+j+2))

    """
    j, x, b = jxb
    a = jnp.dot(x, b)
    x = jnp.roll(x, 1) * jax.lax.select_n(
        1 + (i + 1 < j) - (i + 1 > j),
        jax.lax.zeros_like_array(x),
        (1 + 1 / (2 * j - 1)),
        (1 + 2 / (2 * i + 1)) * (1 + 1 / (j - 1)) * (1 - 3 / (i + j + 2)),
    )
    return (j, x, b), a


@jax.jit
def dct2dlt(b):
    """JAX implementation of dct2dlt"""

    n = b.size
    i = jax.lax.iota(int, n)
    _, x = jax.lax.scan(_dct2dlt_init, 1.0, i)
    _, a = jax.lax.scan(_dct2dlt_iter, (i, x, b), i)
    # apply dct normalisation
    return a / (2 * n)


@jax.jit
def _dlt2dct_init(x, j):
    """compute initial row of M matrix"""
    k = j // 2 * 2
    y = x * (1 - 1 / (k + 2))
    return y, (k - j + 1) * x


@jax.jit
def _dlt2dct_iter(jxa, i):
    """compute next row of M matrix

    uses the recurrence
    M[i+1, j] = M[i, j-1] * (1 - 1 / (i + j + 1))

    """
    j, x, a = jxa
    b = jnp.dot(x, a)
    x = (j > i) * jnp.roll(x, 1) * (1 - 1 / (i + j + 1))
    return (j, x, a), b


@jax.jit
def dlt2dct(a):
    """JAX implementation of dlt2dct"""

    n = a.size
    i = jax.lax.iota(int, n)
    _, x = jax.lax.scan(_dlt2dct_init, 1.0, i)
    _, b = jax.lax.scan(_dlt2dct_iter, (i, x, a), i)
    # apply dct normalisation
    return b * (2 * n)


flt.generic.dct.register(jax.Array, jax.scipy.fft.dct)
flt.generic.idct.register(jax.Array, jax.scipy.fft.idct)
flt.generic.dct2dlt.register(jax.Array, dct2dlt)
flt.generic.dlt2dct.register(jax.Array, dlt2dct)
