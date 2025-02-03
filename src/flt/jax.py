import jax
import jax.numpy as jnp
import jax.scipy.fft

import flt.generic


@jax.jit
def dct2dlt(b):
    """JAX implementation of dct2dlt"""

    def first(x, j):
        """compute initial row of L matrix"""
        even = j % 2 == 0
        y = jax.lax.select(even, x * (1 - 4 / (j + 3)), x)
        y = jax.lax.select(j == 0, 2 * y, y)
        z = jax.lax.select(even, x, 0.0)
        return y, z

    def rest(jxb, i):
        """
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

    n = b.shape[-1]
    i = jax.lax.iota(int, n)
    _, x = jax.lax.scan(first, 1.0, i)
    _, a = jax.lax.scan(rest, (i, x, b), i)
    return a


@jax.jit
def dlt2dct(a):
    """JAX implementation of dlt2dct"""

    def first(x, j):
        """compute initial row of M matrix"""
        k = j // 2 * 2
        y = x * (1 - 1 / (k + 2))
        return y, (k - j + 1) * x

    def rest(jxa, i):
        """
        uses the recurrence
        M[i+1, j] = M[i, j-1] * (1 - 1 / (i + j + 1))
        """
        j, x, a = jxa
        b = jnp.dot(x, a)
        x = (j > i) * jnp.roll(x, 1) * (1 - 1 / (i + j + 1))
        return (j, x, a), b

    n = a.shape[-1]
    i = jax.lax.iota(int, n)
    _, x = jax.lax.scan(first, 1.0, i)
    _, b = jax.lax.scan(rest, (i, x, a), i)
    return b


flt.generic.dct.register(jax.Array, jax.scipy.fft.dct)
flt.generic.idct.register(jax.Array, jax.scipy.fft.idct)
flt.generic.dct2dlt.register(jax.Array, dct2dlt)
flt.generic.dlt2dct.register(jax.Array, dlt2dct)
