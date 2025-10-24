# flt

**Fast Legendre transform for NumPy and JAX**

This is a minimal Python package for fast discrete Legendre transforms (DLTs).
The implementation uses a recursive version of the matrix relations by Alpert &
Rokhlin (1991) to compute the DLT via a discrete cosine transform (DCT).

The package can be installed using pip:

    pip install flt

For more information, please see the [documentation].

Current functionality covers the absolutely minimal use case. Please open an
issue on GitHub if you would like to see anything added.

[documentation]: https://flt.readthedocs.io/

## Array backends

The `flt` package supports generic array backends via single dispatch.
Currently available implementations are:

- NumPy+SciPy (install with `pip install flt[numpy]`)
- JAX (install with `pip install flt[jax]`)

Other implementations are easily added, even from third-party code, and will be
picked up by the `flt` methods automatically.

## Example

The main functionality of the `flt` module in contained in the pair `flt.dlt`
and `flt.idlt` of discrete Legendre transforms:

```py
>>> import jax
>>> import flt
>>> key = jax.random.key(42)
>>> x = jax.random.uniform(key, shape=(100,))
>>> a = flt.dlt(x)
>>> y = flt.idlt(a)
>>> jax.numpy.allclose(x, y)
Array(True, dtype=bool)
```
