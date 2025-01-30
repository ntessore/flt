``flt`` — Fast Legendre Transform
=================================

This is a minimal Python package for fast discrete Legendre transforms (DLTs).
The implementation uses a recursive version of the matrix relations by Alpert &
Rokhlin (1991) to compute the DLT via a discrete cosine transform (DCT).
[1]_ [2]_

The package can be installed using pip::

    pip install flt

Current functionality covers the absolutely minimal use case.  Please
open an issue on GitHub if you would like to see anything added.


Array backends
--------------

The :mod:`flt` functions support generic array backends via single
dispatch.  Currently available implementations are:

* NumPy
* JAX

Other implementations are easily added, even from third-party code, and
will be picked up by the :mod:`flt` functions automatically.


Example
-------

The main functionality of the :mod:`flt` module is the pair :func:`flt.dlt` and
:func:`flt.idlt` of discrete Legendre transforms::

    >>> import jax
    >>> import flt
    >>> key = jax.random.key(42)
    >>> x = jax.random.uniform(key, shape=(100,))
    >>> a = flt.dlt(x)
    >>> y = flt.idlt(a)
    >>> jax.numpy.allclose(x, y)
    Array(True, dtype=bool)


References
----------

.. [1] Alpert, B. K., & Rokhlin, V., 1991, SIAM Journal on Scientific and
       Statistical Computing, 12, 158. doi:10.1137/0912009

.. [2] Tessore N., Loureiro A., Joachimi B., von Wietersheim-Kramsta M.,
       Jeffrey N., 2023, OJAp, 6, 11. doi:10.21105/astro.2302.01942


API
---

.. autofunction:: flt.dlt
.. autofunction:: flt.idlt
.. autofunction:: flt.theta
.. autofunction:: flt.dct2dlt
.. autofunction:: flt.dlt2dct
