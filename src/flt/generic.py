"""
Generic implementation of the FLT.
"""

import functools


def _find_implementation(a: object) -> bool:
    """
    Try and import specific FLT implementations based on array type.
    """
    from sys import modules

    if numpy := modules.get("numpy"):
        if isinstance(a, numpy.ndarray):
            __import__("flt.numpy")
            return True

    if jax := modules.get("jax"):
        if isinstance(a, jax.Array):
            __import__("flt.jax")
            return True

    return False


def _dispatch(fn, a, *args, **kwargs):
    """
    Try to dispatch a function call to an implementation.
    """
    cls = type(a)
    if _find_implementation(a):
        impl = fn.dispatch(cls)
        if impl != fn:
            return impl(a, *args, **kwargs)
    msg = f"{fn.__name__} not implemented for array type {cls.__module__}.{cls.__qualname__}"
    raise NotImplementedError(msg)


@functools.singledispatch
def dct(x):
    return _dispatch(dct, x)


@functools.singledispatch
def idct(x):
    return _dispatch(idct, x)


@functools.singledispatch
def dct2dlt(a):
    """
    Convert DCT coefficients to DLT coefficients.

    Parameters
    ----------
    a : (n,) array
        DCT coefficients.

    Returns
    -------
    b : (n,) array
        DLT coefficients.

    """

    return _dispatch(dct2dlt, a)


@functools.singledispatch
def dlt2dct(b):
    """
    Convert DLT coefficients to DCT coefficients.

    Parameters
    ----------
    b : (n,) array
        DLT coefficients.

    Returns
    -------
    a : (n,) array
        DCT coefficients.

    """

    return _dispatch(dlt2dct, b)


@functools.singledispatch
def dlt(x):
    r"""
    Discrete Legendre transform.

    Takes a function in :math:`\mathtt{n}` points and returns its
    discrete Legendre transform coefficients from :math:`0` to
    :math:`\mathtt{lmax} = \mathtt{n}-1`.

    The function must be given at the points :math:`\cos(\theta)`
    returned by :func:`flt.theta`.

    Parameters
    ----------
    x : (n,) array_like
        Function values.

    Returns
    -------
    a : (n,) array_like
        Legendre coefficients :math:`0` to :math:`\mathtt{lmax}`.

    See Also
    --------
    flt.idlt : the inverse operation
    flt.theta : compute the angles at which the function is evaluated

    Notes
    -----
    The discrete Legendre transform takes a function
    :math:`f(\cos\theta)` over the domain :math:`\theta \in [0, \pi]`
    and returns the coefficients of the series

    .. math::

        f(\cos\theta) = \sum_{l=0}^{l_{\max}} a_l \, P_l(\cos\theta) \;,

    where :math:`P_l(\cos\theta)` is a Legendre polynomial.

    The computation is done in two steps: First, the function is
    transformed with a discrete cosine transform (DCT-II).  Second, the
    DCT coefficients are transformed to the DLT coefficients using a
    recursive version of the matrix relation given by [1]_.

    References
    ----------
    .. [1] Alpert, B. K., & Rokhlin, V. (1991). A fast algorithm for the
       evaluation of Legendre expansions. SIAM Journal on Scientific and
       Statistical Computing, 12(1), 158-179.

    """

    if x.ndim != 1:
        raise TypeError("array must be 1d")

    return dct2dlt(dct(x))


@functools.singledispatch
def idlt(a):
    r"""
    Inverse discrete Legendre transform.

    Takes the :math:`\mathtt{n} = \mathtt{lmax}+1` coefficients of a DLT
    and returns the corresponding function in :math:`\mathtt{n}` points.

    The function will be given at the points :math:`\cos(\theta)`
    returned by :func:`flt.theta`.

    Parameters
    ----------
    a : (n,) array_like
        DLT coefficients from :math:`0` to :math:`\mathtt{lmax}`.

    Returns
    -------
    x : (n,) array_like
        Function values.

    See Also
    --------
    flt.dlt : the forward operation
    flt.theta : compute the angles at which the function is evaluated

    Notes
    -----
    The inverse discrete Legendre transform returns a function
    :math:`f(\cos\theta)` over the domain :math:`\theta \in [0, \pi]`
    given the coefficients of the series

    .. math::

        f(\cos\theta) = \sum_{l=0}^{l_{\max}} a_l \, P_l(\cos\theta) \;,

    where :math:`P_l(\cos\theta)` is a Legendre polynomial.

    The computation is done in two steps: First, the DLT coefficients
    are transformed to the coefficients of a discrete cosine transform
    (DCT-II) using a recursive version of the matrix relation given by
    [1]_.  Second, the inverse discrete cosine transform (IDCT-II) is
    computed.

    References
    ----------
    .. [1] Alpert, B. K., & Rokhlin, V. (1991). A fast algorithm for the
       evaluation of Legendre expansions. SIAM Journal on Scientific and
       Statistical Computing, 12(1), 158-179.

    """

    if a.ndim != 1:
        raise TypeError("array must be 1d")

    return idct(dlt2dct(a))


def theta(n, *, xp=None):
    r"""
    Compute angles for DLT function values.

    Returns :math:`n` angles :math:`\theta_0, \ldots, \theta_{n-1}` at
    which the function :math:`f(\cos\theta)` is evaluated in
    :func:`flt.dlt` or :func:`flt.idlt`.

    The returned angles are distributed either over the open interval
    :math:`(0, \theta)`.

    Parameters
    ----------
    n : int
        Number of nodes.
    xp : array namespace, optional
        Return array from this array namespace.  By default, ``numpy``
        is used.

    Returns
    -------
    theta : array_like (n,)
        Angles in radians.

    """

    if xp is None:
        import numpy as xp

    return xp.pi / n * (xp.arange(n, dtype=float) + 0.5)
