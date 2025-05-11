"""
Generic implementation of the FLT.
"""

import functools
import numpy as np


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


def _dispatch(func):
    """
    Try to dispatch a function call to an implementation.
    """

    dispatch = None

    @functools.wraps(func)
    def resolver(x, *args, **kwargs):
        cls = type(x)
        if _find_implementation(x):
            impl = dispatch(cls)
            if impl is not resolver:
                return impl(x, *args, **kwargs)
        typ = f"{cls.__module__}.{cls.__qualname__}"
        msg = f"{resolver.__name__} not implemented for array type {typ}"
        raise NotImplementedError(msg)

    wrapper = functools.singledispatch(resolver)
    dispatch = wrapper.dispatch

    return wrapper


@_dispatch
def dct(x):
    """
    Discrete cosine transform (DCT-II).
    """


@_dispatch
def idct(x):
    """
    Inverse discrete cosine transform (IDCT-II).
    """


@_dispatch
def dct1(x):
    """
    Type-1 discrete cosine cosine transform (DCT-I).
    """


@_dispatch
def idct1(x):
    """
    Type-1 inverse discrete cosine transform (IDCT-I).
    """


@_dispatch
def dct2dlt(b):
    """
    Convert DCT coefficients to DLT coefficients.

    Parameters
    ----------
    b : (n,) array
        DCT coefficients.

    Returns
    -------
    a : (n,) array
        DLT coefficients.

    """


@_dispatch
def dlt2dct(a):
    """
    Convert DLT coefficients to DCT coefficients.

    Parameters
    ----------
    a : (n,) array
        DLT coefficients.

    Returns
    -------
    b : (n,) array
        DCT coefficients.

    """


@functools.singledispatch
def dlt(x, closed=False):
    r"""
    Discrete Legendre transform.

    Takes a function in :math:`\mathtt{n}` points and returns its
    discrete Legendre transform coefficients from :math:`0` to
    :math:`\mathtt{lmax} = \mathtt{n}-1`.

    The function must be given at the points :math:`\cos(\theta)`
    returned by :func:`flt.theta`.  These can be distributed either over
    the open interval :math:`\theta \in (0, \pi)`, or over the closed
    interval :math:`\theta \in [0, \pi]`, in which case :math:`\theta_0
    = 0` and :math:`\theta_{n-1} = \pi`.

    Parameters
    ----------
    x : (n,) array_like
        Function values.
    closed : bool, optional
        Compute DLT over open (``closed=False``) or closed
        (``closed=True``) interval.

    Returns
    -------
    a : (n,) array_like
        Legendre coefficients :math:`0` to :math:`\mathtt{lmax}`.

    Warnings
    --------
    Not all array implementations support the closed transform.

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

    The computation is done in two steps:

    First, the function is transformed to the coefficients of a discrete
    cosine transform (DCT) using a DCT-II for the open interval, or a
    DCT-I for the closed interval.

    Second, the DCT coefficients are transformed to the DLT coefficients
    using a recursive version of the matrix relation given by [1]_.

    References
    ----------
    .. [1] Alpert, B. K., & Rokhlin, V. (1991). A fast algorithm for the
       evaluation of Legendre expansions. SIAM Journal on Scientific and
       Statistical Computing, 12(1), 158-179.

    """

    if x.ndim != 1:
        raise TypeError("array must be 1d")
    n = x.shape[-1]

    if closed:
        # DCT-I with manual normalisation
        b = dct1(x) / (2 * (n - 1))
        # for DCT-I, last coefficient needs to be divided by 2
        b = b / np.floor(np.linspace(1.0, 2.0, n))
    else:
        # DCT-II with manual normalisation
        b = dct(x) / (2 * n)

    return dct2dlt(b)


@functools.singledispatch
def idlt(a, closed=False):
    r"""
    Inverse discrete Legendre transform.

    Takes the :math:`\mathtt{n} = \mathtt{lmax}+1` coefficients of a DLT
    and returns the corresponding function in :math:`\mathtt{n}` points.

    The function will be given at the points :math:`\cos(\theta)`
    returned by :func:`flt.theta`.  These can be distributed either over
    the open interval :math:`\theta \in (0, \pi)`, or over the closed
    interval :math:`\theta \in [0, \pi]`, in which case :math:`\theta_0
    = 0` and :math:`\theta_{n-1} = \pi`.

    Parameters
    ----------
    a : (n,) array_like
        DLT coefficients from :math:`0` to :math:`\mathtt{lmax}`.

    Returns
    -------
    x : (n,) array_like
        Function values.
    closed : bool, optional
        Compute DLT over open (``closed=False``) or closed
        (``closed=True``) interval.

    Warnings
    --------
    Not all array implementations support the closed transform.

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

    The computation is done in two steps:

    First, the DLT coefficients are transformed to the coefficients of a
    discrete cosine transform (DCT) using a recursive version of the
    matrix relation given by [1]_.

    Second, the function values are computed using the inverse DCT-II
    for the open interval, or the inverse DCT-I for the closed interval.

    References
    ----------
    .. [1] Alpert, B. K., & Rokhlin, V. (1991). A fast algorithm for the
       evaluation of Legendre expansions. SIAM Journal on Scientific and
       Statistical Computing, 12(1), 158-179.

    """

    if a.ndim != 1:
        raise TypeError("array must be 1d")
    n = a.shape[-1]

    b = dlt2dct(a)

    if closed:
        # for IDCT-I, last coefficient needs to be multiplied by 2
        b = b * np.floor(np.linspace(1.0, 2.0, n))
        # IDCT-I with manual normalisation
        x = idct1(b) * (2 * (n - 1))
    else:
        # IDCT-II with manual normalisation
        x = idct(b) * (2 * n)

    return x


def theta(n, closed=False, *, xp=None):
    r"""
    Compute angles for DLT function values.

    Returns :math:`n` angles :math:`\theta_0, \ldots, \theta_{n-1}` at
    which the function :math:`f(\cos\theta)` is evaluated in
    :func:`flt.dlt` or :func:`flt.idlt`.

    The returned angles can be distributed either over the open interval
    :math:`(0, \theta)`, or over the closed interval :math:`[0, \pi]`,
    in which case :math:`\theta_0 = 0, \theta_{n-1} = \pi`.

    Parameters
    ----------
    n : int
        Number of nodes.
    closed : bool, optional
        Compute angles over open (``closed=False``) or closed
        (``closed=True``) interval.
    xp : array namespace, optional
        Return array from this array namespace.  By default, ``numpy``
        is used.

    Returns
    -------
    theta : array_like (n,)
        Angles in radians.

    """

    if xp is None:
        xp = np

    if closed:
        t = xp.linspace(0.0, xp.pi, n)
    else:
        t = xp.linspace(0.0, xp.pi, n + 1)
        t = (t[:-1] + t[1:]) / 2

    return t


def quadrature(n, closed=False, *, xp=None):
    r"""
    Compute a quadrature rule for *n* points.

    Returns a set of weights to numerically integrate the function given
    at the *n* nodes :math:`\cos(\theta)` returned by :func:`flt.theta`.

    The quadrature is exact for band-limited functions, and corresponds
    to the monopole of the FLT, so that ``flt.quadrature(n) @ x`` is
    equal to ``2 * flt.dlt(x)[0]`` to numerical precision.

    >>> n = 20
    >>> x = np.random.rand(n)
    >>> flt.quadrature(n) @ x
    1.0377474091871635
    >>> 2 * flt.dlt(x)[0]
    1.0377474091871632

    Parameters
    ----------
    n : int
        Number of nodes.
    closed : bool, optional
        Compute weights for the open (``closed=False``) or closed
        (``closed=True``) interval.
    xp : array namespace, optional
        Return an array from this array namespace.  By default,
        ``numpy`` is used.

    Returns
    -------
    weights : array_like (n,)
        Weights of the quadrature rule.

    Notes
    -----
    Computes the weights :math:`w_1, \ldots, w_n` of the quadrature rule

    .. math::

        \int_{0}^{\pi} \! f(\cos\theta) \, d\theta
        \approx \sum_{i=1}^{n} w_i \, f(\cos\theta_i) \;,

    where the points :math:`\theta_1, \ldots, \theta_n` are those of the
    function :func:`flt.theta`.  The integral is twice the monopole of
    the FLT,

    .. math::

        \int_{0}^{\pi} \! f(\cos\theta) \, d\theta
        = \int_{0}^{\pi} \! f(\cos\theta) \, P_0(\cos\theta) \, d\theta
        = 2 a_0 \;.

    For band-limited functions, the FLT computes the monopole exactly
    (to numerical precision), and the function returns the corresponding
    set of quadrature weights: for the open interval, this is Fejér's
    first quadrature rule; for the closed interval, it is
    Clenshaw--Curtis quadrature.

    """

    if xp is None:
        xp = np

    b = 4 / (1 - xp.arange(0, n, 2) ** 2)
    b = xp.stack([b, xp.zeros_like(b)], axis=1)
    b = b.reshape(-1)[:n]
    if closed:
        w = idct1(b)
        w = w / (1 + (xp.arange(n) % (n - 1) == 0))
    else:
        w = idct(b)
    return w
