# flt: fast Legendre transform
#
# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
#
# cython: language_level=3, boundscheck=False, embedsignature=True
#
'''

Discrete Legendre Transform (:mod:`flt`)
========================================

This is a minimal Python package for fast discrete Legendre transforms (DLTs).
The implementation uses a recursive version of the matrix relations by Alpert &
Rokhlin (1991) to compute the DLT via a discrete cosine transform (DCT).

The package can be installed using pip::

    pip install flt

Current functionality covers the absolutely minimal use case.  Please open an
issue on GitHub if you would like to see anything added.


Reference/API
-------------

.. autosummary::
   :toctree: api
   :nosignatures:

   dlt
   idlt
   dltmtx
   idltmtx
   theta

'''

__all__ = [
    'dlt',
    'idlt',
    'dltmtx',
    'idltmtx',
    'theta',
]


import numpy as np
from scipy.fft import dct, idct

cdef extern from "dctdlt.c":
    void dctdlt(unsigned int, const double*, double*)
    void dltdct(unsigned int, const double*, double*)


def dlt(a, closed=False):
    r'''discrete Legendre transform

    Takes a function in :math:`\mathtt{n}` points and returns its discrete
    Legendre transform coefficients from :math:`0` to :math:`\mathtt{lmax}
    = \mathtt{n}-1`.

    The function must be given at the points :math:`\cos(\theta)` returned by
    :func:`flt.theta`.  These can be distributed either over the open interval
    :math:`\theta \in (0, \pi)`, or over the closed interval :math:`\theta \in
    [0, \pi]`, in which case :math:`\theta_0 = 0`\ and :math:`\theta_{n-1} =
    \pi`.

    Parameters
    ----------
    a : (n,) array_like
        Function values.
    closed : bool, optional
        Compute DLT over open (``closed=False``) or closed (``closed=True``)
        interval.

    Returns
    -------
    b : (n,) array_like
        Legendre coefficients :math:`0` to :math:`\mathtt{lmax}`.

    See Also
    --------
    flt.idlt : the inverse operation
    flt.theta : compute the angles at which the function is evaluated

    Notes
    -----
    The discrete Legendre transform takes a function :math:`f(\cos\theta)` over
    the domain :math:`\theta \in [0, \pi]` and returns the coefficients of the
    series

    .. math::

        f(\cos\theta) = \sum_{l=0}^{l_{\max}} c_l \, P_l(\cos\theta) \;,

    where :math:`P_l(\cos\theta)` is a Legendre polynomial.

    The computation is done in two steps:

    First, the function is transformed to the coefficients of a discrete cosine
    transform (DCT) using an inverse DCT-III for the open interval, or an
    inverse DCT-I for the closed interval.

    Second, the DCT coefficients are transformed to the DLT coefficientsu using
    a recursive version of the matrix relation given by [1]_.

    References
    ----------
    .. [1] Alpert, B. K., & Rokhlin, V. (1991). A fast algorithm for the
       evaluation of Legendre expansions. SIAM Journal on Scientific and
       Statistical Computing, 12(1), 158-179.

    '''

    # length n of the transform
    if np.ndim(a) != 1:
        raise TypeError('array must be 1d')
    n = np.shape(a)[-1]

    # type of the DCT depends on open or closed interval
    if closed:
        dcttype = 1
    else:
        dcttype = 3

    # compute the DCT coefficients
    a = idct(a, type=dcttype, axis=-1, norm=None)

    # this holds the DLT coefficients
    b = np.empty(n, dtype=float)

    # these are memviews on a and b for C interop
    cdef double[::1] a_ = a
    cdef double[::1] b_ = b

    # transform DCT coefficients to DLT coefficients using C function
    dctdlt(n, &a_[0], &b_[0])

    # done
    return b


def idlt(b, closed=False):
    r'''inverse discrete Legendre transform

    Takes the :math:`\mathtt{n} = \mathtt{lmax}+1` coefficients of a DLT and
    returns the corresponding function in :math:`\mathtt{n}` points.

    The function will be given at the points :math:`\cos(\theta)` returned by
    :func:`flt.theta`.  These can be distributed either over the open interval
    :math:`\theta \in (0, \pi)`, or over the closed interval :math:`\theta \in
    [0, \pi]`, in which case :math:`\theta_0 = 0` and :math:`\theta_{n-1} =
    \pi`.

    Parameters
    ----------
    b : (n,) array_like
        DLT coefficients from :math:`0` to :math:`\mathtt{lmax}`.
    closed : bool, optional
        Compute function over open (``closed=False``) or closed
        (``closed=True``) interval.

    Returns
    -------
    a : (n,) array_like
        Function values.

    See Also
    --------
    flt.dlt : the forward operation
    flt.theta : compute the angles at which the function is evaluated

    Notes
    -----
    The inverse discrete Legendre transform returns a function
    :math:`f(\cos\theta)` over the domain :math:`\theta \in [0, \pi]` given the
    coefficients of the series

    .. math::

        f(\cos\theta) = \sum_{l=0}^{l_{\max}} c_l \, P_l(\cos\theta) \;,

    where :math:`P_l(\cos\theta)` is a Legendre polynomial.

    The computation is done in two steps:

    First, the DLT coefficients are transformed to the coefficients of a
    discrete cosine transform (DCT) using a recursive version of the matrix
    relation given by [1]_.

    Second, the function values are computed using a DCT-III for the open
    interval, or a DCT-I for the closed interval.

    References
    ----------
    .. [1] Alpert, B. K., & Rokhlin, V. (1991). A fast algorithm for the
       evaluation of Legendre expansions. SIAM Journal on Scientific and
       Statistical Computing, 12(1), 158-179.

    '''

    # length n of the transform
    if np.ndim(b) != 1:
        raise TypeError('array must be 1d')
    n = np.shape(b)[-1]

    # type of the DCT depends on open or closed interval
    if closed:
        dcttype = 1
    else:
        dcttype = 3

    # this holds the DCT coefficients
    a = np.empty(n, dtype=float)

    # these are memviews on a and b for C interop
    cdef double[::1] a_ = a
    cdef double[::1] b_ = b

    # transform DLT coefficients to DCT coefficients using C function
    dltdct(n, &b_[0], &a_[0])

    # perform the DCT
    xi = dct(a, type=dcttype, axis=-1, norm=None)

    # done
    return xi


def dltmtx(n, closed=False):
    r'''discrete Legendre transform matrix

    Computes a matrix that performs the discrete Legendre transform
    :func:`flt.dlt` when multiplied by a vector of function values.

    Parameters
    ----------
    n : int
        Length of the transform.
    closed : bool, optional
        Compute DLT over open (``closed=False``) or closed (``closed=True``)
        interval.

    Returns
    -------
    m : (n, n) array_like
        Discrete Legendre transform matrix.

    See Also
    --------
    flt.dlt : the equivalent operation
    flt.theta : compute the angles at which the function is evaluated

    Notes
    -----
    The discrete Legendre transform :func:`flt.dlt` performs the transformation
    in place and does not compute the matrix :func:`flt.dltmtx`.

    '''

    # type of the DCT depends on open or closed interval
    if closed:
        dcttype = 1
    else:
        dcttype = 3

    # compute the DCT matrix row by row (not column by column)
    a = idct(np.eye(n), type=dcttype, axis=1, norm=None)

    # this holds the DLT matrix
    b = np.empty((n, n), dtype=float)

    # these are memviews on a and b for C interop
    cdef double[:, ::1] a_ = a
    cdef double[:, ::1] b_ = b

    # transform DCT row to DLT row using C function
    for i in range(n):
        dctdlt(n, &a_[i, 0], &b_[i, 0])

    # return transpose since we worked on rows, not columns
    return b.T


def idltmtx(n, closed=False):
    r'''inverse discrete Legendre transform matrix

    Computes a matrix that performs the inverse discrete Legendre transform
    :func:`flt.idlt` when multiplied by a vector of coefficients.

    Parameters
    ----------
    n : int
        Length of the transform.
    closed : bool, optional
        Compute inverse DLT over open (``closed=False``) or closed
        (``closed=True``) interval.

    Returns
    -------
    m : (n, n) array_like
        Inverse discrete Legendre transform matrix.

    See Also
    --------
    flt.idlt : the equivalent operation
    flt.theta : compute the angles at which the function is evaluated

    Notes
    -----
    The inverse discrete Legendre transform :func:`flt.idlt` performs the
    transformation in place and does not compute the matrix
    :func:`flt.idltmtx`.

    '''

    # type of the DCT depends on open or closed interval
    if closed:
        dcttype = 1
    else:
        dcttype = 3

    # this is the input matrix
    b = np.eye(n, dtype=float)

    # this holds the DLT part of the matrix
    a = np.empty((n, n), dtype=float)

    # these are memviews on a and b for C interop
    cdef double[:, ::1] a_ = a
    cdef double[:, ::1] b_ = b

    # transform DLT row to DCT row using C function
    for i in range(n):
        dltdct(n, &b_[i, 0], &a_[i, 0])

    # multiply by DCT matrix
    # return transpose since we worked on rows, not columns
    return dct(a, type=dcttype, axis=1, norm=None).T


def theta(n, closed=False):
    r'''compute angles for DLT function values

    Returns :math:`n` angles :math:`\theta_0, \ldots, \theta_{n-1}` at which
    the function :math:`f(\cos\theta)` is evaluated in :func:`flt.dlt` or
    :func:`flt.idlt`.

    The returned angles can be distributed either over the open interval
    :math:`(0, \theta)`, or over the closed interval :math:`[0, \pi]`, in which
    case :math:`\theta_0 = 0, \theta_{n-1} = \pi`.

    Parameters
    ----------
    n : int
        Number of nodes.

    Returns
    -------
    theta : array_like (n,)
        Angles in radians.
    closed : bool, optional
        Compute angles over open (``closed=False``) or closed (``closed=True``)
        interval.

    '''

    if closed:
        t = np.linspace(0, np.pi, n, dtype=float)
    else:
        t = np.arange(n, dtype=float)
        t += 0.5
        t *= np.pi/n

    return t
