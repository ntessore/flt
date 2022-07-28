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

.. currentmodule:: flt

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
    void dctdlt(unsigned int, unsigned int, const double*,
                unsigned int, double*)
    void dltdct(unsigned int, unsigned int, const double*,
                unsigned int, double*)


def dlt(a, closed=False):
    r'''discrete Legendre transform

    Takes a function in :math:`\mathtt{n}` points and returns its discrete
    Legendre transform coefficients from :math:`0` to :math:`\mathtt{lmax}
    = \mathtt{n}-1`.

    The function must be given at the points :math:`\cos(\theta)` returned by
    :func:`flt.theta`.  These can be distributed either over the open interval
    :math:`\theta \in (0, \pi)`, or over the closed interval :math:`\theta \in
    [0, \pi]`, in which case :math:`\theta_0 = 0` and :math:`\theta_{n-1} =
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

    Second, the DCT coefficients are transformed to the DLT coefficients using
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
    b = idct(a, type=dcttype, axis=-1, norm='backward')

    # fix last coefficient for DCT-I
    if closed:
        b[-1] /= 2

    # memview for C interop
    cdef double[::1] b_ = b

    # transform DCT coefficients to DLT coefficients using C function
    dctdlt(n, 1, &b_[0], 1, &b_[0])

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

    # memviews for C interop
    cdef double[::1] a_ = a
    cdef double[::1] b_ = b

    # transform DLT coefficients to DCT coefficients using C function
    dltdct(n, 1, &b_[0], 1, &a_[0])

    # fix last coefficient for DCT-I
    if closed:
        a[-1] *= 2

    # perform the DCT
    return dct(a, type=dcttype, axis=-1, norm='backward', overwrite_x=True)


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

    # compute the DCT matrix
    a = idct(np.eye(n), type=dcttype, axis=0, norm=None, overwrite_x=True)

    # memview for C interop
    cdef double[:, ::1] a_ = a

    # transform DCT column to DLT column using C function
    for i in range(n):
        dctdlt(n, n, &a_[0, i], n, &a_[0, i])

    # done
    return a


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
    a = np.eye(n, dtype=float)

    # memview for C interop
    cdef double[:, ::1] a_ = a

    # transform DLT unit column to DCT column using C function
    for i in range(n):
        dltdct(n, n, &a_[0, i], n, &a_[0, i])

    # multiply by DCT matrix
    return dct(a, type=dcttype, axis=0, norm=None, overwrite_x=True)


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
