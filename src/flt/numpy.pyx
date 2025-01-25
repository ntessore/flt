# cython: language_level=3, boundscheck=False, embedsignature=True

import numpy as np
import scipy.fft

import flt.generic

cdef extern:
    void _dct2dlt "dctdlt" (unsigned int, unsigned int, const double*,
                            unsigned int, double*)
    void _dlt2dct "dltdct" (unsigned int, unsigned int, const double*,
                            unsigned int, double*)


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

    # length n of the transform
    n = a.size

    # output array
    b = np.empty(n, dtype=float)

    # memview for C interop
    cdef double[::1] _a = a
    cdef double[::1] _b = b

    # transform DCT coefficients to DLT coefficients using C function
    _dct2dlt(n, 1, &_a[0], 1, &_b[0])

    # done
    return b


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

    # length n of the transform
    n = b.size

    # output array
    a = np.empty(n, dtype=float)

    # memview for C interop
    cdef double[::1] _a = a
    cdef double[::1] _b = b

    # transform DLT coefficients to DCT coefficients using C function
    _dlt2dct(n, 1, &_b[0], 1, &_a[0])

    # done
    return a


flt.generic.dct.register(np.ndarray, scipy.fft.dct)
flt.generic.idct.register(np.ndarray, scipy.fft.idct)
flt.generic.dct2dlt.register(np.ndarray, dct2dlt)
flt.generic.dlt2dct.register(np.ndarray, dlt2dct)
