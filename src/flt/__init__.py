"""

Discrete Legendre Transform (:mod:`flt`)
========================================

This is a minimal Python package for fast discrete Legendre transforms
(DLTs).  The implementation uses a recursive version of the matrix
relations by Alpert & Rokhlin (1991) to compute the DLT via a discrete
cosine transform (DCT).

The package can be installed using pip::

    pip install flt

Current functionality covers the absolutely minimal use case.  Please
open an issue on GitHub if you would like to see anything added.


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

"""

__all__ = [
    "dlt",
    "idlt",
    "dltmtx",
    "idltmtx",
    "theta",
]

from flt.numpy import (
    dlt,
    idlt,
    dltmtx,
    idltmtx,
    theta,
)
