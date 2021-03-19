// dctdlt.c
// ========
// discrete Legendre transform via DCT
//
// author: Nicolas Tessore <n.tessore@ucl.ac.uk>
// license: MIT
//
// Synopsis
// --------
// The `dctdlt` and `dltdct` functions convert the coefficients of a discrete
// cosine transform (DCT) to the coefficients of a discrete Legendre transform
// (DLT) and vice versa [1].
//
// References
// ----------
// [1] Alpert, B. K., & Rokhlin, V. (1991). A fast algorithm for the evaluation
//     of Legendre expansions. SIAM Journal on Scientific and Statistical
//     Computing, 12(1), 158-179.
//

#define DCTDLT_VERSION 20210318L


// dctdlt
// ======
// convert DCT coefficients to DLT coefficients
//
// Parameters
// ----------
// n : unsigned int
//     Length of the input array.
// stride_in : unsigned int
//     Stride of the input array.
// dct : (n,) array of double
//     Input DCT coefficients.
// stride_out : unsigned int
//     Stride of the output array.
// dlt : (n,) array of double, output
//     Output DLT coefficients.
//
void dctdlt(unsigned int n, unsigned int stride_in, const double* dct,
            unsigned int stride_out, double* dlt)
{
    double a, b;
    unsigned int k, l;

    // first row
    a = 1.;
    b = a;
    *dlt = b*dct[0];
    for(k = 2; k < n; k += 2)
    {
        b *= (k-3.)/(k+1.);
        *dlt += 2*b*dct[k*stride_in];
    }

    // remaining rows
    for(l = 1; l < n; ++l)
    {
        dlt += stride_out;
        a /= (1. - 0.5/l);
        b = a;
        *dlt = b*dct[l*stride_in];
        for(k = l+2; k < n; k += 2)
        {
            b *= (k*(k+l-2.)*(k-l-3.))/((k-2.)*(k+l+1.)*(k-l));
            *dlt += b*dct[k*stride_in];
        }
    }
}


// dltdct
// ======
// convert DLT coefficients to DCT coefficients
//
// Parameters
// ----------
// n : unsigned int
//     Length of the input array.
// stride_in : unsigned int
//     Stride of the input array.
// dlt : (n,) array of double
//     Input DLT coefficients.
// stride_out : unsigned int
//     Stride of the output array.
// dct : (n,) array of double, output
//     Output DCT coefficients.
//
void dltdct(unsigned int n, unsigned int stride_in, const double* dlt,
            unsigned int stride_out, double* dct)
{
    double a, b;
    unsigned int k, l;

    // first row
    a = 1.;
    b = a;
    *dct = b*dlt[0];
    for(l = 2; l < n; l += 2)
    {
        b *= ((l-1.)*(l-1.))/(l*l);
        *dct += b*dlt[l*stride_in];
    }

    // remaining rows
    for(k = 1; k < n; ++k)
    {
        dct += stride_out;
        a *= (1. - 0.5/k);
        b = a;
        *dct = b*dlt[k*stride_in];
        for(l = k+2; l < n; l += 2)
        {
            b *= ((l-k-1.)*(l+k-1.))/((l-k)*(l+k));
            *dct += b*dlt[l*stride_in];
        }
    }
}
