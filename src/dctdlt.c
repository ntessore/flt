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

#define DCTDLT_VERSION 20250124L


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
    unsigned int i, j;

    a = 0.5/n;
    b = 1./n;
    if(n > 0)
    {
        *dlt = dct[0] / (2.*n);
        for(j = 2; j < n; j += 2)
        {
            b *= (1 - 4./(j+1.));
            *dlt += b*dct[j*stride_in];
        }
    }
    for(i = 1; i < n; i += 1)
    {
        dlt += stride_out;
        a /= (1. - 0.5/i);
        b = a;
        *dlt = b*dct[i*stride_in];
        for(j = i+2; j < n; j += 2)
        {
            b *= (1. + 2./(j-2.)) * (1. - 3./(j+i+1.)) * (1. - 3./(j-i));
            *dlt += b*dct[j*stride_in];
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
    unsigned int i, j;

    a = 2.*n;
    b = 2.*n;
    if(n > 0)
    {
        *dct = b * dlt[0];
        for(j = 2; j < n; j += 2)
        {
            b *= (1. - 1./j)*(1. - 1./j);
            *dct += b*dlt[j*stride_in];
        }
    }
    for(i = 1; i < n; i += 1)
    {
        dct += stride_out;
        a *= (1. - 0.5/i);
        b = a;
        *dct = b*dlt[i*stride_in];
        for(j = i+2; j < n; j += 2)
        {
            b *= (1. - 1./(j-i)) * (1. - 1./(j+i));
            *dct += b*dlt[j*stride_in];
        }
    }
}
