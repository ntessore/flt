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
// dct : (n,) array of double
//     Input DCT coefficients.
// dlt : (n,) array of double, output
//     Output DLT coefficients.
//
void dctdlt(unsigned int n, const double* dct, double* dlt)
{
    double a, b;
    unsigned int k, l;

    // first row
    a = 1.;
    b = a;
    dlt[0] = 0.5*b*dct[0];
    for(k = 2; k < n; k += 2)
    {
        b *= (k-3.)/(k+1.);
        dlt[0] += b*dct[k];
    }

    // remaining rows
    for(l = 1; l < n; ++l)
    {
        a /= (1. - 0.5/l);
        b = a;
        dlt[l] = b*dct[l];
        for(k = l+2; k < n; k += 2)
        {
            b *= (k*(k+l-2.)*(k-l-3.))/((k-2.)*(k+l+1.)*(k-l));
            dlt[l] += b*dct[k];
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
// dlt : (n,) array of double
//     Input DLT coefficients.
// dct : (n,) array of double, output
//     Output DCT coefficients.
//
void dltdct(unsigned int n, const double* dlt, double* dct)
{
    double a, b;
    unsigned int k, l;

    // first row
    a = 1.;
    b = a;
    dct[0] = b*dlt[0];
    for(l = 2; l < n; l += 2)
    {
        b *= ((l-1.)*(l-1.))/(l*l);
        dct[0] += b*dlt[l];
    }

    // remaining rows
    for(k = 1; k < n; ++k)
    {
        a *= (1. - 0.5/k);
        b = a;
        dct[k] = b*dlt[k];
        for(l = k+2; l < n; l += 2)
        {
            b *= ((l-k-1.)*(l+k-1.))/((l-k)*(l+k));
            dct[k] += b*dlt[l];
        }
    }
}
