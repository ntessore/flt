#include <stdio.h>

extern void dctdlt(unsigned int n, unsigned int stride_in, const double* dct,
                   unsigned int stride_out, double* dlt);

extern void dltdct(unsigned int n, unsigned int stride_in, const double* dlt,
                   unsigned int stride_out, double* dct);

static const int n = 10;


int main()
{
    double I[n*n];
    double M[n*n];
    double L[n*n];

    unsigned int i, j, k;

    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            I[i*n+j] = (i == j);
            M[i*n+j] = i*j;
            L[i*n+j] = i*j;
        }
    }

    for (j = 0; j < n; ++j)
        dltdct(n, n, I+j, n, M+j);

    for (j = 0; j < n; ++j)
        dctdlt(n, n, I+j, n, L+j);

    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            I[i*n+j] = 0;
            for (k = 0; k < n; ++k) {
                I[i*n+j] += M[i*n+k]*L[k*n+j];
            }
        }
    }

    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            printf(" % .4f", M[i*n+j]);
        }
        printf("\n");
    }
    printf("\n");

    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            printf(" % .4f", L[i*n+j]);
        }
        printf("\n");
    }
    printf("\n");

    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            printf(" % .4f", I[i*n+j]);
        }
        printf("\n");
    }
    printf("\n");

    return 0;
}
