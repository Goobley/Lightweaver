#include "LuSolve.hpp"
#include "Constants.hpp"
#include "LwMisc.hpp"
#include <cmath>
#include <exception>


void lu_decompose(F64View2D A, I32View index, f64* d)
{
    constexpr f64 Tiny = 1e-20;
    const int N = A.shape(0);
    auto vv = F64Arr(N);
    *d = 1.0;

    for (int i = 0; i < N; ++i)
    {
        f64 big = 0.0;
        for (int j = 0; j < N; ++j)
        {
            big = max(big, abs(A(i,j)));
        }
        if (big == 0.0)
            throw std::runtime_error("Singular Matrix");

        vv(i) = 1.0 / big;
    }

    for (int j = 0; j < N; ++j)
    {
        for (int i = 0; i < j; ++i)
        {
            f64 sum = A(i,j);
            for (int k = 0; k < i; ++k)
                sum -= A(i,k) * A(k,j);
            A(i,j) = sum;
        }

        int iMax = 0;
        f64 big = 0.0;
        for (int i = j; i < N; ++i)
        {
            f64 sum = A(i,j);
            for (int k = 0; k < j; ++k)
                sum -= A(i,k) * A(k,j);
            A(i,j) = sum;
            big = max_idx(big, vv(i) * abs(sum), iMax, i);
        }
        if (j != iMax)
        {
            for (int k = 0; k < N; ++k)
            {
                f64 temp = A(iMax, k);
                A(iMax, k) = A(j, k);
                A(j, k) = temp;
            }
            *d = -(*d);
            vv(iMax) = vv(j);
        }
        index(j) = iMax;
        if (A(j,j) == 0.0)
            A(j,j) = Tiny;

        if (j != N)
        {
            f64 temp = 1.0 / A(j,j);
            for (int i = j + 1; i < N; ++i)
                A(i,j) *= temp;
        }
    }
}

void lu_backsub(F64View2D A, I32View index, F64View b)
{
    const int N = A.shape(0);

    int ii  = -1;
    for (int i = 0; i < N; ++i)
    {
        int ip = index(i);
        f64 sum = b(ip);
        b(ip) = b(i);
        if (ii >= 0)
        {
            for (int j = ii; j < i; ++j)
                sum -= A(i,j) * b(j);
        }
        else if (sum != 0.0)
        {
            ii = i;
        }
        b(i) = sum;
    }

    for (int i = N - 1; i >= 0; --i)
    {
        f64 sum = b(i);
        for (int j = i + 1; j < N; ++j)
            sum -= A(i,j) * b(j);
        b(i) = sum / A(i,i);
    }
}

void solve_lin_eq(F64View2D A, F64View b, bool improve)
{
    const int N = A.shape(0);

    F64Arr1D residual;
    F64Arr2D ACopy;
    F64Arr1D bCopy;

    if (improve)
    {
        ACopy = A;
        bCopy = b;
    }

    f64 d;
    auto index = I32Arr(N);
    lu_decompose(A, index, &d);

    lu_backsub(A, index, b);
    if (improve)
    {
        residual = bCopy;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                residual(i) -= ACopy(i,j) * b(j);

        lu_backsub(A, index, residual);
        for (int i = 0; i < N; ++i)
            b(i) += residual(i);
    }
}