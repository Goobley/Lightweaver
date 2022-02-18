#ifndef CMO_BEZIER_HPP
#define CMO_BEZIER_HPP

#include "Constants.hpp"
#include "Simd.hpp"
#include <cmath>


namespace Bezier
{
#if defined(__STDC_IEC_559__) || (defined(_WIN32) && defined(_M_AMD64))
// NOTE(cmo): This checks to ensure we have a valid IEE754 environment~ish
inline ForceInline f64 my_copysign(f64 a, f64 b)
{
    // NOTE(cmo): Copy sign of b onto a.
    // The standard library one doesn't seem to inline in Windows
    union Pun
    {
        double f;
        uint64_t i;
    };
    Pun pa, pb;
    pa.f = abs(a);
    pb.f = b;

    pa.i |= pb.i & 1ULL << 63; // or with sign bit of b
    return pa.f;
}
#else
inline f64 my_copysign(f64 a, f64 b)
{
    return std::copysign(a, b);
}
#endif

#ifdef USE_FRITSCH_DERIVATIVE
inline f64 cent_deriv(f64 dsup, f64 dsdn, f64 chiup, f64 chic, f64 chidn)
{
    /* --- Derivative Fritsch & Butland (1984) --- */

    double fim1, fi, alpha, wprime;

    fim1 = (chic - chiup) / dsup;
    fi = (chidn - chic) / dsdn;

    if (fim1 * fi > 0.0)
    {
        alpha = 1.0 / 3.0 * (1.0 + dsdn / (dsdn + dsup));
        wprime = (fim1 * fi) / ((1.0 - alpha) * fim1 + alpha * fi);
    }
    else
    {
        wprime = 0.0;
    }
    return wprime;
}
#else
inline ForceInline f64 cent_deriv(f64 dsuw, f64 dsdw, f64 yuw, f64 y0, f64 ydw)
{
    // Steffen (1990) derivatives
    const f64 S0 = (ydw - y0) / dsdw;
    const f64 Suw = (y0 - yuw) / dsuw;
    const f64 P0 = abs((Suw * dsdw + S0 * dsuw) / (dsdw + dsuw));
    return (my_copysign(1.0, S0) + my_copysign(1.0, Suw)) * min(abs(Suw), min(abs(S0), 0.5 * P0));
}
#endif

inline void cent_deriv(F64View2D& wprime, f64 dsup, f64 dsdn, const F64View2D& chiup, const F64View2D& chic, const F64View2D& chidn)
{
    for (int j = 0; j < 4; j++)
        for (int i = 0; i < 4; i++)
            wprime(j, i) = cent_deriv(dsup, dsdn, chiup(j, i), chic(j, i), chidn(j, i));
}

inline void cent_deriv(F64View& wprime, f64 dsup, f64 dsdn, const F64View& chiup, const F64View& chic, const F64View& chidn)
{
    for (int i = 0; i < 4; i++)
        wprime(i) = cent_deriv(dsup, dsdn, chiup(i), chic(i), chidn(i));
}

inline ForceInline void Bezier3_coeffs(f64 dt, f64* alpha, f64* beta, f64* gamma, f64* delta, f64* edt)
{
    /* ---

    Integration coeffs. for cubic Bezier interpolants
    Use Taylor expansion if dtau is small

    Coefficients go with:
        alpha: Suw
        beta: S0
        gamma: Cuw
        delta: C0
        edt: exp(-dtau)

     --- */

    f64 dt2 = square(dt);
    f64 dt3 = dt2 * dt;

    if (dt < 5e-2)
    {
        *edt = 1.0 - dt + 0.5 * dt2 - dt3 / 6.0;
        // NOTE(cmo): To get the correct taylor expansion need to take terms up
        // to dt^6 due to the 1/dt^3 term. Then truncate at dt^3.
        *alpha = 0.25 * dt - 0.2 * dt2 + dt3 / 12.0;
        *beta = 0.25 * dt - 0.05 * dt2 + dt3 / 120.0;
        *gamma = 0.25 * dt - 0.15 * dt2 + 0.05 * dt3;
        *delta = 0.25 * dt - 0.1 * dt2 + 0.025 * dt3;
    }
    else if (dt > 30.0)
    {
        *edt = 0.0;
        *alpha = 6.0 / dt3;
        *beta = (-6.0 + 6.0 * dt - 3.0 * dt2 + dt3) / dt3;
        *gamma = 3.0 * (2.0 * dt - 6.0) / dt3;
        *delta = 3.0 * (6.0 - 4.0 * dt + dt2) / dt3;
    }
    else
    {
        *edt = exp(-dt);

        *alpha = (6.0 - *edt * (6.0 + 6.0 * dt + 3 * dt2 + dt3)) / dt3;
        *beta = (6.0 * *edt - 6.0 + 6.0 * dt - 3.0 * dt2 + dt3) / dt3;
        *gamma = 3.0 * (2.0 * dt - 6.0 + *edt * (6.0 + 4.0 * dt + dt2)) / dt3;
        *delta = 3.0 * (6.0 - 4.0 * dt + dt2 - 2.0 * *edt * (3.0 + dt)) / dt3;
    }
}

constexpr bool LimitControlPoints = false;
constexpr f64 limit_control_point(f64 c)
{
    if (LimitControlPoints)
        return max(c, 0.0);
    else
        return c;
}
}

#else
#endif