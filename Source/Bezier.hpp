#ifndef CMO_BEZIER_HPP
#define CMO_BEZIER_HPP

#include "Constants.hpp"
#include <cmath>


namespace Bezier
{
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

inline void cent_deriv(f64 wprime[4][4], f64 dsup, f64 dsdn, f64 chiup[4][4], f64 chic[4][4], f64 chidn[4][4])
{
    for (int j = 0; j < 4; j++)
        for (int i = 0; i < 4; i++)
            wprime[j][i] = cent_deriv(dsup, dsdn, chiup[j][i], chic[j][i], chidn[j][i]);
}

inline void cent_deriv(f64 wprime[4], f64 dsup, f64 dsdn, f64 chiup[4], f64 chic[4], f64 chidn[4])
{
    for (int i = 0; i < 4; i++)
        wprime[i] = cent_deriv(dsup, dsdn, chiup[i], chic[i], chidn[i]);
}

inline void Bezier3_coeffs(f64 dt, f64* alpha, f64* beta, f64* gamma, f64* eps, f64* edt)
{
    /* ---

     Integration coeffs. for cubic Bezier interpolants
     Use Taylor expansion if dtau is small

     --- */

    double dt2 = dt * dt, dt3 = dt2 * dt, dt4;

    if (dt >= 5.e-2)
    {
        //
        *edt = exp(-dt);

        *alpha = (-6.0 + 6.0 * dt - 3.0 * dt2 + dt3 + 6.0 * edt[0]) / dt3;
        dt3 = 1.0 / dt3;
        *beta = (6.0 + (-6.0 - dt * (6.0 + dt * (3.0 + dt))) * edt[0]) * dt3;
        *gamma = 3.0 * (6.0 + (-4.0 + dt) * dt - 2.0 * (3.0 + dt) * edt[0]) * dt3;
        *eps = 3.0 * (edt[0] * (6.0 + dt2 + 4.0 * dt) + 2.0 * dt - 6.0) * dt3;
    }
    else
    {
        dt4 = dt2 * dt2;
        *edt = 1.0 - dt + 0.5 * dt2 - dt3 / 6.0 + dt4 / 24.0;
        //
        *alpha = 0.25 * dt - 0.05 * dt2 + dt3 / 120.0 - dt4 / 840.0;
        *beta = 0.25 * dt - 0.20 * dt2 + dt3 / 12.0 - dt4 / 42.0;
        *gamma = 0.25 * dt - 0.10 * dt2 + dt3 * 0.025 - dt4 / 210.0;
        *eps = 0.25 * dt - 0.15 * dt2 + dt3 * 0.05 - dt4 / 84.0;
    }
}
}

#else
#endif