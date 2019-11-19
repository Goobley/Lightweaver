#include "Formal.hpp"
#include "Background.hpp"
#include "Faddeeva.hh"
#include "JasPP.hpp"

#include <cmath>
#include <fenv.h>
#include <iostream>
#include <utility>
#include <vector>
#include <algorithm>
#include <x86intrin.h>

// Public domain polyfill for feenableexcept on OS X
// http://www-personal.umich.edu/~williams/archive/computation/fe-handling-example.c

inline int feenableexcept(unsigned int excepts)
{
    static fenv_t fenv;
    unsigned int new_excepts = excepts & FE_ALL_EXCEPT;
    // previous masks
    unsigned int old_excepts;

    if (fegetenv(&fenv))
    {
        return -1;
    }
    old_excepts = fenv.__control & FE_ALL_EXCEPT;

    // unmask
    fenv.__control &= ~new_excepts;
    fenv.__mxcsr &= ~(new_excepts << 7);

    return fesetenv(&fenv) ? -1 : old_excepts;
}

inline int fedisableexcept(unsigned int excepts)
{
    static fenv_t fenv;
    unsigned int new_excepts = excepts & FE_ALL_EXCEPT;
    // all previous masks
    unsigned int old_excepts;

    if (fegetenv(&fenv))
    {
        return -1;
    }
    old_excepts = fenv.__control & FE_ALL_EXCEPT;

    // mask
    fenv.__control |= new_excepts;
    fenv.__mxcsr |= new_excepts << 7;

    return fesetenv(&fenv) ? -1 : old_excepts;
}

void print_complex(std::complex<f64> cmp, WofZType wofz)
{
    using Faddeeva::w;
    std::cout << cmp << std::endl;
    std::cout << w(cmp) << std::endl;
}

void planck_nu(long Nspace, double* T, double lambda, double* Bnu)
{
    namespace C = Constants;
    constexpr f64 hc_k = C::HC / (C::KBoltzmann * C::NM_TO_M);
    const f64 hc_kla = hc_k / lambda;
    constexpr f64 twoh_c2 = (2.0 * C::HC) / cube(C::NM_TO_M);
    const f64 twohnu3_c2 = twoh_c2 / cube(lambda);
    constexpr f64 MAX_EXPONENT = 150.0;

    for (int k = 0; k < Nspace; k++)
    {
        f64 hc_Tkla = hc_kla / T[k];
        if (hc_Tkla <= MAX_EXPONENT)
            Bnu[k] = twohnu3_c2 / (exp(hc_Tkla) - 1.0);
        else
            Bnu[k] = 0.0;
    }
}

inline f64 voigt_H(f64 a, f64 v)
{
    using Faddeeva::w;
    using namespace std::complex_literals;
    auto z = (v + a * 1i);
    return w(z).real();
}

inline std::complex<f64> voigt_HF(f64 a, f64 v)
{
    using Faddeeva::w;
    using namespace std::complex_literals;
    auto z = (v + a * 1i);
    return w(z);
}

void Transition::compute_phi(const Atmosphere& atmos, F64View aDamp, F64View vBroad)
{
    namespace C = Constants;
    if (type == TransitionType::CONTINUUM)
        return;

    constexpr f64 sign[] = { -1.0, 1.0 };

    // Why is there still no constexpr math in std? :'(
    const f64 sqrtPi = sqrt(C::Pi);

    wphi.fill(0.0);

    for (int la = 0; la < wavelength.shape(0); ++la)
    {
        const f64 vBase = (wavelength(la) - lambda0) * C::CLight / lambda0;
        const f64 wla = wlambda(la);
        for (int mu = 0; mu < phi.shape(1); ++mu)
        {
            const f64 wlamu = wla * 0.5 * atmos.wmu(mu);
            for (int toObs = 0; toObs < 2; ++toObs)
            {
                const f64 s = sign[toObs];
                for (int k = 0; k < atmos.Nspace; ++k)
                {
                    const f64 vk = (vBase + s * atmos.vlosMu(mu, k)) / vBroad(k);
                    const f64 p = voigt_H(aDamp(k), vk) / (sqrtPi * vBroad(k));
                    phi(la, mu, toObs, k) = p;
                    wphi(k) += p * wlamu;
                }
            }
        }
    }

    for (int k = 0; k < wphi.shape(0); ++k)
    {
        wphi(k) = 1.0 / wphi(k);
    }
}

void Transition::compute_polarised_profiles(
    const Atmosphere& atmos, F64View aDamp, F64View vBroad, const ZeemanComponents& z)
{
    namespace C = Constants;
    if (type == TransitionType::CONTINUUM)
        return;

    if (!polarised)
        return;

    constexpr f64 sign[] = { -1.0, 1.0 };

    const f64 larmor = C::QElectron / (4.0 * C::Pi * C::MElectron) * (lambda0 * C::NM_TO_M);
    const f64 sqrtPi = sqrt(C::Pi);

    assert((bool)atmos.B && "Must provide magnetic field when computing polarised profiles");
    assert((bool)atmos.cosGamma && "Must call Atmosphere::update_projections before computing polarised profiles");
    F64Arr vB(atmos.Nspace);
    F64Arr sv(atmos.Nspace);
    for (int k = 0; k < atmos.Nspace; ++k)
    {
        vB(k) = larmor * atmos.B(k) / vBroad(k);
        sv(k) = 1.0 / (sqrt(C::Pi) * vBroad(k));
    }
    phi.fill(0.0);
    wphi.fill(0.0);
    phiQ.fill(0.0);
    phiU.fill(0.0);
    phiV.fill(0.0);
    psiQ.fill(0.0);
    psiU.fill(0.0);
    psiV.fill(0.0);

    for (int la = 0; la < wavelength.shape(0); ++la)
    {
        const f64 vBase = (wavelength(la) - lambda0) * C::CLight / lambda0;
        const f64 wla = wlambda(la);
        for (int mu = 0; mu < phi.shape(1); ++mu)
        {
            const f64 wlamu = wla * 0.5 * atmos.wmu(mu);
            for (int toObs = 0; toObs < 2; ++toObs)
            {
                const f64 s = sign[toObs];
                for (int k = 0; k < atmos.Nspace; ++k)
                {
                    const f64 vk = (vBase + s * atmos.vlosMu(mu, k)) / vBroad(k);
                    f64 phi_sb = 0.0, phi_pi = 0.0, phi_sr = 0.0;
                    f64 psi_sb = 0.0, psi_pi = 0.0, psi_sr = 0.0;

                    for (int nz = 0; nz < z.alpha.shape(0); ++nz)
                    {
                        auto HF = voigt_HF(aDamp(k), vk - z.shift(nz) * vB(k));
                        auto H = HF.real();
                        auto F = HF.imag(); // NOTE(cmo): Not sure if the 0.5 should be here -- don't think so. Think
                            // it's accounted for later.

                        switch (z.alpha(nz))
                        {
                        case -1:
                        {
                            phi_sb += z.strength(nz) * H;
                            psi_sb += z.strength(nz) * F;
                        }
                        break;
                        case 0:
                        {
                            phi_pi += z.strength(nz) * H;
                            psi_pi += z.strength(nz) * F;
                        }
                        break;
                        case 1:
                        {
                            phi_sr += z.strength(nz) * H;
                            psi_sr += z.strength(nz) * F;
                        }
                        break;
                        }
                    }
                    f64 sin2_gamma = 1.0 - square(atmos.cosGamma(mu, k));
                    f64 cos_2chi = atmos.cos2chi(mu, k);
                    f64 sin_2chi = atmos.sin2chi(mu, k);
                    f64 cos_gamma = atmos.cosGamma(mu, k);

                    f64 phi_sigma = phi_sr + phi_sb;
                    f64 phi_delta = 0.5 * phi_pi - 0.25 * phi_sigma;
                    phi(la, mu, toObs, k) += (phi_delta * sin2_gamma + 0.5 * phi_sigma) * sv(k);

                    phiQ(la, mu, toObs, k) += s * phi_delta * sin2_gamma * cos_2chi * sv(k);
                    phiU(la, mu, toObs, k) += phi_delta * sin2_gamma * sin_2chi * sv(k);
                    phiV(la, mu, toObs, k) += s * 0.5 * (phi_sr - phi_sb) * cos_gamma * sv(k);

                    f64 psi_sigma = psi_sr + psi_sb;
                    f64 psi_delta = 0.5 * psi_pi - 0.25 * psi_sigma;

                    psiQ(la, mu, toObs, k) += s * psi_delta * sin2_gamma * cos_2chi * sv(k);
                    psiU(la, mu, toObs, k) += psi_delta * sin2_gamma * sin_2chi * sv(k);
                    psiV(la, mu, toObs, k) += s * 0.5 * (psi_sr - psi_sb) * cos_gamma * sv(k);

                    wphi(k) += wlamu * phi(la, mu, toObs, k);
                }
            }
        }
    }
    for (int k = 0; k < wphi.shape(0); ++k)
    {
        wphi(k) = 1.0 / wphi(k);
    }
}

inline void w2(f64 dtau, f64* w)
{
    f64 expdt;

    if (dtau < 5.0E-4)
    {
        w[0] = dtau * (1.0 - 0.5 * dtau);
        w[1] = square(dtau) * (0.5 - dtau / 3.0);
    }
    else if (dtau > 50.0)
    {
        w[1] = w[0] = 1.0;
    }
    else
    {
        expdt = exp(-dtau);
        w[0] = 1.0 - expdt;
        w[1] = w[0] - dtau * expdt;
    }
}

inline void w3(f64 dtau, f64* w)
{
    f64 expdt, delta;

    if (dtau < 5.0E-4)
    {
        w[0] = dtau * (1.0 - 0.5 * dtau);
        delta = square(dtau);
        w[1] = delta * (0.5 - dtau / 3.0);
        delta *= dtau;
        w[2] = delta * (1.0 / 3.0 - 0.25 * dtau);
    }
    else if (dtau > 50.0)
    {
        w[1] = w[0] = 1.0;
        w[2] = 2.0;
    }
    else
    {
        expdt = exp(-dtau);
        w[0] = 1.0 - expdt;
        w[1] = w[0] - dtau * expdt;
        w[2] = 2.0 * w[1] - square(dtau) * expdt;
    }
}

struct FormalData
{
    Atmosphere* atmos;
    F64View chi;
    F64View S;
    F64View I;
    F64View Psi;
};

struct FormalDataStokes
{
    Atmosphere* atmos;
    F64View2D chi;
    F64View2D S;
    F64View2D I;
    FormalData fdIntens;
};

void piecewise_linear_1d_impl(FormalData* fd, f64 zmu, bool toObs, f64 Istart)
{
    JasUnpack((*fd), chi, S, Psi, I, atmos);
    const auto& height = atmos->height;
    const int Ndep = chi.shape(0);
    bool computeOperator = Psi.shape(0) != 0;

    /* --- Distinguish between rays going from BOTTOM to TOP
            (to_obs == TRUE), and vice versa --      -------------- */

    // NOTE(cmo): I admit, on some level, the directions of these derivatives (uw -
    // dw) feels odd, but they're consistent with the RH implementation. The only
    // change that would really occur if these were flipped would be I(k) = ... -
    // w[1] * dS_uw, but really this is a holdover from when this was parabolic. May
    // adjust, but not really planning on using thisFS

    int dk = -1;
    int k_start = Ndep - 1;
    int k_end = 0;
    if (!toObs)
    {
        dk = 1;
        k_start = 0;
        k_end = Ndep - 1;
    }
    f64 dtau_uw = zmu * (chi(k_start) + chi(k_start + dk)) * abs(height(k_start) - height(k_start + dk));
    f64 dS_uw = (S(k_start) - S(k_start + dk)) / dtau_uw;

    /* --- Boundary conditions --                        -------------- */
    f64 I_upw = Istart;

    I(k_start) = I_upw;
    if (computeOperator)
        Psi(k_start) = 0.0;

    /* --- Solve transfer along ray --                   -------------- */

    f64 w[2];
    for (int k = k_start + dk; k != k_end; k += dk)
    {
        w2(dtau_uw, w);

        /* --- Piecewise linear here --               -------------- */
        f64 dtau_dw = zmu * (chi(k) + chi(k + dk)) * abs(height(k) - height(k + dk));
        f64 dS_dw = (S(k) - S(k + dk)) / dtau_dw;

        I(k) = (1.0 - w[0]) * I_upw + w[0] * S(k) + w[1] * dS_uw;

        if (computeOperator)
            Psi(k) = w[0] - w[1] / dtau_uw;

        /* --- Re-use downwind quantities for next upwind position -- --- */
        I_upw = I(k);
        dS_uw = dS_dw;
        dtau_uw = dtau_dw;
    }

    /* --- Piecewise linear integration at end of ray -- ---------- */
    w2(dtau_uw, w);
    I(k_end) = (1.0 - w[0]) * I_upw + w[0] * S(k_end) + w[1] * dS_uw;
    if (computeOperator)
    {
        Psi(k_end) = w[0] - w[1] / dtau_uw;
        for (int k = 0; k < Psi.shape(0); ++k)
            Psi(k) /= chi(k);
    }
}

void piecewise_linear_1d(FormalData* fd, int mu, bool toObs, f64 wav)
{
    JasUnpack((*fd), atmos, I, chi);
    f64 zmu = 0.5 / atmos->muz(mu);
    auto height = atmos->height;

    int dk = -1;
    int kStart = atmos->Nspace - 1;
    int kEnd = 0;
    if (!toObs)
    {
        dk = 1;
        kStart = 0;
        kEnd = atmos->Nspace - 1;
    }
    f64 dtau_uw = zmu * (chi(kStart) + chi(kStart + dk)) * abs(height(kStart) - height(kStart + dk));

    f64 Iupw = 0.0;
    if (toObs && atmos->lowerBc == THERMALISED)
    {
        f64 Bnu[2];
        int Nspace = atmos->Nspace;
        planck_nu(2, &atmos->temperature(Nspace - 2), wav, Bnu);
        Iupw = Bnu[1] - (Bnu[0] - Bnu[1]) / dtau_uw;
    }
    else if (!toObs && atmos->upperBc == THERMALISED)
    {
        f64 Bnu[2];
        planck_nu(2, &atmos->temperature(0), wav, Bnu);
        Iupw = Bnu[0] - (Bnu[1] - Bnu[0]) / dtau_uw;
    }

    piecewise_linear_1d_impl(fd, zmu, toObs, Iupw);
}

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

void piecewise_bezier3_1d_impl(FormalData* fd, f64 zmu, bool toObs, f64 Istart)
{
    JasUnpack((*fd), atmos, chi, S, I, Psi);
    const auto& height = atmos->height;
    const int Ndep = atmos->Nspace;
    // bool computeOperator = Psi.shape(0) != 0;
    bool computeOperator = bool(Psi);

    /* --- Distinguish between rays going from BOTTOM to TOP
            (to_obs == TRUE), and vice versa --      -------------- */

    int dk = -1;
    int k_start = Ndep - 1;
    int k_end = 0;
    if (!toObs)
    {
        dk = 1;
        k_start = 0;
        k_end = Ndep - 1;
    }

    /* --- Boundary conditions --                        -------------- */
    f64 I_upw = Istart;

    I(k_start) = I_upw;
    if (computeOperator)
        Psi(k_start) = 0.0;

    int k = k_start + dk;
    f64 ds_uw = abs(height(k) - height(k - dk)) * zmu;
    f64 ds_dw = abs(height(k + dk) - height(k)) * zmu;
    f64 dx_uw = (chi(k) - chi(k - dk)) / ds_uw;
    f64 dx_c = Bezier::cent_deriv(ds_uw, ds_dw, chi(k - dk), chi(k), chi(k + dk));
    f64 E = max(chi(k) - (ds_uw / 3.0) * dx_c, 0.0);
    f64 F = max(chi(k - dk) + (ds_uw / 3.0) * dx_uw, 0.0);
    f64 dtau_uw = ds_uw * (chi(k) + chi(k - dk) + E + F) * 0.25;
    f64 dS_uw = (S(k) - S(k - dk)) / dtau_uw;

    f64 ds_dw2 = 0.0;
    auto dx_downwind = [&ds_dw, &ds_dw2, &chi, &k, dk] {
        return Bezier::cent_deriv(ds_dw, ds_dw2, chi(k), chi(k + dk), chi(k + 2 * dk));
    };
    f64 dtau_dw = 0.0;
    auto dS_central
        = [&dtau_uw, &dtau_dw, &S, &k, dk] { return Bezier::cent_deriv(dtau_uw, dtau_dw, S(k - dk), S(k), S(k + dk)); };
    for (; k != k_end - dk; k += dk)
    {
        ds_dw2 = abs(height(k + 2 * dk) - height(k + dk)) * zmu;
        f64 dx_dw = dx_downwind();
        E = max(chi(k) + (ds_dw / 3.0) * dx_c, 0.0);
        F = max(chi(k + dk) - (ds_dw / 3.0) * dx_dw, 0.0);
        dtau_dw = ds_dw * (chi(k) + chi(k + dk) + E + F) * 0.25;

        f64 alpha, beta, gamma, eps, edt;
        Bezier::Bezier3_coeffs(dtau_uw, &alpha, &beta, &gamma, &eps, &edt);

        f64 dS_c = dS_central();

        E = max(S(k) - (dtau_uw / 3.0) * dS_c, 0.0);
        F = max(S(k - dk) + (dtau_uw / 3.0) * dS_uw, 0.0);

        I(k) = I_upw * edt + alpha * S(k) + beta * S(k - dk) + gamma * E + eps * F;
        if (computeOperator)
            Psi(k) = alpha + gamma;

        I_upw = I(k);
        ds_uw = ds_dw;
        ds_dw = ds_dw2;
        dx_uw = dx_c;
        dx_c = dx_dw;
        dtau_uw = dtau_dw;
        dS_uw = dS_c;
    }
    // NOTE(cmo): Need to handle last 2 points here
    k = k_end - dk;
    ds_dw = abs(height(k + dk) - height(k)) * zmu;
    f64 dx_dw = (chi(k + dk) - chi(k)) / ds_dw;
    E = max(chi(k) + (ds_dw / 3.0) * dx_c, 0.0);
    F = max(chi(k + dk) - (ds_dw / 3.0) * dx_dw, 0.0);
    dtau_dw = ds_dw * (chi(k) + chi(k + dk) + E + F) * 0.25;

    f64 alpha, beta, gamma, eps, edt;
    Bezier::Bezier3_coeffs(dtau_uw, &alpha, &beta, &gamma, &eps, &edt);

    f64 dS_c = dS_central();

    E = max(S(k) - dtau_uw / 3.0 * dS_c, 0.0);
    F = max(S(k - dk) + dtau_uw / 3.0 * dS_uw, 0.0);

    I(k) = I_upw * edt + alpha * S(k) + beta * S(k - dk) + gamma * E + eps * F;
    if (computeOperator)
        Psi(k) = alpha + gamma;
    I_upw = I(k);

    // Piecewise linear on end
    k = k_end;
    dtau_uw = 0.5 * zmu * (chi(k) + chi(k - dk)) * abs(height(k) - height(k - dk));
    // NOTE(cmo): See note in the linear formal solver if wondering why -w[1] is
    // used in I(k). Basically, the derivative (dS_uw) was taken in the other
    // direction there. In some ways this is nicer, as the operator and I take
    // the same form, but it doesn't really make any significant difference
    dS_uw = (S(k) - S(k - dk)) / dtau_uw;
    f64 w[2];
    w2(dtau_uw, w);
    I(k) = (1.0 - w[0]) * I_upw + w[0] * S(k) - w[1] * dS_uw;

    if (computeOperator)
    {
        Psi(k) = w[0] - w[1] / dtau_uw;
        for (int k = 0; k < Psi.shape(0); ++k)
            Psi(k) /= chi(k);
    }
}

void piecewise_bezier3_1d(FormalData* fd, int mu, bool toObs, f64 wav)
{
    JasUnpack((*fd), atmos, chi);
    // This is 1.0 here, as we are normally effectively rolling in the averaging
    // factor for dtau, whereas it's explicit in this solver
    f64 zmu = 1.0 / atmos->muz(mu);
    auto height = atmos->height;

    int dk = -1;
    int kStart = atmos->Nspace - 1;
    int kEnd = 0;
    if (!toObs)
    {
        dk = 1;
        kStart = 0;
        kEnd = atmos->Nspace - 1;
    }
    f64 dtau_uw = 0.5 * zmu * (chi(kStart) + chi(kStart + dk)) * abs(height(kStart) - height(kStart + dk));

    f64 Iupw = 0.0;
    if (toObs && atmos->lowerBc == THERMALISED)
    {
        f64 Bnu[2];
        int Nspace = atmos->Nspace;
        planck_nu(2, &atmos->temperature(Nspace - 2), wav, Bnu);
        Iupw = Bnu[1] - (Bnu[0] - Bnu[1]) / dtau_uw;
    }
    else if (!toObs && atmos->upperBc == THERMALISED)
    {
        f64 Bnu[2];
        planck_nu(2, &atmos->temperature(0), wav, Bnu);
        Iupw = Bnu[0] - (Bnu[1] - Bnu[0]) / dtau_uw;
    }

    piecewise_bezier3_1d_impl(fd, zmu, toObs, Iupw);
}

void SIMD_MatInv(float* src)
{
    /* ---

        Very fast in-place 4x4 Matrix inversion using SIMD instrutions
        Only works with 32-bits floats. It uses Cramer's rule.

        Provided by Intel

        Requires SSE instructions but all x86 machines since
        Pentium III have them.

        --                                            ------------------ */
    // NOTE(cmo): This can also be done equivalently for f64 with avx/avx2 on newer cpus

    __m128 minor0, minor1, minor2, minor3;
    __m128 row0, row1, row2, row3;
    __m128 det, tmp1;

    // -----------------------------------------------
    tmp1 = _mm_loadh_pi(_mm_loadl_pi(tmp1, (__m64*)(src)), (__m64*)(src + 4));
    row1 = _mm_loadh_pi(_mm_loadl_pi(row1, (__m64*)(src + 8)), (__m64*)(src + 12));
    row0 = _mm_shuffle_ps(tmp1, row1, 0x88);
    row1 = _mm_shuffle_ps(row1, tmp1, 0xDD);
    tmp1 = _mm_loadh_pi(_mm_loadl_pi(tmp1, (__m64*)(src + 2)), (__m64*)(src + 6));
    row3 = _mm_loadh_pi(_mm_loadl_pi(row3, (__m64*)(src + 10)), (__m64*)(src + 14));
    row2 = _mm_shuffle_ps(tmp1, row3, 0x88);
    row3 = _mm_shuffle_ps(row3, tmp1, 0xDD);
    // -----------------------------------------------
    tmp1 = _mm_mul_ps(row2, row3);
    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
    minor0 = _mm_mul_ps(row1, tmp1);
    minor1 = _mm_mul_ps(row0, tmp1);
    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
    minor0 = _mm_sub_ps(_mm_mul_ps(row1, tmp1), minor0);
    minor1 = _mm_sub_ps(_mm_mul_ps(row0, tmp1), minor1);
    minor1 = _mm_shuffle_ps(minor1, minor1, 0x4E);
    // -----------------------------------------------
    tmp1 = _mm_mul_ps(row1, row2);
    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
    minor0 = _mm_add_ps(_mm_mul_ps(row3, tmp1), minor0);
    minor3 = _mm_mul_ps(row0, tmp1);
    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
    minor0 = _mm_sub_ps(minor0, _mm_mul_ps(row3, tmp1));
    minor3 = _mm_sub_ps(_mm_mul_ps(row0, tmp1), minor3);
    minor3 = _mm_shuffle_ps(minor3, minor3, 0x4E);
    // -----------------------------------------------
    tmp1 = _mm_mul_ps(_mm_shuffle_ps(row1, row1, 0x4E), row3);
    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
    row2 = _mm_shuffle_ps(row2, row2, 0x4E);
    minor0 = _mm_add_ps(_mm_mul_ps(row2, tmp1), minor0);
    minor2 = _mm_mul_ps(row0, tmp1);
    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
    minor0 = _mm_sub_ps(minor0, _mm_mul_ps(row2, tmp1));
    minor2 = _mm_sub_ps(_mm_mul_ps(row0, tmp1), minor2);
    minor2 = _mm_shuffle_ps(minor2, minor2, 0x4E);
    // -----------------------------------------------
    tmp1 = _mm_mul_ps(row0, row1);
    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
    minor2 = _mm_add_ps(_mm_mul_ps(row3, tmp1), minor2);
    minor3 = _mm_sub_ps(_mm_mul_ps(row2, tmp1), minor3);
    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
    minor2 = _mm_sub_ps(_mm_mul_ps(row3, tmp1), minor2);
    minor3 = _mm_sub_ps(minor3, _mm_mul_ps(row2, tmp1));
    // -----------------------------------------------
    tmp1 = _mm_mul_ps(row0, row3);
    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
    minor1 = _mm_sub_ps(minor1, _mm_mul_ps(row2, tmp1));
    minor2 = _mm_add_ps(_mm_mul_ps(row1, tmp1), minor2);
    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
    minor1 = _mm_add_ps(_mm_mul_ps(row2, tmp1), minor1);
    minor2 = _mm_sub_ps(minor2, _mm_mul_ps(row1, tmp1));
    // -----------------------------------------------
    tmp1 = _mm_mul_ps(row0, row2);
    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0xB1);
    minor1 = _mm_add_ps(_mm_mul_ps(row3, tmp1), minor1);
    minor3 = _mm_sub_ps(minor3, _mm_mul_ps(row1, tmp1));
    tmp1 = _mm_shuffle_ps(tmp1, tmp1, 0x4E);
    minor1 = _mm_sub_ps(minor1, _mm_mul_ps(row3, tmp1));
    minor3 = _mm_add_ps(_mm_mul_ps(row1, tmp1), minor3);
    // -----------------------------------------------
    det = _mm_mul_ps(row0, minor0);
    det = _mm_add_ps(_mm_shuffle_ps(det, det, 0x4E), det);
    det = _mm_add_ss(_mm_shuffle_ps(det, det, 0xB1), det);
    tmp1 = _mm_rcp_ss(det);
    det = _mm_sub_ss(_mm_add_ss(tmp1, tmp1), _mm_mul_ss(det, _mm_mul_ss(tmp1, tmp1)));
    det = _mm_shuffle_ps(det, det, 0x00);
    minor0 = _mm_mul_ps(det, minor0);
    _mm_storel_pi((__m64*)(src), minor0);
    _mm_storeh_pi((__m64*)(src + 2), minor0);
    minor1 = _mm_mul_ps(det, minor1);
    _mm_storel_pi((__m64*)(src + 4), minor1);
    _mm_storeh_pi((__m64*)(src + 6), minor1);
    minor2 = _mm_mul_ps(det, minor2);
    _mm_storel_pi((__m64*)(src + 8), minor2);
    _mm_storeh_pi((__m64*)(src + 10), minor2);
    minor3 = _mm_mul_ps(det, minor3);
    _mm_storel_pi((__m64*)(src + 12), minor3);
    _mm_storeh_pi((__m64*)(src + 14), minor3);
}

bool gluInvertMatrix(const double m[16], double invOut[16])
{
    double inv[16], det;
    int i;

    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14]
        + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];

    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14]
        - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];

    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13]
        + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];

    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13]
        - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];

    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14]
        - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];

    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14]
        + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];

    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13]
        - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];

    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13]
        + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];

    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7]
        - m[13] * m[3] * m[6];

    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14]
        - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];

    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13]
        + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];

    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13]
        - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];

    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7]
        + m[9] * m[3] * m[6];

    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7]
        - m[8] * m[3] * m[6];

    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7]
        + m[8] * m[3] * m[5];

    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6]
        - m[8] * m[2] * m[5];

    det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

    if (det == 0)
        return false;

    det = 1.0 / det;

    for (i = 0; i < 16; i++)
        invOut[i] = inv[i] * det;

    return true;
}

void stokes_K(int k, const F64View2D& chi, f64 chiI, f64 K[4][4])
{
    for (int j = 0; j < 4; ++j)
        for (int i = 0; i < 4; ++i)
            K[j][i] = 0.0;
    K[0][1] = chi(1, k);
    K[0][2] = chi(2, k);
    K[0][3] = chi(3, k);

    K[1][2] = chi(6, k);
    K[1][3] = chi(5, k);
    K[2][3] = chi(4, k);

    for (int j = 0; j < 3; ++j)
    {
        for (int i = j + 1; i < 4; ++i)
        {
            K[j][i] /= chiI;
            K[i][j] = K[j][i];
        }
    }

    K[1][3] *= -1.0;
    K[2][1] *= -1.0;
    K[3][2] *= -1.0;
}

inline void prod(f64 a[4][4], f64 b[4][4], f64 c[4][4])
{
    for (int j = 0; j < 4; ++j)
        for (int i = 0; i < 4; ++i)
            c[j][i] = 0.0;

    for (int j = 0; j < 4; ++j)
        for (int i = 0; i < 4; ++i)
            for (int k = 0; k < 4; ++k)
                c[j][i] += a[k][i] * b[j][k];
}

inline void prod(f64 a[4][4], f64 b[4], f64 c[4])
{
    for (int i = 0; i < 4; ++i)
        c[i] = 0.0;

    for (int i = 0; i < 4; ++i)
        for (int k = 0; k < 4; ++k)
            c[i] += a[i][k] * b[k];
}

inline void prod(f32 a[4][4], f64 b[4], f64 c[4])
{
    for (int i = 0; i < 4; ++i)
        c[i] = 0.0;

    for (int i = 0; i < 4; ++i)
        for (int k = 0; k < 4; ++k)
            c[i] += f64(a[i][k]) * b[k];
}

#define GLU_MAT 1
#define JAIME_ORDER 0
void piecewise_stokes_bezier3_1d_impl(FormalDataStokes* fd, f64 zmu, bool toObs, f64 Istart[4], bool polarisedFrequency)
{
    JasUnpack((*fd), atmos, chi, S, I);
    const auto& height = atmos->height;
    const int Ndep = atmos->Nspace;

    // clang-format off
    constexpr f64 id[4][4] = { { 1.0, 0.0, 0.0, 0.0 },
                               { 0.0, 1.0, 0.0, 0.0 },
                               { 0.0, 0.0, 1.0, 0.0 },
                               { 0.0, 0.0, 0.0, 1.0 } };
    // clang-format on

    auto slice_s4 = [&S](int k, f64 slice[4]) {
        for (int i = 0; i < 4; ++i)
        {
            slice[i] = S(i, k);
        }
    };

    /* --- Distinguish between rays going from BOTTOM to TOP
            (to_obs == TRUE), and vice versa --      -------------- */

    int dk = -1;
    int k_start = Ndep - 1;
    int k_end = 0;
    if (!toObs)
    {
        dk = 1;
        k_start = 0;
        k_end = Ndep - 1;
    }

    /* --- Boundary conditions --                        -------------- */

    for (int n = 0; n < 4; ++n)
        I(n, k_start) = Istart[n];

    int k = k_start + dk;
    f64 ds_uw = abs(height(k) - height(k - dk)) * zmu;
    f64 ds_dw = abs(height(k + dk) - height(k)) * zmu;
    f64 dx_uw = (chi(0, k) - chi(0, k - dk)) / ds_uw;
    f64 dx_c = Bezier::cent_deriv(ds_uw, ds_dw, chi(0, k - dk), chi(0, k), chi(0, k + dk));
    f64 c1 = max(chi(0, k) - (ds_uw / 3.0) * dx_c, 0.0);
    f64 c2 = max(chi(0, k - dk) + (ds_uw / 3.0) * dx_uw, 0.0);
    f64 dtau_uw = ds_uw * (chi(0, k) + chi(0, k - dk) + c1 + c2) * 0.25;

    f64 Ku[4][4], K0[4][4], Su[4], S0[4];
    f64 dKu[4][4], dK0[4][4], dSu[4], dS0[4];
    stokes_K(k_start, chi, chi(0, k_start), Ku);
    stokes_K(k, chi, chi(0, k), K0);
    // memset(Ku[0], 0, 16*sizeof(f64));
    // memset(K0[0], 0, 16*sizeof(f64));
    slice_s4(k_start, Su);
    slice_s4(k, S0);

    for (int n = 0; n < 4; ++n)
    {
        dSu[n] = (S0[n] - Su[n]) / dtau_uw;
        for (int m = 0; m < 4; ++m)
            dKu[n][m] = (K0[n][m] - Ku[n][m]) / dtau_uw;
    }

    f64 ds_dw2 = 0.0;
    f64 dtau_dw = 0.0;
    auto dx_downwind = [&ds_dw, &ds_dw2, &chi, &k, dk] {
        return Bezier::cent_deriv(ds_dw, ds_dw2, chi(0, k), chi(0, k + dk), chi(0, k + 2 * dk));
    };

    f64 Kd[4][4], A[4][4], Ma[4][4], Mb[4][4], Mc[4][4], V0[4], V1[4], Sd[4];
#if GLU_MAT
    f64 Md[4][4], Mdi[4][4];
#else
    f32 Md[4][4];
#endif
    for (; k != k_end - dk; k += dk)
    {
        ds_dw2 = abs(height(k + 2 * dk) - height(k + dk)) * zmu;
        f64 dx_dw = dx_downwind();
        c1 = max(chi(0, k) + (ds_dw / 3.0) * dx_c, 0.0);
        c2 = max(chi(0, k + dk) - (ds_dw / 3.0) * dx_dw, 0.0);
        dtau_dw = ds_dw * (chi(0, k) + chi(0, k + dk) + c1 + c2) * 0.25;

        f64 alpha, beta, gamma, edt, eps;
        Bezier::Bezier3_coeffs(dtau_uw, &alpha, &beta, &gamma, &eps, &edt);

        stokes_K(k + dk, chi, chi(0, k + dk), Kd);
        // memset(Kd[0], 0, 16*sizeof(f64));
        slice_s4(k + dk, Sd);

        Bezier::cent_deriv(dK0, dtau_uw, dtau_dw, Ku, K0, Kd);
        Bezier::cent_deriv(dS0, dtau_uw, dtau_dw, Su, S0, Sd);

        prod(Ku, Ku, Ma); // Ma = Ku @ Ku
        prod(K0, K0, A); // A = K0 @ K0

        // c1 = S0[0] - (dtau_uw/3.0) * dS0[0];
        // c2 = Su[0] + (dtau_uw/3.0) * dSu[0];
        // I(0, k) = I(0, k-dk) * edt + alpha * S0[0] + beta * Su[0] + gamma * c1 + eps * c2;

        for (int j = 0; j < 4; ++j)
        {
            for (int i = 0; i < 4; ++i)
            {
                // A in paper (LHS of system)
                Md[j][i] = id[j][i] + alpha * K0[j][i]
                    - gamma * -(dtau_uw / 3.0 * (A[j][i] + dK0[j][i] + K0[j][i]) + K0[j][i]);

                // Terms to be multiplied by I(:,k-dk) in B: (exp(-dtau) + beta*Ku + epsilon*\bar{f}_k)
                Ma[j][i] = edt * id[j][i] - beta * Ku[j][i]
                    + eps * (dtau_uw / 3.0 * (Ma[j][i] + dKu[j][i] + Ku[j][i]) - Ku[j][i]);

                // Terms to be multiplied by S(:,k-dk) in B i.e. f_k
                Mb[j][i] = beta * id[j][i] + eps * (id[j][i] - dtau_uw / 3.0 * Ku[j][i]);

                // Terms to be multiplied by S(:,k) in B i.e. e_k
                Mc[j][i] = alpha * id[j][i] + gamma * (id[j][i] + dtau_uw / 3.0 * K0[j][i]);
            }
        }

        // printf("%e, %e, %e, %e\n", K0[0][0], K0[1][0], K0[0][1], K0[3][2]);

        for (int i = 0; i < 4; ++i)
        {
            V0[i] = 0.0;
            for (int j = 0; j < 4; ++j)
                V0[i] += Ma[i][j] * I(j, k - dk) + Mb[i][j] * Su[j] + Mc[i][j] * S0[j];

                // NOTE(cmo): I think the direction on the original control points here was wrong
#if JAIME_ORDER
            V0[i] += dtau_uw / 3.0 * (gamma * dS0[i] - eps * dSu[i]);
#else
            V0[i] += dtau_uw / 3.0 * (eps * dSu[i] - gamma * dS0[i]);
#endif
        }

#if GLU_MAT
        gluInvertMatrix(Md[0], Mdi[0]);
        prod(Mdi, V0, V1);
#else
        SIMD_MatInv(Md[0]);
        prod(Md, V0, V1);
#endif

        for (int i = 0; i < 4; ++i)
            I(i, k) = V1[i];

        memcpy(Su, S0, 4 * sizeof(f64));
        memcpy(S0, Sd, 4 * sizeof(f64));
        memcpy(dSu, dS0, 4 * sizeof(f64));

        memcpy(Ku[0], K0[0], 16 * sizeof(f64));
        memcpy(K0[0], Kd[0], 16 * sizeof(f64));
        memcpy(dKu[0], dK0[0], 16 * sizeof(f64));

        dtau_uw = dtau_dw;
        ds_uw = ds_dw;
        ds_dw = ds_dw2;
        dx_uw = dx_c;
        dx_c = dx_dw;
    }
    // NOTE(cmo): Need to handle last 2 points here
    k = k_end - dk;
    f64 dx_dw = (chi(0, k + dk) - chi(0, k)) / ds_dw;
    c1 = max(chi(0, k) + (ds_dw / 3.0) * dx_c, 0.0);
    c2 = max(chi(0, k + dk) - (ds_dw / 3.0) * dx_dw, 0.0);
    dtau_dw = ds_dw * (chi(0, k) + chi(0, k + dk) + c1 + c2) * 0.25;

    f64 alpha, beta, gamma, edt, eps;
    Bezier::Bezier3_coeffs(dtau_uw, &alpha, &beta, &gamma, &eps, &edt);

    stokes_K(k + dk, chi, chi(0, k + dk), Kd);
    // memset(Kd[0], 0, 16*sizeof(f64));
    slice_s4(k + dk, Sd);

    Bezier::cent_deriv(dK0, dtau_uw, dtau_dw, Ku, K0, Kd);
    Bezier::cent_deriv(dS0, dtau_uw, dtau_dw, Su, S0, Sd);

    prod(Ku, Ku, Ma); // Ma = Ku @ Ku
    prod(K0, K0, A); // A = K0 @ K0

    // c1 = max(S(0, k) - (dtau_uw/3.0) * dS0[0], 0.0);
    // c2 = max(S(0, k-dk) + (dtau_uw/3.0) * dSu[0], 0.0);
    // I(0, k) = I(0, k-dk) * edt + alpha * S(0, k) + beta * S(0, k-dk) + gamma * c1 + eps * c2;

    for (int j = 0; j < 4; ++j)
    {
        for (int i = 0; i < 4; ++i)
        {
            // A in paper (LHS of system)
            Md[j][i]
                = id[j][i] + alpha * K0[j][i] - gamma * -(dtau_uw / 3.0 * (A[j][i] + dK0[j][i] + K0[j][i]) + K0[j][i]);

            // Terms to be multiplied by I(:,k-dk) in B: (exp(-dtau) + beta + \bar{f}_k)
            Ma[j][i] = edt * id[j][i] - beta * Ku[j][i]
                + eps * (dtau_uw / 3.0 * (Ma[j][i] + dKu[j][i] + Ku[j][i]) - Ku[j][i]);

            // Terms to be multiplied by S(:,k-dk) in B i.e. f_k
            Mb[j][i] = beta * id[j][i] + eps * (id[j][i] - dtau_uw / 3.0 * Ku[j][i]);

            // Terms to be multiplied by S(:,k) in B i.e. e_k
            Mc[j][i] = alpha * id[j][i] + gamma * (id[j][i] + dtau_uw / 3.0 * K0[j][i]);
        }
    }

    for (int i = 0; i < 4; ++i)
    {
        V0[i] = 0.0;
        for (int j = 0; j < 4; ++j)
            V0[i] += Ma[i][j] * I(j, k - dk) + Mb[i][j] * Su[j] + Mc[i][j] * S0[j];

#if JAIME_ORDER
        V0[i] += dtau_uw / 3.0 * (gamma * dS0[i] - eps * dSu[i]);
#else
        V0[i] += dtau_uw / 3.0 * (eps * dSu[i] - gamma * dS0[i]);
#endif
    }

#if GLU_MAT
    gluInvertMatrix(Md[0], Mdi[0]);
    prod(Mdi, V0, V1);
#else
    SIMD_MatInv(Md[0]);
    prod(Md, V0, V1);
#endif

    for (int i = 0; i < 4; ++i)
        I(i, k) = V1[i];

    memcpy(Su, S0, 4 * sizeof(f64));
    memcpy(S0, Sd, 4 * sizeof(f64));
    memcpy(dSu, dS0, 4 * sizeof(f64));

    memcpy(Ku[0], K0[0], 16 * sizeof(f64));
    memcpy(K0[0], Kd[0], 16 * sizeof(f64));
    memcpy(dKu[0], dK0[0], 16 * sizeof(f64));

    dtau_uw = dtau_dw;
    ds_uw = ds_dw;
    ds_dw = ds_dw2;
    dx_uw = dx_c;
    dx_c = dx_dw;

    // Piecewise linear on end
    k = k_end;
    dtau_uw = 0.5 * zmu * (chi(0, k) + chi(0, k - dk)) * abs(height(k) - height(k - dk));

    f64 w[2];
    w2(dtau_uw, w);
    for (int n = 0; n < 4; ++n)
        V0[n] = w[0] * S(n, k) - w[1] * dSu[n];

    for (int n = 0; n < 4; ++n)
    {
        for (int m = 0; m < 4; ++m)
        {
            A[n][m] = -w[1] / dtau_uw * Ku[n][m];
            Md[n][m] = (w[0] - w[1] / dtau_uw) * K0[n][m];
        }
        A[n][n] = 1.0 - w[0];
        Md[n][n] = 1.0;
    }

    for (int n = 0; n < 4; ++n)
        for (int m = 0; m < 4; ++m)
            V0[n] += A[n][m] * I(m, k - dk);

#if GLU_MAT
    gluInvertMatrix(Md[0], Mdi[0]);
    prod(Mdi, V0, V1);
#else
    SIMD_MatInv(Md[0]);
    prod(Md, V0, V1);
#endif

    for (int n = 0; n < 4; ++n)
        I(n, k) = V1[n];
}

void piecewise_stokes_bezier3_1d(FormalDataStokes* fd, int mu, bool toObs, f64 wav, bool polarisedFrequency)
{
    if (!polarisedFrequency)
    {
        piecewise_bezier3_1d(&fd->fdIntens, mu, toObs, wav);
        return;
    }

    JasUnpack((*fd), atmos, chi);
    // This is 1.0 here, as we are normally effectively rolling in the averaging
    // factor for dtau, whereas it's explicit in this solver
    f64 zmu = 1.0 / atmos->muz(mu);
    auto height = atmos->height;

    int dk = -1;
    int kStart = atmos->Nspace - 1;
    int kEnd = 0;
    if (!toObs)
    {
        dk = 1;
        kStart = 0;
        kEnd = atmos->Nspace - 1;
    }
    f64 dtau_uw = 0.5 * zmu * (chi(0, kStart) + chi(0, kStart + dk)) * abs(height(kStart) - height(kStart + dk));

    f64 Iupw[4] = { 0.0, 0.0, 0.0, 0.0 };
    if (toObs && atmos->lowerBc == THERMALISED)
    {
        f64 Bnu[2];
        int Nspace = atmos->Nspace;
        planck_nu(2, &atmos->temperature(Nspace - 2), wav, Bnu);
        Iupw[0] = Bnu[1] - (Bnu[0] - Bnu[1]) / dtau_uw;
    }
    else if (!toObs && atmos->upperBc == THERMALISED)
    {
        f64 Bnu[2];
        planck_nu(2, &atmos->temperature(0), wav, Bnu);

        Iupw[0] = Bnu[0] - (Bnu[1] - Bnu[0]) / dtau_uw;
    }

    piecewise_stokes_bezier3_1d_impl(fd, zmu, toObs, Iupw, polarisedFrequency);
}

struct IntensityCoreData
{
    Atmosphere* atmos;
    Spectrum* spect;
    FormalData* fd;
    Background* background;
    std::vector<Atom*>* activeAtoms;
    F64Arr* JDag;
    F64View chiTot;
    F64View etaTot;
    F64View Uji;
    F64View Vij;
    F64View Vji;
    F64View I;
    F64View S;
    F64View Ieff;
    F64View PsiStar;
};

struct StokesCoreData
{
    Atmosphere* atmos;
    Spectrum* spect;
    FormalDataStokes* fd;
    Background* background;
    std::vector<Atom*>* activeAtoms;
    F64Arr* JDag;
    F64View2D chiTot;
    F64View2D etaTot;
    F64View Uji;
    F64View Vij;
    F64View Vji;
    F64View2D I;
    F64View2D S;
};

namespace GammaFsCores
{
f64 intensity_core(IntensityCoreData& data, int la)
{
    JasUnpack(*data, atmos, spect, fd, background, activeAtoms, JDag);
    JasUnpack(data, chiTot, etaTot, Uji, Vij, Vji);
    JasUnpack(data, I, S, Ieff, PsiStar);
    const int Nspace = atmos.Nspace;
    const int Nrays = atmos.Nrays;
    const int Nspect = spect.wavelength.shape(0);
    JDag = spect.J(la);
    // JDag.fill(0.0);
    F64View J = spect.J(la);
    J.fill(0.0);

    for (int a = 0; a < activeAtoms.size(); ++a)
        activeAtoms[a]->setup_wavelength(la);

    // NOTE(cmo): If we only have continua then opacity is angle independent
    bool continuaOnly = true;
    for (int a = 0; a < activeAtoms.size(); ++a)
    {
        auto& atom = *activeAtoms[a];
        for (int kr = 0; kr < atom.Ntrans; ++kr)
        {
            auto& t = *atom.trans[kr];
            if (!t.active(la))
                continue;
            continuaOnly = continuaOnly && (t.type == CONTINUUM);
        }
    }
    // continuaOnly = false;

    f64 dJMax = 0.0;
    for (int mu = 0; mu < Nrays; ++mu)
    {
        for (int toObsI = 0; toObsI < 2; toObsI += 1)
        {
            bool toObs = (bool)toObsI;
            // const f64 sign = If toObs Then 1.0 Else -1.0 End;
            if (!continuaOnly || (continuaOnly && (mu == 0 && toObsI == 0)))
            {
                chiTot.fill(0.0);
                etaTot.fill(0.0);

                for (int a = 0; a < activeAtoms.size(); ++a)
                {
                    auto& atom = *activeAtoms[a];
                    atom.zero_angle_dependent_vars();
                    for (int kr = 0; kr < atom.Ntrans; ++kr)
                    {
                        // continue;
                        auto& t = *atom.trans[kr];
                        // if (kr == atom.Ntrans-4 || kr == atom.Ntrans-5)
                        //     continue;
                        if (!t.active(la))
                            continue;

                        // if (t.type == LINE)
                        //     printf("Line: %d, %d\n", kr, la);

                        t.uv(la, mu, toObs, Uji, Vij, Vji);

                        for (int k = 0; k < Nspace; ++k)
                        {
                            f64 chi = atom.n(t.i, k) * Vij(k) - atom.n(t.j, k) * Vji(k);
                            f64 eta = atom.n(t.j, k) * Uji(k);

                            atom.chi(t.i, k) += chi;
                            atom.chi(t.j, k) -= chi;
                            atom.U(t.j, k) += Uji(k);
                            atom.V(t.i, k) += Vij(k);
                            atom.V(t.j, k) += Vji(k);
                            chiTot(k) += chi;
                            etaTot(k) += eta;
                            atom.eta(k) += eta;
                        }
                    }
                }
                    // Do LTE atoms here
                for (int k = 0; k < Nspace; ++k)
                {
                    chiTot(k) += background.chi(la, k);
                    S(k) = (etaTot(k) + background.eta(la, k) + background.sca(la, k) * JDag(k)) / chiTot(k);
                }

                // if (la == 500)
                // {
                //     printf("---------%d------------\n", la);
                // }
                // for (int k = 0; k < Nspace; ++k)
                // {
                //     S(k) = (etaTot(k) + background.eta(la, k) + background.sca(la, k) * JDag(k)) / chiTot(k);
                //     // S(k) = background.eta(la, k) / chiTot(k);
                //     // if (la == 500)
                //     // {
                //     //     printf("%.2e, ", S(k));
                //     // }
                // }
                // if (la==1314)
                // {
                //     printf("Eta ---------%d------------\n", la);
                //     for (int k = 0; k < Nspace; ++k)
                //     {
                //         printf("%.2e, ", background.eta(la, k));
                //     }
                //     printf("\nEta ---------%d------------\n\n", la);
                //     printf("Chi ---------%d------------\n", la);
                //     for (int k = 0; k < Nspace; ++k)
                //     {
                //         printf("%.2e, ", chiTot(k));
                //     }
                //     printf("\nChi ---------%d------------\n\n", la);
                //     printf("S ---------%d------------\n", la);
                //     for (int k = 0; k < Nspace; ++k)
                //     {
                //         printf("%.2e, ", S(k));
                //     }
                //     printf("\nS ---------%d------------\n\n", la);
                // }
            }

            piecewise_bezier3_1d(&fd, mu, toObs, spect.wavelength(la));
            // piecewise_linear_1d(&fd, mu, toObs, spect.wavelength(la));
            spect.I(la, mu) = I(0);

            for (int k = 0; k < Nspace; ++k)
            {
                J(k) += 0.5 * atmos.wmu(mu) * I(k);
            }

            if (spect.JRest && spect.hPrdActive && spect.hPrdActive(la))
            {
                int hPrdLa = spect.la_to_hPrdLa(la);
                for (int k = 0; k < Nspace; ++k)
                {
                    const auto& coeffs = spect.JCoeffs(hPrdLa, mu, toObs, k);
                    for (const auto& c : coeffs)
                    {
                        spect.JRest(c.idx, k) += 0.5 * atmos.wmu(mu) * c.frac * I(k);
                    }
                }
            }

            for (int a = 0; a < activeAtoms.size(); ++a)
            {
                auto& atom = *activeAtoms[a];
                for (int k = 0; k < Nspace; ++k)
                {
                    Ieff(k) = I(k) - PsiStar(k) * atom.eta(k);
                }

                for (int kr = 0; kr < atom.Ntrans; ++kr)
                {
                    auto& t = *atom.trans[kr];
                    if (!t.active(la))
                        continue;

                    const f64 wmu = 0.5 * atmos.wmu(mu);
                    t.uv(la, mu, toObs, Uji, Vij, Vji);

                    for (int k = 0; k < Nspace; ++k)
                    {
                        const f64 wlamu = atom.wla(kr, k) * wmu;

                        f64 integrand = (Uji(k) + Vji(k) * Ieff(k)) - (PsiStar(k) * atom.chi(t.i, k) * atom.U(t.j, k));
                        atom.Gamma(t.i, t.j, k) += integrand * wlamu;

                        integrand = (Vij(k) * Ieff(k)) - (PsiStar(k) * atom.chi(t.j, k) * atom.U(t.i, k));
                        atom.Gamma(t.j, t.i, k) += integrand * wlamu;
                        t.Rij(k) += I(k) * Vij(k) * wlamu;
                        t.Rji(k) += (Uji(k) + I(k) * Vij(k)) * wlamu;
                    }
                }
            }
        }
    }
    for (int k = 0; k < Nspace; ++k)
    {
        f64 dJ = abs(1.0 - JDag(k) / J(k));
        dJMax = max(dJ, dJMax);
    }
    return dJMax;
}

f64 stokes_fs_core(StokesCoreData& data, int la, bool updateJ)
{
    JasUnpack(*data, atmos, spect, fd, background, activeAtoms, JDag);
    JasUnpack(data, chiTot, etaTot, Uji, Vij, Vji, I, S);

    const int Nspace = atmos.Nspace;
    const int Nrays = atmos.Nrays;
    F64View J = spect.J(la);
    if (updateJ)
    {
        JDag = spect.J(la);
        J.fill(0.0);
    }

    for (int a = 0; a < activeAtoms.size(); ++a)
        activeAtoms[a]->setup_wavelength(la);

    // NOTE(cmo): If we only have continua then opacity is angle independent
    bool continuaOnly = true;
    for (int a = 0; a < activeAtoms.size(); ++a)
    {
        auto& atom = *activeAtoms[a];
        for (int kr = 0; kr < atom.Ntrans; ++kr)
        {
            auto& t = *atom.trans[kr];
            if (!t.active(la))
                continue;
            continuaOnly = continuaOnly && (t.type == CONTINUUM);
        }
    }

    f64 dJMax = 0.0;
    for (int mu = 0; mu < Nrays; ++mu)
    {
        for (int toObsI = 0; toObsI < 2; toObsI += 1)
        {
            bool toObs = (bool)toObsI;
            bool polarisedFrequency = false;
            // const f64 sign = If toObs Then 1.0 Else -1.0 End;
            if (!continuaOnly || (continuaOnly && (mu == 0 && toObsI == 0)))
            {
                chiTot.fill(0.0);
                etaTot.fill(0.0);
                S.fill(0.0);

                for (int a = 0; a < activeAtoms.size(); ++a)
                {
                    auto& atom = *activeAtoms[a];
                    for (int kr = 0; kr < atom.Ntrans; ++kr)
                    {
                        auto& t = *atom.trans[kr];
                        if (!t.active(la))
                            continue;

                        t.uv(la, mu, toObs, Uji, Vij, Vji);

                        for (int k = 0; k < Nspace; ++k)
                        {
                            f64 chi = atom.n(t.i, k) * Vij(k) - atom.n(t.j, k) * Vji(k);
                            f64 eta = atom.n(t.j, k) * Uji(k);

                            chiTot(0, k) += chi;
                            etaTot(0, k) += eta;

// NOTE(cmo): Even with this stuff commented out and B set to 0, it still yeilds the same incorrect result. Must be
// something wrong with the FS as this should be identical to no-pol case.
#if 1
                            if (t.type == TransitionType::LINE && t.polarised)
                            {
                                polarisedFrequency = true;
                                int lt = t.lt_idx(la);
                                f64 chiNoProfile = chi / t.phi(lt, mu, toObs, k);
                                chiTot(1, k) += chiNoProfile * t.phiQ(lt, mu, toObs, k);
                                chiTot(2, k) += chiNoProfile * t.phiU(lt, mu, toObs, k);
                                chiTot(3, k) += chiNoProfile * t.phiV(lt, mu, toObs, k);
                                chiTot(4, k) += chiNoProfile * t.psiQ(lt, mu, toObs, k);
                                chiTot(5, k) += chiNoProfile * t.psiU(lt, mu, toObs, k);
                                chiTot(6, k) += chiNoProfile * t.psiV(lt, mu, toObs, k);

                                f64 etaNoProfile = eta / t.phi(lt, mu, toObs, k);
                                etaTot(1, k) += etaNoProfile * t.phiQ(lt, mu, toObs, k);
                                etaTot(2, k) += etaNoProfile * t.phiU(lt, mu, toObs, k);
                                etaTot(3, k) += etaNoProfile * t.phiV(lt, mu, toObs, k);
                            }
#endif
                        }
                    }
                }
                // Do LTE atoms here

                for (int k = 0; k < Nspace; ++k)
                {
                    chiTot(0, k) += background.chi(la, k);
                    S(0, k) = (etaTot(0, k) + background.eta(la, k) + background.sca(la, k) * JDag(k)) / chiTot(0, k);
                }
                if (polarisedFrequency)
                {
                    for (int n = 1; n < 4; ++n)
                    {
                        for (int k = 0; k < Nspace; ++k)
                        {
                            S(n, k) = etaTot(n, k) / chiTot(0, k);
                        }
                    }
                }
            }

#if 1
            piecewise_stokes_bezier3_1d(&fd, mu, toObs, spect.wavelength(la), polarisedFrequency);
            spect.I(la, mu) = I(0, 0);
            spect.Quv(0, la, mu) = I(1, 0);
            spect.Quv(1, la, mu) = I(2, 0);
            spect.Quv(2, la, mu) = I(3, 0);
#else
            // NOTE(cmo): Checking with the normal FS and just using the first row of ezach of the matrices does indeed
            // produce the correct result
            piecewise_bezier3_1d(&fd.fdIntens, mu, toObs, spect.wavelength(la));
            spect.I(la, mu) = I(0, 0);
            spect.Quv(0, la, mu) = 0.0;
            spect.Quv(1, la, mu) = 0.0;
            spect.Quv(2, la, mu) = 0.0;
#endif

            if (updateJ)
            {
                for (int k = 0; k < Nspace; ++k)
                {
                    J(k) += 0.5 * atmos.wmu(mu) * I(0, k);
                }
            }
        }
    }
    if (updateJ)
    {
        for (int k = 0; k < Nspace; ++k)
        {
            f64 dJ = abs(1.0 - JDag(k) / J(k));
            dJMax = max(dJ, dJMax);
        }
    }
    return dJMax;
}
}

f64 gamma_matrices_formal_sol(Context& ctx)
{
    // feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
    JasUnpack(*ctx, atmos, spect, background);
    JasUnpack(ctx, activeAtoms);

    const int Nspace = atmos.Nspace;
    const int Nrays = atmos.Nrays;
    const int Nspect = spect.wavelength.shape(0);
    // auto Iplus = spect.I;

    F64Arr chiTot = F64Arr(Nspace);
    F64Arr etaTot = F64Arr(Nspace);
    F64Arr S = F64Arr(Nspace);
    F64Arr Uji = F64Arr(Nspace);
    F64Arr Vij = F64Arr(Nspace);
    F64Arr Vji = F64Arr(Nspace);
    F64Arr I = F64Arr(Nspace);
    F64Arr PsiStar = F64Arr(Nspace);
    F64Arr Ieff = F64Arr(Nspace);
    F64Arr JDag = F64Arr(Nspace);
    FormalData fd;
    fd.atmos = &atmos;
    fd.chi = chiTot;
    fd.S = S;
    fd.Psi = PsiStar;
    fd.I = I;
    IntensityCoreData iCore;
    JasPackPtr(iCore, atmos, spect, fd, background, activeAtoms, JDag);
    JasPack(iCore, chiTot, etaTot, Uji, Vij, Vji);
    JasPack(iCore, I, S, Ieff, PsiStar);

    printf("%d, %d, %d\n", Nspace, Nrays, Nspect);

    if (spect.JRest)
        spect.JRest.fill(0.0);

    for (auto& a : activeAtoms)
    {
        a->zero_rates();
    }

    f64 dJMax = 0.0;
    for (int la = 0; la < Nspect; ++la)
    {
        f64 dJ = GammaFsCores::intensity_core(iCore, la);
        dJMax = max(dJ, dJMax);
    }
    for (int a = 0; a < activeAtoms.size(); ++a)
    {
        auto& atom = *activeAtoms[a];
        for (int k = 0; k < Nspace; ++k)
        {
            for (int i = 0; i < atom.Nlevel; ++i)
            {
                atom.Gamma(i, i, k) = 0.0;
                f64 gammaDiag = 0.0;
                for (int j = 0; j < atom.Nlevel; ++j)
                {
                    gammaDiag += atom.Gamma(j, i, k);
                }
                atom.Gamma(i, i, k) = -gammaDiag;
            }
        }
    }
    return dJMax;
}

f64 formal_sol_full_stokes(Context& ctx)
{
    // feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
    JasUnpack(*ctx, atmos, spect, background);
    JasUnpack(ctx, activeAtoms);

    if (!atmos.B)
        assert(false && "Magnetic field required");

    const int Nspace = atmos.Nspace;
    const int Nrays = atmos.Nrays;
    const int Nspect = spect.wavelength.shape(0);
    // auto Iplus = spect.I;

    F64Arr2D chiTot = F64Arr2D(7, Nspace);
    F64Arr2D etaTot = F64Arr2D(4, Nspace);
    F64Arr2D S = F64Arr2D(4, Nspace);
    F64Arr Uji = F64Arr(Nspace);
    F64Arr Vij = F64Arr(Nspace);
    F64Arr Vji = F64Arr(Nspace);
    F64Arr2D I = F64Arr2D(4, Nspace);
    F64Arr JDag = F64Arr(Nspace);
    FormalDataStokes fd;
    fd.atmos = &atmos;
    fd.chi = chiTot;
    fd.S = S;
    fd.I = I;
    fd.fdIntens.atmos = fd.atmos;
    fd.fdIntens.chi = fd.chi(0);
    fd.fdIntens.S = fd.S(0);
    fd.fdIntens.I = fd.I(0);
    StokesCoreData core;
    JasPackPtr(core, atmos, spect, fd, background, activeAtoms, JDag);
    JasPack(core, chiTot, etaTot, Uji, Vij, Vji, I, S);

    printf("%d, %d, %d\n", Nspace, Nrays, Nspect);

    f64 dJMax = 0.0;
    for (int la = 0; la < Nspect; ++la)
    {
        f64 dJ = GammaFsCores::stokes_fs_core(core, la, true);
        dJMax = max(dJ, dJMax);
    }
    return dJMax;
}

namespace EscapeProbability
{

void compute_phi_mu_1(const Transition& t, const Atmosphere& atmos, int lt, F64View vBroad, F64View phi)
{
    namespace C = Constants;
    if (t.type == TransitionType::CONTINUUM)
        return;

    // Why is there still no constexpr math in std? :'(
    const f64 sqrtPi = sqrt(C::Pi);

    const f64 vBase = (t.wavelength(lt) - t.lambda0) * C::CLight / t.lambda0;
    // const f64 wla = t.wlambda(la);
    for (int k = 0; k < atmos.Nspace; ++k)
    {
        const f64 vk = (vBase + atmos.vlos(k)) / vBroad(k);
        const f64 p = voigt_H(t.aDamp(k), vk) / (sqrtPi * vBroad(k));
        phi(k) = p;
        // wphi(k) += p * wla;
    }
}

void uv_mu_1(const Atom& atom, const Transition& t, int lt, F64View phi, F64View Uji, F64View Vij, F64View Vji)
{
    namespace C = Constants;

    if (t.type == TransitionType::LINE)
    {
        constexpr f64 hc_4pi = 0.25 * C::HC / C::Pi;
        const f64 gij = t.Bji / t.Bij;
        for (int k = 0; k < Vij.shape(0); ++k)
        {
            Vij(k) = hc_4pi * t.Bij * phi(k);
            Vji(k) = gij * Vij(k);
            Uji(k) = t.Aji / t.Bji * Vji(k);
        }
    }
    else
    {
        constexpr f64 hc = 2 * C::HC / cube(C::NM_TO_M);
        const f64 hcl = hc / cube(t.wavelength(lt));
        const f64 a = t.alpha(lt);
        constexpr f64 hc_k = C::HC / (C::KBoltzmann * C::NM_TO_M);
        const f64 hc_kl = hc_k / t.wavelength(lt);
        for (int k = 0; k < Vij.shape(0); ++k)
        {
            const f64 gij = atom.nStar(t.i, k) / atom.nStar(t.j, k) * exp(-hc_kl / atom.atmos->temperature(k));
            Vij(k) = a;
            Vji(k) = gij * Vij(k);
            Uji(k) = hcl * Vji(k);
        }
    }
}

f64 escape_probability(bool line, f64 tau, f64 tauC, f64 alpha, f64* dq)
{
    namespace C = Constants;
    *dq = 0.0;
    if (tauC > 50.0)
        return 0.0;

    f64 etc = exp(-tauC);
    if (line)
    {
        f64 beta = 2.0 * C::Pi;
        f64 q = etc / (2.0 + beta * tau);
        *dq = -(tauC * beta + 2.0 * tauC / tau + beta) * q / (beta * tau + 2.0);
        return q;
    }

    f64 beta = max(3.0 * (tau + tauC) / alpha, 1.0);
    f64 b3 = cube(beta);
    f64 q = exp(-b3 * (tau + tauC) - alpha * (beta - 1.0)) / (2 * beta);
    *dq = -b3 * q;
    return q;
}

f64 escape_formal_sol(const Atmosphere& atmos, f64 lambda, F64View chi, F64View chiB, F64View S, F64View P, F64View Q,
    F64View Lambda, bool line)
{
    namespace C = Constants;
    // NOTE(cmo): This is a Feautrier style method, i.e. P = I+ + I-, Q = I+ - I-
    F64Arr tau(atmos.Nspace);
    F64Arr tauB(atmos.Nspace);

    tau(0) = 0.0;
    tauB(0) = 0.0;
    for (int k = 1; k < atmos.Nspace-1; ++k)
    {
        f64 zz = abs(atmos.height(k - 1) - atmos.height(k + 1)) * 0.5;
        tauB(k) += tauB(k - 1) + chiB(k) * zz;
        tau(k) += tau(k - 1) + chi(k) * zz + tauB(k);
    }
    tau(0) = 0.5 * tau(1);
    tauB(0) = 0.5 * tauB(1);
    tau(atmos.Nspace - 1) = 2.0 * tau(atmos.Nspace - 2);
    tauB(atmos.Nspace - 1) = 2.0 * tauB(atmos.Nspace - 2);

    P(atmos.Nspace - 1) = S(atmos.Nspace - 1);
    Q(atmos.Nspace - 1) = 0.0;
    Lambda(atmos.Nspace - 1) = 1.0;

    f64 sum = 0.0;
    for (int k = atmos.Nspace - 2; k > 1; --k)
    {
        f64 t = tau(k);
        f64 tb = tauB(k);

        f64 alpha = C::HC / C::KBoltzmann / lambda / atmos.temperature(k);
        f64 dp;
        f64 ep = escape_probability(line, t, tb, alpha, &dp);

        Lambda(k) = 1.0 - 2.0 * ep;
        f64 dx = 0.5 * log((tau(k + 1) + tauB(k + 1)) / (tau(k - 1) + tauB(k - 1)));
        f64 h = -S(k) * dp * (tau(k) * dx);
        sum += h;

        P(k) = S(k) * (1.0 - 2.0 * ep) + sum;
        Q(k) = -S(k) * 2.0 * ep + sum;
    }

    P(0) = P(1);
    Lambda(0) = Lambda(1);
    Q(0) = Q(1);
    f64 Iplus = P[0];
    return Iplus;
}

void gamma_matrices_escape_prob(Atom* a, Background& background, const Atmosphere& atmos)
{
    // JasUnpack(*ctx, atmos, background);
    // JasUnpack(ctx, activeAtoms);

    // F64Arr muzOld{atmos.muz};
    // F64Arr wmuOld{atmos.wmu};
    // int NraysOld = atmos.Nrays;

    // atmos.Nrays = 1;
    // atmos.muz(0) = 1.0;
    // atmos.wmu(0) = 1.0;
    auto atom = *a;

    F64Arr chi(atmos.Nspace);
    F64Arr eta(atmos.Nspace);
    F64Arr Uji(atmos.Nspace);
    F64Arr Vij(atmos.Nspace);
    F64Arr Vji(atmos.Nspace);
    F64Arr phi(atmos.Nspace);
    F64Arr S(atmos.Nspace);
    F64Arr P(atmos.Nspace);
    F64Arr Q(atmos.Nspace);
    F64Arr Lambda(atmos.Nspace);

    for (int kr = 0; kr < atom.Ntrans; ++kr)
    {
        auto& t = *atom.trans[kr];
        int lt = 0;
        bool line = false;
        if (t.type == TransitionType::LINE)
        {
            lt = t.wavelength.shape(0) / 2;
            line = true;
            compute_phi_mu_1(t, atmos, lt, atom.vBroad, phi);
        }
        int la = lt + t.Nblue;

        auto chiB = background.chi(la);
        auto etaB = background.eta(la);

        if (t.type == TransitionType::LINE)
        {
            for (int k = 0; k < atmos.Nspace; ++k)
            {
                uv_mu_1(atom, t, lt, phi, Uji, Vij, Vji);
                f64 x = atom.n(t.i, k) * Vij(k) - atom.n(t.j, k) * Vji(k);
                chi(k) = x;
                f64 n = atom.n(t.j, k) * Uji(k);
                S(k) = (n + etaB(k)) / (chi(k) + chiB(k));
            }
            // do FS
            escape_formal_sol(atmos, t.wavelength(lt), chi, chiB, S, P, Q, Lambda, true);
            // Add to Gamma
            for (int k = 0; k < atmos.Nspace; ++k)
            {
                f64 Ieff = P(k) - S(k) * Lambda(k);
                atom.Gamma(t.j, t.i, k) += t.Bij * Ieff;
                atom.Gamma(t.i, t.j, k) += t.Aji * (1 - Lambda(k)) + t.Bji * Ieff;
            }
        }
        else
        {
            // We should avoid sampling the continuum more than every... 10nm?
            // But remember to sample final chunk, or the weighting will be off
            f64 wlaSum = 0.0;
            f64 prevWl = 0.0;
            int Nlambda = t.wavelength.shape(0);
            for (int ltc = 0; ltc < Nlambda; ++ltc)
            {
                wlaSum += t.wlambda(ltc);

                if (t.wavelength(ltc) - prevWl < 10.0 && ltc != Nlambda - 1)
                    continue;

                prevWl = t.wavelength(ltc);
                for (int k = 0; k < atmos.Nspace; ++k)
                {
                    uv_mu_1(atom, t, lt, phi, Uji, Vij, Vji);
                    f64 x = atom.n(t.i, k) * Vij(k) - atom.n(t.j, k) * Vji(k);
                    chi(k) = x;
                    f64 n = atom.n(t.j, k) * Uji(k);
                    S(k) = (n + etaB(k)) / (chi(k) + chiB(k));
                }
                escape_formal_sol(atmos, t.wavelength(ltc), chi, chiB, S, P, Q, Lambda, false);
                // NOTE(cmo): This method is pretty basic, in that we pretend
                // the continuum is constant over each chunk and use that value.
                // For anything left over, we assume it's equal to the last
                // (reddest) point in the continuum.
                for (int k = 0; k < atmos.Nspace; ++k)
                {
                    f64 Ieff = P(k) - S(k) * Lambda(k);
                    f64 integrand = (Uji(k) + Vji(k) * Ieff) - (Lambda(k) * Uji(k));
                    atom.Gamma(t.i, t.j, k) += integrand * wlaSum;

                    integrand = (Vij(k) * Ieff) - (Lambda(k) * Uji(k));
                    atom.Gamma(t.j, t.i, k) += integrand * wlaSum;
                }
                wlaSum = 0.0;
            }
        }
    }
    for (int k = 0; k < atmos.Nspace; ++k)
    {
        for (int i = 0; i < atom.Nlevel; ++i)
        {
            atom.Gamma(i, i, k) = 0.0;
            f64 gammaDiag = 0.0;
            for (int j = 0; j < atom.Nlevel; ++j)
            {
                gammaDiag += atom.Gamma(j, i, k);
            }
            atom.Gamma(i, i, k) = -gammaDiag;
        }
    }

    // atmos.Nrays = NraysOld;
    // for (int i = 0; i < atmos.Nrays; ++i)
    // {
    //     atmos.muz(i) = muzOld(i);
    //     atmos.wmu(i) = wmuOld(i);
    // }
}
}

void stat_eq(Atom* atomIn)
{
    auto& atom = *atomIn;
    int Nlevel = atom.Nlevel;
    const int Nspace = atom.n.shape(1);

    auto nk = F64Arr(Nlevel);
    auto Gamma = F64Arr2D(Nlevel, Nlevel);

    for (int k = 0; k < Nspace; ++k)
    {
        for (int i = 0; i < Nlevel; ++i)
        {
            nk(i) = atom.n(i, k);
            for (int j = 0; j < Nlevel; ++j)
                Gamma(i, j) = atom.Gamma(i, j, k);
        }

        int iEliminate = 0;
        f64 nMax = 0.0;
        for (int i = 0; i < Nlevel; ++i)
            nMax = max_idx(nMax, nk(i), iEliminate, i);

        for (int i = 0; i < Nlevel; ++i)
        {
            Gamma(iEliminate, i) = 1.0;
            nk(i) = 0.0;
        }
        nk(iEliminate) = atom.nTotal(k);

        solve_lin_eq(Gamma, nk);
        // int one = 1;
        // int* ipiv = (int*)malloc(Nlevel * sizeof(int));
        // int info = 0;
        // solve(&Nlevel, &one, Gamma.data.data(), &Nlevel, ipiv, nk.data.data(), &Nlevel, &info);
        for (int i = 0; i < Nlevel; ++i)
            atom.n(i, k) = nk(i);
    }
}

namespace PrdCores
{
void total_depop_rate(const Transition* trans, const Atom& atom, F64View Pj)
{
    const int Nspace = trans->Rij.shape(0);

    for (int k = 0; k < Nspace; ++k)
    {
        Pj(k) = trans->Qelast(k);
        for (int i = 0; i < atom.C.shape(0); ++i)
            Pj(k) += atom.C(i, trans->j, k);

        for (auto& t : atom.trans)
        {
            if (t->j == trans->j)
                Pj(k) += t->Rji(k);
            if (t->i == trans->j)
                Pj(k) += t->Rij(k);
        }
    }
}


constexpr f64 PrdQWing = 4.0;
constexpr f64 PrdQCore = 4.0;
constexpr f64 PrdQSpread = 5.0;
constexpr f64 PrdDQ = 0.25;

/*
    * Gouttebroze's fast approximation for
    *  GII(q_abs, q_emit) = PII(q_abs, q_emit) / phi(q_emit)

    * See: P. Gouttebroze, 1986, A&A 160, 195
    *      H. Uitenbroek,  1989, A&A, 216, 310-314 (cross redistribution)
    */

inline f64 G_zero(f64 x)
{
    return 1.0 / (abs(x) + sqrt(square(x) + 1.273239545));
}

f64 GII(f64 adamp, f64 q_emit, f64 q_abs)
{
    constexpr f64 waveratio = 1.0;
    namespace C = Constants;
    f64 gii, pcore, aq_emit, umin, epsilon, giiwing, u1, phicore, phiwing;

    /* --- Symmetrize with respect to emission frequency --   --------- */

    if (q_emit < 0.0)
    {
        q_emit = -q_emit;
        q_abs = -q_abs;
    }
    pcore = 0.0;
    gii = 0.0;

    /* --- Core region --                                     --------- */

    if (q_emit < PrdQWing)
    {
        if ((q_abs < -PrdQWing) || (q_abs > q_emit + waveratio * PrdQSpread))
            return gii;
        if (abs(q_abs) <= q_emit)
            gii = G_zero(q_emit);
        else
            gii = exp(square(q_emit) - square(q_abs)) * G_zero(q_abs);

        if (q_emit >= PrdQCore)
        {
            phicore = exp(-square(q_emit));
            phiwing = adamp / (sqrt(C::Pi) * (square(adamp) + square(q_emit)));
            pcore = phicore / (phicore + phiwing);
        }
    }
    /* --- Wing region --                                     --------- */

    if (q_emit >= PrdQCore)
    {
        aq_emit = waveratio * q_emit;
        if (q_emit >= PrdQWing)
        {
            if (abs(q_abs - aq_emit) > waveratio * PrdQSpread)
                return gii;
            pcore = 0.0;
        }
        umin = abs((q_abs - aq_emit) / (1.0 + waveratio));
        giiwing = (1.0 + waveratio) * (1.0 - 2.0 * umin * G_zero(umin)) * exp(-square(umin));

        if (waveratio == 1.0)
        {
            epsilon = q_abs / aq_emit;
            giiwing *= (2.75 - (2.5 - 0.75 * epsilon) * epsilon);
        }
        else
        {
            u1 = abs((q_abs - aq_emit) / (waveratio - 1.0));
            giiwing -= abs(1.0 - waveratio) * (1.0 - 2.0 * u1 * G_zero(u1)) * exp(-square(u1));
        }
        /* --- Linear combination of core- and wing contributions ------- */

        giiwing = giiwing / (2.0 * waveratio * sqrt(C::Pi));
        gii = pcore * gii + (1.0 - pcore) * giiwing;
    }
    return gii;
}

void prd_scatter(Transition* t, F64View Pj, const Atom& atom, const Atmosphere& atmos, const Spectrum& spect)
{
    auto& trans = *t;

    namespace C = Constants;
    const int Nlambda = trans.wavelength.shape(0);

    bool initialiseGii = (!trans.gII) || (trans.gII(0, 0, 0) < 0.0);
    constexpr int maxFineGrid = max(3 * PrdQWing, 2 * PrdQSpread) / PrdDQ + 1;
    if (!trans.gII)
        trans.gII = F64Arr3D(Nlambda, atmos.Nspace, maxFineGrid);

    // Reset Rho
    trans.rhoPrd.fill(1.0);

    F64Arr Jk(Nlambda);
    F64Arr qAbs(Nlambda);
    F64Arr JFine(maxFineGrid);
    F64Arr qp(maxFineGrid);
    F64Arr wq(maxFineGrid);

    for (int k = 0; k < atmos.Nspace; ++k)
    {
        f64 gamma = atom.n(trans.i, k) / atom.n(trans.j, k) * trans.Bij / Pj(k);
        f64 Jbar = trans.Rij(k) / trans.Bij;

        if (spect.JRest)
        {
            for (int la = 0; la < Nlambda; ++la)
            {
                int prdLa = spect.la_to_prdLa(la + trans.Nblue);
                Jk(la) = spect.JRest(prdLa, k);
                // Jk(la) = spect.JRest(la + trans.Nblue, k);
            }
        }
        else
        {
            for (int la = 0; la < Nlambda; ++la)
            {
                Jk(la) = spect.J(la + trans.Nblue, k);
            }
        }
        // Local mean intensity in doppler units
        for (int la = 0; la < Nlambda; ++la)
        {
            qAbs(la) = (trans.wavelength(la) - trans.lambda0) * C::CLight / (trans.lambda0 * atom.vBroad(k));
        }

        for (int la = 0; la < Nlambda; ++la)
        {
            f64 qEmit = qAbs(la);

            int q0, qN;
            if (abs(qEmit) < PrdQCore)
            {
                q0 = -PrdQWing;
                qN = PrdQWing;
            }
            else if (abs(qEmit) < PrdQWing)
            {
                if (qEmit > 0.0)
                {
                    q0 = -PrdQWing;
                    qN = qEmit + PrdQSpread;
                }
                else
                {
                    q0 = qEmit - PrdQSpread;
                    qN = PrdQWing;
                }
            }
            else
            {
                q0 = qEmit - PrdQSpread;
                qN = qEmit + PrdQSpread;
            }
            int Np = int((f64)(qN - q0) / PrdDQ) + 1;
            qp(0) = q0;
            for (int lap = 1; lap < Np; ++lap)
                qp(lap) = qp(lap - 1) + PrdDQ;

            linear(qAbs, Jk, qp.slice(0, Np), JFine);

            if (initialiseGii)
            {
                wq.fill(PrdDQ);
                wq(0) = 5.0 / 12.0 * PrdDQ;
                wq(1) = 13.0 / 12.0 * PrdDQ;
                wq(Np - 1) = 5.0 / 12.0 * PrdDQ;
                wq(Np - 2) = 13.0 / 12.0 * PrdDQ;
                for (int lap = 0; lap < Np; ++lap)
                    trans.gII(la, k, lap) = GII(trans.aDamp(k), qEmit, qp(lap)) * wq(lap);
            }
            F64View gII = trans.gII(la, k);

            f64 gNorm = 0.0;
            f64 scatInt = 0.0;
            for (int lap = 0; lap < Np; ++lap)
            {
                gNorm += gII(lap);
                scatInt += JFine(lap) * gII(lap);
            }
            trans.rhoPrd(la, k) += gamma * (scatInt / gNorm - Jbar);
        }
    }
}
}

f64 formal_sol_update_rates(Context& ctx, I32View wavelengthIdxs)
{
    JasUnpack(*ctx, atmos, spect, background);
    JasUnpack(ctx, activeAtoms);

    const int Nspace = atmos.Nspace;
    const int Nrays = atmos.Nrays;

    F64Arr chiTot = F64Arr(Nspace);
    F64Arr etaTot = F64Arr(Nspace);
    F64Arr S = F64Arr(Nspace);
    F64Arr Uji = F64Arr(Nspace);
    F64Arr Vij = F64Arr(Nspace);
    F64Arr Vji = F64Arr(Nspace);
    F64Arr I = F64Arr(Nspace);
    F64Arr Ieff = F64Arr(Nspace);
    F64Arr JDag = F64Arr(Nspace);
    FormalData fd;
    fd.atmos = &atmos;
    fd.chi = chiTot;
    fd.S = S;
    fd.I = I;

    printf("%d, %d\n", Nspace, Nrays);

    for (auto& a : activeAtoms)
    {
        for (auto& t : a->trans)
        {
            if (t->rhoPrd)
            {
                t->zero_rates();
            }
        }
    }
    if (spect.JRest)
        spect.JRest.fill(0.0);

    f64 dJMax = 0.0;

    for (int i = 0; i < wavelengthIdxs.shape(0); ++i)
    {
        const f64 la = wavelengthIdxs(i);
        JDag = spect.J(la);
        F64View J = spect.J(la);
        J.fill(0.0);

        for (int a = 0; a < activeAtoms.size(); ++a)
            activeAtoms[a]->setup_wavelength(la);

        for (int mu = 0; mu < Nrays; ++mu)
        {
            for (int toObsI = 0; toObsI < 2; toObsI += 1)
            {
                bool toObs = (bool)toObsI;

                chiTot.fill(0.0);
                etaTot.fill(0.0);

                for (int a = 0; a < activeAtoms.size(); ++a)
                {
                    auto& atom = *activeAtoms[a];
                    for (int kr = 0; kr < atom.Ntrans; ++kr)
                    {
                        auto& t = *atom.trans[kr];
                        if (!t.active(la))
                            continue;

                        t.uv(la, mu, toObs, Uji, Vij, Vji);

                        for (int k = 0; k < Nspace; ++k)
                        {
                            f64 chi = atom.n(t.i, k) * Vij(k) - atom.n(t.j, k) * Vji(k);
                            f64 eta = atom.n(t.j, k) * Uji(k);

                            chiTot(k) += chi;
                            etaTot(k) += eta;
                            atom.eta(k) += eta;
                        }
                    }
                }
                // Do LTE atoms here

                for (int k = 0; k < Nspace; ++k)
                {
                    chiTot(k) += background.chi(la, k);
                    S(k) = (etaTot(k) + background.eta(la, k) + background.sca(la, k) * JDag(k)) / chiTot(k);
                }

                piecewise_bezier3_1d(&fd, mu, toObs, spect.wavelength(la));
                spect.I(la, mu) = I(0);

                for (int k = 0; k < Nspace; ++k)
                {
                    J(k) += 0.5 * atmos.wmu(mu) * I(k);
                }

                if (spect.JRest && spect.hPrdActive && spect.hPrdActive(la))
                {
                    int hPrdLa = spect.la_to_hPrdLa(la);
                    for (int k = 0; k < Nspace; ++k)
                    {
                        const auto& coeffs = spect.JCoeffs(hPrdLa, mu, toObs, k);
                        for (const auto& c : coeffs)
                        {
                            spect.JRest(c.idx, k) += 0.5 * atmos.wmu(mu) * c.frac * I(k);
                        }
                    }
                }

                for (int a = 0; a < activeAtoms.size(); ++a)
                {
                    auto& atom = *activeAtoms[a];
                    for (int kr = 0; kr < atom.Ntrans; ++kr)
                    {
                        auto& t = *atom.trans[kr];
                        if (!t.active(la) || !t.rhoPrd)
                            continue;

                        const f64 wmu = 0.5 * atmos.wmu(mu);
                        t.uv(la, mu, toObs, Uji, Vij, Vji);

                        for (int k = 0; k < Nspace; ++k)
                        {
                            const f64 wlamu = atom.wla(kr, k) * wmu;
                            t.Rij(k) += I(k) * Vij(k) * wlamu;
                            t.Rji(k) += (Uji(k) + I(k) * Vij(k)) * wlamu;
                        }
                    }
                }
            }
        }
        for (int k = 0; k < Nspace; ++k)
        {
            f64 dJ = abs(1.0 - JDag(k) / J(k));
            dJMax = max(dJ, dJMax);
        }
    }
    return dJMax;
}

f64 formal_sol_update_rates(Context& ctx, const std::vector<int>& wavelengthIdxs)
{
    // TODO(cmo): Really need to fix the const-correctness on these arrays, if possible.
    I32View wavelengthView(const_cast<int*>(wavelengthIdxs.data()), wavelengthIdxs.size());
    return formal_sol_update_rates(ctx, wavelengthView);
}

f64 redistribute_prd_lines(Context& ctx, int maxIter, f64 tol)
{
    struct PrdData
    {
        Transition* line;
        const Atom& atom;
        Ng ng;

        PrdData(Transition* l, const Atom& a, Ng&& n)
            : line(l), atom(a), ng(n)
        {}
    };
    JasUnpack(*ctx, atmos, spect);
    JasUnpack(ctx, activeAtoms);
    std::vector<PrdData> prdLines;
    prdLines.reserve(10);
    for (auto& a : activeAtoms)
    {
        for (auto& t : a->trans)
        {
            if (t->rhoPrd)
            {
                // t->zero_rates();
                prdLines.emplace_back(PrdData(t, *a, Ng(0, 0, 0, t->rhoPrd.flatten())));
            }
        }
    }

    const int Nspect = spect.wavelength.shape(0);
    auto& idxsForFs = spect.hPrdIdxs;
    std::vector<int> prdIdxs;
    if (spect.hPrdIdxs.size() == 0)
    {
        prdIdxs.reserve(Nspect);
        for (int la = 0; la < Nspect; ++la)
        {
            bool prdLinePresent = false;
            for (auto& p : prdLines)
                prdLinePresent = (p.line->active(la) || prdLinePresent);
            if (prdLinePresent)
                prdIdxs.emplace_back(la);
        }
        idxsForFs = prdIdxs;
    }

    int iter = 0;
    f64 dRho = 0.0;
    F64Arr Pj(atmos.Nspace);
    while (iter < maxIter)
    {
        for (auto& p : prdLines)
        {
            PrdCores::total_depop_rate(p.line, p.atom, Pj);
            PrdCores::prd_scatter(p.line, Pj, p.atom, atmos, spect);
            p.ng.accelerate(p.line->rhoPrd.flatten());
            dRho = max(dRho, p.ng.max_change());
        }

        formal_sol_update_rates(ctx, idxsForFs);

        if (dRho < tol)
            break;

        ++iter;
    }

    return dRho;
}

void configure_hprd_coeffs(Context& ctx)
{
    namespace C = Constants;
    struct PrdData
    {
        Transition* line;
        const Atom& atom;

        PrdData(Transition* l, const Atom& a)
            : line(l), atom(a)
        {}
    };
    JasUnpack(*ctx, atmos, spect);
    JasUnpack(ctx, activeAtoms);
    std::vector<PrdData> prdLines;
    prdLines.reserve(10);
    for (auto& a : activeAtoms)
    {
        for (auto& t : a->trans)
        {
            if (t->rhoPrd)
            {
                prdLines.emplace_back(PrdData(t, *a));
            }
        }
    }

    if (prdLines.size() == 0)
        return;

    const int Nspect = spect.wavelength.shape(0);
    spect.prdActive = BoolArr(false, Nspect);
    spect.la_to_prdLa = I32Arr(0, Nspect);
    auto& prdLambdas = spect.prdIdxs;
    prdLambdas.clear();
    prdLambdas.reserve(Nspect);
    for (int la = 0; la < Nspect; ++la)
    {
        bool prdLinePresent = false;
        for (auto& p : prdLines)
            prdLinePresent = (p.line->active(la) || prdLinePresent);
        if (prdLinePresent)
        {
            prdLambdas.emplace_back(la);
            spect.prdActive(la) = true;
            spect.la_to_prdLa(la) = prdLambdas.size()-1;
        }
    }


    // NOTE(cmo): We can't simply store the prd wavelengths and then only
    // compute JRest from those. JRest can be only prdIdxs long, but we need to
    // compute all of the contributors to each of these prdIdx.
    // NOTE(cmo): This might be overcomplicating the problem. STiC is happy to
    // only compute these terms for the prd idxs. But I worry if a high Doppler
    // shift were to brind intensity into the prdLine region. I don't think it's
    // massively likely, but 500km/s is 0.8nm @ 500nm, and I don't want the line
    // wings missing Jbar that they should have.
    // NOTE(cmo): Okay, I need to think about this one some more (note that in
    // its current state, something is exploding, but that's neither here nor
    // there in realm of design). Currently when we compute the udpated the J
    // for use internal to the prd_scatter function, we only loop over the prd
    // wavelengths. Therefore it wouldn't be consistent to add in contributions
    // from outside that range. We would therefore have to extend the wavelength
    // range over which the FS was being calculated to continue using this wider
    // definition. I don't know if that's worthwhile, especially as, for the
    // most part, PRD is making lines narrower rather than wider. i.e.
    // conecentrating the intensity towards the core, which is where this method
    // will be fine anyway. This is already an approximation, so it's probably
    // best not to overcomplicate it

    auto check_lambda_scatter_into_prd_region = [&](int la)
    {
        constexpr f64 sign[] = {-1.0, 1.0};
        for (int mu = 0; mu < atmos.Nrays; ++mu)
        {
            for (int toObs = 0; toObs <= 1; ++toObs)
            {
                for (int k = 0; k < atmos.Nspace; ++k)
                {
                    const f64 s = sign[toObs];
                    const f64 fac = 1.0 + atmos.vlosMu(mu, k) * s / C::CLight;
                    int prevIndex = max(la-1, 0);
                    int nextIndex = min(la+1, (int)spect.wavelength.shape(0)-1);
                    const f64 prevLambda = spect.wavelength(prevIndex) * fac;
                    const f64 nextLambda = spect.wavelength(nextIndex) * fac;

                    int i = la;
                    for (; spect.wavelength(i) > prevLambda && i >= 0; --i);
                    for (; i < spect.wavelength.shape(0); ++i)
                    {
                        const f64 lambdaI = spect.wavelength(i);
                        if (spect.prdActive(i))
                            return true;
                        else if (lambdaI > nextLambda)
                            break;
                    }
                }
            }
        }
        return false;
    };

    auto& hPrdIdxs = spect.hPrdIdxs;
    hPrdIdxs.clear();
    hPrdIdxs.reserve(Nspect);
    spect.la_to_hPrdLa = I32Arr(0, Nspect);
    spect.hPrdActive = BoolArr(false, Nspect);
    for (int la = 0; la < Nspect; ++la)
    {
        if (check_lambda_scatter_into_prd_region(la))
        {
            hPrdIdxs.emplace_back(la);
            spect.hPrdActive(la) = true;
            spect.la_to_hPrdLa(la) = hPrdIdxs.size()-1;
        }
    }

    spect.JRest = F64Arr2D(0.0, prdLambdas.size(), atmos.Nspace);
    spect.JCoeffs = Prd::JCoeffVec(hPrdIdxs.size(), atmos.Nrays, 2, atmos.Nspace);
    constexpr f64 sign[] = {-1.0, 1.0};

    for (auto idx : hPrdIdxs)
    {
        for (int mu = 0; mu < atmos.Nrays; ++mu)
        {
            for (int toObs = 0; toObs <= 1; ++toObs)
            {
                for (int k = 0; k < atmos.Nspace; ++k)
                {
                    int hPrdLa = spect.la_to_hPrdLa(idx);
                    auto coeffVec = spect.JCoeffs(hPrdLa, mu, toObs);
                    const f64 s = sign[toObs];

                    const f64 fac = 1.0 + atmos.vlosMu(mu, k) * s / C::CLight;
                    int prevIndex = max(idx-1, 0);
                    int nextIndex = min(idx+1, Nspect-1);
                    const f64 prevLambda = spect.wavelength(prevIndex) * fac;
                    const f64 lambdaRest = spect.wavelength(idx) * fac;
                    const f64 nextLambda = spect.wavelength(nextIndex) * fac;
                    bool doLowerHalf = true, doUpperHalf = true;
                    // These will only be equal on the ends. And we can't be at both ends at the same time.
                    if (prevIndex == idx)
                    {
                        doLowerHalf = false;
                        for (int i = 0; i < Nspect; ++i)
                        {
                            if (spect.wavelength(i) <= lambdaRest && spect.prdActive(i))
                                coeffVec(k).emplace_back(Prd::JInterpCoeffs{spect.la_to_prdLa(i), 1.0});
                            else
                                break;
                        }
                    }
                    else if (nextIndex == idx)
                    {
                        // NOTE(cmo): By doing this part here, there's a strong
                        // likelihood that the indices for the final point will
                        // not be monotonic, but the cost of this is probably
                        // siginificantly lower than sorting all of the arrays,
                        // especially as it is likely that this case won't be
                        // used as I don't expect a PRD line right on the edge
                        // of the wavelength window in most scenarios
                        doUpperHalf = false;
                        for (int i = Nspect-1; i >= 0; --i)
                        {
                            if (spect.wavelength(i) > lambdaRest && spect.prdActive(i))
                                coeffVec(k).emplace_back(Prd::JInterpCoeffs{spect.la_to_prdLa(i), 1.0});
                            else
                                break;
                        }
                    }

                    int i  = idx;
                    // NOTE(cmo): If the shift is s.t. spect.wavelength(idx) > prevLambda, then we need to roll back 
                    for (; spect.wavelength(i) > prevLambda && i >= 0; --i);


                    // NOTE(cmo): Upper bound goes all the way to the top, but we will break out early when possible.
                    for (; i < Nspect; ++i)
                    {
                        const f64 lambdaI = spect.wavelength(i);
                        // NOTE(cmo): Early termination condition
                        if (lambdaI > nextLambda)
                            break;

                        // NOTE(cmo): Don't do these if this is an edge case and was previously handled with constant extrapolation
                        if (doLowerHalf && spect.prdActive(i) && lambdaI > prevLambda && lambdaI <= lambdaRest)
                        {
                            const f64 frac = (lambdaI - prevLambda) / (lambdaRest - prevLambda);
                            coeffVec(k).emplace_back(Prd::JInterpCoeffs{spect.la_to_prdLa(i), frac});
                        }
                        else if (doUpperHalf && spect.prdActive(i) && lambdaI > lambdaRest && lambdaI < nextLambda)
                        {
                            const f64 frac = (lambdaI - lambdaRest) / (nextLambda - lambdaRest);
                            coeffVec(k).emplace_back(Prd::JInterpCoeffs{spect.la_to_prdLa(i), 1.0 - frac});
                        }
                    }
                }
            }
        }
    }

    for (auto& p : prdLines)
    {
        auto& wavelength = p.line->wavelength;
        p.line->hPrdCoeffs = Prd::RhoCoeffVec(wavelength.shape(0), atmos.Nrays, 2, atmos.Nspace);
        auto& coeffs = p.line->hPrdCoeffs;
        for (int lt = 0; lt < wavelength.shape(0); ++lt)
        {
            for (int mu = 0; mu < atmos.Nrays; ++mu)
            {
                for (int toObs = 0; toObs <= 1; ++toObs)
                {
                    for (int k = 0; k < atmos.Nspace; ++k)
                    {
                        const f64 s = sign[toObs];
                        const f64 lambdaRest = wavelength(lt) * (1.0 + atmos.vlosMu(mu, k) * s / C::CLight);
                        auto& c = coeffs(lt, mu, toObs, k);
                        if (lambdaRest <= wavelength(0))
                        {
                            c.frac = 0.0;
                            c.i0 = 0;
                            c.i1 = 1;
                        }
                        else if (lambdaRest >= wavelength(wavelength.shape(0)-1))
                        {
                            c.frac = 1.0;
                            c.i0 = wavelength.shape(0) - 2;
                            c.i1 = wavelength.shape(0) - 1;
                        }
                        else
                        {
                            auto it = std::upper_bound(wavelength.data, wavelength.data + wavelength.shape(0), lambdaRest) - 1;
                            c.frac = (lambdaRest - *it) / (*(it+1) - *it);
                            c.i0 = it - wavelength.data;
                            c.i1 = c.i0 + 1;
                        }
                    }
                }
            }
        }
    }
}