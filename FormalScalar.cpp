#include "Lightweaver.hpp"
#include "Bezier.hpp"
#include "JasPP.hpp"

#include <cmath>
#include <fenv.h>
#include <iostream>
#include <utility>
#include <vector>
#include <algorithm>

using namespace LwInternal;

#ifdef __APPLE__
// Public domain polyfill for feenableexcept on OS X
// http://www-personal.umich.edu/~williams/archive/computation/fe-handling-example.c
int feenableexcept(unsigned int excepts)
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
#endif

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

void piecewise_linear_1d_impl(FormalData* fd, f64 zmu, bool toObs, f64 Istart)
{
    JasUnpack((*fd), chi, S, Psi, I, atmos);
    const auto& height = atmos->height;
    const int Ndep = atmos->Nspace;
    bool computeOperator = bool(Psi);

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


void piecewise_bezier3_1d_impl(FormalData* fd, f64 zmu, bool toObs, f64 Istart)
{
    JasUnpack((*fd), atmos, chi, S, I, Psi);
    const auto& height = atmos->height;
    const int Ndep = atmos->Nspace;
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

namespace LwInternal
{
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

bool continua_only(const IntensityCoreData& data, int la)
{
    JasUnpack(*data, activeAtoms, lteAtoms);
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
    for (int a = 0; a < lteAtoms.size(); ++a)
    {
        auto& atom = *lteAtoms[a];
        for (int kr = 0; kr < atom.Ntrans; ++kr)
        {
            auto& t = *atom.trans[kr];
            if (!t.active(la))
                continue;
            continuaOnly = continuaOnly && (t.type == CONTINUUM);
        }
    }
    return continuaOnly;
}

void gather_opacity_emissivity(IntensityCoreData* data, bool computeOperator, int la, int mu, bool toObs)
{
    JasUnpack(*(*data), activeAtoms, lteAtoms);
    JasUnpack((*data), Uji, Vij, Vji, chiTot, etaTot);
    const int Nspace = data->atmos->Nspace;

    for (int a = 0; a < activeAtoms.size(); ++a)
    {
        auto& atom = *activeAtoms[a];
        atom.zero_angle_dependent_vars();
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

                if (computeOperator)
                {
                    atom.chi(t.i, k) += chi;
                    atom.chi(t.j, k) -= chi;
                    atom.U(t.j, k) += Uji(k);
                    atom.V(t.i, k) += Vij(k);
                    atom.V(t.j, k) += Vji(k);
                }
                chiTot(k) += chi;
                etaTot(k) += eta;
                atom.eta(k) += eta;
            }
        }
    }
    for (int a = 0; a < lteAtoms.size(); ++a)
    {
        auto& atom = *lteAtoms[a];
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
            }
        }
    }
}

f64 intensity_core(IntensityCoreData& data, int la, FsMode mode)
{
    JasUnpack(*data, atmos, spect, fd, background);
    JasUnpack(*data, activeAtoms, lteAtoms, JDag);
    JasUnpack(data, chiTot, etaTot, Uji, Vij, Vji);
    JasUnpack(data, I, S, Ieff, PsiStar);
    const int Nspace = atmos.Nspace;
    const int Nrays = atmos.Nrays;
    const int Nspect = spect.wavelength.shape(0);
    const bool updateJ = mode & FsMode::UpdateJ;
    const bool updateRates = mode & FsMode::UpdateRates;
    const bool prdRatesOnly = mode & FsMode::PrdOnly;
    const bool computeOperator = bool(PsiStar);

    JDag = spect.J(la);
    F64View J = spect.J(la);
    if (updateJ)
        J.fill(0.0);

    for (int a = 0; a < activeAtoms.size(); ++a)
        activeAtoms[a]->setup_wavelength(la);
    for (int a = 0; a < lteAtoms.size(); ++a)
        lteAtoms[a]->setup_wavelength(la);

    // NOTE(cmo): If we only have continua then opacity is angle independent
    const bool continuaOnly = continua_only(data, la);

    f64 dJMax = 0.0;
    for (int mu = 0; mu < Nrays; ++mu)
    {
        for (int toObsI = 0; toObsI < 2; toObsI += 1)
        {
            bool toObs = (bool)toObsI;
            if (!continuaOnly || (continuaOnly && (mu == 0 && toObsI == 0)))
            {
                chiTot.fill(0.0);
                etaTot.fill(0.0);

                gather_opacity_emissivity(&data, computeOperator, la, mu, toObs);

                for (int k = 0; k < Nspace; ++k)
                {
                    chiTot(k) += background.chi(la, k);
                    S(k) = (etaTot(k) + background.eta(la, k) + background.sca(la, k) * JDag(k)) / chiTot(k);
                }
            }

            piecewise_bezier3_1d(&fd, mu, toObs, spect.wavelength(la));
            // piecewise_linear_1d(&fd, mu, toObs, spect.wavelength(la));
            spect.I(la, mu) = I(0);

            if (updateJ)
            {
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
            }

            if (updateJ || computeOperator)
            {
                for (int a = 0; a < activeAtoms.size(); ++a)
                {
                    auto& atom = *activeAtoms[a];

                    if (computeOperator)
                    {
                        for (int k = 0; k < Nspace; ++k)
                        {
                            Ieff(k) = I(k) - PsiStar(k) * atom.eta(k);
                        }
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

                            if (computeOperator)
                            {
                                f64 integrand = (Uji(k) + Vji(k) * Ieff(k)) - (PsiStar(k) * atom.chi(t.i, k) * atom.U(t.j, k));
                                atom.Gamma(t.i, t.j, k) += integrand * wlamu;

                                integrand = (Vij(k) * Ieff(k)) - (PsiStar(k) * atom.chi(t.j, k) * atom.U(t.i, k));
                                atom.Gamma(t.j, t.i, k) += integrand * wlamu;
                            }

                            if ((updateRates && !prdRatesOnly)
                                || (prdRatesOnly && t.rhoPrd))
                            {
                                t.Rij(k) += I(k) * Vij(k) * wlamu;
                                t.Rji(k) += (Uji(k) + I(k) * Vij(k)) * wlamu;
                            }
                        }
                    }
                }
            }
            if (updateRates && !prdRatesOnly)
            {
                for (int a = 0; a < lteAtoms.size(); ++a)
                {
                    auto& atom = *lteAtoms[a];

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
                            t.Rij(k) += I(k) * Vij(k) * wlamu;
                            t.Rji(k) += (Uji(k) + I(k) * Vij(k)) * wlamu;
                        }
                    }
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

f64 formal_sol_gamma_matrices(Context& ctx)
{
    // feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
    JasUnpack(*ctx, atmos, spect, background);
    JasUnpack(ctx, activeAtoms, lteAtoms);

    const int Nspace = atmos.Nspace;
    const int Nrays = atmos.Nrays;
    const int Nspect = spect.wavelength.shape(0);

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
    JasPackPtr(iCore, atmos, spect, fd, background);
    JasPackPtr(iCore, activeAtoms, lteAtoms, JDag);
    JasPack(iCore, chiTot, etaTot, Uji, Vij, Vji);
    JasPack(iCore, I, S, Ieff, PsiStar);

    if (spect.JRest)
        spect.JRest.fill(0.0);

    for (auto& a : activeAtoms)
    {
        a->zero_rates();
    }
    for (auto& a : lteAtoms)
    {
        a->zero_rates();
    }

    f64 dJMax = 0.0;
    FsMode mode = (UpdateJ | UpdateRates);
    for (int la = 0; la < Nspect; ++la)
    {
        f64 dJ = intensity_core(iCore, la, mode);
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

f64 formal_sol_update_rates(Context& ctx)
{
    JasUnpack(*ctx, atmos, spect, background);
    JasUnpack(ctx, activeAtoms, lteAtoms);

    const int Nspace = atmos.Nspace;
    const int Nrays = atmos.Nrays;
    const int Nspect = spect.wavelength.shape(0);

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
    IntensityCoreData iCore;
    JasPackPtr(iCore, atmos, spect, fd, background);
    JasPackPtr(iCore, activeAtoms, lteAtoms, JDag);
    JasPack(iCore, chiTot, etaTot, Uji, Vij, Vji);
    JasPack(iCore, I, S, Ieff);

    if (spect.JRest)
        spect.JRest.fill(0.0);

    for (auto& a : activeAtoms)
    {
        a->zero_rates();
    }
    for (auto& a : lteAtoms)
    {
        a->zero_rates();
    }

    f64 dJMax = 0.0;
    FsMode mode = (UpdateJ | UpdateRates);
    for (int la = 0; la < Nspect; ++la)
    {
        f64 dJ = intensity_core(iCore, la, mode);
        dJMax = max(dJ, dJMax);
    }
    return dJMax;
}

f64 formal_sol_update_rates_fixed_J(Context& ctx)
{
    JasUnpack(*ctx, atmos, spect, background);
    JasUnpack(ctx, activeAtoms, lteAtoms);

    const int Nspace = atmos.Nspace;
    const int Nrays = atmos.Nrays;
    const int Nspect = spect.wavelength.shape(0);

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
    IntensityCoreData iCore;
    JasPackPtr(iCore, atmos, spect, fd, background);
    JasPackPtr(iCore, activeAtoms, lteAtoms, JDag);
    JasPack(iCore, chiTot, etaTot, Uji, Vij, Vji);
    JasPack(iCore, I, S, Ieff);

    for (auto& a : activeAtoms)
    {
        a->zero_rates();
    }
    for (auto& a : lteAtoms)
    {
        a->zero_rates();
    }

    f64 dJMax = 0.0;
    FsMode mode = (UpdateRates);
    for (int la = 0; la < Nspect; ++la)
    {
        f64 dJ = intensity_core(iCore, la, mode);
        dJMax = max(dJ, dJMax);
    }
    return dJMax;
}

f64 formal_sol(Context& ctx)
{
    JasUnpack(*ctx, atmos, spect, background);
    JasUnpack(ctx, activeAtoms, lteAtoms);

    const int Nspace = atmos.Nspace;
    const int Nrays = atmos.Nrays;
    const int Nspect = spect.wavelength.shape(0);

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
    IntensityCoreData iCore;
    JasPackPtr(iCore, atmos, spect, fd, background);
    JasPackPtr(iCore, activeAtoms, lteAtoms, JDag);
    JasPack(iCore, chiTot, etaTot, Uji, Vij, Vji);
    JasPack(iCore, I, S, Ieff);

    FsMode mode = FsMode::FsOnly;
    for (int la = 0; la < Nspect; ++la)
    {
        intensity_core(iCore, la, mode);
    }
    return 0.0;
}