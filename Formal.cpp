#include "JasPP.hpp"
#include "Formal.hpp"
#include "Faddeeva.hh"

#include <cmath>
#include <vector>
#include <fenv.h>
#include <iostream>

// Public domain polyfill for feenableexcept on OS X
// http://www-personal.umich.edu/~williams/archive/computation/fe-handling-example.c


inline int feenableexcept(unsigned int excepts)
{
    static fenv_t fenv;
    unsigned int new_excepts = excepts & FE_ALL_EXCEPT;
    // previous masks
    unsigned int old_excepts;

    if (fegetenv(&fenv)) {
        return -1;
    }
    old_excepts = fenv.__control & FE_ALL_EXCEPT;

    // unmask
    fenv.__control &= ~new_excepts;
    fenv.__mxcsr   &= ~(new_excepts << 7);

    return fesetenv(&fenv) ? -1 : old_excepts;
}

inline int fedisableexcept(unsigned int excepts)
{
    static fenv_t fenv;
    unsigned int new_excepts = excepts & FE_ALL_EXCEPT;
    // all previous masks
    unsigned int old_excepts;

    if (fegetenv(&fenv)) {
        return -1;
    }
    old_excepts = fenv.__control & FE_ALL_EXCEPT;

    // mask
    fenv.__control |= new_excepts;
    fenv.__mxcsr   |= new_excepts << 7;

    return fesetenv(&fenv) ? -1 : old_excepts;
}

void print_complex(std::complex<f64> cmp, WofZType wofz)
{
    using Faddeeva::w;
    std::cout << cmp << std::endl;
    std::cout << w(cmp) << std::endl;
}

void planck_nu(long Nspace, double *T, double lambda, double *Bnu)
{
    namespace C = Constants;
    constexpr f64 hc_k =  C::HC / (C::KBoltzmann * C::NM_TO_M);
    const f64 hc_kla = hc_k / lambda;
    constexpr f64 twoh_c2 = (2.0 * C::HC) / cube(C::NM_TO_M);
    const f64 twohnu3_c2 = twoh_c2 / cube(lambda);
    constexpr int MAX_EXPONENT = 150;

    for (int k = 0; k < Nspace; k++) {
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
    auto z = (v + a*1i);
    return w(z).real();
}

void Transition::compute_phi(const Atmosphere& atmos, F64View aDamp, F64View vBroad)
{
    namespace C = Constants;
    if (type == TransitionType::CONTINUUM)
        return;

    constexpr f64 sign[] = {-1.0, 1.0};

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

inline void w2(f64 dtau, f64 *w)
{
    f64 expdt;

    if (dtau < 5.0E-4) {
        w[0] = dtau * (1.0 - 0.5 * dtau);
        w[1] = square(dtau) * (0.5 - dtau / 3.0);
    } else if (dtau > 50.0) {
        w[1] = w[0] = 1.0;
    } else {
        expdt = exp(-dtau);
        w[0] = 1.0 - expdt;
        w[1] = w[0] - dtau * expdt;
    }
}

inline void w3(f64 dtau, f64 *w) {
    f64 expdt, delta;

    if (dtau < 5.0E-4) {
        w[0] = dtau * (1.0 - 0.5 * dtau);
        delta = square(dtau);
        w[1] = delta * (0.5 - dtau / 3.0);
        delta *= dtau;
        w[2] = delta * (1.0 / 3.0 - 0.25 * dtau);
    } else if (dtau > 50.0) {
        w[1] = w[0] = 1.0;
        w[2] = 2.0;
    } else {
        expdt = exp(-dtau);
        w[0] = 1.0 - expdt;
        w[1] = w[0] - dtau * expdt;
        w[2] = 2.0 * w[1] - square(dtau) * expdt;
    }
}

void piecewise_linear_1d_impl(f64 zmu, bool toObs, f64 Istart, 
                              F64View height, F64View chi, F64View S, 
                              F64View I, F64View Psi)
{
    const int Ndep = chi.shape(0);
    bool computeOperator = Psi.shape(0) != 0;

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
    f64 dtau_uw = zmu * (chi(k_start) + chi(k_start + dk)) 
                        * abs(height(k_start) - height(k_start + dk));
    f64 dS_uw = (S(k_start) - S(k_start + dk)) / dtau_uw;

    /* --- Boundary conditions --                        -------------- */
    f64 I_upw = Istart;

    I(k_start) = I_upw;
    if (computeOperator)
        Psi(k_start) = 0.0;

    /* --- Solve transfer along ray --                   -------------- */

    f64 w[2];
    for (int k = k_start + dk; k != k_end; k += dk) {
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

void piecewise_linear_1d(Atmosphere* atmos, int mu, bool toObs, f64 wav, 
                         F64View chi, F64View S, F64View I, F64View Psi)
{
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
    f64 dtau_uw = zmu * (chi(kStart) + chi(kStart + dk)) 
                        * abs(height(kStart) - height(kStart + dk));

    f64 Iupw = 0.0;
    if (toObs && atmos->lowerBc == THERMALISED)
    {
        f64 Bnu[2];
        int Nspace = atmos->Nspace;
        planck_nu(2, &atmos->temperature(Nspace-2), wav, Bnu);
        Iupw = Bnu[1] - (Bnu[0] - Bnu[1]) / dtau_uw;
    }
    else if (!toObs && atmos->upperBc == THERMALISED)
    {
        f64 Bnu[2];
        planck_nu(2, &atmos->temperature(0), wav, Bnu);
        Iupw = Bnu[0] - (Bnu[1] - Bnu[0]) / dtau_uw;
    }

    piecewise_linear_1d_impl(zmu, toObs, Iupw, height, chi, S, I, Psi);
}


f64 gamma_matrices_formal_sol(Context ctx)
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

    printf("%d, %d, %d\n", Nspace, Nrays, Nspect);

    f64 dJMax = 0.0;
    for (int la = 0; la < Nspect; ++la)
    {
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
                // const f64 sign = If toObs Then 1.0 Else -1.0 End;
                chiTot.fill(0.0);
                etaTot.fill(0.0);

                for (int a = 0; a < activeAtoms.size(); ++a)
                {
                    auto& atom = *activeAtoms[a];
                    atom.zero_angle_dependent_vars();
                    for (int kr = 0; kr < atom.Ntrans; ++kr)
                    {
                        auto& t = *atom.trans[kr];
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
                    S(k) = (etaTot(k) + background.eta(la, k)) / chiTot(k);
                }

                piecewise_linear_1d(&atmos, mu, toObs, spect.wavelength(la), chiTot, S, I, PsiStar);
                spect.I(la, mu) = I(0);

                for (int k = 0; k < Nspace; ++k)
                {
                    J(k) += 0.5 * atmos.wmu(mu) * I(k);
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

                            f64 integrand = (Uji(k) + Vji(k) * Ieff(k)) 
                                    - (PsiStar(k) * atom.chi(t.i, k) * atom.U(t.j, k));
                            atom.Gamma(t.i, t.j, k) += integrand * wlamu;

                            integrand = (Vij(k) * Ieff(k)) 
                                      - (PsiStar(k) * atom.chi(t.j, k) * atom.U(t.i, k));
                            atom.Gamma(t.j, t.i, k) += integrand * wlamu;
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
    for (int a = 0; a < activeAtoms.size(); ++a)
    {
        auto& atom = *activeAtoms[a];
        for (int k = 0; k < Nspace; ++k)
        {
            for (int i = 0; i < atom.Nlevel; ++i)
            {
                atom.Gamma(i,i,k) = 0.0;
                f64 gammaDiag = 0.0;
                for (int j = 0; j < atom.Nlevel; ++j)
                {
                    gammaDiag += atom.Gamma(j,i,k);
                }
                atom.Gamma(i,i,k) = -gammaDiag;
            }
        }
    }
    return dJMax;
}

void stat_eq(Atom* atomIn, DgesvType solve)
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
            atom.n(i,k) = nk(i);
    }
}