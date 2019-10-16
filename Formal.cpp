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

void piecewise_bezier3_1d_impl(f64 zmu, bool toObs, f64 Istart,
                               F64View height, F64View chi,
                               F64View S, F64View I, F64View Psi)
{
    // TODO(cmo)

}

void piecewise_bezier3_1d(Atmosphere* atmos, int mu, bool toObs, f64 wav, 
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

    piecewise_bezier3_1d_impl(zmu, toObs, Iupw, height, chi, S, I, Psi);
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

f64 escape_formal_sol(const Atmosphere& atmos, f64 lambda, F64View chi, F64View chiB, F64View S, F64View P, F64View Q, F64View Lambda, bool line)
{
    namespace C = Constants;
    // NOTE(cmo): This is a Feautrier style method, i.e. P = I+ + I-, Q = I+ - I-
    F64Arr tau(atmos.Nspace);
    F64Arr tauB(atmos.Nspace);

    tau(0) = 0.0;
    tauB(0) = 0.0;
    for (int k = 1; k < atmos.Nspace; ++k)
    {
        f64 zz = abs(atmos.height(k-1) - atmos.height(k+1)) * 0.5;
        tauB(k) += tauB(k-1) + chiB(k) * zz;
        tau(k) += tau(k-1) + chi(k) * zz + tauB(k);
    }
    tau(0) = 0.5*tau(1);
    tauB(0) = 0.5*tauB(1);
    tau(atmos.Nspace-1) = 0.5*tau(atmos.Nspace-2);
    tauB(atmos.Nspace-1) = 0.5*tauB(atmos.Nspace-2);

    P(atmos.Nspace-1) = S(atmos.Nspace-1);
    Q(atmos.Nspace-1) = 0.0;
    Lambda(atmos.Nspace-1) = 1.0;

    f64 sum = 0.0;
    for (int k = atmos.Nspace-2; k > 1; --k)
    {
        f64 t = tau(k);
        f64 tb = tauB(k);

        f64 alpha = C::HC / C::KBoltzmann / lambda / atmos.temperature(k);
        f64 dp;
        f64 ep = escape_probability(line, t, tb, alpha, &dp);

        Lambda(k) = 1.0 - 2.0 * ep;
        f64 dx = 0.5 * log((tau(k+1) + tauB(k+1)) / (tau(k-1) + tauB(k-1)));
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

void gamma_matrices_escape_prob(Atom* a, Background& background, 
                                const Atmosphere& atmos)
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
                f64 x = atom.n(t.i,k) * Vij(k) - atom.n(t.j,k) * Vji(k);
                chi(k) = x;
                f64 n = atom.n(t.j,k) * Uji(k);
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

                if (t.wavelength(ltc) - prevWl < 10.0 
                    && ltc != Nlambda-1)
                    continue;

                prevWl = t.wavelength(ltc);
                for (int k = 0; k < atmos.Nspace; ++k)
                {
                    uv_mu_1(atom, t, lt, phi, Uji, Vij, Vji);
                    f64 x = atom.n(t.i,k) * Vij(k) - atom.n(t.j,k) * Vji(k);
                    chi(k) = x;
                    f64 n = atom.n(t.j,k) * Uji(k);
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
                    f64 integrand = (Uji(k) + Vji(k) * Ieff) 
                            - (Lambda(k) * Uji(k));
                    atom.Gamma(t.i, t.j, k) += integrand * wlaSum;

                    integrand = (Vij(k) * Ieff) 
                                - (Lambda(k) * Uji(k));
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
            atom.Gamma(i,i,k) = 0.0;
            f64 gammaDiag = 0.0;
            for (int j = 0; j < atom.Nlevel; ++j)
            {
                gammaDiag += atom.Gamma(j,i,k);
            }
            atom.Gamma(i,i,k) = -gammaDiag;
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
            atom.n(i,k) = nk(i);
    }
}