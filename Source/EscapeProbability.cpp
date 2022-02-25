#include "Lightweaver.hpp"

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
        const f64 vk = (vBase + atmos.vz(k)) / vBroad(k);
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
        if (t.type == TransitionType::LINE)
        {
            lt = t.wavelength.shape(0) / 2;
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