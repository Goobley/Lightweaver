#include "Lightweaver.hpp"
#include "Bezier.hpp"
#include "JasPP.hpp"
#include <string.h>
#include <exception>

using namespace LwInternal;

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

    if (!(bool)atmos.B)
        throw std::runtime_error("Must provide magnetic field when computing polarised profiles");
    if (!(bool)atmos.cosGamma)
        throw std::runtime_error("Must call Atmosphere::update_projections before computing polarised profiles");
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

void stokes_K(int k, const F64View2D& chi, f64 chiI, F64View2D& K)
{
    K.fill(0.0);
    K(0, 1) = chi(1, k);
    K(0, 2) = chi(2, k);
    K(0, 3) = chi(3, k);

    K(1, 2) = chi(6, k);
    K(1, 3) = chi(5, k);
    K(2, 3) = chi(4, k);

    for (int j = 0; j < 3; ++j)
    {
        for (int i = j + 1; i < 4; ++i)
        {
            K(j, i) /= chiI;
            K(i, j) = K(j, i);
        }
    }

    K(1, 3) *= -1.0;
    K(2, 1) *= -1.0;
    K(3, 2) *= -1.0;
}

inline void prod(const F64View2D& a, const F64View2D& b, F64View2D& c)
{
    c.fill(0.0);

    for (int j = 0; j < 4; ++j)
        for (int i = 0; i < 4; ++i)
            for (int k = 0; k < 4; ++k)
                c(j, i) += a(k, i) * b(j, k);
}

inline void prod(const F64View2D& a, const F64View& b, F64View& c)
{
    c.fill(0.0);

    for (int i = 0; i < 4; ++i)
        for (int k = 0; k < 4; ++k)
            c(i) += a(i, k) * b(k);
}

#define StackStoredView2D(name, dim0, dim1) f64 name ## Storage[dim0*dim1]; auto name = F64View2D(name ## Storage, dim0, dim1);
#define StackStoredView(name, dim0) f64 name ## Storage[dim0]; auto name = F64View(name ## Storage, dim0);

void piecewise_stokes_bezier3_1d_impl(FormalDataStokes* fd, f64 zmu, bool toObs,
                                      f64 Istart[4], bool polarisedFrequency)
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

    auto slice_s4 = [&S](int k, F64View& slice) {
        for (int i = 0; i < 4; ++i)
        {
            slice(i) = S(i, k);
        }
    };

    // NOTE(cmo): Set up directions and loop bounds
    int dk = -1;
    int k_start = Ndep - 1;
    int k_end = 0;
    if (!toObs)
    {
        dk = 1;
        k_start = 0;
        k_end = Ndep - 1;
    }

    for (int n = 0; n < 4; ++n)
        I(n, k_start) = Istart[n];

    // NOTE(cmo): Set up values and storage at initial point (one from first end)
    int k = k_start + dk;
    f64 ds_uw = abs(height(k) - height(k - dk)) * zmu;
    f64 ds_dw = abs(height(k + dk) - height(k)) * zmu;
    f64 dx_uw = (chi(0, k) - chi(0, k - dk)) / ds_uw;
    f64 dx_c = Bezier::cent_deriv(ds_uw, ds_dw, chi(0, k - dk), chi(0, k), chi(0, k + dk));
    f64 c1 = Bezier::limit_control_point(chi(0, k) - (ds_uw / 3.0) * dx_c);
    f64 c2 = Bezier::limit_control_point(chi(0, k - dk) + (ds_uw / 3.0) * dx_uw);
    f64 dtau_uw = ds_uw * (chi(0, k) + chi(0, k - dk) + c1 + c2) * 0.25;

    StackStoredView2D(Ku, 4, 4);
    StackStoredView2D(dKu, 4, 4);
    StackStoredView2D(K0, 4, 4);
    StackStoredView2D(dK0, 4, 4);
    StackStoredView(Su, 4);
    StackStoredView(dSu, 4);
    StackStoredView(S0, 4);
    StackStoredView(dS0, 4);
    stokes_K(k_start, chi, chi(0, k_start), Ku);
    stokes_K(k, chi, chi(0, k), K0);
    slice_s4(k_start, Su);
    slice_s4(k, S0);

    for (int n = 0; n < 4; ++n)
    {
        dSu(n) = (S0(n) - Su(n)) / dtau_uw;
        for (int m = 0; m < 4; ++m)
            dKu(n, m) = (K0(n, m) - Ku(n, m)) / dtau_uw;
    }

    f64 ds_dw2 = 0.0;
    f64 dtau_dw = 0.0;
    f64 dx_dw = 0.0;
    auto dx_downwind = [&ds_dw, &ds_dw2, &chi, &k, dk] () {
        return Bezier::cent_deriv(ds_dw, ds_dw2, chi(0, k), chi(0, k + dk), chi(0, k + 2 * dk));
    };

    StackStoredView2D(Kd, 4, 4);
    StackStoredView2D(K02, 4, 4);
    StackStoredView2D(Ku2, 4, 4);
    StackStoredView2D(Ma, 4, 4);
    StackStoredView2D(Mb, 4, 4);
    StackStoredView2D(Mc, 4, 4);
    StackStoredView2D(Md, 4, 4);
    StackStoredView(V0, 4);
    StackStoredView(Sd, 4);
    for (; k != k_end + dk; k += dk)
    {
        if (k == k_end)
        {
            // Assume linear on the end, so drop dw point stuff
            for (int n = 0; n < 4; ++n)
            {
                dS0(n) = (S0(n) - Su(n)) / dtau_uw;
                for (int m = 0; m < 4; ++m)
                    dK0(n, m) = (K0(n, m) - Ku(n, m)) / dtau_uw;
            }
        }
        else
        {
            if (k_end - k == dk)
            {
                dx_dw = (chi(0, k + dk) - chi(0, k)) / ds_dw;
            }
            else
            {
                ds_dw2 = abs(height(k + 2 * dk) - height(k + dk)) * zmu;
                dx_dw = dx_downwind();
            }
            c1 = Bezier::limit_control_point(chi(0, k) + (ds_dw / 3.0) * dx_c);
            c2 = Bezier::limit_control_point(chi(0, k+dk) - (ds_dw / 3.0) * dx_dw);
            dtau_dw = ds_dw * (chi(0, k) + chi(0, k + dk) + c1 + c2) * 0.25;

            stokes_K(k + dk, chi, chi(0, k + dk), Kd);
            slice_s4(k + dk, Sd);

            Bezier::cent_deriv(dK0, dtau_uw, dtau_dw, Ku, K0, Kd);
            Bezier::cent_deriv(dS0, dtau_uw, dtau_dw, Su, S0, Sd);
        }


        prod(Ku, Ku, Ku2); // Ku2 = Ku @ Ku
        prod(K0, K0, K02); // K02 = K0 @ K0

        f64 alpha, beta, gamma, delta, edt;
        Bezier::Bezier3_coeffs(dtau_uw, &alpha, &beta, &gamma, &delta, &edt);

        for (int j = 0; j < 4; ++j)
        {
            for (int i = 0; i < 4; ++i)
            {
                // Defined in little memo Jaime sent
                f64 d = dtau_uw / 3.0 * (Ku2(j, i) + Ku(j, i) - dKu(j, i)) - Ku(j, i);
                f64 e = dtau_uw / 3.0 * (K02(j, i) + K0(j, i) - dK0(j, i)) + K0(j, i);
                // A in paper (LHS of system)
                Md(j, i) = id[j][i] + beta * K0(j, i) + delta * e;

                // Terms to be multiplied by I(:,k-dk)
                Ma(j, i) = edt * id[j][i] - alpha * Ku(j, i) + gamma * d;

                // Terms to be multiplied by S(:,k-dk)
                Mb(j, i) = alpha * id[j][i] + gamma * (id[j][i] - (dtau_uw / 3.0) * Ku(j, i));

                // Terms to be multiplied by S(:,k)
                Mc(j, i) = beta * id[j][i] + delta * (id[j][i] + (dtau_uw / 3.0) * K0(j, i));
            }
        }

        // Build complete RHS
        for (int i = 0; i < 4; ++i)
        {
            V0(i) = 0.0;
            for (int j = 0; j < 4; ++j)
                V0(i) += Ma(i, j) * I(j, k - dk) + Mb(i, j) * Su(j) + Mc(i, j) * S0(j);

            // Extra terms that just chill on the end of the RHS
            V0(i) += (dtau_uw / 3.0) * (gamma * dSu(i) - delta * dS0(i));
        }

        solve_lin_eq(Md, V0);

        for (int i = 0; i < 4; ++i)
            I(i, k) = V0(i);

        // NOTE(cmo): Shuffle everything along to avoid recomputing things
        memcpy(SuStorage, S0Storage, 4 * sizeof(f64));
        memcpy(S0Storage, SdStorage, 4 * sizeof(f64));
        memcpy(dSuStorage, dS0Storage, 4 * sizeof(f64));

        memcpy(KuStorage, K0Storage, 16 * sizeof(f64));
        memcpy(K0Storage, KdStorage, 16 * sizeof(f64));
        memcpy(dKuStorage, dK0Storage, 16 * sizeof(f64));

        dtau_uw = dtau_dw;
        ds_uw = ds_dw;
        ds_dw = ds_dw2;
        dx_uw = dx_c;
        dx_c = dx_dw;
    }
}

namespace LwInternal
{
void piecewise_stokes_bezier3_1d(FormalDataStokes* fd, int la, int mu, bool toObs,
                                 const F64View1D& wave, bool polarisedFrequency)
{
    const f64 wav = wave(la);
    if (!polarisedFrequency)
    {
        piecewise_bezier3_1d(&fd->fdIntens, la, mu, toObs, wave);
        return;
    }

    JasUnpack((*fd), atmos, chi);
    f64 zmu = 1.0 / atmos->muz(mu);
    auto height = atmos->height;

    int dk = -1;
    int kStart = atmos->Nspace - 1;
    if (!toObs)
    {
        dk = 1;
        kStart = 0;
    }
    f64 dtau_uw = 0.5 * zmu * (chi(0, kStart) + chi(0, kStart + dk)) * abs(height(kStart) - height(kStart + dk));

    f64 Iupw[4] = { 0.0, 0.0, 0.0, 0.0 };
    if (toObs)
    {
        if (atmos->zLowerBc.type == THERMALISED)
        {
            f64 Bnu[2];
            int Nspace = atmos->Nspace;
            planck_nu(2, &atmos->temperature(Nspace - 2), wav, Bnu);
            Iupw[0] = Bnu[1] - (Bnu[0] - Bnu[1]) / dtau_uw;
        }
        else if (atmos->zLowerBc.type == CALLABLE)
        {
            int muIdx = atmos->zLowerBc.idxs(mu, int(toObs));
            if (muIdx == -1)
            {
                // NOTE(cmo): This shouldn't be possible, so I won't try to
                // recover.
                printf("Error in boundary condition indexing\n");
                assert(false);
            }
            Iupw[0] = atmos->zLowerBc.bcData(la, muIdx, 0);
        }
    }
    else
    {
        if (atmos->zUpperBc.type == THERMALISED)
        {
            f64 Bnu[2];
            planck_nu(2, &atmos->temperature(0), wav, Bnu);
            Iupw[0] = Bnu[0] - (Bnu[1] - Bnu[0]) / dtau_uw;
        }
        else if (atmos->zUpperBc.type == CALLABLE)
        {
            int muIdx = atmos->zUpperBc.idxs(mu, int(toObs));
            if (muIdx == -1)
            {
                // NOTE(cmo): This shouldn't be possible, so I won't try to
                // recover.
                printf("Error in boundary condition indexing\n");
                assert(false);
            }
            Iupw[0] = atmos->zUpperBc.bcData(la, muIdx, 0);
        }
    }

    piecewise_stokes_bezier3_1d_impl(fd, zmu, toObs, Iupw, polarisedFrequency);
}
}

namespace GammaFsCores
{
f64 stokes_fs_core(StokesCoreData& data, int la, bool updateJ, bool upOnly)
{
    JasUnpack(*data, atmos, spect, fd, background);
    JasUnpack(*data, activeAtoms, detailedAtoms, JDag, J20Dag);
    JasUnpack(data, chiTot, etaTot, Uji, Vij, Vji, I, S);
    JasUnpack(data, J20);

    const int Nspace = atmos.Nspace;
    const int Nrays = atmos.Nrays;
    const f64 inv2root2 = 1.0 / (2.0 * sqrt(2.0));
    F64View J = spect.J(la);
    if (updateJ)
    {
        JDag = spect.J(la);
        J.fill(0.0);
        if (J20)
        {
            J20Dag = J20(la);
            J20(la).fill(0.0);
        }
    }

    for (int a = 0; a < activeAtoms.size(); ++a)
        activeAtoms[a]->setup_wavelength(la);
    for (int a = 0; a < detailedAtoms.size(); ++a)
        detailedAtoms[a]->setup_wavelength(la);

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
    for (int a = 0; a < detailedAtoms.size(); ++a)
    {
        auto& atom = *detailedAtoms[a];
        for (int kr = 0; kr < atom.Ntrans; ++kr)
        {
            auto& t = *atom.trans[kr];
            if (!t.active(la))
                continue;
            continuaOnly = continuaOnly && (t.type == CONTINUUM);
        }
    }
    // NOTE(cmo): Need full integration if we're doing J20.
    if (J20)
        continuaOnly = false;

    int toObsStart = 0;
    int toObsEnd = 2;
    if (upOnly)
        toObsStart = 1;

    f64 dJMax = 0.0;
    for (int mu = 0; mu < Nrays; ++mu)
    {
        // NOTE(cmo): Whilst these don't directly match LL04, they do nicely
        // match Trujillo Bueno 2001
        // See also Fluri & Stenflo 1999, Del Pino Aléman et al 2014.
        const f64 mu2 = square(atmos.muz(mu));
        const f64 wJ20_I = inv2root2 * (3.0 * mu2 - 1.0);
        const f64 wJ20_Q = inv2root2 * 3.0 * (mu2 - 1.0); // -3/(2sqrt2) sin^2\theta
        for (int toObsI = toObsStart; toObsI < toObsEnd; toObsI += 1)
        {
            bool toObs = (bool)toObsI;
            bool polarisedFrequency = J20 || false;
            if (!continuaOnly || (continuaOnly && (mu == 0 && toObsI == toObsStart)))
            {
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

                            chiTot(0, k) += chi;
                            etaTot(0, k) += eta;

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
                        }
                    }
                }
                for (int a = 0; a < detailedAtoms.size(); ++a)
                {
                    auto& atom = *detailedAtoms[a];
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
                        }
                    }
                }

                if (J20)
                {
                    F64View sca = background.sca(la);
                    for (int k = 0; k < Nspace; ++k)
                    {
                        etaTot(0, k) += wJ20_I * sca(k) * J20Dag(k);
                        etaTot(1, k) += wJ20_Q * sca(k) * J20Dag(k);
                    }
                }

                F64View chiBg = background.chi(la);
                F64View etaBg = background.eta(la);
                F64View sca = background.sca(la);
                for (int k = 0; k < Nspace; ++k)
                {
                    chiTot(0, k) += chiBg(k);
                    S(0, k) = (etaTot(0, k) + etaBg(k) + sca(k) * JDag(k)) / chiTot(0, k);
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
            switch (atmos.Ndim)
            {
                case 1:
                {
                    piecewise_stokes_bezier3_1d(&fd, la, mu, toObs,
                                                spect.wavelength,
                                                polarisedFrequency);
                    spect.I(la, mu, 0) = I(0, 0);
                    spect.Quv(0, la, mu, 0) = I(1, 0);
                    spect.Quv(1, la, mu, 0) = I(2, 0);
                    spect.Quv(2, la, mu, 0) = I(3, 0);
                } break;

                default:
                {
                    printf("Unexpected Ndim %d\n", atmos.Ndim);
                } break;
            }
#else
            // NOTE(cmo): Checking with the normal FS and just using the first row of ezach of the matrices does indeed
            // produce the correct result
            piecewise_bezier3_1d(&fd.fdIntens, mu, toObs, spect.wavelength(la));
            spect.I(la, mu) = I(0, 0);
            spect.Quv(0, la, mu) = 0.0;
            spect.Quv(1, la, mu) = 0.0;
            spect.Quv(2, la, mu) = 0.0;
#endif

            // TODO(cmo): Rates?
            if (updateJ)
            {
                const f64 wmu = atmos.wmu(mu);
                for (int k = 0; k < Nspace; ++k)
                {
                    J(k) += 0.5 * wmu * I(0, k);
                }
                if (J20)
                {
                    const f64 wmuJ20_I = wJ20_I * wmu;
                    const f64 wmuJ20_Q = wJ20_Q * wmu;
                    for (int k = 0; k < Nspace; ++k)
                        J20(la, k) += wmuJ20_I * I(0, k) + wmuJ20_Q * I(1, k);
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

IterationResult formal_sol_full_stokes_impl(Context& ctx, bool updateJ, bool upOnly,
                                            ExtraParams params)
{
    JasUnpack(*ctx, atmos, spect, background);
    JasUnpack(ctx, activeAtoms, detailedAtoms);

    if (!atmos.B)
        throw std::runtime_error("Magnetic field required");

    const int Nspace = atmos.Nspace;
    const int Nspect = spect.wavelength.shape(0);

    F64View2D J20;
    F64Arr J20Dag;
    if (params.contains("J20"))
    {
        J20 = params.get_as<F64View2D>("J20");
        J20Dag = F64Arr(Nspace);
    }

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
    fd.fdIntens.interp = ctx.interpFn.interp_2d;
    StokesCoreData core;
    JasPackPtr(core, atmos, spect, fd, background);
    JasPackPtr(core, activeAtoms, detailedAtoms, JDag, J20Dag);
    JasPack(core, chiTot, etaTot, Uji, Vij, Vji, I, S);
    JasPack(core, J20);

    f64 dJMax = 0.0;
    int maxIdx = 0;
    for (int la = 0; la < Nspect; ++la)
    {
        f64 dJ = GammaFsCores::stokes_fs_core(core, la, updateJ, upOnly);
        dJMax = max_idx(dJ, dJMax, maxIdx, la);
    }
    IterationResult result{};
    result.updatedJ = updateJ;
    if (updateJ)
    {
        result.dJMax = dJMax;
        result.dJMaxIdx = maxIdx;
    }
    return result;
}

IterationResult formal_sol_full_stokes(Context& ctx, bool updateJ, bool upOnly,
                                       ExtraParams params)
{
    return ctx.iterFns.full_stokes_fs(ctx, updateJ, upOnly, params);
}