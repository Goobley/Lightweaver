#include "LwInternal.hpp"
#include "Lightweaver.hpp"
#include "Bezier.hpp"
#include "JasPP.hpp"
#include "Simd.hpp"
#include "ThreadStorage.hpp"
#include "TaskSetWrapper.hpp"

#include <cmath>
#include <iostream>
#include <utility>
#include <vector>
#include <algorithm>
#include <chrono>

#include "SimdFullIterationTemplates.hpp"

#ifdef CMO_BASIC_PROFILE
using hrc = std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;
#endif

using namespace LwInternal;
using LwInternal::FormalData;
using LwInternal::IntensityCoreData;

void Transition::compute_phi_la(const Atmosphere& atmos, const F64View& aDamp,
                                const F64View& vBroad, int lt)
{
    namespace C = Constants;

    constexpr f64 sign[] = { -1.0, 1.0 };
    // Why is there still no constexpr math in std? :'(
    const f64 sqrtPi = sqrt(C::Pi);

    const f64 vBase = (wavelength(lt) - lambda0) * C::CLight / lambda0;
    for (int mu = 0; mu < phi.shape(1); ++mu)
    {
        for (int toObs = 0; toObs < 2; ++toObs)
        {
            const f64 s = sign[toObs];
            for (int k = 0; k < atmos.Nspace; ++k)
            {
                const f64 vk = (vBase + s * atmos.vlosMu(mu, k)) / vBroad(k);
                const f64 p = voigt_H(aDamp(k), vk) / (sqrtPi * vBroad(k));
                phi(lt, mu, toObs, k) = p;
            }
        }
    }
}

void Transition::compute_phi(const Atmosphere& atmos, F64View aDamp, F64View vBroad)
{
    if (type == TransitionType::CONTINUUM)
        return;

    if (bound_parallel_compute_phi)
    {
        bound_parallel_compute_phi(atmos, aDamp, vBroad);
        return;
    }

    for (int la = 0; la < wavelength.shape(0); ++la)
    {
        compute_phi_la(atmos, aDamp, vBroad, la);
    }
}

void Transition::compute_phi_parallel(LwInternal::ThreadData* threading, const Atmosphere& atmos,
                                      F64View aDamp, F64View vBroad)
{
    if (type == TransitionType::CONTINUUM)
        return;

    struct LineProfileData
    {
        Transition* t;
        const Atmosphere* atmos;
        F64View* aDamp;
        F64View* vBroad;
    };

    LineProfileData data;
    data.t = this;
    data.atmos = &atmos;
    data.aDamp = &aDamp;
    data.vBroad = &vBroad;
    auto compute_profile = [](void* data, enki::TaskScheduler* s,
                               enki::TaskSetPartition p, u32 threadId)
    {
        LineProfileData* d = (LineProfileData*)data;
        for (i64 la = p.start; la < p.end; ++la)
            d->t->compute_phi_la(*(d->atmos), *(d->aDamp), *(d->vBroad), la);
    };

    enki::TaskScheduler* sched = &threading->sched;
    {
        LwTaskSet lineProfile((void*)&data, sched, wavelength.shape(0),
                              1, compute_profile);
        sched->AddTaskSetToPipe(&lineProfile);
        sched->WaitforTask(&lineProfile);
    }
}

void Transition::compute_wphi(const Atmosphere& atmos)
{
    namespace C = Constants;
    if (type == TransitionType::CONTINUUM)
        return;

    wphi.fill(0.0);
    for (int la = 0; la < wavelength.shape(0); ++la)
    {
        const f64 wla = wlambda(la);
        for (int mu = 0; mu < phi.shape(1); ++mu)
        {
            const f64 wlamu = wla * 0.5 * atmos.wmu(mu);
            for (int toObs = 0; toObs < 2; ++toObs)
            {
                for (int k = 0; k < atmos.Nspace; ++k)
                {
                    const f64 p = phi(la, mu, toObs, k);
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
    // NOTE(cmo): This is the Auer & Paletou (1994) method, direction is due to
    // integral along tau

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
    f64 rcpDtau_uw = 1.0 / dtau_uw;
    f64 dS_uw = (S(k_start) - S(k_start + dk)) * rcpDtau_uw;

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
        f64 rcpDtau_dw = 1.0 / dtau_dw;
        f64 dS_dw = (S(k) - S(k + dk)) * rcpDtau_dw;

        I(k) = (1.0 - w[0]) * I_upw + w[0] * S(k) + w[1] * dS_uw;

        if (computeOperator)
            Psi(k) = w[0] - w[1] * rcpDtau_uw;

        /* --- Re-use downwind quantities for next upwind position -- --- */
        I_upw = I(k);
        dS_uw = dS_dw;
        dtau_uw = dtau_dw;
        rcpDtau_uw = rcpDtau_dw;
    }

    /* --- Piecewise linear integration at end of ray -- ---------- */
    w2(dtau_uw, w);
    I(k_end) = (1.0 - w[0]) * I_upw + w[0] * S(k_end) + w[1] * dS_uw;
    if (computeOperator)
    {
        Psi(k_end) = w[0] - w[1] * rcpDtau_uw;
        for (int k = 0; k < Psi.shape(0); ++k)
            Psi(k) /= chi(k);
    }
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

    f64 Cuw = Bezier::limit_control_point(chi(k - dk) + (ds_uw / 3.0) * dx_uw);
    f64 C0 = Bezier::limit_control_point(chi(k) - (ds_uw / 3.0) * dx_c);

    // NOTE(cmo): Average chi over the uw-0 interval -- the Bezier3 integral
    f64 dtau_uw = ds_uw * (chi(k) + chi(k - dk) + Cuw + C0) * 0.25;
    f64 dS_uw = (S(k) - S(k - dk)) / dtau_uw;

    f64 ds_dw2 = 0.0;
    auto dx_downwind = [&ds_dw, &ds_dw2, &chi, &k, dk] {
        return Bezier::cent_deriv(ds_dw, ds_dw2, chi(k), chi(k + dk), chi(k + 2 * dk));
    };
    f64 dtau_dw = 0.0;
    auto dS_central
        = [&dtau_uw, &dtau_dw, &S, &k, dk] () { return Bezier::cent_deriv(dtau_uw, dtau_dw, S(k - dk), S(k), S(k + dk)); };

    for (; k != k_end - dk; k += dk)
    {
        ds_dw2 = abs(height(k + 2 * dk) - height(k + dk)) * zmu;
        f64 dx_dw = dx_downwind();
        Cuw = Bezier::limit_control_point(chi(k) + (ds_dw / 3.0) * dx_c);
        C0 = Bezier::limit_control_point(chi(k + dk) - (ds_dw / 3.0) * dx_dw);
        dtau_dw = ds_dw * (chi(k) + chi(k + dk) + Cuw + C0) * 0.25;

        f64 alpha, beta, gamma, delta, edt;
        Bezier::Bezier3_coeffs(dtau_uw, &alpha, &beta, &gamma, &delta, &edt);

        f64 dS_c = dS_central();

        Cuw = Bezier::limit_control_point(S(k - dk) + (dtau_uw / 3.0) * dS_uw);
        C0 = Bezier::limit_control_point(S(k) - (dtau_uw / 3.0) * dS_c);

        I(k) = I_upw * edt + alpha * S(k - dk) + beta * S(k) + gamma * Cuw + delta * C0;
        if (computeOperator)
            Psi(k) = beta + delta;

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
    Cuw = Bezier::limit_control_point(chi(k) + (ds_dw / 3.0) * dx_c);
    C0 = Bezier::limit_control_point(chi(k + dk) - (ds_dw / 3.0) * dx_dw);
    // TODO(cmo): Use this quantity to compute the final point without falling back to w2? (Make derivatives constant?)
    dtau_dw = ds_dw * (chi(k) + chi(k + dk) + Cuw + C0) * 0.25;

    f64 alpha, beta, gamma, delta, edt;
    Bezier::Bezier3_coeffs(dtau_uw, &alpha, &beta, &gamma, &delta, &edt);

    f64 dS_c = dS_central();

    Cuw = Bezier::limit_control_point(S(k - dk) + dtau_uw / 3.0 * dS_uw);
    C0 = Bezier::limit_control_point(S(k) - dtau_uw / 3.0 * dS_c);

    I(k) = I_upw * edt + alpha * S(k - dk) + beta * S(k) + gamma * Cuw + delta * C0;
    if (computeOperator)
        Psi(k) = beta + delta;
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

f64 besser_control_point_1d(f64 hM, f64 hP, f64 yM, f64 yO, f64 yP)
{
    const f64 dM = (yO - yM) / hM;
    const f64 dP = (yP - yO) / hP;

    if (dM * dP <= 0.0)
        return yO;

    f64 yOp = (hM * dP + hP * dM) / (hM + hP);
    f64 cM = yO - 0.5 * hM * yOp;
    f64 cP = yO + 0.5 * hP * yOp;

    // NOTE(cmo): We know dM and dP have the same sign, so if dM is positive, M < O < P
    f64 minYMO = yM;
    f64 maxYMO = yO;
    f64 minYOP = yO;
    f64 maxYOP = yP;
    if (dM < 0.0)
    {
        minYMO = yO;
        maxYMO = yM;
        minYOP = yP;
        maxYOP = yO;
    }

    if (cM < minYMO || cM > maxYMO)
        return yM;

    if (cP < minYOP || cP > maxYOP)
    {
        cP = yP;
        yOp = (cP - yO) / (0.5 * hP);
        cM = yO - 0.5 * hM * yOp;
    }

    return cM;
}

struct BesserCoeffs1d
{
    f64 M;
    f64 O;
    f64 C;
    f64 edt;
};

BesserCoeffs1d besser_coeffs_1d(f64 t)
{
    if (t < 0.14)
    // if (t < 0.05)
    {
        f64 m = (t * (t * (t * (t * (t * (t * ((140.0 - 18.0 * t) * t - 945.0) + 5400.0) - 25200.0) + 90720.0) - 226800.0) + 302400.0)) / 907200.0;
        f64 o = (t * (t * (t * (t * (t * (t * ((10.0 - t) * t - 90.0) + 720.0) - 5040.0) + 30240.0) - 151200.0) + 604800.0)) / 1814400.0;
        f64 c = (t * (t * (t * (t * (t * (t * ((35.0 - 4.0 * t) * t - 270.0) + 1800.0) - 10080.0) + 45360.0) - 151200.0) + 302400.0)) / 907200.0;
        f64 edt = 1.0 - t + 0.5 * square(t) - cube(t) / 6.0 + t * cube(t) / 24.0 - square(t) * cube(t) / 120.0 + cube(t) * cube(t) / 720.0 - cube(t) * cube(t) * t / 5040.0;
        return BesserCoeffs1d{m, o, c, edt};
    }
    else
    {
        f64 t2 = square(t);
        f64 edt = exp(-t);
        f64 m = (2.0 - edt * (t2 + 2.0 * t + 2.0)) / t2;
        f64 o = 1.0 - 2.0 * (edt + t - 1.0) / t2;
        f64 c = 2.0 * (t - 2.0 + edt * (t + 2.0)) / t2;
        return BesserCoeffs1d{m, o, c, edt};
    }
}

void piecewise_besser_1d_impl(FormalData* fd, f64 zmu, bool toObs, f64 Istart)
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

    for (; k != k_end; k += dk)
    {
        f64 ds_uw = abs(height(k) - height(k - dk)) * zmu;
        f64 ds_dw = abs(height(k + dk) - height(k)) * zmu;

        f64 chi_uw = chi(k - dk);
        f64 chiLocal = chi(k);
        f64 chi_dw = chi(k + dk);
        f64 chiC = besser_control_point_1d(ds_uw, ds_dw, chi_uw, chiLocal, chi_dw);

        f64 dtauUw = (1.0 / 3.0) * (chi_uw + chiC + chiLocal) * ds_uw;
        f64 dtauDw = 0.5 * (chiLocal + chi_dw) * ds_dw;

        f64 Suw = S(k - dk);
        f64 SLocal = S(k);
        f64 Sdw = S(k + dk);
        f64 SC = besser_control_point_1d(dtauUw, dtauDw, Suw, SLocal, Sdw);

        auto coeffs = besser_coeffs_1d(dtauUw);
        I(k) = I_upw * coeffs.edt + coeffs.M * Suw + coeffs.O * SLocal + coeffs.C * SC;
        if (computeOperator)
            Psi(k) = coeffs.O + coeffs.C;

        I_upw = I(k);
    }
    // Piecewise linear on end
    k = k_end;
    f64 dtau_uw = 0.5 * zmu * (chi(k) + chi(k - dk)) * abs(height(k) - height(k - dk));
    // NOTE(cmo): See note in the linear formal solver if wondering why -w[1] is
    // used in I(k). Basically, the derivative (dS_uw) was taken in the other
    // direction there. In some ways this is nicer, as the operator and I take
    // the same form, but it doesn't really make any significant difference
    f64 dS_uw = (S(k) - S(k - dk)) / dtau_uw;
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
void piecewise_linear_1d(FormalData* fd, int la, int mu, bool toObs, const F64View1D& wave)
{
    const f64 wav = wave(la);
    JasUnpack((*fd), atmos, chi);
    f64 zmu = 0.5 / atmos->muz(mu);
    auto height = atmos->height;

    int dk = -1;
    int kStart = atmos->Nspace - 1;
    if (!toObs)
    {
        dk = 1;
        kStart = 0;
    }
    f64 dtau_uw = zmu * (chi(kStart) + chi(kStart + dk)) * abs(height(kStart) - height(kStart + dk));

    f64 Iupw = 0.0;
    if (toObs)
    {
        if (atmos->zLowerBc.type == THERMALISED)
        {
            f64 Bnu[2];
            int Nspace = atmos->Nspace;
            planck_nu(2, &atmos->temperature(Nspace - 2), wav, Bnu);
            Iupw = Bnu[1] - (Bnu[0] - Bnu[1]) / dtau_uw;
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
            Iupw = atmos->zLowerBc.bcData(la, muIdx, 0);
        }
    }
    else
    {
        if (atmos->zUpperBc.type == THERMALISED)
        {
            f64 Bnu[2];
            planck_nu(2, &atmos->temperature(0), wav, Bnu);
            Iupw = Bnu[0] - (Bnu[1] - Bnu[0]) / dtau_uw;
        }
        else if (atmos->zUpperBc.type == CALLABLE)
        {
            int muIdx = atmos->zUpperBc.idxs(mu, int(toObs));
            if (muIdx == -1)
            {
                printf("Error in boundary condition indexing\n");
                assert(false);
                Iupw = 0.0;
            }
            else
                Iupw = atmos->zUpperBc.bcData(la, muIdx, 0);
        }
    }

    piecewise_linear_1d_impl(fd, zmu, toObs, Iupw);
}

void piecewise_bezier3_1d(FormalData* fd, int la, int mu, bool toObs, const F64View1D& wave)
{
    const f64 wav = wave(la);
    JasUnpack((*fd), atmos, chi);
    // This is 1.0 here, as we are normally effectively rolling in the averaging
    // factor for dtau, whereas it's explicit in this solver
    f64 zmu = 1.0 / atmos->muz(mu);
    auto height = atmos->height;

    int dk = -1;
    int kStart = atmos->Nspace - 1;
    if (!toObs)
    {
        dk = 1;
        kStart = 0;
    }
    f64 dtau_uw = 0.5 * zmu * (chi(kStart) + chi(kStart + dk)) * abs(height(kStart) - height(kStart + dk));

    f64 Iupw = 0.0;
    if (toObs)
    {
        if (atmos->zLowerBc.type == THERMALISED)
        {
            // NOTE(cmo): Trying to fulfill diffusion condition.
            f64 Bnu[2];
            int Nspace = atmos->Nspace;
            planck_nu(2, &atmos->temperature(Nspace - 2), wav, Bnu);
            Iupw = Bnu[1] - (Bnu[0] - Bnu[1]) / dtau_uw;
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
            Iupw = atmos->zLowerBc.bcData(la, muIdx, 0);
        }
    }
    else
    {
        if (atmos->zUpperBc.type == THERMALISED)
        {
            f64 Bnu[2];
            planck_nu(2, &atmos->temperature(0), wav, Bnu);
            Iupw = Bnu[0] - (Bnu[1] - Bnu[0]) / dtau_uw;
        }
        else if (atmos->zUpperBc.type == CALLABLE)
        {
            int muIdx = atmos->zUpperBc.idxs(mu, int(toObs));
            if (muIdx == -1)
            {
                printf("Error in boundary condition indexing\n");
                assert(false);
                Iupw = 0.0;
            }
            else
                Iupw = atmos->zUpperBc.bcData(la, muIdx, 0);
        }
    }

    piecewise_bezier3_1d_impl(fd, zmu, toObs, Iupw);
}

void piecewise_besser_1d(FormalData* fd, int la, int mu, bool toObs, const F64View1D& wave)
{
    const f64 wav = wave(la);
    JasUnpack((*fd), atmos, chi);
    // This is 1.0 here, as we are normally effectively rolling in the averaging
    // factor for dtau, whereas it's explicit in this solver
    f64 zmu = 1.0 / atmos->muz(mu);
    auto height = atmos->height;

    int dk = -1;
    int kStart = atmos->Nspace - 1;
    if (!toObs)
    {
        dk = 1;
        kStart = 0;
    }
    f64 dtau_uw = 0.5 * zmu * (chi(kStart) + chi(kStart + dk)) * abs(height(kStart) - height(kStart + dk));

    f64 Iupw = 0.0;
    if (toObs)
    {
        if (atmos->zLowerBc.type == THERMALISED)
        {
            f64 Bnu[2];
            int Nspace = atmos->Nspace;
            planck_nu(2, &atmos->temperature(Nspace - 2), wav, Bnu);
            Iupw = Bnu[1] - (Bnu[0] - Bnu[1]) / dtau_uw;
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
            Iupw = atmos->zLowerBc.bcData(la, muIdx, 0);
        }
    }
    else
    {
        if (atmos->zUpperBc.type == THERMALISED)
        {
            f64 Bnu[2];
            planck_nu(2, &atmos->temperature(0), wav, Bnu);
            Iupw = Bnu[0] - (Bnu[1] - Bnu[0]) / dtau_uw;
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
            Iupw = atmos->zUpperBc.bcData(la, muIdx, 0);
        }
    }

    piecewise_besser_1d_impl(fd, zmu, toObs, Iupw);
}
}

IterationResult formal_sol_iteration_matrices_scalar(Context& ctx, bool lambdaIterate, ExtraParams params)
{
    FsMode mode = (FsMode::UpdateJ | FsMode::UpdateRates);
    if (lambdaIterate)
        mode = mode | FsMode::PureLambdaIteration;

    return LwInternal::formal_sol_iteration_matrices_impl<SimdType::Scalar>(ctx, mode, params);
}

IterationResult formal_sol_gamma_matrices(Context& ctx, bool lambdaIterate, ExtraParams params)
{
    return ctx.iterFns.fs_iter(ctx, lambdaIterate, params);
}

IterationResult formal_sol_scalar(Context& ctx, bool upOnly, ExtraParams params)
{
    FsMode mode = FsMode::FsOnly;
    if (upOnly)
        mode = mode | FsMode::UpOnly;
    return formal_sol_impl<SimdType::Scalar>(ctx, mode, params);
}

IterationResult formal_sol(Context& ctx, bool upOnly, ExtraParams params)
{
    return ctx.iterFns.simple_fs(ctx, upOnly, params);
}