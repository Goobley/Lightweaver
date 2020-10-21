#include "LwInternal.hpp"
#include "Lightweaver.hpp"
#include "Bezier.hpp"
#include "JasPP.hpp"
#include "ThreadStorage.hpp"

#include <cmath>
#include <fenv.h>
#include <iostream>
#include <utility>
#include <vector>
#include <algorithm>
#include <chrono>

#ifdef CMO_BASIC_PROFILE
using hrc = std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;
#endif

using namespace LwInternal;
using LwInternal::FormalData;
using LwInternal::IntensityCoreData;

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

void Transition::compute_phi_la(const Atmosphere& atmos, const F64View& aDamp,
                                const F64View& vBroad, int lt)
{
    namespace C = Constants;

    constexpr f64 sign[] = { -1.0, 1.0 };
    // Why is there still no constexpr math in std? :'(
    const f64 sqrtPi = sqrt(C::Pi);

    const f64 vBase = (wavelength(lt) - lambda0) * C::CLight / lambda0;
    const f64 wla = wlambda(lt);
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

    LineProfileData* data = (LineProfileData*)malloc(sizeof(LineProfileData));
    data->t = this;
    data->atmos = &atmos;
    data->aDamp = &aDamp;
    data->vBroad = &vBroad;
    auto compute_profile = [](void* data, scheduler* s,
                               sched_task_partition p, sched_uint threadId)
    {
        LineProfileData* d = (LineProfileData*)data;
        for (i64 la = p.start; la < p.end; ++la)
            d->t->compute_phi_la(*(d->atmos), *(d->aDamp), *(d->vBroad), la);
    };

    {
        sched_task lineProfile;
        scheduler_add(&threading->sched, &lineProfile, compute_profile,
                      (void*)data, wavelength.shape(0), 1);
        scheduler_join(&threading->sched, &lineProfile);
    }

    free(data);
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
void piecewise_linear_1d(FormalData* fd, int la, int mu, bool toObs, f64 wav)
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
            Iupw = atmos->zLowerBc.bcData(la, mu, 0);
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
            Iupw = atmos->zUpperBc.bcData(la, mu, 0);
        }
    }

    piecewise_linear_1d_impl(fd, zmu, toObs, Iupw);
}

void piecewise_bezier3_1d(FormalData* fd, int la, int mu, bool toObs, f64 wav)
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
            Iupw = atmos->zLowerBc.bcData(la, mu, 0);
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
            Iupw = atmos->zUpperBc.bcData(la, mu, 0);
        }
    }

    piecewise_bezier3_1d_impl(fd, zmu, toObs, Iupw);
}

void piecewise_besser_1d(FormalData* fd, int la, int mu, bool toObs, f64 wav)
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
            Iupw = atmos->zLowerBc.bcData(la, mu, 0);
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
            Iupw = atmos->zUpperBc.bcData(la, mu, 0);
        }
    }

    piecewise_besser_1d_impl(fd, zmu, toObs, Iupw);
}

bool continua_only(const IntensityCoreData& data, int la)
{
    JasUnpack(*data, activeAtoms, detailedAtoms);
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
    return continuaOnly;
}

void gather_opacity_emissivity(IntensityCoreData* data, bool computeOperator, int la, int mu, bool toObs)
{
    JasUnpack(*(*data), activeAtoms, detailedAtoms);
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
                    atom.eta(k) += eta;
                }
                chiTot(k) += chi;
                etaTot(k) += eta;
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

                chiTot(k) += chi;
                etaTot(k) += eta;
            }
        }
    }
}

f64 intensity_core(IntensityCoreData& data, int la, FsMode mode)
{
    JasUnpack(*data, atmos, spect, fd, background);
    JasUnpack(*data, activeAtoms, detailedAtoms, JDag);
    JasUnpack(data, chiTot, etaTot, Uji, Vij, Vji);
    JasUnpack(data, I, S, Ieff, PsiStar);
    const LwFsFn formal_solver = data.formal_solver;
    const int Nspace = atmos.Nspace;
    const int Nrays = atmos.Nrays;
    const int Nspect = spect.wavelength.shape(0);
    const bool updateJ = mode & FsMode::UpdateJ;
    const bool updateRates = mode & FsMode::UpdateRates;
    const bool prdRatesOnly = mode & FsMode::PrdOnly;
    const bool lambdaIterate = mode & FsMode::PureLambdaIteration;
    const bool computeOperator = bool(PsiStar);
    const bool storeDepthData = (data.depthData && data.depthData->fill);

    JDag = spect.J(la);
    F64View J = spect.J(la);
    if (updateJ)
        J.fill(0.0);

    for (int a = 0; a < activeAtoms.size(); ++a)
        activeAtoms[a]->setup_wavelength(la);
    for (int a = 0; a < detailedAtoms.size(); ++a)
        detailedAtoms[a]->setup_wavelength(la);

    // NOTE(cmo): If we only have continua then opacity is angle independent
    const bool continuaOnly = continua_only(data, la);
    // printf("Core: %d, %d\n", atmos.xLowerBc.type, atmos.xUpperBc.type);
    //     printf("%d, %d, %d, %d, %d, %d\n", atmos.zLowerBc.type,
    //                                        atmos.zUpperBc.type,
    //                                        atmos.xLowerBc.type,
    //                                        atmos.xUpperBc.type,
    //                                        atmos.yLowerBc.type,
    //                                        atmos.yUpperBc.type);

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
                if (storeDepthData)
                {
                    auto& depth = *data.depthData;
                    if (!continuaOnly)
                    {
                        for (int k = 0; k < Nspace; ++k)
                        {
                            depth.chi(la, mu, toObsI, k) = chiTot(k);
                            depth.eta(la, mu, toObsI, k) = etaTot(k);
                        }
                    }
                    else
                    {
                        for (int mu = 0; mu < Nrays; ++mu)
                            for (int toObsI = 0; toObsI < 2; toObsI += 1)
                                for (int k = 0; k < Nspace; ++k)
                                {
                                    depth.chi(la, mu, toObsI, k) = chiTot(k);
                                    depth.eta(la, mu, toObsI, k) = etaTot(k);
                                }
                    }
                }
            }

            switch (atmos.Ndim)
            {
                case 1:
                {
                    // piecewise_bezier3_1d(&fd, la, mu, toObs, spect.wavelength(la));
                    // piecewise_besser_1d(&fd, la, mu, toObs, spect.wavelength(la));
                    // piecewise_linear_1d(&fd, la, mu, toObs, spect.wavelength(la));
                    formal_solver(&fd, la, mu, toObs, spect.wavelength(la));
                    spect.I(la, mu, 0) = I(0);
                } break;

                case 2:
                {
                    // piecewise_linear_2d(&fd, la, mu, toObs, spect.wavelength(la));
                    // piecewise_besser_2d(&fd, la, mu, toObs, spect.wavelength(la));
                    // piecewise_parabolic_2d(&fd, la, mu, toObs, spect.wavelength(la));
                    formal_solver(&fd, la, mu, toObs, spect.wavelength(la));
                    auto I2 = I.reshape(atmos.Nz, atmos.Nx);
                    for (int j = 0; j < atmos.Nx; ++j)
                        spect.I(la, mu, j) = I2(0, j);
                } break;

                default:
                    printf("Unexpected Ndim!\n");
            }

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
                        if (lambdaIterate)
                            PsiStar.fill(0.0);

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
                for (int a = 0; a < detailedAtoms.size(); ++a)
                {
                    auto& atom = *detailedAtoms[a];

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
            if (storeDepthData)
            {
                auto& depth = *data.depthData;
                for (int k = 0; k < Nspace; ++k)
                    depth.I(la, mu, toObsI, k) = I(k);
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

f64 formal_sol_gamma_matrices(Context& ctx, bool lambdaIterate)
{
    // feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
    JasUnpack(*ctx, atmos, spect, background, depthData);
    JasUnpack(ctx, activeAtoms, detailedAtoms);
    // build_intersection_list(&atmos);

    const int Nspace = atmos.Nspace;
    const int Nspect = spect.wavelength.shape(0);

    if (ctx.Nthreads <= 1)
    {
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
        fd.interp = ctx.interpFn.interp_2d;
        IntensityCoreData iCore;
        JasPackPtr(iCore, atmos, spect, fd, background, depthData);
        JasPackPtr(iCore, activeAtoms, detailedAtoms, JDag);
        JasPack(iCore, chiTot, etaTot, Uji, Vij, Vji);
        JasPack(iCore, I, S, Ieff, PsiStar);
        iCore.formal_solver = ctx.formalSolver.solver;

        if (spect.JRest)
            spect.JRest.fill(0.0);

        for (auto& a : activeAtoms)
        {
            a->zero_rates();
        }
        for (auto& a : detailedAtoms)
        {
            a->zero_rates();
        }

        f64 dJMax = 0.0;
        FsMode mode = (FsMode::UpdateJ | FsMode::UpdateRates);
        if (lambdaIterate)
            mode = mode | FsMode::PureLambdaIteration;

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
    else
    {
#ifdef CMO_BASIC_PROFILE
        hrc::time_point startTime = hrc::now();
#endif
        auto& cores = ctx.threading.intensityCores;

        if (spect.JRest)
            spect.JRest.fill(0.0);

        for (auto& core : cores.cores)
        {
            for (auto& a : *core->activeAtoms)
            {
                a->zero_rates();
                a->zero_Gamma();
            }
            for (auto& a : *core->detailedAtoms)
            {
                a->zero_rates();
            }
        }
#ifdef CMO_BASIC_PROFILE
        hrc::time_point preMidTime = hrc::now();
#endif

        struct FsTaskData
        {
            IntensityCoreData* core;
            f64 dJ;
            i64 dJIdx;
            bool lambdaIterate;
        };
        FsTaskData* taskData = (FsTaskData*)malloc(ctx.Nthreads * sizeof(FsTaskData));
        for (int t = 0; t < ctx.Nthreads; ++t)
        {
            taskData[t].core = cores.cores[t];
            taskData[t].dJ = 0.0;
            taskData[t].dJIdx = 0;
            taskData[t].lambdaIterate = lambdaIterate;
        }

        auto fs_task = [](void* data, scheduler* s,
                          sched_task_partition p, sched_uint threadId)
        {
            auto& td = ((FsTaskData*)data)[threadId];
            FsMode mode = (FsMode::UpdateJ | FsMode::UpdateRates);
            if (td.lambdaIterate)
                mode = mode | FsMode::PureLambdaIteration;

            for (i64 la = p.start; la < p.end; ++la)
            {
                f64 dJ = intensity_core(*td.core, la, mode);
                td.dJ = max_idx(td.dJ, dJ, td.dJIdx, la);
            }
        };

        {
            sched_task formalSolutions;
            scheduler_add(&ctx.threading.sched, &formalSolutions,
                          fs_task, (void*)taskData, Nspect, 4);
            scheduler_join(&ctx.threading.sched, &formalSolutions);
        }
#ifdef CMO_BASIC_PROFILE
        hrc::time_point midTime = hrc::now();
#endif

        f64 dJMax = 0.0;
        i64 maxIdx = 0;
        for (int t = 0; t < ctx.Nthreads; ++t)
            dJMax = max_idx(dJMax, taskData[t].dJ, maxIdx, taskData[t].dJIdx);


        ctx.threading.intensityCores.accumulate_Gamma_rates_parallel(ctx);

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
#ifdef CMO_BASIC_PROFILE
        hrc::time_point endTime = hrc::now();
        int f = duration_cast<nanoseconds>(preMidTime - startTime).count();
        int s = duration_cast<nanoseconds>(midTime - preMidTime).count();
        int t = duration_cast<nanoseconds>(endTime - midTime).count();
        printf("[FS]  First: %d ns, Second: %d ns, Third: %d ns, Ratio: %.3e\n", f, s, t, (f64)f/(f64)s);
#endif
        return dJMax;
    }
}

f64 formal_sol_update_rates(Context& ctx)
{
    JasUnpack(*ctx, atmos, spect, background, depthData);
    JasUnpack(ctx, activeAtoms, detailedAtoms);

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
    fd.interp = ctx.interpFn.interp_2d;
    IntensityCoreData iCore;
    JasPackPtr(iCore, atmos, spect, fd, background, depthData);
    JasPackPtr(iCore, activeAtoms, detailedAtoms, JDag);
    JasPack(iCore, chiTot, etaTot, Uji, Vij, Vji);
    JasPack(iCore, I, S, Ieff);
    iCore.formal_solver = ctx.formalSolver.solver;

    if (spect.JRest)
        spect.JRest.fill(0.0);

    for (auto& a : activeAtoms)
    {
        a->zero_rates();
    }
    for (auto& a : detailedAtoms)
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
    JasUnpack(*ctx, atmos, spect, background, depthData);
    JasUnpack(ctx, activeAtoms, detailedAtoms);

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
    fd.interp = ctx.interpFn.interp_2d;
    IntensityCoreData iCore;
    JasPackPtr(iCore, atmos, spect, fd, background, depthData);
    JasPackPtr(iCore, activeAtoms, detailedAtoms, JDag);
    JasPack(iCore, chiTot, etaTot, Uji, Vij, Vji);
    JasPack(iCore, I, S, Ieff);
    iCore.formal_solver = ctx.formalSolver.solver;

    for (auto& a : activeAtoms)
    {
        a->zero_rates();
    }
    for (auto& a : detailedAtoms)
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
    JasUnpack(*ctx, atmos, spect, background, depthData);
    JasUnpack(ctx, activeAtoms, detailedAtoms);

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
    fd.interp = ctx.interpFn.interp_2d;
    IntensityCoreData iCore;
    JasPackPtr(iCore, atmos, spect, fd, background, depthData);
    JasPackPtr(iCore, activeAtoms, detailedAtoms, JDag);
    JasPack(iCore, chiTot, etaTot, Uji, Vij, Vji);
    JasPack(iCore, I, S, Ieff);
    iCore.formal_solver = ctx.formalSolver.solver;

    FsMode mode = FsMode::FsOnly;
    for (int la = 0; la < Nspect; ++la)
    {
        intensity_core(iCore, la, mode);
    }
    return 0.0;
}