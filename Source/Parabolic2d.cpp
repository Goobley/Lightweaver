#include "Constants.hpp"
#include "LwAtmosphere.hpp"
#include "LwInternal.hpp"
#include "Lightweaver.hpp"
#include "JasPP.hpp"
#include "Utils.hpp"
#include "LwFormalInterface.hpp"

#include <algorithm>
#include <limits>

using namespace LwInternal;

// NOTE(cmo): Our substepping approach is insufficient when x spacing << z
// spacing, as the periodic boundaries end up unbalanced, since they assume only
// 1 wraparound possible. -- Getting negative intensities in this case, after first pop update.

namespace CmoParabolic
{

void w3(double dtau, double *w)
{
  double expdt, delta;

  if (dtau < 5.0E-4) {
    w[0]   = dtau*(1.0 - 0.5*dtau);
    delta  = square(dtau);
    w[1]   = delta*(0.5 - dtau/3.0);
    delta *= dtau;
    w[2]   = delta*(1.0/3.0 - 0.25*dtau);
  } else if (dtau > 50.0) {
    w[1] = w[0] = 1.0;
    w[2] = 2.0;
  } else {
    expdt = exp(-dtau);
    w[0]  = 1.0 - expdt;
    w[1]  = w[0] - dtau*expdt;
    w[2]  = 2.0*w[1] - square(dtau) * expdt;
  }
}


static int linReplace = 0;
static int totalInterp = 0;
f64 interp_param_lin(const IntersectionData& grid, const IntersectionResult& loc,
                 const F64View2D& param)
{
    // TODO(cmo): This is only linear for now. Probably copy out small range to
    // contiguous buffer in future.
    f64 result;
    // totalInterp += 1;
    switch (loc.axis)
    {
        case InterpolationAxis::None:
        {
            int x = int(loc.fractionalX);
            int z = int(loc.fractionalZ);

            result = param(z, x);
        } break;

        case InterpolationAxis::X:
        {
            int xm, xp, z;
            f64 frac;

            xm = int(loc.fractionalX);
            xp = xm + 1;
            frac = loc.fractionalX - xm;
            z = int(loc.fractionalZ);

            result = (1.0 - frac) * param(z, xm) + frac * param(z, xp);
        } break;

        case InterpolationAxis::Z:
        {
            int zm = int(loc.fractionalZ);
            int zp = zm + 1;
            f64 frac = loc.fractionalZ - zm;
            int x = int(loc.fractionalX);

            result = (1.0 - frac) * param(zm, x) + frac * param(zp, x);
        } break;
    }
    if (result < 0.0)
        int Break = 1;
    return result;
}

f64 interp_param(const IntersectionData& grid, const IntersectionResult& loc,
                      const F64View2D& param)
{
    constexpr f64 Eps = 1.0e-6;
    bool resultSet = false;
    f64 result;

    // totalInterp += 1;
    f64 x, xim, xi, xip, xipp, yim, yi, yip, yipp;
    switch (loc.axis)
    {
        case InterpolationAxis::None:
        {
            int x = int(loc.fractionalX);
            int z = int(loc.fractionalZ);

            result = param(z, x);
            resultSet = true;
        } break;

        case InterpolationAxis::X:
        {
            const F64View& xp = grid.x;
            const int Nx = xp.shape(0);
            int i = int(loc.fractionalX);
            f64 frac = loc.fractionalX - i;

            if (i == Nx - 1)
            {
                --i;
                frac = 1.0;
            }
            x = (1.0 - frac) * xp(i) + frac * xp(i+1);
            const int zIdx = int(loc.fractionalZ);

            if (i == 0 || i == Nx - 2)
            {
                result = (1.0 - frac) * param(zIdx, i) + frac * param(zIdx, i+1);
                resultSet = true;
                break;
            }

            xim = xp(i-1);
            xi = xp(i);
            xip = xp(i+1);
            xipp = xp(i+2);

            yim = param(zIdx, i-1);
            yi = param(zIdx, i);
            yip = param(zIdx, i+1);
            yipp = param(zIdx, i+2);

        } break;

        case InterpolationAxis::Z:
        {
            const F64View& xp = grid.z;
            const int Nx = xp.shape(0);
            int i = int(loc.fractionalZ);
            f64 frac = loc.fractionalZ - i;

            if (i == Nx - 1)
            {
                --i;
                frac = 1.0;
            }
            x = (1.0 - frac) * xp(i) + frac * xp(i+1);
            const int zIdx = int(loc.fractionalX);

            if (i == 0 || i == Nx - 2)
            {
                result = (1.0 - frac) * param(i, zIdx) + frac * param(i+1, zIdx);
                resultSet = true;
                break;
            }

            xim = xp(i-1);
            xi = xp(i);
            xip = xp(i+1);
            xipp = xp(i+2);

            yim = param(i-1, zIdx);
            yi = param(i, zIdx);
            yip = param(i+1, zIdx);
            yipp = param(i+2, zIdx);

        } break;
    }

    if (resultSet)
        return result;

    const f64 him = xi - xim;
    const f64 hi = xip - xi;
    const f64 hip = xipp - xip;
    f64 q2 = yim * ((x - xi) * (x - xip)) / (him * (him + hi));
    q2 -= yi * ((x - xim) * (x - xip)) / (him * hi);
    q2 += yip * ((x - xim) * (x - xi)) / ((him + hi) * hi);

    f64 q3 = yi * ((x - xip) * (x - xipp)) / (hi * (hi + hip));
    q3 -= yip * ((x - xi) * (x - xipp)) / (hi * hip);
    q3 += yipp * ((x - xi) * (x - xip)) / ((hi + hip) * hip);

    const f64 H = him + hi + hip;
    f64 yyim = - ((2*him + hi)*H + him*(him + hi)) / (him*(him + hi)*H) * yim;
    yyim += ((him + hi)*H) / (him*hi*(hi + hip)) * yi;
    yyim -= (him*H) / ((him + hi)*hi*hip) * yip;
    yyim += (him*(him + hi)) / ((hi + hip)*hip*H) * yipp;

    f64 yyi = - (hi*(hi + hip)) / (him*(him + hi)*H) * yim;
    yyi += (hi*(hi + hip) - him*(2*hi + hip)) / (him*hi*(hi + hip)) * yi;
    yyi += (him*(hi + hip)) / ((him + hi)*hi*hip) * yip;
    yyi -= (him*hi) / ((hi + hip)*hip*H) * yipp;

    f64 yyip = (hi*hip) / (him*(him + hi)*H) * yim;
    yyip -= (hip*(him + hi)) / (him*hi*(hi + hip)) * yi;
    yyip += ((him + 2*hi)*hip - (him + hi)*hi) / ((him + hi)*hi*hip) * yip;
    yyip += ((him + hi)*hi) / ((hi + hip)*hip*H) * yipp;

    f64 yyipp = - ((hi + hip)*hip) / (him*(him + hi)*H) * yim;
    yyipp += (hip*H) / (him*hi*(hi + hip)) * yi;
    yyipp -= ((hi + hip) * H) / ((him + hi)*hi*hip) * yip;
    yyipp += ((2*hip + hi)*H + hip*(hi + hip)) / ((hi + hip)*hip*H) * yipp;

    // NOTE(cmo): Smoothness indicators
    const f64 beta2 = square(hi + hip) * square(abs(yyip - yyi) / hi - abs(yyi - yyim) / him);
    const f64 beta3 = square(him + hi) * square(abs(yyipp - yyip) / hip - abs(yyip - yyi) / hi);


    // NOTE(cmo): Linear weights
    const f64 gamma2 = - (x - xipp) / (xipp - xim);
    const f64 gamma3 = (x - xim) / (xipp - xim);

    // NOTE(cmo): Non-linear weights
    const f64 alpha2 = gamma2 / (Eps + beta2);
    const f64 alpha3 = gamma3 / (Eps + beta3);

    const f64 omega2 = alpha2 / (alpha2 + alpha3);
    const f64 omega3 = alpha3 / (alpha2 + alpha3);

    // NOTE(cmo): Interpolated value
    result = omega2 * q2 + omega3 * q3;

    if (result < 0.0)
    {
        // linReplace += 1;
        return interp_param_lin(grid, loc, param);
    }

        // int Break = 1;
    return result;
}

void piecewise_parabolic_2d(FormalData* fd, int la, int mu, bool toObs, f64 wav)
{
    auto& atmos = fd->atmos;
    // printf(".............................................\n");
    // printf("%d %d %d %d\n", atmos->Nspace, atmos->Nx, atmos->Ny, atmos->Nz);
    assert(bool(atmos->intersections));
    totalInterp = 0;
    linReplace = 0;

    if (atmos->xLowerBc.type != PERIODIC || atmos->xUpperBc.type != PERIODIC)
    {
        // printf("%d, %d, %d, %d, %d, %d\n", atmos->zLowerBc.type,
        //                                    atmos->zUpperBc.type,
        //                                    atmos->xLowerBc.type,
        //                                    atmos->xUpperBc.type,
        //                                    atmos->yLowerBc.type,
        //                                    atmos->yUpperBc.type
        // );
        printf("Only supporting periodic x BCs for now! %d, %d\n", atmos->xLowerBc.type,
                                                                   atmos->xUpperBc.type);
        assert(false);
    }

    f64 muz = If toObs Then atmos->muz(mu) Else -atmos->muz(mu) End;
    // NOTE(cmo): We invert the sign of mux, because for muz it is done
    // implicitly, and both need to be the additive inverse so the ray for
    // !toObs is the opposite of the toObs ray.
    f64 mux = If toObs Then atmos->mux(mu) Else -atmos->mux(mu) End;


    // NOTE(cmo): As always, assume toObs
    int dk = -1;
    int kStart = atmos->Nz - 1;
    int kEnd = 0;
    if (!toObs)
    {
        dk = 1;
        kStart = 0;
        kEnd = atmos->Nz - 1;
    }

    // NOTE(cmo): Assume mux >= 0 and correct if not
    // NOTE(cmo): L->R
    int dj = 1;
    int jStart = 0;
    int jEnd = atmos->Nx - 1;
    if (mux < 0)
    {
        dj = -1;
        jStart = jEnd;
        jEnd = 0;
    }
    // printf("%s, %d, %d\n", If toObs Then "toObs" Else "away" End, kStart, kEnd);
    // printf(".............................................\n");

    F64View2D I = fd->I.reshape(atmos->Nz, atmos->Nx);
    I.fill(0.0);
    F64View2D chi = fd->chi.reshape(atmos->Nz, atmos->Nx);
    F64View2D S = fd->S.reshape(atmos->Nz, atmos->Nx);
    F64View2D Psi;
    const bool computeOperator = bool(fd->Psi);
    if (computeOperator)
        Psi = fd->Psi.reshape(atmos->Nz, atmos->Nx);
    Psi.fill(0.0);
    F64View2D temperature = atmos->temperature.reshape(atmos->Nz, atmos->Nx);

    for (int j = jStart; j != jEnd + dj; j += dj)
        for (int k = kStart; k != kEnd + dk; k += dk)
            if (chi(k, j) < 0.0)
            {
                printf("chi %e %d %d\n", chi(k, j), k, j);
                assert(false);
            }

    RadiationBc bcType = If toObs
                         Then atmos->zLowerBc.type
                         Else atmos->zUpperBc.type End;
    IntersectionData gridData {atmos->x,
                               atmos->z,
                               mux,
                               muz,
                               atmos->x(0) - (atmos->x(1) - atmos->x(0)),
                               toObs,
                               dj,
                               jStart,
                               jEnd,
                               dk,
                               kStart,
                               kEnd};

    auto& intersections = atmos->intersections.intersections;
    int k = kStart;
    // NOTE(cmo): Handle BC in starting plane
    for (int j = jStart; j != jEnd + dj; j += dj)
    {
        I(k, j) = 0.0;

        if (bcType == THERMALISED)
        {
            auto dwIntersection = intersections(mu, (int)toObs, k, j).dwIntersection;
            f64 chiDw = interp_param(gridData, dwIntersection, chi);
            f64 dtauDw = 0.5 * abs(dwIntersection.distance) * (chi(k, j) + chiDw);
            f64 temperatureDw = interp_param(gridData, dwIntersection, temperature);
            f64 Bnu[2];
            int Nz = atmos->Nz;
            if (toObs)
            {
                f64 temp[2];
                temp[0] = temperatureDw;
                temp[1] = temperature(k, j);
                planck_nu(2, temp, wav, Bnu);
                I(k, j) = Bnu[1] - (Bnu[0] - Bnu[1]) / dtauDw;
            }
            else
            {
                f64 temp[2];
                temp[0] = temperature(k, j);
                temp[1] = temperatureDw;
                planck_nu(2, temp, wav, Bnu);
                I(k, j) = Bnu[0] - (Bnu[1] - Bnu[0]) / dtauDw;
            }
        }
        // TODO(cmo): Handle other Bcs!
        if (computeOperator)
            Psi(k, j) = 0.0;
    }
    k += dk;

    for (; k != kEnd; k += dk)
    {
        for (int j = jStart; j != jEnd + dj; j += dj)
        {
            // if (la == 0)
            //     plot_intersections(fd, j, k, mu, toObs);
            if (j == 4 && k == 57)
                int BreakHere  = 1;
            int longCharIdx = intersections(mu, (int)toObs, k, j).longCharIdx;
            auto uwIntersection = intersections(mu, (int)toObs, k, j).uwIntersection;
            auto dwIntersection = intersections(mu, (int)toObs, k, j).dwIntersection;
            f64 origDistance = uwIntersection.distance;
            if (longCharIdx < 0)
            // if (true)
            {
                f64 dsUw = uwIntersection.distance;
                f64 dsDw = dwIntersection.distance;

                f64 chiDw = interp_param(gridData, dwIntersection, chi);
                f64 chiUw = interp_param(gridData, uwIntersection, chi);
                f64 chiLocal = chi(k, j);
                f64 dtauUw = (0.5) * (chiUw + chiLocal) * dsUw;
                f64 dtauDw = (0.5) * (chiLocal + chiDw) * dsDw;

                f64 Suw = interp_param(gridData, uwIntersection, S);
                f64 Sdw = interp_param(gridData, dwIntersection, S);
                f64 SLocal = S(k, j);
                f64 dSuw = (Suw - SLocal) / dtauUw;
                f64 dSdw = (SLocal - Sdw) / dtauDw;

                f64 c1 = (dSuw * dtauDw + dSdw * dtauUw);
                f64 c2 = dSuw - dSdw;

                f64 w[3];
                w3(dtauUw, w);

                f64 Iuw = interp_param_lin(gridData, uwIntersection, I);
                if (Iuw < 0.0)
                {
                    printf("uw neg %d, %d, %d, %d, %d\n", j, k, la, mu, toObs);
                    assert(false);
                }

                I(k, j) = (1.0 - w[0]) * Iuw + w[0] * SLocal + (w[1]*c1 + w[2]*c2) / (dtauUw + dtauDw);
                if (I(k, j) < 0.0)
                {
                    I(k, j) = (1.0 - w[0]) * Iuw + w[0] * SLocal + w[1] * dSuw;
                    if (computeOperator)
                    {
                        Psi(k, j) = w[0] - w[1] / dtauUw;
                    }
                }
                else
                {
                    if (computeOperator)
                    {
                        f64 c3 = dtauUw - dtauDw;
                        Psi(k, j) = w[0] + (w[1]*c3 - w[2]) / (dtauUw * dtauDw);
                    }
                }

            }
            else
            {
                auto& substeps = atmos->intersections.substeps[longCharIdx];
                f64 Iuw = interp_param_lin(gridData, substeps.steps[0], I);
                if (Iuw < 0.0)
                {
                    printf("lc uw neg %d, %d, %d\n", j, k, mu);
                    assert(false);
                }
                for (int stepIdx = 1; stepIdx < substeps.steps.size() - 1; ++stepIdx)
                {
                    const auto& stepUw = substeps.steps[stepIdx - 1];
                    const auto& step = substeps.steps[stepIdx];
                    const auto& stepDw = substeps.steps[stepIdx + 1];
                    f64 dsUw = stepUw.distance;
                    f64 dsDw = step.distance;
                    f64 chiUw = interp_param(gridData, stepUw, chi);
                    f64 chiLocal = interp_param(gridData, step, chi);
                    f64 chiDw = interp_param(gridData, stepDw, chi);
                    f64 dtauUw = 0.5 * (chiUw + chiLocal) * dsUw;
                    f64 dtauDw = 0.5 * (chiDw + chiLocal) * dsDw;
                    f64 Suw = interp_param(gridData, stepUw, S);
                    f64 SLocal = interp_param(gridData, step, S);
                    f64 Sdw = interp_param(gridData, stepDw, S);

                    f64 dSuw = (Suw - SLocal) / dtauUw;
                    f64 dSdw = (SLocal - Sdw) / dtauDw;
                    f64 c1 = (dSuw * dtauDw + dSdw * dtauUw);
                    f64 c2 = dSuw - dSdw;

                    f64 w[3];
                    w3(dtauUw, w);

                    f64 IuwPrev = Iuw;
                    Iuw = (1.0 - w[0]) * Iuw + w[0] * SLocal + (w[1]*c1 + w[2]*c2) / (dtauUw + dtauDw);
                    if (Iuw < 0.0)
                        Iuw = (1.0 - w[0]) * IuwPrev + w[0] * SLocal + w[1] * dSuw;

                }

                int stepIdx = substeps.steps.size() - 1;
                const auto& stepUw = substeps.steps[stepIdx - 1];
                const auto& step = substeps.steps[stepIdx]; // same as uwIntersection
                f64 dsUw = stepUw.distance;
                f64 dsDw = step.distance;

                f64 chiUw = interp_param(gridData, stepUw, chi);
                f64 chiLocal = interp_param(gridData, step, chi);
                f64 chiDw = chi(k, j);

                f64 dtauUw = 0.5 * (chiUw + chiLocal) * dsUw;
                f64 dtauDw = 0.5 * (chiDw + chiLocal) * dsDw;

                f64 Suw = interp_param(gridData, stepUw, S);
                f64 SLocal = interp_param(gridData, step, S);
                f64 Sdw = S(k, j);

                f64 dSuw = (Suw - SLocal) / dtauUw;
                f64 dSdw = (SLocal - Sdw) / dtauDw;
                f64 c1 = (dSuw * dtauDw + dSdw * dtauUw);
                f64 c2 = dSuw - dSdw;

                f64 w[3];
                w3(dtauUw, w);

                f64 IuwPrev = Iuw;
                Iuw = (1.0 - w[0]) * Iuw + w[0] * SLocal + (w[1]*c1 + w[2]*c2) / (dtauUw + dtauDw);
                if (Iuw < 0.0)
                    Iuw = (1.0 - w[0]) * IuwPrev + w[0] * SLocal + w[1] * dSuw;

                const auto& uw = step;
                const auto& dw = dwIntersection;
                dsUw = uw.distance;
                dsDw = dw.distance;
                chiUw = interp_param(gridData, uw, chi);
                chiLocal = chi(k, j);
                chiDw = interp_param(gridData, dw, chi);
                dtauUw = 0.5 * (chiUw + chiLocal) * dsUw;
                dtauDw = 0.5 * (chiDw + chiLocal) * dsDw;

                Suw = interp_param(gridData, uw, S);
                SLocal = S(k, j);
                Sdw = interp_param(gridData, dw, S);

                dSuw = (Suw - SLocal) / dtauUw;
                dSdw = (SLocal - Sdw) / dtauDw;
                c1 = (dSuw * dtauDw + dSdw * dtauUw);
                c2 = dSuw - dSdw;

                w3(dtauUw, w);

                I(k, j) = (1.0 - w[0]) * Iuw + w[0] * SLocal + (w[1]*c1 + w[2]*c2) / (dtauUw + dtauDw);
                if (I(k, j) < 0.0)
                {
                    I(k, j) = (1.0 - w[0]) * Iuw + w[0] * SLocal + w[1] * dSuw;
                    if (computeOperator)
                    {
                        Psi(k, j) = w[0] - w[1] / dtauUw;
                    }
                }
                else
                {
                    if (computeOperator)
                    {
                        f64 c3 = dtauUw - dtauDw;
                        Psi(k, j) = w[0] + (w[1]*c3 - w[2]) / (dtauUw * dtauDw);
                    }
                }
            }
        }
    }
    k = kEnd;

    for (int j = jStart; j != jEnd + dj; j += dj)
    {
        // auto uwIntersection = uw_intersection_2d(gridData, j, k);
        int substepIdx = intersections(mu, (int)toObs, k, j).substepIdx;
        auto uwIntersection = intersections(mu, (int)toObs, k, j).uwIntersection;
        f64 origDistance = abs(uwIntersection.distance);
        if (substepIdx < 0)
        // if (true)
        {
            f64 chiUw = interp_param(gridData, uwIntersection, chi);
            f64 dtau = 0.5 * (chiUw + chi(k, j)) * abs(uwIntersection.distance);
            f64 Suw = interp_param(gridData, uwIntersection, S);
            f64 Iuw = interp_param(gridData, uwIntersection, I);

            f64 w[2];
            w2(dtau, w);
            f64 c1 = (Suw - S(k, j)) / dtau;
            I(k, j) = (1.0 - w[0]) * Iuw + w[0] * S(k, j) + w[1] * c1;

            if (computeOperator)
                Psi(k, j) = w[0] - w[1] / dtau;
        }
        else
        {
#if 1
            auto& substeps = atmos->intersections.substeps[substepIdx];
            f64 Iuw = interp_param(gridData, substeps.steps[0], I);
            for (int stepIdx = 1; stepIdx < substeps.steps.size(); ++stepIdx)
            {
                const auto& stepUw = substeps.steps[stepIdx-1];
                const auto& step = substeps.steps[stepIdx];
                f64 chiUw = interp_param(gridData, stepUw, chi);
                f64 chiLocal = interp_param(gridData, step, chi);
                f64 dtau = 0.5 * (chiUw + chiLocal) * step.distance;
                f64 Suw = interp_param(gridData, stepUw, S);
                f64 SLocal = interp_param(gridData, step, S);

                f64 w[2];
                w2(dtau, w);
                f64 c1 = (Suw - SLocal) / dtau;
                Iuw = (1.0 - w[0]) * Iuw + w[0] * SLocal + w[1] * c1;
            }
            f64 chiUw = interp_param(gridData, uwIntersection, chi);
            f64 dtau = 0.5 * (chiUw + chi(k, j)) * (uwIntersection.distance);
            f64 Suw = interp_param(gridData, uwIntersection, S);

            f64 w[2];
            w2(dtau, w);
            f64 c1 = (Suw - S(k, j)) / dtau;
            I(k, j) = (1.0 - w[0]) * Iuw + w[0] * S(k, j) + w[1] * c1;

            if (computeOperator)
                Psi(k, j) = w[0] - w[1] / dtau;
#else
            auto& substeps = atmos->intersections.substeps[substepIdx];
            f64 Iuw = interp_param(gridData, uwIntersection, I);
            f64 accumDist = 0.0;
            // for (const auto& step : substeps.steps)
            for (int stepIdx = 0; stepIdx < substeps.steps.size()-1; ++stepIdx)
            {
                const auto& step = substeps.steps[stepIdx];
                const auto& dwStep = substeps.steps[stepIdx+1];
                f64 dsUw = step.distance;
                f64 dsDw = dwStep.distance;

                f64 chiUw = interp_param(gridData, uwIntersection, chi);
                f64 chiLocal = interp_param(gridData, step, chi);
                f64 chiDw = interp_param(gridData, dwStep, chi);
                f64 chiC = besser_control_point(dsUw, dsDw, chiUw, chiLocal, chiDw);

                f64 dtau = (1.0 / 3.0) * (chiUw + chiLocal + chiC) * dsUw;

                f64 Suw = interp_param(gridData, uwIntersection, S);
                f64 SLocal = interp_param(gridData, step, S);
                f64 Sdw = interp_param(gridData, dwStep, S);
                f64 SC = besser_control_point(dsUw, dsDw, Suw, SLocal, Sdw);
                auto coeffs = besser_coeffs(dtau);

                Iuw = coeffs.edt * Iuw + coeffs.M * Suw + coeffs.O * SLocal + coeffs.C * SC;

                accumDist += step.distance;
                uwIntersection = step;
            }
            const auto& step = substeps.steps.back();
            f64 dsUw = step.distance;
            f64 dsDw = origDistance - (accumDist + step.distance);
            f64 chiUw = interp_param(gridData, uwIntersection, chi);
            f64 chiLocal = interp_param(gridData, step, chi);
            f64 chiDw = chi(j, k);
            f64 chiC = besser_control_point(dsUw, dsDw, chiUw, chiLocal, chiDw);

            f64 dtau = (1.0 / 3.0) * (chiUw + chiLocal + chiC) * dsUw;

            f64 Suw = interp_param(gridData, uwIntersection, S);
            f64 SLocal = interp_param(gridData, step, S);
            f64 Sdw = S(j, k);
            f64 SC = besser_control_point(dsUw, dsDw, Suw, SLocal, Sdw);
            auto coeffs = besser_coeffs(dtau);

            Iuw = coeffs.edt * Iuw + coeffs.M * Suw + coeffs.O * SLocal + coeffs.C * SC;

            dsUw = dsDw;
            chiUw = interp_param(gridData, uwIntersection, chi);
            dtau = 0.5 * (chiUw + chi(j, k)) * dsDw;
            Suw = interp_param(gridData, uwIntersection, S);

            f64 w[2];
            w2(dtau, w);
            f64 c1 = (Suw - S(j, k)) / dtau;
            I(j, k) = (1.0 - w[0]) * Iuw + w[0] * S(j, k) + w[1] * c1;

            if (computeOperator)
                Psi(j, k) = w[0] - w[1] / dtau;
#endif
        }

    }


    if (computeOperator)
    {
        // for (int k = 0; k < atmos->Nspace; ++k)
        //     if (fd->Psi(k) <= 0.0)
        //     {
        //         printf("%d, %e\n", k, fd->Psi(k));
        //         assert(false);
        //     }

        for (int k = 0; k < atmos->Nspace; ++k)
            fd->Psi(k) /= fd->chi(k);

    }
    // if (linReplace > 0)
    //     printf("%d, %d, %e%%\n", linReplace, totalInterp, 100.0 * (f64)linReplace / (f64)totalInterp);
}
}

extern "C"
{
FormalSolver fs_provider()
{
    return FormalSolver{1, 1, "piecewise_parabolic_2d", CmoParabolic::piecewise_parabolic_2d};
}
}