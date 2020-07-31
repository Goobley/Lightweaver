#include "LwInternal.hpp"
#include "Lightweaver.hpp"
#include "JasPP.hpp"
#include "Utils.hpp"
#define PLYGHT_IMPL
#include "Plyght/Plyght.hpp"

#include <limits>

using namespace LwInternal;

namespace LwInternal
{
struct IntersectionData
{
    F64View x;
    F64View z;
    f64 mux;
    f64 muz;
    f64 xWrapVal;
    bool toObs;
    int xStep;
    int xStart;
    int xEnd;
    int zStep;
    int zStart;
    int zEnd;
};

// enum class InterpolationAxis
// {
//     None,
//     X,
//     Z
// };

// struct IntersectionResult
// {
//     InterpolationAxis axis;
//     f64 fractionalX;
//     f64 fractionalZ;
//     f64 distance;
// };

struct Ray
{
    f64 ox;
    f64 oz;
    f64 mux;
    f64 muz;
};

// parametrised intersection (t) with a plane of constant x
// These rely on proper IEEE754 nan/inf handling
f64 x_plane_intersection(f64 offset, const Ray& ray)
{
    f64 t = -(ray.ox - offset) / (ray.mux);
    return t;
}

// parametrised intersection (t) with a plane of constant z
f64 z_plane_intersection(f64 offset, const Ray& ray)
{
    f64 t = -(ray.oz - offset) / (ray.muz);
    return t;
}

int directional_modulo_index(const IntersectionData& grid, int x)
{
    return ((x + 1) % grid.x.shape(0)) - 1;
}

f64 fmod_pos(f64 a, f64 b)
{
    return a - std::floor(a / b) * b;
}

f64 directional_fmodulo(const IntersectionData& grid, f64 x)
{
    return grid.xWrapVal + fmod_pos(x - grid.xWrapVal,
                                    grid.x(grid.x.shape(0) - 1) - grid.xWrapVal);
}

// NOTE(cmo): This gives the intersection downwind of the point (xp, zp)
IntersectionResult dw_intersection_2d(const IntersectionData& grid, int zp, int xp)
{
    bool wraparound = false;
    Ray ray{grid.x(xp), grid.z(zp), grid.mux, grid.muz};
    f64 tx = 0.0;
    // if (xp == grid.xEnd - grid.xStep)
    if (xp == 4 && zp == 1)
        int BreakHere = 1;
    if (xp == grid.xEnd)
    {
        tx = x_plane_intersection(grid.x(xp) + grid.xStep
                                  * abs(grid.x(xp - grid.xStep) - grid.x(xp)),
                                  ray);
        wraparound = true;
    }
    else
    {
        tx = x_plane_intersection(grid.x(xp + grid.xStep), ray);
    }
    f64 tz = z_plane_intersection(grid.z(zp + grid.zStep), ray);
    if (tx * tz < 0)
        int BreakHere = 1;

    if (abs(tx) < abs(tz))
    {
        f64 fracX = xp + grid.xStep;
        if (wraparound)
            fracX = directional_modulo_index(grid, fracX);
        f64 fracZ = zp + grid.zStep * (tx / tz);
        return IntersectionResult(InterpolationAxis::Z, fracZ, fracX, tx);
    }
    else if (abs(tz) < abs(tx))
    {
        f64 fracZ = zp + grid.zStep;
        f64 fracX = xp + grid.xStep * (tz / tx);
        if (wraparound)
        {
            tx = tz;
            f64 xIntersection = ray.ox + ray.mux * tx;
            xIntersection = directional_fmodulo(grid, xIntersection);
            if (xIntersection < grid.x(0))
            {
                fracX = (xIntersection - grid.xWrapVal) / (grid.x(0) - grid.xWrapVal) - 1.0;
                if (fracX < -1.0)
                {
                    int BreakHere = 1;
                    assert(false);
                }
            }
            else
            {
                int xpIdx = hunt(grid.x, xIntersection);
                if (grid.x(xpIdx) == xIntersection)
                    return IntersectionResult(InterpolationAxis::None, fracZ, (f64)xpIdx, tz);

                xpIdx += 1;
                int xmIdx = xpIdx - 1;
                fracX = xmIdx + abs(grid.x(xmIdx) - xIntersection)
                                / abs(grid.x(xmIdx) - grid.x(xpIdx));
            }
        }
        return IntersectionResult(InterpolationAxis::X, fracZ, fracX, tz);
    }
    else
    {
        f64 fracZ = zp + grid.zStep;
        f64 fracX = xp + grid.xStep;
        if (wraparound)
            fracX = (f64)directional_modulo_index(grid, fracX);
        return IntersectionResult(InterpolationAxis::None, fracZ, fracX, tz);
    }
}

IntersectionResult uw_intersection_2d(const IntersectionData& grid, int zp, int xp)
{
    bool wraparound = false;
    Ray ray{grid.x(xp), grid.z(zp), grid.mux, grid.muz};
    f64 tx = 0.0;
    // if (xp == grid.xStart || xp == grid.xStart + grid.xStep)
    if (xp == grid.xStart)
    {
        wraparound = true;
        tx = std::numeric_limits<f64>::infinity();
    }
    else
    {
        tx = x_plane_intersection(grid.x(xp - grid.xStep), ray);
    }
    f64 tz = z_plane_intersection(grid.z(zp - grid.zStep), ray);
    if (std::isfinite(tx) && tx * tz < 0)
        int BreakHere = 1;

    if (abs(tx) < abs(tz))
    {
        f64 fracX = xp - grid.xStep;
        f64 fracZ = zp - grid.zStep * (tx / tz);
        return IntersectionResult(InterpolationAxis::Z, fracZ, fracX, tx);
    }
    else if (abs(tz) < abs(tx))
    {
        f64 fracZ = zp - grid.zStep;
        f64 fracX = 0.0;
        if (!wraparound)
        {
            fracX = xp - grid.xStep * (tz / tx);
        }
        else
        {
            tx = tz;
            f64 xIntersection = ray.ox + ray.mux * tx;
            xIntersection = directional_fmodulo(grid, xIntersection);
            if (xIntersection < grid.x(0))
            {
                fracX = (xIntersection - grid.xWrapVal) / (grid.x(0) - grid.xWrapVal) - 1.0;
                if (fracX < -1.0)
                {
                    int BreakHere = 1;
                    assert(false);
                }
            }
            else
            {
                int xpIdx = hunt(grid.x, xIntersection);
                if (grid.x(xpIdx) == xIntersection)
                    return IntersectionResult(InterpolationAxis::None, fracZ, (f64)xpIdx, tz);

                xpIdx += 1;
                int xmIdx = xpIdx - 1;
                fracX = xmIdx + abs(grid.x(xmIdx) - xIntersection)
                                / abs(grid.x(xmIdx) - grid.x(xpIdx));
            }
        }
        return IntersectionResult(InterpolationAxis::X, fracZ, fracX, tz);
    }
    else
    {
        f64 fracZ = zp - grid.zStep;
        f64 fracX = xp - grid.xStep;
        return IntersectionResult(InterpolationAxis::None, fracZ, fracX, tz);
    }
}

f64 interp_param(const IntersectionData& grid, const IntersectionResult& loc,
                 const F64View2D& param)
{
    // TODO(cmo): This is only linear for now. Probably copy out small range to
    // contiguous buffer in future.
    switch (loc.axis)
    {
        case InterpolationAxis::None:
        {
            int x = int(loc.fractionalX);
            if (x == -1)
                x = param.shape(1) - 1;
            int z = int(loc.fractionalZ);

            f64 result = param(z, x);
            return result;
        } break;

        case InterpolationAxis::X:
        {
            int xm, xp, z;
            f64 frac;

            if (loc.fractionalX < 0.0)
            {
                xm = grid.x.shape(0) - 1;
                xp = 0;
                frac = loc.fractionalX + 1.0;
                z = int(loc.fractionalZ);
                if (frac < 0.0)
                    printf("%f\n", frac);
            }
            else
            {
                xm = int(loc.fractionalX);
                xp = xm + 1;
                frac = loc.fractionalX - xm;
                z = int(loc.fractionalZ);
            }

            f64 result = (1.0 - frac) * param(z, xm) + frac * param(z, xp);
            return result;
        } break;

        case InterpolationAxis::Z:
        {
            int zm = int(loc.fractionalZ);
            int zp = zm + 1;
            f64 frac = loc.fractionalZ - zm;
            int x = int(loc.fractionalX);
            if (x == -1)
                x = param.shape(1) - 1;

            f64 result = (1.0 - frac) * param(zm, x) + frac * param(zp, x);
            return result;
        } break;
    }
}

f64 frac_idx_1d(const F64View1D& view, f64 idx)
{
    int xm, xp;
    f64 frac;

    if (idx < 0.0)
    {
        xm = view.shape(0) - 1;
        xp = 0.0;
        frac = idx + 1.0;
    }
    else
    {
        xm = int(idx);
        xp = xm + 1;
        frac = idx - xm;
    }

    if (frac == 0.0)
        return view(xm);

    f64 result = (1.0 - frac) * view(xm) + frac * view(xp);
    return result;
}

f64 frac_loc_1d(const F64View1D& view, f64 wrapVal, f64 idx)
{
    f64 xm, xp;
    f64 frac;

    if (idx == int(idx))
        return view(int(idx));

    if (idx < 0.0)
    {
        xm = wrapVal;
        xp = view(0);
        frac = idx + 1.0;
    }
    else
    {
        xm = view(int(idx));
        xp = view(int(idx) + 1);
        frac = idx - int(idx);
    }

    if (frac == 0.0)
        return view(xm);

    f64 result = (1.0 - frac) * xm + frac * xp;
    return result;
}

// struct InterpolationStencil
// {
//     IntersectionResult uwIntersection;
//     IntersectionResult dwIntersection;
//     bool longChar;
// };

void build_intersection_list(Atmosphere* atmos)
{
    if (atmos->Ndim != 2)
        return;

    if (atmos->xLowerBc.type != PERIODIC || atmos->xUpperBc.type != PERIODIC)
    {
        printf("Only supporting periodic x BCs for now!\n");
        assert(false);
    }

    atmos->intersections.init(atmos->Nrays, atmos->Nz, atmos->Nx);
    auto& intersections = atmos->intersections.intersections;

    for (int mu = 0; mu < atmos->muz.shape(0); ++mu)
    {
        for (int toObsI = 0; toObsI < 2; ++toObsI)
        {
            bool toObs = (bool)toObsI;
            f64 muz = If toObs Then atmos->muz(mu) Else -atmos->muz(mu) End;
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

            int k = kStart;
            // NOTE(cmo): Handle BC in starting plane
            for (int j = jStart; j != jEnd + dj; j += dj)
            {
                IntersectionResult uw(InterpolationAxis::None, k, j, 0.0);
                auto dw = dw_intersection_2d(gridData, k, j);
                dw.distance = abs(dw.distance);
                intersections(mu, toObsI, k, j) = InterpolationStencil{uw, dw, -1};
            }
            k += dk;

            for (; k != kEnd; k += dk)
            {
                for (int j = jStart; j != jEnd + dj; j += dj)
                {
                    if (j == 4 && k == 14)
                        int BreakHere = 1;
                    auto uw = uw_intersection_2d(gridData, k, j);
                    auto dw = dw_intersection_2d(gridData, k, j);
                    uw.distance = abs(uw.distance);
                    dw.distance = abs(dw.distance);
                    bool longChar = (j == jStart);
                    int longCharIdx = -1;
                    if (longChar)
                    {
                        // if (uw.fractionalX < 0.0)
                        // {
                        //     // NOTE(cmo): We intersect within the ghost cell, so no passing through any extra planes

                        // }
                        // else
                        assert(uw.axis != InterpolationAxis::Z);
                        if (uw.fractionalX > 0.0)
                        {
                            atmos->intersections.substeps.emplace_back(SubstepIntersections{});
                            auto& substeps = atmos->intersections.substeps.back();
                            longCharIdx = atmos->intersections.substeps.size() - 1;

                            int zp = int(uw.fractionalZ);
                            f64 zInt = gridData.z(zp);
                            f64 xInt = frac_loc_1d(gridData.x, gridData.xWrapVal, uw.fractionalX);
                            Ray ray{xInt, zInt, gridData.mux, gridData.muz};

                            int xp = int(uw.fractionalX + gridData.xStep);
                            f64 accumDist = 0.0;
                            while (xp != gridData.xEnd + gridData.xStep)
                            {
                                // Compute intersection with plane at gridData.x(xp)
                                f64 tx = x_plane_intersection(gridData.x(xp), ray);
                                f64 fracZ = zp + gridData.zStep * abs(tx / uw.distance);
                                tx = abs(tx) - accumDist;
                                substeps.steps.emplace_back(IntersectionResult(InterpolationAxis::Z, fracZ, f64(xp), tx));
                                xp += gridData.xStep;
                                accumDist += tx;
                            }
                        }
                    }
                    intersections(mu, toObsI, k, j) = InterpolationStencil{uw, dw, longCharIdx};
                }
            }

            k = kEnd;
            for (int j = jStart; j != jEnd + dj; j += dj)
            {
                auto uw = uw_intersection_2d(gridData, k, j);
                uw.distance = abs(uw.distance);
                IntersectionResult dw(InterpolationAxis::None, k, j, 0.0);
                intersections(mu, toObsI, k, j) = InterpolationStencil{uw, dw, -1};
            }
        }
    }
}

void piecewise_linear_2d(FormalData* fd, int la, int mu, bool toObs, f64 wav)
{
    auto& atmos = fd->atmos;
    // printf(".............................................\n");
    // printf("%d %d %d %d\n", atmos->Nspace, atmos->Nx, atmos->Ny, atmos->Nz);
    assert(bool(atmos->intersections));

    if (atmos->xLowerBc.type != PERIODIC || atmos->xUpperBc.type != PERIODIC)
    {
        printf("Only supporting periodic x BCs for now!\n");
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
    F64View2D temperature = atmos->temperature.reshape(atmos->Nz, atmos->Nx);

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
        // I(j, k) = 0.0;

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

    for (; k != kEnd + dk; k += dk)
    {
        for (int j = jStart; j != jEnd + dj; j += dj)
        {
            // auto uwIntersection = uw_intersection_2d(gridData, j, k);
            int substepIdx = intersections(mu, (int)toObs, k, j).substepIdx;
            auto uwIntersection = intersections(mu, (int)toObs, k, j).uwIntersection;
            f64 origDistance = uwIntersection.distance;
            if (substepIdx < 0)
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
                auto& substeps = atmos->intersections.substeps[substepIdx];
                f64 Iuw = interp_param(gridData, uwIntersection, I);
                f64 accumDist = 0.0;
                for (const auto& step : substeps.steps)
                {
                    f64 chiUw = interp_param(gridData, uwIntersection, chi);
                    f64 chiLocal = interp_param(gridData, step, chi);
                    f64 dtau = 0.5 * (chiUw + chiLocal) * step.distance;
                    f64 Suw = interp_param(gridData, uwIntersection, S);
                    f64 SLocal = interp_param(gridData, step, S);
                    accumDist += step.distance;

                    f64 w[2];
                    w2(dtau, w);
                    f64 c1 = (Suw - SLocal) / dtau;
                    Iuw = (1.0 - w[0]) * Iuw + w[0] * SLocal + w[1] * c1;
                    uwIntersection = step;
                }
                f64 chiUw = interp_param(gridData, uwIntersection, chi);
                f64 dtau = 0.5 * (chiUw + chi(k, j)) * (origDistance - accumDist);
                f64 Suw = interp_param(gridData, uwIntersection, S);

                f64 w[2];
                w2(dtau, w);
                f64 c1 = (Suw - S(k, j)) / dtau;
                I(k, j) = (1.0 - w[0]) * Iuw + w[0] * S(k, j) + w[1] * c1;

                if (computeOperator)
                    Psi(k, j) = w[0] - w[1] / dtau;
            }

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
}

f64 besser_control_point(f64 hM, f64 hP, f64 yM, f64 yO, f64 yP)
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

struct BesserCoeffs
{
    f64 M;
    f64 O;
    f64 C;
    f64 edt;
};

BesserCoeffs besser_coeffs(f64 t)
{
    if (t < 0.14)
    // if (t < 0.05)
    {
        f64 m = (t * (t * (t * (t * (t * (t * ((140.0 - 18.0 * t) * t - 945.0) + 5400.0) - 25200.0) + 90720.0) - 226800.0) + 302400.0)) / 907200.0;
        f64 o = (t * (t * (t * (t * (t * (t * ((10.0 - t) * t - 90.0) + 720.0) - 5040.0) + 30240.0) - 151200.0) + 604800.0)) / 1814400.0;
        f64 c = (t * (t * (t * (t * (t * (t * ((35.0 - 4.0 * t) * t - 270.0) + 1800.0) - 10080.0) + 45360.0) - 151200.0) + 302400.0)) / 907200.0;
        f64 edt = 1.0 - t + 0.5 * square(t) - cube(t) / 6.0 + t * cube(t) / 24.0 - square(t) * cube(t) / 120.0 + cube(t) * cube(t) / 720.0 - cube(t) * cube(t) * t / 5040.0;
        return BesserCoeffs{m, o, c, edt};
    }
    else
    {
        f64 t2 = square(t);
        f64 edt = exp(-t);
        f64 m = (2.0 - edt * (t2 + 2.0 * t + 2.0)) / t2;
        f64 o = 1.0 - 2.0 * (edt + t - 1.0) / t2;
        f64 c = 2.0 * (t - 2.0 + edt * (t + 2.0)) / t2;
        return BesserCoeffs{m, o, c, edt};
    }
}

void piecewise_besser_2d(FormalData* fd, int la, int mu, bool toObs, f64 wav)
{
    // NOTE(cmo): Implmentation of BESSER method following Stepan & Trujillo Bueno (2013) A&A, 557, A143. This is a 2D variant for scalar intensity, but following the same limiting principles, used to integrate chi and S.
    auto& atmos = fd->atmos;
    // printf(".............................................\n");
    // printf("%d %d %d %d\n", atmos->Nspace, atmos->Nx, atmos->Ny, atmos->Nz);
    assert(bool(atmos->intersections));

    if (atmos->xLowerBc.type != PERIODIC || atmos->xUpperBc.type != PERIODIC)
    {
        printf("Only supporting periodic x BCs for now!\n");
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
    I.fill(-1.0);
    F64View2D chi = fd->chi.reshape(atmos->Nz, atmos->Nx);
    F64View2D S = fd->S.reshape(atmos->Nz, atmos->Nx);
    F64View2D Psi;
    const bool computeOperator = bool(fd->Psi);
    if (computeOperator)
        Psi = fd->Psi.reshape(atmos->Nz, atmos->Nx);
    Psi.fill(0.0);
    F64View2D temperature = atmos->temperature.reshape(atmos->Nz, atmos->Nx);

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
            int substepIdx = intersections(mu, (int)toObs, k, j).substepIdx;
            auto uwIntersection = intersections(mu, (int)toObs, k, j).uwIntersection;
            auto dwIntersection = intersections(mu, (int)toObs, k, j).dwIntersection;
            f64 origDistance = uwIntersection.distance;
            if (substepIdx < 0)
            // if (true)
            {
                f64 dsUw = uwIntersection.distance;
                f64 dsDw = dwIntersection.distance;

                f64 chiDw = interp_param(gridData, dwIntersection, chi);
                f64 chiUw = interp_param(gridData, uwIntersection, chi);
                f64 chiLocal = chi(k, j);
                f64 chiC = besser_control_point(dsUw, dsDw, chiUw, chiLocal, chiDw);
                f64 dtauUw = (1.0 / 3.0) * (chiUw + chiLocal + chiC) * dsUw;
                f64 dtauDw = (0.5) * (chiLocal + chiDw) * dsDw;

                f64 Suw = interp_param(gridData, uwIntersection, S);
                f64 Sdw = interp_param(gridData, dwIntersection, S);
                f64 SLocal = S(k, j);
                f64 SC = besser_control_point(dtauUw, dtauDw, Suw, SLocal, Sdw);

                f64 Iuw = interp_param(gridData, uwIntersection, I);
                if (Iuw < 0.0)
                {
                    printf("uw neg %d, %d, %d\n", k, j, mu);
                    assert(false);
                }
                auto coeffs = besser_coeffs(dtauUw);

                // f64 minS = min(Suw, Sdw);
                // f64 maxS = max(Suw, Sdw);
                // if (SC < minS || SC > maxS)
                // {
                //     printf("%d, %d, %d\n", j, k, mu);

                //     assert(false);
                // }
                // f64 w[2];
                // w2(dtauUw, w);
                // f64 c1 = (Suw - S(j, k)) / dtauUw;
                // f64 IjkLinear = (1.0 - w[0]) * Iuw + w[0] * S(j, k) + w[1] * c1;

                I(k, j) = coeffs.edt * Iuw + coeffs.M * Suw + coeffs.O * SLocal + coeffs.C * SC;

                // f64 tol = (abs(IjkLinear - I(j,k)) / I(j,k));
                // if (tol > 5e-2 && IjkLinear > 1e-9)
                // {
                //     printf("%d, %d, %d\n", j, k, mu);
                //     assert(false);
                // }

                if (computeOperator)
                {
                    Psi(k, j) = coeffs.O + coeffs.C;
                    // f64 PsiLinear = w[0] - w[1] / dtauUw;
                    // f64 psiTol = abs(PsiLinear - Psi(j,k)) / Psi(j,k);
                    // if (psiTol > 5e-2 && PsiLinear > 1e-6)
                    // {
                    //     printf("%d, %d, %d\n", j, k, mu);
                    //     assert(false);
                    // }
                }
            }
            else
            {
#if 0
                auto& substeps = atmos->intersections.substeps[substepIdx];
                f64 Iuw = interp_param(gridData, uwIntersection, I);
                f64 accumDist = 0.0;
                for (const auto& step : substeps.steps)
                {
                    f64 chiUw = interp_param(gridData, uwIntersection, chi);
                    f64 chiDw = interp_param(gridData, step, chi);
                    f64 dtau = 0.5 * (chiUw + chiDw) * step.distance;
                    // f64 dtau = 0.5 * (chiUw + chiDw) * abs(step.distance - accumDist);
                    f64 Suw = interp_param(gridData, uwIntersection, S);
                    f64 Sdw = interp_param(gridData, step, S);

                    f64 w[2];
                    w2(dtau, w);
                    f64 c1 = (Suw - Sdw) / dtau;
                    Iuw = (1.0 - w[0]) * Iuw + w[0] * Sdw + w[1] * c1;
                    accumDist += step.distance;
                    uwIntersection = step;
                }
                f64 chiUw = interp_param(gridData, uwIntersection, chi);
                f64 dtau = 0.5 * (chiUw + chi(k, j)) * (origDistance - accumDist);
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

                    accumDist += dsUw;
                    uwIntersection = step;
                }
                const auto& step = substeps.steps.back();
                f64 dsUw = step.distance;
                f64 dsDw = origDistance - (accumDist + step.distance);
                f64 chiUw = interp_param(gridData, uwIntersection, chi);
                f64 chiLocal = interp_param(gridData, step, chi);
                f64 chiDw = chi(k, j);
                f64 chiC = besser_control_point(dsUw, dsDw, chiUw, chiLocal, chiDw);

                f64 dtau = (1.0 / 3.0) * (chiUw + chiLocal + chiC) * dsUw;

                f64 Suw = interp_param(gridData, uwIntersection, S);
                f64 SLocal = interp_param(gridData, step, S);
                f64 Sdw = S(k, j);
                f64 SC = besser_control_point(dsUw, dsDw, Suw, SLocal, Sdw);
                auto coeffs = besser_coeffs(dtau);

                Iuw = coeffs.edt * Iuw + coeffs.M * Suw + coeffs.O * SLocal + coeffs.C * SC;

                const auto& uw = step;
                const auto& dw = intersections(mu, (int)toObs, k, j).dwIntersection;
                dsUw = dsDw;
                dsDw = dw.distance;
                chiUw = interp_param(gridData, uw, chi);
                chiLocal = chi(k, j);
                chiDw = interp_param(gridData, dw, chi);
                dtau = (1.0 / 3.0) * (chiUw + chiLocal + chiC) * dsUw;

                Suw = interp_param(gridData, uw, S);
                SLocal = S(k, j);
                Sdw = interp_param(gridData, dw, S);
                SC = besser_control_point(dsUw, dsDw, Suw, SLocal, Sdw);
                coeffs = besser_coeffs(dtau);

                I(k, j) = coeffs.edt * Iuw + coeffs.M * Suw + coeffs.O * SLocal + coeffs.C * SC;

                if (computeOperator)
                    Psi(k, j) = coeffs.O + coeffs.C;
#endif
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
            f64 dtau = 0.5 * (chiUw + chi(k, j)) * uwIntersection.distance;
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
            f64 Iuw = interp_param(gridData, uwIntersection, I);
            f64 accumDist = 0.0;
            for (const auto& step : substeps.steps)
            {
                f64 chiUw = interp_param(gridData, uwIntersection, chi);
                f64 chiDw = interp_param(gridData, step, chi);
                f64 dtau = 0.5 * (chiUw + chiDw) * step.distance;
                // f64 dtau = 0.5 * (chiUw + chiDw) * abs(step.distance - accumDist);
                f64 Suw = interp_param(gridData, uwIntersection, S);
                f64 Sdw = interp_param(gridData, step, S);

                f64 w[2];
                w2(dtau, w);
                f64 c1 = (Suw - Sdw) / dtau;
                Iuw = (1.0 - w[0]) * Iuw + w[0] * Sdw + w[1] * c1;
                accumDist += step.distance;
                uwIntersection = step;
            }
            f64 chiUw = interp_param(gridData, uwIntersection, chi);
            f64 dtau = 0.5 * (chiUw + chi(k, j)) * (origDistance - accumDist);
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
}

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

constexpr bool WaitForKeyDown = true;
void interactive_wait()
{
    if (WaitForKeyDown)
    {
        getchar();
    }
}

struct Point
{
    f64 x;
    f64 y;
};

Point grid_location(const Atmosphere& atmos, const IntersectionResult& inter, f64 wrapVal)
{
    f64 x, z;
    switch (inter.axis)
    {
        case InterpolationAxis::X:
        {
            z = atmos.z(int(inter.fractionalZ));
            x = frac_loc_1d(atmos.x, wrapVal, inter.fractionalX);
        } break;
        case InterpolationAxis::Z:
        {
            int xIdx = int(inter.fractionalX);
            if (xIdx == -1)
                xIdx = atmos.x.shape(0) - 1;
            x = atmos.x(xIdx);
            z = frac_idx_1d(atmos.z, inter.fractionalZ);
        } break;
        case InterpolationAxis::None:
        {
            int xIdx = int(inter.fractionalX);
            if (xIdx == -1)
                xIdx = atmos.x.shape(0) - 1;
            x = atmos.x(xIdx);
            z = atmos.z(int(inter.fractionalZ));
        } break;
    }
    return Point{x, z};
}

void plot_intersections(FormalData* fd, int j, int k, int mu, bool toObs)
{
    printf("%d %d\n", j, k);
    plyght().start_frame().plot();
    auto& atmos = *(fd->atmos);
#if 0
    f64 wrapVal = atmos.x(0) - (atmos.x(1) - atmos.x(0));

    f64 zMin = atmos.z(0);
    f64 zMax = atmos.z(atmos.z.shape(0)-1);
    f64 xMin = atmos.x(0);
    f64 xMax = atmos.x(atmos.x.shape(0)-1);
    f64 xs[3];
    f64 ys[3];

    for (int x = 0; x < atmos.x.shape(0); ++x)
    {
        xs[0] = xs[1] = atmos.x(x);
        ys[0] = zMin;
        ys[1] = zMax;
        plyght().line_style("C0")
                .line(xs, ys, 2);
    }

    for (int z = 0; z < atmos.z.shape(0); ++z)
    {
        xs[0] = xMin;
        xs[1] = xMax;
        ys[0] = ys[1] = atmos.z(z);
        plyght().line_style("C1")
                .line(xs, ys, 2);

    }

    int my = mu;
    // for (int my = 0; my < atmos.Nrays; ++my)
    // {
        auto inter = atmos.intersections.intersections(my, (int)toObs, j, k);
        auto uw = inter.uwIntersection;
        auto dw = inter.dwIntersection;
        auto ptUw = grid_location(atmos, uw, wrapVal);
        auto ptDw = grid_location(atmos, dw, wrapVal);
        xs[0] = ptUw.x;
        xs[1] = atmos.x(j);
        xs[2] = ptDw.x;
        ys[0] = ptUw.y;
        ys[1] = atmos.z(k);
        ys[2] = ptDw.y;
        plyght().line_style("C3")
                .line(xs, ys, 3);
        plyght().line_style("or")
                .line(&ptUw.x, &ptUw.y, 1);
        plyght().line_style("og")
                .line(&ptDw.x, &ptDw.y, 1);
        plyght().line_style("ob")
                .line(&atmos.x(j), &atmos.z(k), 1);
        // printf("%e, %e, %e, %e\n", xs[0], ys[0], xs[1], ys[1]);
        printf("%e, %e, %e, %e\n", uw.fractionalX, uw.fractionalZ,
                                   dw.fractionalX, dw.fractionalZ);
    // }
#else
    F64Arr xx(410);
    F64Arr tempLine(82);
    for (int i = 0; i < xx.shape(0); ++i)
        xx(i) = i;
    auto temp = atmos.vturb.reshape(atmos.Nx, atmos.Nz);
    for (int i = 0; i < 82; ++i)
        tempLine(i) = temp(2, i);

    plyght().line(xx.data(), tempLine.data(), 82);
#endif

    plyght().end_frame();
    interactive_wait();
}

void piecewise_parabolic_2d(FormalData* fd, int la, int mu, bool toObs, f64 wav)
{
    auto& atmos = fd->atmos;
    // printf(".............................................\n");
    // printf("%d %d %d %d\n", atmos->Nspace, atmos->Nx, atmos->Ny, atmos->Nz);
    assert(bool(atmos->intersections));

    if (atmos->xLowerBc.type != PERIODIC || atmos->xUpperBc.type != PERIODIC)
    {
        printf("Only supporting periodic x BCs for now!\n");
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
    I.fill(-1.0);
    F64View2D chi = fd->chi.reshape(atmos->Nz, atmos->Nx);
    F64View2D S = fd->S.reshape(atmos->Nz, atmos->Nx);
    F64View2D Psi;
    const bool computeOperator = bool(fd->Psi);
    if (computeOperator)
        Psi = fd->Psi.reshape(atmos->Nz, atmos->Nx);
    Psi.fill(0.0);
    F64View2D temperature = atmos->temperature.reshape(atmos->Nz, atmos->Nx);

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
            // if (toObs)
            //     int BreakHere  = 1;
            int substepIdx = intersections(mu, (int)toObs, k, j).substepIdx;
            auto uwIntersection = intersections(mu, (int)toObs, k, j).uwIntersection;
            auto dwIntersection = intersections(mu, (int)toObs, k, j).dwIntersection;
            f64 origDistance = uwIntersection.distance;
            if (substepIdx < 0)
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

                f64 Iuw = interp_param(gridData, uwIntersection, I);
                if (Iuw < 0.0)
                {
                    printf("uw neg %d, %d, %d\n", j, k, mu);
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
                auto& substeps = atmos->intersections.substeps[substepIdx];
                f64 Iuw = interp_param(gridData, uwIntersection, I);
                f64 accumDist = 0.0;
                for (const auto& step : substeps.steps)
                {
                    f64 chiUw = interp_param(gridData, uwIntersection, chi);
                    f64 chiDw = interp_param(gridData, step, chi);
                    f64 dtau = 0.5 * (chiUw + chiDw) * step.distance;
                    f64 Suw = interp_param(gridData, uwIntersection, S);
                    f64 Sdw = interp_param(gridData, step, S);
                    accumDist += step.distance;

                    f64 w[2];
                    w2(dtau, w);
                    f64 c1 = (Suw - Sdw) / dtau;
                    Iuw = (1.0 - w[0]) * Iuw + w[0] * Sdw + w[1] * c1;
                    uwIntersection = step;
                }
                f64 chiUw = interp_param(gridData, uwIntersection, chi);
                f64 dtau = 0.5 * (chiUw + chi(k, j)) * (origDistance - accumDist);
                f64 Suw = interp_param(gridData, uwIntersection, S);

                f64 w[2];
                w2(dtau, w);
                f64 c1 = (Suw - S(k, j)) / dtau;
                I(k, j) = (1.0 - w[0]) * Iuw + w[0] * S(k, j) + w[1] * c1;

                if (computeOperator)
                    Psi(k, j) = w[0] - w[1] / dtau;
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
            f64 Iuw = interp_param(gridData, uwIntersection, I);
            f64 accumDist = 0.0;
            for (const auto& step : substeps.steps)
            {
                f64 chiUw = interp_param(gridData, uwIntersection, chi);
                f64 chiDw = interp_param(gridData, step, chi);
                f64 dtau = 0.5 * (chiUw + chiDw) * step.distance;
                f64 Suw = interp_param(gridData, uwIntersection, S);
                f64 Sdw = interp_param(gridData, step, S);
                accumDist += step.distance;

                f64 w[2];
                w2(dtau, w);
                f64 c1 = (Suw - Sdw) / dtau;
                Iuw = (1.0 - w[0]) * Iuw + w[0] * Sdw + w[1] * c1;
                uwIntersection = step;
            }
            f64 chiUw = interp_param(gridData, uwIntersection, chi);
            f64 dtau = 0.5 * (chiUw + chi(k, j)) * (origDistance - accumDist);
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
}
}