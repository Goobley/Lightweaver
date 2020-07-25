#include "LwInternal.hpp"
#include "Lightweaver.hpp"
#include "JasPP.hpp"
#include "Utils.hpp"

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
IntersectionResult dw_intersection_2d(const IntersectionData& grid, int xp, int zp)
{
    bool wraparound = false;
    Ray ray{grid.x(xp), grid.z(zp), grid.mux, grid.muz};
    f64 tx = 0.0;
    // if (xp == grid.xEnd - grid.xStep)
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

    if (abs(tx) < abs(tx))
    {
        f64 fracX = xp + grid.xStep;
        if (wraparound)
            fracX = directional_modulo_index(grid, fracX);
        f64 fracZ = zp + grid.zStep * (tx / tz);
        return IntersectionResult(InterpolationAxis::Z, fracX, fracZ, tx);
    }
    else if (abs(tz) < abs(tx))
    {
        f64 fracZ = zp + grid.zStep;
        f64 fracX = xp + grid.xStep * (tz / tx);
        if (wraparound)
            fracX = directional_modulo_index(grid, fracX);
        return IntersectionResult(InterpolationAxis::X, fracX, fracZ, tz);
    }
    else
    {
        f64 fracZ = zp + grid.zStep;
        f64 fracX = xp + grid.xStep;
        if (wraparound)
            fracX = directional_modulo_index(grid, fracX);
        return IntersectionResult(InterpolationAxis::None, fracX, fracZ, tz);
    }
}

IntersectionResult uw_intersection_2d(const IntersectionData& grid, int xp, int zp)
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

    if (abs(tx) < abs(tz))
    {
        f64 fracX = xp - grid.xStep;
        f64 fracZ = zp - grid.zStep * (tx / tz);
        return IntersectionResult(InterpolationAxis::Z, fracX, fracZ, tx);
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
            if (xIntersection < 0)
            {
                int xpIdx = 0;
                fracX = 0 - abs(grid.xWrapVal - xIntersection) / abs(grid.xWrapVal - grid.x(xpIdx));
                if (fracX < -1.0)
                {
                    int BreakHere = 1;
                }
            }
            else
            {
                int xpIdx = hunt(grid.x, xIntersection);
                if (grid.x(xpIdx) == xIntersection)
                    return IntersectionResult(InterpolationAxis::None, xpIdx, fracZ, tz);

                xpIdx += 1;
                int xmIdx = xpIdx - 1;
                fracX = xmIdx + abs(grid.x(xmIdx) - xIntersection)
                                / abs(grid.x(xmIdx) - grid.x(xpIdx));
            }
        }
        return IntersectionResult(InterpolationAxis::X, fracX, fracZ, tz);
    }
    else
    {
        f64 fracZ = zp - grid.zStep;
        f64 fracX = xp - grid.xStep;
        return IntersectionResult(InterpolationAxis::None, fracX, fracZ, tz);
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
                x = param.shape(0) - 1;
            int z = int(loc.fractionalZ);

            f64 result = param(x, z);
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

            f64 result = (1.0 - frac) * param(xm, z) + frac * param(xp, z);
            return result;
        } break;

        case InterpolationAxis::Z:
        {
            int zm = int(loc.fractionalZ);
            int zp = zm + 1;
            f64 frac = loc.fractionalZ - zm;
            int x = int(loc.fractionalX);

            f64 result = (1.0 - frac) * param(x, zm) + frac * param(x, zp);
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

    atmos->intersections.init(atmos->Nrays, atmos->Nx, atmos->Nz);
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
                IntersectionResult uw(InterpolationAxis::None, j, k, 0.0);
                auto dw = dw_intersection_2d(gridData, j, k);
                intersections(mu, toObsI, j, k) = InterpolationStencil{uw, dw, -1};
            }
            k += dk;

            for (; k != kEnd; k += dk)
            {
                for (int j = jStart; j != jEnd + dj; j += dj)
                {
                    auto uw = uw_intersection_2d(gridData, j, k);
                    auto dw = dw_intersection_2d(gridData, j, k);
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
                            f64 xInt = frac_idx_1d(gridData.x, uw.fractionalX);
                            Ray ray{xInt, zInt, gridData.mux, gridData.muz};

                            int xp = int(uw.fractionalX + gridData.xStep);
                            while (xp != gridData.xEnd + gridData.xStep)
                            {
                                // Compute intersection with plane at gridData.x(xp)
                                f64 tx = x_plane_intersection(gridData.x(xp), ray);
                                f64 fracZ = zp + gridData.zStep * abs(tx / uw.distance);
                                substeps.steps.emplace_back(IntersectionResult(InterpolationAxis::Z, f64(xp), fracZ, tx));
                                xp += gridData.xStep;
                            }
                        }
                    }
                    intersections(mu, toObsI, j, k) = InterpolationStencil{uw, dw, longCharIdx};
                }
            }

            k = kEnd;
            for (int j = jStart; j != jEnd + dj; j += dj)
            {
                auto uw = uw_intersection_2d(gridData, j, k);
                IntersectionResult dw(InterpolationAxis::None, j, k, 0.0);
                intersections(mu, toObsI, j, k) = InterpolationStencil{uw, dw, -1};
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

    F64View2D I = fd->I.reshape(atmos->Nx, atmos->Nz);
    I.fill(0.0);
    F64View2D chi = fd->chi.reshape(atmos->Nx, atmos->Nz);
    F64View2D S = fd->S.reshape(atmos->Nx, atmos->Nz);
    F64View2D Psi;
    const bool computeOperator = bool(fd->Psi);
    if (computeOperator)
        Psi = fd->Psi.reshape(atmos->Nx, atmos->Nz);
    F64View2D temperature = atmos->temperature.reshape(atmos->Nx, atmos->Nz);

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
            auto dwIntersection = intersections(mu, (int)toObs, j, k).dwIntersection;
            f64 chiDw = interp_param(gridData, dwIntersection, chi);
            f64 dtauDw = 0.5 * abs(dwIntersection.distance) * (chi(j, k) + chiDw);
            f64 temperatureDw = interp_param(gridData, dwIntersection, temperature);
            f64 Bnu[2];
            int Nz = atmos->Nz;
            if (toObs)
            {
                f64 temp[2];
                temp[0] = temperature(j, k);
                temp[1] = temperatureDw;
                planck_nu(2, temp, wav, Bnu);
                I(j, k) = Bnu[1] - (Bnu[0] - Bnu[1]) / dtauDw;
            }
            else
            {
                f64 temp[2];
                temp[0] = temperature(j, k);
                temp[1] = temperatureDw;
                planck_nu(2, temp, wav, Bnu);
                // I(j, k) = Bnu[0] - (Bnu[1] - Bnu[0]) / dtauDw;
                I(j, k) = Bnu[1] - (Bnu[0] - Bnu[1]) / dtauDw;
            }
        }
        // TODO(cmo): Handle other Bcs!
        if (computeOperator)
            Psi(j, k) = 0.0;
    }
    k += dk;

    for (; k != kEnd + dk; k += dk)
    {
        for (int j = jStart; j != jEnd + dj; j += dj)
        {
            // auto uwIntersection = uw_intersection_2d(gridData, j, k);
            int substepIdx = intersections(mu, (int)toObs, j, k).substepIdx;
            auto uwIntersection = intersections(mu, (int)toObs, j, k).uwIntersection;
            f64 origDistance = abs(uwIntersection.distance);
            if (substepIdx < 0)
            {
                f64 chiUw = interp_param(gridData, uwIntersection, chi);
                f64 dtau = 0.5 * (chiUw + chi(j, k)) * abs(uwIntersection.distance);
                f64 Suw = interp_param(gridData, uwIntersection, S);
                f64 Iuw = interp_param(gridData, uwIntersection, I);

                f64 w[2];
                w2(dtau, w);
                f64 c1 = (Suw - S(j, k)) / dtau;
                I(j, k) = (1.0 - w[0]) * Iuw + w[0] * S(j, k) + w[1] * c1;

                if (computeOperator)
                    Psi(j, k) = w[0] - w[1] / dtau;
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
                    f64 dtau = 0.5 * (chiUw + chiDw) * abs(step.distance - accumDist);
                    f64 Suw = interp_param(gridData, uwIntersection, S);
                    f64 Sdw = interp_param(gridData, step, S);
                    accumDist += abs(step.distance);

                    f64 w[2];
                    w2(dtau, w);
                    f64 c1 = (Suw - Sdw) / dtau;
                    Iuw = (1.0 - w[0]) * Iuw + w[0] * Sdw + w[1] * c1;
                    uwIntersection = step;
                }
                f64 chiUw = interp_param(gridData, uwIntersection, chi);
                f64 dtau = 0.5 * (chiUw + chi(j, k)) * abs(origDistance - accumDist);
                f64 Suw = interp_param(gridData, uwIntersection, S);

                f64 w[2];
                w2(dtau, w);
                f64 c1 = (Suw - S(j, k)) / dtau;
                I(j, k) = (1.0 - w[0]) * Iuw + w[0] * S(j, k) + w[1] * c1;

                if (computeOperator)
                    Psi(j, k) = w[0] - w[1] / dtau;
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

    if (dM * dP <= 0)
        return yO;

    f64 yOp = (hM * dP + hP * dM) / (hM + hP);
    f64 cM = yO - 0.5 * hM * yOp;
    f64 cP = yO + 0.5 * hP * yOp;

    // NOTE(cmo): We know dM and dP have the same sign, so if dM is positive, M < O < P
    f64 minYMO = yM;
    f64 maxYMO = yO;
    f64 minYOP = yO;
    f64 maxYOP = yP;
    if (dM < 0)
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
    {
        f64 m = (t * (t * (t * (t * (t * (t * ((140.0 - 18.0 * t) * t - 945.0) + 5400.0) - 25200.0) + 90720.0) - 226800.0) + 302400.0)) / 907200.0;
        f64 o = (t * (t * (t * (t * (t * (t * ((10.0 - t) * t - 90.0) + 720.0) - 5040.0) + 30240.0) - 151200.0) + 604800.0)) / 1814400.0;
        f64 c = (t * (t * (t * (t * (t * (t * ((35.0 - 4.0 * t) * t - 270.0) + 1800.0) - 10080.0) + 45360.0) - 151200.0) + 302400.0)) / 907200.0;
        f64 edt = 1.0 - t + 0.5 * square(t) - cube(t) / 6.0 + t * cube(t) / 24.0 - square(t) * cube(t) / 120.0 + cube(t) * cube(t) / 720.0;
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

    F64View2D I = fd->I.reshape(atmos->Nx, atmos->Nz);
    I.fill(0.0);
    F64View2D chi = fd->chi.reshape(atmos->Nx, atmos->Nz);
    F64View2D S = fd->S.reshape(atmos->Nx, atmos->Nz);
    F64View2D Psi;
    const bool computeOperator = bool(fd->Psi);
    if (computeOperator)
        Psi = fd->Psi.reshape(atmos->Nx, atmos->Nz);
    F64View2D temperature = atmos->temperature.reshape(atmos->Nx, atmos->Nz);

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
            auto dwIntersection = intersections(mu, (int)toObs, j, k).dwIntersection;
            f64 chiDw = interp_param(gridData, dwIntersection, chi);
            f64 dtauDw = 0.5 * abs(dwIntersection.distance) * (chi(j, k) + chiDw);
            f64 temperatureDw = interp_param(gridData, dwIntersection, temperature);
            f64 Bnu[2];
            int Nz = atmos->Nz;
            if (toObs)
            {
                f64 temp[2];
                temp[0] = temperature(j, k);
                temp[1] = temperatureDw;
                planck_nu(2, temp, wav, Bnu);
                I(j, k) = Bnu[1] - (Bnu[0] - Bnu[1]) / dtauDw;
            }
            else
            {
                f64 temp[2];
                temp[0] = temperature(j, k);
                temp[1] = temperatureDw;
                planck_nu(2, temp, wav, Bnu);
                // I(j, k) = Bnu[0] - (Bnu[1] - Bnu[0]) / dtauDw;
                I(j, k) = Bnu[1] - (Bnu[0] - Bnu[1]) / dtauDw;
            }
        }
        // TODO(cmo): Handle other Bcs!
        if (computeOperator)
            Psi(j, k) = 0.0;
    }
    k += dk;

    for (; k != kEnd; k += dk)
    {
        for (int j = jStart; j != jEnd + dj; j += dj)
        {
            int substepIdx = intersections(mu, (int)toObs, j, k).substepIdx;
            auto uwIntersection = intersections(mu, (int)toObs, j, k).uwIntersection;
            auto dwIntersection = intersections(mu, (int)toObs, j, k).uwIntersection;
            f64 origDistance = abs(uwIntersection.distance);
            if (substepIdx < 0)
            {
                f64 dsUw = abs(uwIntersection.distance);
                f64 dsDw = abs(dwIntersection.distance);

                f64 chiDw = interp_param(gridData, dwIntersection, chi);
                f64 chiUw = interp_param(gridData, uwIntersection, chi);
                f64 chiLocal = chi(j, k);
                f64 chiC = besser_control_point(dsUw, dsDw, chiUw, chiLocal, chiDw);
                f64 dtau = (1.0 / 3.0) * (chiUw + chiLocal + chiC) * dsUw;

                f64 Suw = interp_param(gridData, uwIntersection, S);
                f64 Sdw = interp_param(gridData, dwIntersection, S);
                f64 SLocal = S(j, k);
                f64 SC = besser_control_point(dsUw, dsDw, Suw, SLocal, Sdw);

                f64 Iuw = interp_param(gridData, uwIntersection, I);
                auto coeffs = besser_coeffs(dtau);

                I(j, k) = coeffs.edt * Iuw + coeffs.M * Suw + coeffs.O * SLocal + coeffs.C * SC;

                if (computeOperator)
                    Psi(j, k) = coeffs.O + coeffs.C;
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
                    f64 dtau = 0.5 * (chiUw + chiDw) * abs(step.distance - accumDist);
                    f64 Suw = interp_param(gridData, uwIntersection, S);
                    f64 Sdw = interp_param(gridData, step, S);
                    accumDist += abs(step.distance);

                    f64 w[2];
                    w2(dtau, w);
                    f64 c1 = (Suw - Sdw) / dtau;
                    Iuw = (1.0 - w[0]) * Iuw + w[0] * Sdw + w[1] * c1;
                    uwIntersection = step;
                }
                f64 chiUw = interp_param(gridData, uwIntersection, chi);
                f64 dtau = 0.5 * (chiUw + chi(j, k)) * abs(origDistance - accumDist);
                f64 Suw = interp_param(gridData, uwIntersection, S);

                f64 w[2];
                w2(dtau, w);
                f64 c1 = (Suw - S(j, k)) / dtau;
                I(j, k) = (1.0 - w[0]) * Iuw + w[0] * S(j, k) + w[1] * c1;

                if (computeOperator)
                    Psi(j, k) = w[0] - w[1] / dtau;
            }

        }
    }
    k = kEnd;

    for (int j = jStart; j != jEnd + dj; j += dj)
    {
        // auto uwIntersection = uw_intersection_2d(gridData, j, k);
        int substepIdx = intersections(mu, (int)toObs, j, k).substepIdx;
        auto uwIntersection = intersections(mu, (int)toObs, j, k).uwIntersection;
        f64 origDistance = abs(uwIntersection.distance);
        if (substepIdx < 0)
        {
            f64 chiUw = interp_param(gridData, uwIntersection, chi);
            f64 dtau = 0.5 * (chiUw + chi(j, k)) * abs(uwIntersection.distance);
            f64 Suw = interp_param(gridData, uwIntersection, S);
            f64 Iuw = interp_param(gridData, uwIntersection, I);

            f64 w[2];
            w2(dtau, w);
            f64 c1 = (Suw - S(j, k)) / dtau;
            I(j, k) = (1.0 - w[0]) * Iuw + w[0] * S(j, k) + w[1] * c1;

            if (computeOperator)
                Psi(j, k) = w[0] - w[1] / dtau;
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
                f64 dtau = 0.5 * (chiUw + chiDw) * abs(step.distance - accumDist);
                f64 Suw = interp_param(gridData, uwIntersection, S);
                f64 Sdw = interp_param(gridData, step, S);
                accumDist += abs(step.distance);

                f64 w[2];
                w2(dtau, w);
                f64 c1 = (Suw - Sdw) / dtau;
                Iuw = (1.0 - w[0]) * Iuw + w[0] * Sdw + w[1] * c1;
                uwIntersection = step;
            }
            f64 chiUw = interp_param(gridData, uwIntersection, chi);
            f64 dtau = 0.5 * (chiUw + chi(j, k)) * abs(origDistance - accumDist);
            f64 Suw = interp_param(gridData, uwIntersection, S);

            f64 w[2];
            w2(dtau, w);
            f64 c1 = (Suw - S(j, k)) / dtau;
            I(j, k) = (1.0 - w[0]) * Iuw + w[0] * S(j, k) + w[1] * c1;

            if (computeOperator)
                Psi(j, k) = w[0] - w[1] / dtau;
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