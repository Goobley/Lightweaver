#include "LwInternal.hpp"
#include "Lightweaver.hpp"
#include "JasPP.hpp"
#include "Utils.hpp"
// #define PLYGHT_IMPL
// #include "Plyght/Plyght.hpp"

#include <cstdlib>
#include <limits>

using namespace LwInternal;

namespace LwInternal
{

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

f64 grid_fmod_x(const IntersectionData& grid, f64 x)
{
    return grid.x(0) + fmod_pos(x - grid.x(0), grid.x(grid.x.shape(0) - 1) - grid.x(0));
}

IntersectionResult dw_intersection_2d(const IntersectionData& grid, int zp, int xp)
{
    if (xp == grid.xEnd)
    {
        // NOTE(cmo): Periodic grid, wrap-around
        if (grid.periodic)
        {
            xp = grid.xStart;
        }
        // NOTE(cmo): Else return 0 distance, we can only "self-intersect" for
        // non-vertical rays
        // At this point this needs to be caught by logic in the FS.
        else if (abs(grid.muz) != 1.0)
        {
            return IntersectionResult(InterpolationAxis::None, zp, xp, 0.0);
        }
    }

    Ray ray{grid.x(xp), grid.z(zp), grid.mux, grid.muz};
    f64 tx = x_plane_intersection(grid.x(xp + grid.xStep), ray);
    f64 tz = z_plane_intersection(grid.z(zp + grid.zStep), ray);

    if (abs(tx) < abs(tz))
    {
        f64 fracX = xp + grid.xStep;
        f64 fracZ = zp + grid.zStep * (tx / tz);
        return IntersectionResult(InterpolationAxis::Z, fracZ, fracX, tx);
    }
    else if (abs(tz) < abs(tx))
    {
        f64 fracZ = zp + grid.zStep;
        f64 fracX = xp + grid.xStep * (tz / tx);
        return IntersectionResult(InterpolationAxis::X, fracZ, fracX, tz);
    }
    else
    {
        f64 fracX = xp + grid.xStep;
        f64 fracZ = zp + grid.zStep;
        return IntersectionResult(InterpolationAxis::None, fracZ, fracX, tx);
    }
}

IntersectionResult uw_intersection_2d(const IntersectionData& grid, int zp, int xp)
{
    if (xp == grid.xStart)
    {
        // NOTE(cmo): Periodic grid, wrap-around
        if (grid.periodic)
        {
            xp = grid.xEnd;
        }
        // NOTE(cmo): Else return 0 distance, we can only "self-intersect" for
        // non-vertical rays.
        // At this point this needs to be caught by logic in the FS.
        else if (abs(grid.muz) != 1.0)
        {
            return IntersectionResult(InterpolationAxis::None, zp, xp, 0.0);
        }
    }

    Ray ray{grid.x(xp), grid.z(zp), grid.mux, grid.muz};
    f64 tx = x_plane_intersection(grid.x(xp - grid.xStep), ray);
    f64 tz = z_plane_intersection(grid.z(zp - grid.zStep), ray);

    if (abs(tx) < abs(tz))
    {
        f64 fracX = xp - grid.xStep;
        f64 fracZ = zp - grid.zStep * (tx / tz);
        return IntersectionResult(InterpolationAxis::Z, fracZ, fracX, tx);
    }
    else if (abs(tz) < abs(tx))
    {
        f64 fracZ = zp - grid.zStep;
        f64 fracX = xp - grid.xStep * (tz / tx);
        return IntersectionResult(InterpolationAxis::X, fracZ, fracX, tz);
    }
    else
    {
        f64 fracX = xp - grid.xStep;
        f64 fracZ = zp - grid.zStep;
        return IntersectionResult(InterpolationAxis::None, fracZ, fracX, tx);
    }
}

f64 frac_idx(const F64View& param, f64 fracIdx)
{
    int xm = int(fracIdx);
    int xp = xm + 1;

    if (xm == fracIdx)
        return param(xm);

    f64 frac = fracIdx - xm;
    return (1.0 - frac) * param(xm) + frac * param(xp);
}

IntersectionResult uw_intersection_2d_frac_x(const IntersectionData& grid, IntersectionResult start)
{
    if (start.axis != InterpolationAxis::Z)
    {
        printf("Shouldn't be here as z intersection has been hit\n");
        std::abort();
    }

    // NOTE(cmo): Based on the test above, we know, x must be at an intersection
    // i.e. fracX is an integer, and we must not be at an intersection in z, so
    // fracZ is non-integer in R+
    // This can only be reached from code assuming periodic boundaries, so implicit wrap is fine.
    int xp = int(start.fractionalX);
    if (xp == grid.xStart)
        xp = grid.xEnd;

    f64 startX = grid.x(xp);
    f64 startZ = frac_idx(grid.z, start.fractionalZ);

    int zPlaneIdx = If grid.zStep > 0 Then int(start.fractionalZ) Else int(start.fractionalZ) - grid.zStep End;

    Ray ray{startX, startZ, grid.mux, grid.muz};
    f64 tx = x_plane_intersection(grid.x(xp - grid.xStep), ray);
    f64 tz = z_plane_intersection(grid.z(zPlaneIdx), ray);

    if (abs(tx) < abs(tz))
    {
        f64 fracX = xp - grid.xStep;
        // NOTE(cmo): The calculation of fracZ is wrong here, it doesn't take into account starting from a non-integer fracZ
        // this sounds correct on paper...
        // it's not in practice
        // f64 fracThroughZ = start.fractionalZ - int(start.fractionalZ);
        // f64 fracZ = start.fractionalZ - (1.0 - fracThroughZ) * grid.zStep * (tx / tz);
        // The below appears correct, and the intent of the above (but not wrong :P)
        f64 fracThroughZ = abs(zPlaneIdx - start.fractionalZ);
        f64 fracZ = start.fractionalZ - fracThroughZ * grid.zStep * (tx / tz);
        return IntersectionResult(InterpolationAxis::Z, fracZ, fracX, tx);
    }
    else if (abs(tz) < abs(tx))
    {
        f64 fracZ = zPlaneIdx;
        f64 fracX = xp - grid.xStep * (tz / tx);
        return IntersectionResult(InterpolationAxis::X, fracZ, fracX, tz);
    }
    else
    {
        f64 fracX = xp - grid.xStep;
        f64 fracZ = zPlaneIdx;
        return IntersectionResult(InterpolationAxis::None, fracZ, fracX, tx);
    }
}


f64 interp_linear_2d(const IntersectionData& grid, const IntersectionResult& loc,
                     const F64View2D& param)
{
    // TODO(cmo): This is only linear for now. Probably copy out small range to
    // contiguous buffer in future.
    switch (loc.axis)
    {
        case InterpolationAxis::None:
        {
            int x = int(loc.fractionalX);
            int z = int(loc.fractionalZ);

            f64 result = param(z, x);
            return result;
        } break;

        case InterpolationAxis::X:
        {
            int xm, xp, z;
            f64 frac;
            xm = int(loc.fractionalX);
            xp = xm + 1;
            frac = loc.fractionalX - xm;
            z = int(loc.fractionalZ);

            f64 result = (1.0 - frac) * param(z, xm) + frac * param(z, xp);
            return result;
        } break;

        case InterpolationAxis::Z:
        {
            int zm = int(loc.fractionalZ);
            int zp = zm + 1;
            f64 frac = loc.fractionalZ - zm;
            int x = int(loc.fractionalX);

            f64 result = (1.0 - frac) * param(zm, x) + frac * param(zp, x);
            return result;
        } break;

        default:
        {
            // UNREACHABLE
            return 0.0;
        } break;
    }
}

f64 besser_control_point(f64 hM, f64 hP, f64 yM, f64 yO, f64 yP)
{
    const f64 deltaMO = (yO - yM);
    const f64 dM = (yO - yM) / hM;
    const f64 dP = (yP - yO) / hP;

    if (dM * dP <= 0.0)
        return yO;

    f64 yOp = (hM * dP + hP * dM) / (hM + hP);
    f64 cM = yO - 0.5 * hM * yOp;
    f64 cP = yO + 0.5 * hP * yOp;

    // NOTE(cmo): We know dM and dP have the same sign, so if deltaMO is positive, M < O < P
    f64 minYMO = yM;
    f64 maxYMO = yO;
    f64 minYOP = yO;
    f64 maxYOP = yP;
    if (deltaMO < 0.0)
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


f64 interp_besser_2d(const IntersectionData& grid, const IntersectionResult& loc,
                     const F64View2D& param)
{
    switch (loc.axis)
    {
        case InterpolationAxis::None:
        {
            int x = int(loc.fractionalX);
            int z = int(loc.fractionalZ);

            f64 result = param(z, x);
            return result;
        } break;

        case InterpolationAxis::X:
        {
            int xm, xp, z;
            f64 frac;
            xm = int(loc.fractionalX);
            xp = xm + 1;
            frac = loc.fractionalX - xm;
            z = int(loc.fractionalZ);

            if (grid.xStep < 0)
            {
                // NOTE(cmo): xStep is negative so our upwind 3 point stencil is xm+2, xm+1, xm
                // M = xm
                // O = xp
                // P = xp - dx
                if (xp == grid.xStart)
                {
                    f64 result = (1.0 - frac) * param(z, xm) + frac * param(z, xp);
                    return result;
                }
                f64 hM = grid.x(xp) - grid.x(xm);
                f64 hP = grid.x(xp-grid.xStep) - grid.x(xp);
                f64 yM = param(z, xm);
                f64 yO = param(z, xp);
                f64 yP = param(z, xp-grid.xStep);

                f64 cM = besser_control_point(hM, hP, yM, yO, yP);
                f64 u = frac;

                f64 result = square(1.0 - u) * yM + 2.0 * u * (1.0 - u) * cM + square(u) * yO;
                return result;
            }
            else
            {
                // NOTE(cmo): Stencil is xm-1, xm, xm+1
                // M = xp
                // O = xm
                // P = xm - dx
                if (xm == grid.xStart)
                {
                    f64 result = (1.0 - frac) * param(z, xm) + frac * param(z, xp);
                    return result;
                }
                f64 hM = grid.x(xm) - grid.x(xp);
                f64 hP = grid.x(xm-grid.xStep) - grid.x(xm);
                f64 yM = param(z, xp);
                f64 yO = param(z, xm);
                f64 yP = param(z, xm-grid.xStep);

                f64 cM = besser_control_point(hM, hP, yM, yO, yP);
                f64 u = 1.0 - frac;

                f64 result = square(1.0 - u) * yM + 2.0 * u * (1.0 - u) * cM + square(u) * yO;
                return result;
            }

        } break;

        case InterpolationAxis::Z:
        {
            int zm = int(loc.fractionalZ);
            int zp = zm + 1;
            f64 frac = loc.fractionalZ - zm;
            int x = int(loc.fractionalX);

            if (grid.zStep < 0)
            {
                // NOTE(cmo): zStep is negative so our upwind 3 point stencil is zm+2, zm+1, zm
                // M = zm
                // O = zp
                // P = zp - dz
                if (zp == grid.zStart)
                {
                    f64 result = (1.0 - frac) * param(zm, x) + frac * param(zp, x);
                    return result;
                }
                f64 hM = grid.z(zp) - grid.z(zm);
                f64 hP = grid.z(zp-grid.zStep) - grid.z(zp);
                f64 yM = param(zm, x);
                f64 yO = param(zp, x);
                f64 yP = param(zp-grid.zStep, x);

                f64 cM = besser_control_point(hM, hP, yM, yO, yP);
                f64 u = frac;

                f64 result = square(1.0 - u) * yM + 2.0 * u * (1.0 - u) * cM + square(u) * yO;
                return result;
            }
            else
            {
                // NOTE(cmo): Stencil is zm-1, zm, zm+1
                // M = zp
                // O = zm
                // P = zm - dz
                if (zm == grid.zStart)
                {
                    f64 result = (1.0 - frac) * param(zm, x) + frac * param(zp, x);
                    return result;
                }
                f64 hM = grid.z(zm) - grid.z(zp);
                f64 hP = grid.z(zm-grid.zStep) - grid.z(zm);
                f64 yM = param(zp, x);
                f64 yO = param(zm, x);
                f64 yP = param(zm-grid.zStep, x);

                f64 cM = besser_control_point(hM, hP, yM, yO, yP);
                f64 u = 1.0 - frac;

                f64 result = square(1.0 - u) * yM + 2.0 * u * (1.0 - u) * cM + square(u) * yO;
                return result;
            }
        } break;


        default:
        {
            // UNREACHABLE
            return 0.0;
        } break;
    }
}


void piecewise_linear_2d(FormalData* fd, int la, int mu, bool toObs, const F64View1D& wave)
{
    const f64 wav = wave(la);
    auto& atmos = fd->atmos;
    auto& interp_param = fd->interp;
    // printf(".............................................\n");
    // printf("%d %d %d %d\n", atmos->Nspace, atmos->Nx, atmos->Ny, atmos->Nz);
    assert(bool(atmos->intersections));

    bool periodic = false;
    if (atmos->xLowerBc.type == PERIODIC && atmos->xUpperBc.type == PERIODIC)
    {
        periodic = true;
    }
    else if (! (atmos->xLowerBc.type == CALLABLE && atmos->xUpperBc.type == CALLABLE))
    {
        printf("Mixed boundary types not supported on x-axis, and must be CALLABLE or PERIODIC!\n");
        std::abort();
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

    if (!periodic)
    {
        int toObsI = int(toObs);
        // NOTE(cmo): The idea behind this indexing scheme is that we still want
        // Nrays to represent half the unit sphere, but we want the half split
        // by the y-z plane, rather than the half split by the x-y plane that
        // toObs splits by by default.
        // int muIdx = (mu % (atmos->Nrays / 2)) * 2 + toObsI;
        if (mux > 0)
        {
            for (int k = 0; k < atmos->Nz; ++k)
            {
                int muIdx = atmos->xLowerBc.idxs(mu, toObsI);
                if (muIdx == -1)
                {
                    // NOTE(cmo): This shouldn't be possible, so I won't try to
                    // recover.
                    printf("Error in boundary condition indexing\n");
                    assert(false);
                }
                else
                    I(k, 0) = atmos->xLowerBc.bcData(la, muIdx, k);
                if (computeOperator)
                    Psi(k, 0) = 0.0;
            }
        }
        else if (mux < 0)
        {
            for (int k = 0; k < atmos->Nz; ++k)
            {
                int muIdx = atmos->xUpperBc.idxs(mu, toObsI);
                if (muIdx == -1)
                {
                    // NOTE(cmo): This shouldn't be possible, so I won't try to
                    // recover.
                    printf("Error in boundary condition indexing\n");
                    assert(false);
                }
                else
                    I(k, atmos->Nx-1) = atmos->xUpperBc.bcData(la, muIdx, k);
                if (computeOperator)
                    Psi(k, atmos->Nx-1) = 0.0;
            }
        }

        // NOTE(cmo): Account for fixed BCs -> we don't want to touch I(k, jStart),
        // unless mux == 0, i.e. muz == 1.0, in which case we still need to trace
        // the vertical
        if (mux != 0.0)
            jStart += dj;
    }

    auto& currentBc = If toObs
                      Then atmos->zLowerBc
                      Else atmos->zUpperBc End;
    RadiationBc bcType = currentBc.type;
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
                               kEnd,
                               periodic};

    auto& intersections = atmos->intersections.intersections;
    int k = kStart;
    // NOTE(cmo): Handle BC in starting plane
    for (int j = jStart; j != jEnd + dj; j += dj)
    {
        I(j, k) = 0.0;

        switch (bcType)
        {
            case THERMALISED:
            {
                // NOTE(cmo): This becomes a problem for j == jEnd with
                // non-periodic boundary conditions; RH appears to just ignore
                // this column when using fixed boundaries, which makes sense,
                // since we can't truly accumulate the Lambda operator anyway.
                // Nevertheless, this gradient is likely to be similar from the
                // preceeding point to this one, so let's recompute that and use
                // it. Admittedly this is a bit of a HACK, but is likely to
                // introduce very little error.
                auto dwIntersection = intersections(mu, (int)toObs, k, j).dwIntersection;
                if ((!periodic) && (j == jEnd) && (mux != 0.0))
                {
                    // NOTE(cmo): This assumes the atmosphere is wider than one
                    // column... but it has to be really.
                    dwIntersection = intersections(mu, (int)toObs, k, j-dj).dwIntersection;
                }
                f64 chiDw = interp_param(gridData, dwIntersection, chi);
                f64 dtauDw = 0.5 * abs(dwIntersection.distance) * (chi(k, j) + chiDw);
                f64 temperatureDw = interp_param(gridData, dwIntersection, temperature);
                f64 Bnu[2];
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
            } break;

            case CALLABLE:
            {
                int muIdx = currentBc.idxs(mu, int(toObs));
                if (muIdx == -1)
                {
                    // NOTE(cmo): This shouldn't be possible, so I won't try to
                    // recover.
                    printf("Error in boundary condition indexing\n");
                    assert(false);
                }
                I(k, j) = currentBc.bcData(la, muIdx, j);
            } break;

            case ZERO: break;

            default:
            {
                printf("Unsupported z-boundary type");
            } break;
        }

        if (computeOperator)
            Psi(k, j) = 0.0;
    }
    k += dk;

    for (; k != kEnd + dk; k += dk)
    {
        for (int j = jStart; j != jEnd + dj; j += dj)
        {
            // auto uwIntersection = uw_intersection_2d(gridData, j, k);
            int longCharIdx = intersections(mu, (int)toObs, k, j).longCharIdx;
            auto uwIntersection = intersections(mu, (int)toObs, k, j).uwIntersection;
            if (longCharIdx < 0)
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
                auto& substeps = atmos->intersections.substeps[longCharIdx];
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
            }

        }
    }


    if (computeOperator)
    {
        for (int k = 0; k < atmos->Nspace; ++k)
            fd->Psi(k) /= fd->chi(k);

    }
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
    // TODO(cmo): Possibly reduce the number of terms in these a bit, they could
    // end up being more costly than system exp
    if (t < 0.14)
    // if (t < 0.05)
    {
        f64 m = (t * (t * (t * (t * (t * (t * ((140.0 - 18.0 * t) * t - 945.0) + 5400.0) - 25200.0) + 90720.0) - 226800.0) + 302400.0)) / 907200.0;
        f64 o = (t * (t * (t * (t * (t * (t * ((10.0 - t) * t - 90.0) + 720.0) - 5040.0) + 30240.0) - 151200.0) + 604800.0)) / 1814400.0;
        f64 c = (t * (t * (t * (t * (t * (t * ((35.0 - 4.0 * t) * t - 270.0) + 1800.0) - 10080.0) + 45360.0) - 151200.0) + 302400.0)) / 907200.0;
        // f64 edt = 1.0 - t + 0.5 * square(t) - cube(t) / 6.0 + t * cube(t) / 24.0 - square(t) * cube(t) / 120.0 + cube(t) * cube(t) / 720.0 - cube(t) * cube(t) * t / 5040.0;
        f64 edt = (t * (t * (t * (t * (t * (t * ((t / 40320.0 - 1.0 / 5040.0) * t + 1.0 / 720) - 1.0 / 120.0) + 1.0 / 24.0) - 1.0 / 6.0) + 1.0 / 2.0) - 1.0)) + 1.0;
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

void piecewise_besser_2d(FormalData* fd, int la, int mu, bool toObs, const F64View1D& wave)
{
    // NOTE(cmo): Implmentation of BESSER method following Stepan & Trujillo Bueno (2013) A&A, 557, A143. This is a 2D variant for scalar intensity, but following the same limiting principles, used to integrate chi and S.
    const f64 wav = wave(la);
    auto& atmos = fd->atmos;
    auto& interp_param = fd->interp;
    // printf(".............................................\n");
    // printf("%d %d %d %d\n", atmos->Nspace, atmos->Nx, atmos->Ny, atmos->Nz);
    assert(bool(atmos->intersections));

    bool periodic = false;
    if (atmos->xLowerBc.type == PERIODIC && atmos->xUpperBc.type == PERIODIC)
    {
        periodic = true;
    }
    else if (! (atmos->xLowerBc.type == CALLABLE && atmos->xUpperBc.type == CALLABLE))
    {
        printf("Mixed boundary types not supported on x-axis, and must be CALLABLE or PERIODIC!\n");
        std::abort();
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
    {
        Psi = fd->Psi.reshape(atmos->Nz, atmos->Nx);
        Psi.fill(0.0);
    }
    F64View2D temperature = atmos->temperature.reshape(atmos->Nz, atmos->Nx);

    if (!periodic)
    {
        int toObsI = int(toObs);
        // NOTE(cmo): The idea behind this indexing scheme is that we still want
        // Nrays to represent half the unit sphere, but we want the half split
        // by the y-z plane, rather than the half split by the x-y plane that
        // toObs splits by by default.
        // int muIdx = (mu % (atmos->Nrays / 2)) * 2 + toObsI;

        if (mux > 0)
        {
            for (int k = 0; k < atmos->Nz; ++k)
            {
                int muIdx = atmos->xLowerBc.idxs(mu, toObsI);
                if (muIdx == -1)
                {
                    // NOTE(cmo): This shouldn't be possible, so I won't try to
                    // recover.
                    printf("Error in boundary condition indexing\n");
                    assert(false);
                }
                I(k, 0) = atmos->xLowerBc.bcData(la, muIdx, k);
                if (computeOperator)
                    Psi(k, 0) = 0.0;
            }
        }
        else if (mux < 0)
        {
            for (int k = 0; k < atmos->Nz; ++k)
            {
                int muIdx = atmos->xUpperBc.idxs(mu, toObsI);
                if (muIdx == -1)
                {
                    // NOTE(cmo): This shouldn't be possible, so I won't try to
                    // recover.
                    printf("Error in boundary condition indexing\n");
                    assert(false);
                }
                I(k, atmos->Nx-1) = atmos->xUpperBc.bcData(la, muIdx, k);
                if (computeOperator)
                    Psi(k, atmos->Nx-1) = 0.0;
            }
        }

        // NOTE(cmo): Account for fixed BCs -> we don't want to touch I(k, jStart),
        // unless mux == 0, i.e. muz == 1.0, in which case we still need to trace
        // the vertical
        if (mux != 0.0)
            jStart += dj;
    }

    auto& currentBc = If toObs
                      Then atmos->zLowerBc
                      Else atmos->zUpperBc End;
    RadiationBc bcType = currentBc.type;
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
                               kEnd,
                               periodic};

    auto& intersections = atmos->intersections.intersections;
    int k = kStart;
    // NOTE(cmo): Handle BC in starting plane
    for (int j = jStart; j != jEnd + dj; j += dj)
    {
        I(k, j) = 0.0;

        switch (bcType)
        {
            case THERMALISED:
            {
                // NOTE(cmo): This becomes a problem for j == jEnd with
                // non-periodic boundary conditions; RH appears to just ignore
                // this column when using fixed boundaries, which makes sense,
                // since we can't truly accumulate the Lambda operator anyway.
                // Nevertheless, this gradient is likely to be similar from the
                // preceeding point to this one, so let's recompute that and use
                // it. Admittedly this is a bit of a HACK, but is likely to
                // introduce very little error.
                auto dwIntersection = intersections(mu, (int)toObs, k, j).dwIntersection;
                if ((!periodic) && (j == jEnd) && (mux != 0.0))
                {
                    // NOTE(cmo): This assumes the atmosphere is wider than one
                    // column... but it has to be really.
                    dwIntersection = intersections(mu, (int)toObs, k, j-dj).dwIntersection;
                }
                f64 chiDw = interp_param(gridData, dwIntersection, chi);
                f64 dtauDw = 0.5 * abs(dwIntersection.distance) * (chi(k, j) + chiDw);
                f64 temperatureDw = interp_param(gridData, dwIntersection, temperature);
                f64 Bnu[2];
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
            } break;

            case CALLABLE:
            {
                int muIdx = currentBc.idxs(mu, int(toObs));
                if (muIdx == -1)
                {
                    // NOTE(cmo): This shouldn't be possible, so I won't try to
                    // recover.
                    printf("Error in boundary condition indexing\n");
                    printf("%d, %d\n", mu, toObs);
                    assert(false);
                }

                I(k, j) = currentBc.bcData(la, muIdx, j);
            } break;

            case ZERO: break;

            default:
            {
                printf("Unsupported z-boundary type");
            } break;
        }

        if (computeOperator)
            Psi(k, j) = 0.0;
    }
    k += dk;

    for (; k != kEnd; k += dk)
    {
        for (int j = jStart; j != jEnd + dj; j += dj)
        {
            int longCharIdx = intersections(mu, (int)toObs, k, j).longCharIdx;
            auto uwIntersection = intersections(mu, (int)toObs, k, j).uwIntersection;
            auto dwIntersection = intersections(mu, (int)toObs, k, j).dwIntersection;
            if (longCharIdx < 0)
            {
                f64 dsUw = uwIntersection.distance;
                f64 dsDw = dwIntersection.distance;
                if (dsDw == 0.0)
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
                    auto coeffs = besser_coeffs(dtauUw);

                    I(k, j) = coeffs.edt * Iuw + coeffs.M * Suw + coeffs.O * SLocal + coeffs.C * SC;

                    if (computeOperator)
                    {
                        Psi(k, j) = coeffs.O + coeffs.C;
                    }
                }
            }
            else
            {
                auto& substeps = atmos->intersections.substeps[longCharIdx];
                f64 Iuw = interp_param(gridData, substeps.steps[0], I);
                for (int stepIdx = 1; stepIdx < substeps.steps.size()-1; ++stepIdx)
                {
                    const auto& uwStep = substeps.steps[stepIdx-1];
                    const auto& step = substeps.steps[stepIdx];
                    const auto& dwStep = substeps.steps[stepIdx+1];
                    f64 dsUw = uwStep.distance;
                    f64 dsDw = dwStep.distance;

                    f64 chiUw = interp_param(gridData, uwStep, chi);
                    f64 chiLocal = interp_param(gridData, step, chi);
                    f64 chiDw = interp_param(gridData, dwStep, chi);
                    f64 chiC = besser_control_point(dsUw, dsDw, chiUw, chiLocal, chiDw);

                    f64 dtauUw = (1.0 / 3.0) * (chiUw + chiLocal + chiC) * dsUw;
                    f64 dtauDw = (0.5) * (chiLocal + chiDw) * dsDw;

                    f64 Suw = interp_param(gridData, uwStep, S);
                    f64 SLocal = interp_param(gridData, step, S);
                    f64 Sdw = interp_param(gridData, dwStep, S);
                    f64 SC = besser_control_point(dtauUw, dtauDw, Suw, SLocal, Sdw);
                    auto coeffs = besser_coeffs(dtauUw);

                    Iuw = coeffs.edt * Iuw + coeffs.M * Suw + coeffs.O * SLocal + coeffs.C * SC;
                }
                int stepIdx = substeps.steps.size() - 1;
                const auto& uwStep = substeps.steps[stepIdx - 1];
                const auto& step = substeps.steps[stepIdx]; // same as uwIntersection
                f64 dsUw = uwStep.distance;
                f64 dsDw = step.distance;
                f64 chiUw = interp_param(gridData, uwStep, chi);
                f64 chiLocal = interp_param(gridData, step, chi);
                f64 chiDw = chi(k, j);
                f64 chiC = besser_control_point(dsUw, dsDw, chiUw, chiLocal, chiDw);

                f64 dtauUw = (1.0 / 3.0) * (chiUw + chiLocal + chiC) * dsUw;
                f64 dtauDw = (0.5) * (chiLocal + chiDw) * dsDw;

                f64 Suw = interp_param(gridData, uwStep, S);
                f64 SLocal = interp_param(gridData, step, S);
                f64 Sdw = S(k, j);
                f64 SC = besser_control_point(dtauUw, dtauDw, Suw, SLocal, Sdw);
                auto coeffs = besser_coeffs(dtauUw);

                Iuw = coeffs.edt * Iuw + coeffs.M * Suw + coeffs.O * SLocal + coeffs.C * SC;

                const auto& uw = step;
                const auto& dw = dwIntersection;
                dsUw = uw.distance;
                dsDw = dw.distance;
                if (dsDw == 0.0)
                {
                    f64 chiUw = interp_param(gridData, uw, chi);
                    f64 dtau = 0.5 * (chiUw + chi(k, j)) * dsUw;
                    f64 Suw = interp_param(gridData, uw, S);

                    f64 w[2];
                    w2(dtau, w);
                    f64 c1 = (Suw - S(k, j)) / dtau;
                    I(k, j) = (1.0 - w[0]) * Iuw + w[0] * S(k, j) + w[1] * c1;

                    if (computeOperator)
                        Psi(k, j) = w[0] - w[1] / dtau;
                }
                else
                {
                    chiUw = interp_param(gridData, uw, chi);
                    chiLocal = chi(k, j);
                    chiDw = interp_param(gridData, dw, chi);
                    dtauUw = (1.0 / 3.0) * (chiUw + chiLocal + chiC) * dsUw;
                    dtauDw = (0.5) * (chiLocal + chiDw) * dsDw;

                    Suw = interp_param(gridData, uw, S);
                    SLocal = S(k, j);
                    Sdw = interp_param(gridData, dw, S);
                    SC = besser_control_point(dtauUw, dtauDw, Suw, SLocal, Sdw);
                    coeffs = besser_coeffs(dtauUw);

                    I(k, j) = coeffs.edt * Iuw + coeffs.M * Suw + coeffs.O * SLocal + coeffs.C * SC;

                    if (computeOperator)
                        Psi(k, j) = coeffs.O + coeffs.C;
                }
            }

        }
    }
    k = kEnd;

    for (int j = jStart; j != jEnd + dj; j += dj)
    {
        int longCharIdx = intersections(mu, (int)toObs, k, j).longCharIdx;
        auto uwIntersection = intersections(mu, (int)toObs, k, j).uwIntersection;
        if (longCharIdx < 0)
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
            auto& substeps = atmos->intersections.substeps[longCharIdx];
            f64 Iuw = interp_param(gridData, substeps.steps[0], I);
            for (int stepIdx = 1; stepIdx < substeps.steps.size()-1; ++stepIdx)
            {
                const auto& uwStep = substeps.steps[stepIdx-1];
                const auto& step = substeps.steps[stepIdx];
                const auto& dwStep = substeps.steps[stepIdx+1];
                f64 dsUw = uwStep.distance;
                f64 dsDw = dwStep.distance;

                f64 chiUw = interp_param(gridData, uwStep, chi);
                f64 chiLocal = interp_param(gridData, step, chi);
                f64 chiDw = interp_param(gridData, dwStep, chi);
                f64 chiC = besser_control_point(dsUw, dsDw, chiUw, chiLocal, chiDw);

                f64 dtauUw = (1.0 / 3.0) * (chiUw + chiLocal + chiC) * dsUw;
                f64 dtauDw = (0.5) * (chiLocal + chiDw) * dsDw;

                f64 Suw = interp_param(gridData, uwStep, S);
                f64 SLocal = interp_param(gridData, step, S);
                f64 Sdw = interp_param(gridData, dwStep, S);
                f64 SC = besser_control_point(dtauUw, dtauDw, Suw, SLocal, Sdw);
                auto coeffs = besser_coeffs(dtauUw);

                Iuw = coeffs.edt * Iuw + coeffs.M * Suw + coeffs.O * SLocal + coeffs.C * SC;
            }
            int stepIdx = substeps.steps.size() - 1;
            const auto& uwStep = substeps.steps[stepIdx - 1];
            const auto& step = substeps.steps[stepIdx]; // same as uwIntersection
            f64 dsUw = uwStep.distance;
            f64 dsDw = step.distance;
            f64 chiUw = interp_param(gridData, uwStep, chi);
            f64 chiLocal = interp_param(gridData, step, chi);
            f64 chiDw = chi(k, j);
            f64 chiC = besser_control_point(dsUw, dsDw, chiUw, chiLocal, chiDw);

            f64 dtauUw = (1.0 / 3.0) * (chiUw + chiLocal + chiC) * dsUw;
            f64 dtauDw = (0.5) * (chiLocal + chiDw) * dsDw;

            f64 Suw = interp_param(gridData, uwStep, S);
            f64 SLocal = interp_param(gridData, step, S);
            f64 Sdw = S(k, j);
            f64 SC = besser_control_point(dtauUw, dtauDw, Suw, SLocal, Sdw);
            auto coeffs = besser_coeffs(dtauUw);

            Iuw = coeffs.edt * Iuw + coeffs.M * Suw + coeffs.O * SLocal + coeffs.C * SC;

            chiUw = interp_param(gridData, uwIntersection, chi);
            f64 dtau = 0.5 * (chiUw + chi(k, j)) * (uwIntersection.distance);
            Suw = interp_param(gridData, uwIntersection, S);

            f64 w[2];
            w2(dtau, w);
            f64 c1 = (Suw - S(k, j)) / dtau;
            I(k, j) = (1.0 - w[0]) * Iuw + w[0] * S(k, j) + w[1] * c1;

            if (computeOperator)
                Psi(k, j) = w[0] - w[1] / dtau;
        }

    }


    if (computeOperator)
    {
        for (int k = 0; k < atmos->Nspace; ++k)
            fd->Psi(k) /= fd->chi(k);

    }
}

}

void build_intersection_list(Atmosphere* atmos)
{
    if (atmos->Ndim != 2)
        return;

    bool periodic = false;
    if (atmos->xLowerBc.type == PERIODIC && atmos->xUpperBc.type == PERIODIC)
    {
        periodic = true;
    }
    else if (! (atmos->xLowerBc.type == CALLABLE && atmos->xUpperBc.type == CALLABLE))
    {
        printf("Mixed boundary types not supported on x-axis!\n");
        std::abort();
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
                                    kEnd,
                                    periodic};

            int k = kStart;
            // NOTE(cmo): Handle BC in starting plane
            for (int j = jStart; j != jEnd + dj; j += dj)
            {
                IntersectionResult uw(InterpolationAxis::None, k, j, 0.0);
                auto dw = dw_intersection_2d(gridData, k, j);
                dw.distance = abs(dw.distance);

                if ((!periodic) && (j == jEnd) && (mux != 0.0))
                    dw = IntersectionResult(InterpolationAxis::None, k, j, 0.0);

                intersections(mu, toObsI, k, j) = InterpolationStencil{uw, dw, -1, -1};
            }
            k += dk;

            for (; k != kEnd + dk; k += dk)
            {
                for (int j = jStart; j != jEnd + dj; j += dj)
                {
                    auto uw = uw_intersection_2d(gridData, k, j);
                    uw.distance = abs(uw.distance);
                    bool longChar = (periodic &&
                                     j == jStart &&
                                     uw.axis == InterpolationAxis::Z);
                    int longCharIdx = -1;
                    int substepIdx = -1;

                    if (longChar)
                    {
                        atmos->intersections.substeps.emplace_back(SubstepIntersections{});
                        auto& substeps = atmos->intersections.substeps.back();
                        longCharIdx = atmos->intersections.substeps.size() - 1;

                        auto locToUpwind = uw;
                        while (true)
                        {
                            auto uuw = uw_intersection_2d_frac_x(gridData, locToUpwind);
                            uuw.distance = abs(uuw.distance);
                            substeps.steps.emplace_back(uuw);
                            if (uuw.axis != InterpolationAxis::Z)
                            {
                                // NOTE(cmo): As we filled dw->uw, it makes
                                // sense to reverse the vector now.

                                // TODO(cmo): Make this a useful contiguous buffer for solving the RTE along
                                std::reverse(std::begin(substeps.steps), std::end(substeps.steps));
                                substeps.steps.emplace_back(uw);
                                break;
                            }
                            locToUpwind = uuw;
                        }
                    }
                    IntersectionResult dw;
                    if (k != kEnd)
                    {
                        if ((!periodic) && (j == jEnd) && (mux != 0.0))
                        {
                            dw = IntersectionResult(InterpolationAxis::None, k, j, 0.0);
                        }
                        else
                        {
                            dw = dw_intersection_2d(gridData, k, j);
                            dw.distance = abs(dw.distance);
                        }
                    }
                    else
                    {
                        dw = IntersectionResult(InterpolationAxis::None, k, j, 0.0);
                    }
                    intersections(mu, toObsI, k, j) = InterpolationStencil{uw, dw,
                                                                           longCharIdx, substepIdx};
                }
            }
        }
    }
}
