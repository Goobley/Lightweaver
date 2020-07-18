#include "LwInternal.hpp"
#include "Lightweaver.hpp"
#include "JasPP.hpp"
#include "Utils.hpp"

using namespace LwInternal;

namespace LwInternal
{
struct IntersectionData
{
    F64View x;
    F64View z;
    f64 mux;
    f64 muz;
    bool toObs;
    int xStep;
    int zStep;
};

enum class InterpolationAxis
{
    X,
    Z
};

struct IntersectionResult
{
    InterpolationAxis axis;
    f64 fractionalX;
    f64 fractionalZ;
    f64 distance;
};

// NOTE(cmo): This gives the intersection downwind of the point (xp, zp)
IntersectionResult dw_intersection_2d(const IntersectionData& grid, int xp, int zp)
{
    if (xp + grid.xStep >= grid.x.shape(0) || xp + grid.xStep < 0)
        printf("dw_intersection called for a point going outside the grid");
    if (zp + grid.zStep >= grid.z.shape(0) || zp + grid.zStep < 0)
        printf("dw_intersection called for a point going outside the grid");
    f64 dx = abs(grid.x(xp) - grid.x(xp + grid.xStep));
    f64 dz = abs(grid.z(zp) - grid.z(zp + grid.zStep));
    const f64 dmaxX = abs(1.0 / grid.mux * dx);
    const f64 dmaxZ = abs(1.0 / grid.muz * dz);

    f64 dIntersection = -1.0;
    f64 dxInt = -1.0;
    f64 dzInt = -1.0;
    InterpolationAxis axis;
    if (dmaxX < dmaxZ)
    {
        dIntersection = dmaxX;
        dxInt = dx;
        dzInt = sqrt(square(dIntersection) - square(dmaxX));
        axis = InterpolationAxis::Z;
    }
    else
    {
        dIntersection = dmaxZ;
        dzInt = dz;
        dxInt = sqrt(square(dIntersection) - square(dmaxZ));
        axis = InterpolationAxis::X;
    }
    const f64 fractionalX = xp + grid.xStep * (dxInt / dx);
    const f64 fractionalZ = zp + grid.zStep * (dzInt / dz);

    return {axis, fractionalX, fractionalZ, dIntersection};
}

IntersectionResult uw_intersection_2d(const IntersectionData& grid, int xp, int zp)
{
    if (zp - grid.zStep >= grid.z.shape(0) || zp - grid.zStep < 0)
        printf("uw_intersection called for a point going outside the grid");

    auto dwResult = dw_intersection_2d(grid, xp - grid.xStep, zp - grid.zStep);
    IntersectionResult result(dwResult);
    switch (dwResult.axis)
    {
        case InterpolationAxis::X:
        {
            f64 xm = floor(dwResult.fractionalX);
            result.fractionalX = 1.0 - (dwResult.fractionalX - xm) + xm;
            result.fractionalZ = zp - grid.zStep;
        } break;

        case InterpolationAxis::Z:
        {
            f64 zm = floor(dwResult.fractionalZ);
            result.fractionalZ = 1.0 - (dwResult.fractionalZ - zm) + zm;
            result.fractionalX = xp - grid.xStep;

        } break;
    }

    // NOTE(cmo): Do we need to wrap around and onto the previous z plane?
    // TODO(cmo): Need to handle other BCs here too!
    if ((result.fractionalX == 0 || result.fractionalX == grid.x.shape(0) - 1) &&
       (result.axis == InterpolationAxis::Z))
    {
        f64 dz = abs(grid.z(zp) - grid.z(zp - grid.zStep));
        f64 distance = abs(1.0 / grid.muz * dz);
        f64 dx = sqrt(square(distance) - square(dz));

        f64 x = 0.0;
        if (result.fractionalX == 0)
        {
            f64 toXEnd = abs(grid.x(xp) - grid.x(0));
            x = grid.x(grid.x.shape(0) - 1) - dx - toXEnd;
        }
        else
        {
            f64 toXEnd = abs(grid.x(grid.x.shape(0) - 1) - grid.x(xp));
            x = grid.x(0) + dx - toXEnd;
        }
        int xm = hunt(grid.x, x);
        f64 fractionalX = (x - grid.x(xm)) / (grid.x(xm + 1) - grid.x(xm)) + xm;

        result.axis = InterpolationAxis::X;
        result.fractionalX = fractionalX;
        result.fractionalZ = zp - grid.zStep;
        result.distance = distance;
    }

    return result;
}

f64 interp_param(const IntersectionData& grid, const IntersectionResult& loc,
                 F64View2D param)
{
    // TODO(cmo): This is only linear for now. Probably copy out small range to
    // contiguous buffer in future.
    switch (loc.axis)
    {
        case InterpolationAxis::X:
        {
            int xm = int(loc.fractionalX);
            int xp = xm + 1;
            f64 frac = loc.fractionalX - xm;
            int z = int(loc.fractionalZ);

            f64 result = (1.0 - frac) * param(xm, z) + frac * param(xp, z);
            return result;
        } break;

        case InterpolationAxis::Z:
        {
            int zm = int(loc.fractionalZ);
            int zp = zm + 1;
            f64 frac = loc.fractionalZ - zm;
            int x = int(loc.fractionalZ);

            f64 result = (1.0 - frac) * param(x, zm) + frac * param(x, zp);
            return result;
        } break;
    }
}

void piecewise_linear_2d(FormalData* fd, int la, int mu, bool toObs, f64 wav)
{
    auto& atmos = fd->atmos;
    printf(".............................................\n");
    printf("%d %d %d %d\n", atmos->Nspace, atmos->Nx, atmos->Ny, atmos->Nz);
    printf(".............................................\n");

    if (atmos->xLowerBc.type != PERIODIC || atmos->xUpperBc.type != PERIODIC)
    {
        printf("Only supporting periodic x BCs for now!\n");
        assert(false);
    }

    f64 muz = atmos->muz(mu);
    // NOTE(cmo): We invert the sign of mux, because for muz it is done
    // implicitly, and both need to be the additive inverse so we the ray for
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

    F64View2D I = fd->I.reshape(atmos->Nx, atmos->Nz);
    F64View2D chi = fd->chi.reshape(atmos->Nx, atmos->Nz);
    F64View2D S = fd->I.reshape(atmos->Nx, atmos->Nz);
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
                               toObs,
                               dj,
                               dk};
    int k = kStart;
    // NOTE(cmo): Handle BC in starting plane
    for (int j = jStart; j != jEnd; j += dj)
    {
        I(j, k) = 0.0;

        if (bcType == THERMALISED)
        {
            auto dwIntersection = dw_intersection_2d(gridData, j, k);
            f64 chiDw = interp_param(gridData, dwIntersection, chi);
            f64 dtauDw = 0.5 * dwIntersection.distance * (chi(j, k) + chiDw);
            f64 Bnu[2];
            int Nz = atmos->Nz;
            if (toObs)
            {
                planck_nu(2, &temperature(j, k-1), wav, Bnu);
                I(j, k) = Bnu[1] - (Bnu[0] - Bnu[1]) / dtauDw;
            }
            else
            {
                planck_nu(2, &temperature(j, k), wav, Bnu);
                I(j, k) = Bnu[0] - (Bnu[1] - Bnu[0]) / dtauDw;
            }
        }
        // TODO(cmo): Handle other Bcs!
    }
    k += dk;

    for (; k != kEnd; k += dk)
    {
        for (int j = jStart; j != jEnd; j += dj)
        {
            // TODO(cmo): uw_intersection isn't working rn!
            auto uwIntersection = uw_intersection_2d(gridData, j, k);
            f64 chiUw = interp_param(gridData, uwIntersection, chi);
            f64 dtau = 0.5 * (chiUw + chi(j, k));
            f64 Suw = interp_param(gridData, uwIntersection, S);
            f64 Iuw = interp_param(gridData, uwIntersection, I);

            f64 w[2];
            w2(dtau, w);
            f64 c1 = (Suw - S(j, k)) / dtau;
            I(j, k) = (1.0 - w[0]) * Iuw + w[0] * S(j, k) + w[1] * c1;

            if (computeOperator)
                Psi(j, k) = w[0] - w[1] / dtau;
        }
    }

    if (computeOperator)
        for (int k = 0; k < atmos->Nspace; ++k)
            fd->Psi(k) /= fd->chi(k);
}
}