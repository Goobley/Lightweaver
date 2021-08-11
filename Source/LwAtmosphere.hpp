#ifndef CMO_LW_ATMOSPHERE_HPP
#define CMO_LW_ATMOSPHERE_HPP

#include "CmoArray.hpp"

enum RadiationBc
{
    UNINITIALISED,
    ZERO,
    THERMALISED,
    PERIODIC,
    CALLABLE
};

typedef Jasnah::Array2NonOwn<i32> BcIdxs;

struct AtmosphericBoundaryCondition
{
    RadiationBc type;
    F64Arr3D bcData;
    BcIdxs idxs;

    AtmosphericBoundaryCondition() : type(RadiationBc::ZERO), bcData()
    {}

    AtmosphericBoundaryCondition(RadiationBc typ, int Nwave, int Nmu,
                                 int Nspace, BcIdxs indexVector)
        : type(typ),
          bcData(),
          idxs(indexVector)
    {
        if (type == RadiationBc::CALLABLE)
            bcData = F64Arr3D(Nwave, Nmu, Nspace);
    }

    void set_bc_data(F64View3D data)
    {
        for (int la = 0; la < bcData.shape(0); ++la)
            for (int mu = 0; mu < bcData.shape(1); ++mu)
                for (int k = 0; k < bcData.shape(2); ++k)
                    bcData(la, mu, k) = data(la, mu, k);
    }
};

enum class InterpolationAxis
{
    None,
    X,
    Z
};

template <typename T>
bool approx_equal(T a, T b, T eps)
{
    // https://floating-point-gui.de/errors/comparison/

    const T absA = std::abs(a);
    const T absB = std::abs(b);
    const T diff = std::abs(a - b);

    if (a == b)
    {
        return true;
    }
    else if (a == 0.0 || b == 0.0 || (absA + absB < std::numeric_limits<T>::min()))
    {
        // NOTE(cmo): min in std::numeric_limits is the minimum normalised value.
        return diff < (eps * std::numeric_limits<T>::min());
    }
    else
    {
        return diff / min(absA + absB, std::numeric_limits<T>::max()) < eps;
    }
}

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
    bool periodic;
};


struct IntersectionResult
{
    InterpolationAxis axis;
    f64 fractionalZ;
    f64 fractionalX;
    f64 distance;

    static constexpr f64 PlaneTol = 1e-6;

    IntersectionResult() : axis(InterpolationAxis::None),
                           fractionalZ(0.0),
                           fractionalX(0.0),
                           distance(0.0)
    {}

    IntersectionResult(InterpolationAxis a, f64 fracZ, f64 fracX, f64 dist)
        : axis(a),
          fractionalZ(fracZ),
          fractionalX(fracX),
          distance(dist)
    {
        switch (axis)
        {
            case InterpolationAxis::X:
            {
                f64 x = std::round(fractionalX);
                if (approx_equal(x, fractionalX, PlaneTol))
                {
                    axis = InterpolationAxis::None;
                    fractionalX = x;
                }
            } break;

            case InterpolationAxis::Z:
            {
                f64 z = std::round(fractionalZ);
                if (approx_equal(z, fractionalZ, PlaneTol))
                {
                    axis = InterpolationAxis::None;
                    fractionalZ = z;
                }
            } break;

            default:
                break;
        }


    }
};

struct InterpolationStencil
{
    IntersectionResult uwIntersection;
    IntersectionResult dwIntersection;
    int longCharIdx;
    int substepIdx;
};

struct SubstepIntersections
{
    std::vector<IntersectionResult> steps;
};

struct Intersections
{
    Jasnah::Array4Own<InterpolationStencil> intersections;
    std::vector<SubstepIntersections> substeps;

    void init(int mu, int Nx, int Nz)
    {
        intersections = Jasnah::Array4Own<InterpolationStencil>(mu, 2, Nx, Nz);
        substeps.reserve(2 * mu * Nz);
    }

    explicit operator bool()
    {
        return bool(intersections);
    }
};

struct Atmosphere
{
    int Nspace;
    int Nrays;
    int Ndim;
    int Nx;
    int Ny;
    int Nz;
    int Noutgoing;
    F64View x;
    F64View y;
    F64View z;
    F64View height;
    F64View temperature;
    F64View ne;
    F64View vx;
    F64View vy;
    F64View vz;
    F64View2D vlosMu;
    F64View B;
    F64View gammaB;
    F64View chiB;
    F64View2D cosGamma;
    F64View2D cos2chi;
    F64View2D sin2chi;
    F64View vturb;
    F64View nHTot;
    F64View muz;
    F64View muy;
    F64View mux;
    F64View wmu;

    AtmosphericBoundaryCondition xLowerBc;
    AtmosphericBoundaryCondition xUpperBc;
    AtmosphericBoundaryCondition yLowerBc;
    AtmosphericBoundaryCondition yUpperBc;
    AtmosphericBoundaryCondition zLowerBc;
    AtmosphericBoundaryCondition zUpperBc;

    Intersections intersections;

    void update_projections();
};

void build_intersection_list(Atmosphere* atmos);

#else
#endif