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

struct AtmosphericBoundaryCondition
{
    RadiationBc type;
    F64Arr2D bcData;

    AtmosphericBoundaryCondition() : type(RadiationBc::ZERO), bcData()
    {}

    AtmosphericBoundaryCondition(RadiationBc typ, int Nmu, int Nwave)
        : type(typ),
          bcData()
    {
        if (type == RadiationBc::CALLABLE)
            bcData = F64Arr2D(Nmu, Nwave);
    }

    void set_bc_data(F64View2D data)
    {
        for (int mu = 0; mu < bcData.shape(0); ++mu)
            for (int la = 0; la < bcData.shape(1); ++la)
                bcData(mu, la) = data(mu, la);
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

    void update_projections();
};

#else
#endif