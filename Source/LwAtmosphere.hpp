#ifndef CMO_LW_ATMOSPHERE_HPP
#define CMO_LW_ATMOSPHERE_HPP

#include "CmoArray.hpp"

enum RadiationBC
{
    ZERO,
    THERMALISED
};

struct Atmosphere
{
    F64View cmass;
    F64View height;
    F64View tau_ref;
    F64View temperature;
    F64View ne;
    F64View vlos;
    F64View2D vlosMu;
    F64View B;
    F64View gammaB;
    F64View chiB;
    F64View2D cosGamma;
    F64View2D cos2chi;
    F64View2D sin2chi;
    F64View vturb;
    F64View nHtot;
    F64View muz;
    F64View muy;
    F64View mux;
    F64View wmu;
    int Nspace;
    int Nrays;

    enum RadiationBC lowerBc;
    enum RadiationBC upperBc;

    void update_projections();
};

#else
#endif