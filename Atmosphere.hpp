#ifndef ATMOSPHERE_HPP
#define ATMOSPHERE_HPP
#include "CmoArray.hpp"

struct Atmosphere
{
    F64View cmass;
    F64View height;
    F64View tau_ref;
    F64View temperature;
    F64View ne;
    F64View vlos;
    F64View vturb;
    F64View nHtot;
    F64View muz;
    F64View muy;
    F64View mux;
    F64View wmu;
    int Nspace;
    int Nrays;

    void print_tau() const;
};
#else
#endif