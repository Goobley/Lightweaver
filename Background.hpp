#ifndef CMO_BACKGROUND_HPP
#define CMO_BACKGROUND_HPP

#include "Constants.hpp"
#include "CmoArray.hpp"
#include "Atmosphere.hpp"

struct BackgroundData
{
    F64View chPops;
    F64View ohPops;
    F64View h2Pops;
    F64View hMinusPops;
    F64View2D hPops;

    F64View wavelength;
    F64View2D chi;
    F64View2D eta;
    F64View2D scatt;

    Atmosphere* atmos;
};

void linear(F64View xTable, F64View yTable, F64View x, F64View y);
f64 linear(F64View xTable, F64View yTable, f64 x);
void basic_background(BackgroundData* bd);
f64 Gaunt_bf(f64 lambda, f64 n_eff, int charge);

#endif