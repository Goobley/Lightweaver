#ifndef CMO_BACKGROUND_HPP
#define CMO_BACKGROUND_HPP

#include "Lightweaver.hpp"
#include "Constants.hpp"
#include "CmoArray.hpp"

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
};

void basic_background(BackgroundData* bd, Atmosphere* atmos,
                      int laStart=-1, int laEnd=-1);
f64 Gaunt_bf(f64 lambda, f64 n_eff, int charge);

#endif