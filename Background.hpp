#ifndef CMO_BACKGROUND_HPP
#define CMO_BACKGROUND_HPP

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

void basic_background(BackgroundData* bd);

#endif