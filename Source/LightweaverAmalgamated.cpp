#define CMO_FORMAL_INTERFACE_IMPL
#include "Lightweaver.hpp"

#include "FormalInterface.cpp"
#include "LuSolve.cpp"
#include "Atmosphere.cpp"
#include "Background.cpp"
#include "FastBackground.cpp"
#include "FormalScalar.cpp"
#include "FormalScalar2d.cpp"
#include "FormalStokes.cpp"
#include "UpdatePopulations.cpp"
#include "Prd.cpp"
#include "EscapeProbability.cpp"
#include "ThreadStorage.cpp"

// NOTE(cmo): This file does a lot of includes,  #defines, and usings that I'm not super keen on, so I suggest leaving it last
#include "Faddeeva.cc"