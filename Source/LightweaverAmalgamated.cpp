#include "Lightweaver.hpp"

#include "LuSolve.cpp"
#include "Atmosphere.cpp"
#include "Background.cpp"
#include "FormalScalar.cpp"
#include "FormalStokes.cpp"
#include "UpdatePopulations.cpp"
#include "Prd.cpp"
#include "EscapeProbability.cpp"
#include "ThreadStorage.cpp"

// NOTE(cmo): This file does a lot of includes,  #defines, and usings that I'm not super keen on, so I suggest leaving it last
#include "Faddeeva.cc"
#define SCHED_IMPLEMENTATION
#define SCHED_PIPE_SIZE_LOG2 10
#include "TaskScheduler.h"