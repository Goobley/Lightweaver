#ifndef CMO_LW_CONTEXT_HPP
#define CMO_LW_CONTEXT_HPP

#include "ThreadStorage.hpp"
#include "LwInternal.hpp"
#include "LwFormalInterface.hpp"
#include "LwMisc.hpp"
#include "LwAtmosphere.hpp"
#include "LwTransition.hpp"
#include "LwAtom.hpp"

struct DepthData
{
    bool fill;
    F64View4D chi;
    F64View4D eta;
    F64View4D I;
};

struct Context
{
    Atmosphere* atmos;
    Spectrum* spect;
    std::vector<Atom*> activeAtoms;
    std::vector<Atom*> detailedAtoms;
    Background* background;
    DepthData* depthData;
    int Nthreads;
    LwInternal::ThreadData threading;
    FormalSolver formalSolver;
    InterpFn interpFn;
    FsIterationFns iterFns;
    void* methodScratch;

    void initialise_threads()
    {
        threading.initialise(this);
    }

    void update_threads()
    {
        threading.clear(this);
        threading.initialise(this);
    }
};

#else
#endif