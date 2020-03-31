#ifndef CMO_LW_CONTEXT_HPP
#define CMO_LW_CONTEXT_HPP

#include "ThreadStorage.hpp"
#include "LwInternal.hpp"
#include "LwMisc.hpp"
#include "LwAtmosphere.hpp"
#include "LwTransition.hpp"
#include "LwAtom.hpp"

struct Context
{
    Atmosphere* atmos;
    Spectrum* spect;
    std::vector<Atom*> activeAtoms;
    std::vector<Atom*> detailedAtoms;
    Background* background;
    int Nthreads;
    LwInternal::ThreadData threading;

    void initialise_threads()
    {
        threading.initialise(this);
    }

    void update_threads()
    {
        threading.clear();
        threading.initialise(this);
    }
};

#else
#endif