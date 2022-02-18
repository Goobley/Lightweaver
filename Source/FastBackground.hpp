#ifndef CMO_FAST_BACKGROUND_HPP
#define CMO_FAST_BACKGROUND_HPP
#include "Lightweaver.hpp"
#include "CmoArray.hpp"
#include "TaskScheduler.h"
#include "TaskSetWrapper.hpp"
#include "Utils.hpp"
#include <vector>

struct BackgroundContinuum
{
    int i;
    int j;
    int laStart;
    int laEnd;
    F64View alpha;

    BackgroundContinuum(int i_, int j_, f64 minLa, f64 laEdge,
                        F64View crossSection,
                        F64View globalWavelength)
        : i(i_),
          j(j_),
          laStart(),
          laEnd(),
          alpha(crossSection)
    {
        laStart = hunt(globalWavelength, minLa);
        laEnd = hunt(globalWavelength, laEdge) + 1;
    }
};

struct ResonantRayleighLine
{
    f64 Aji;
    f64 gRatio; // g_j / g_0
    f64 lambda0;
    f64 lambdaMax;

    ResonantRayleighLine(f64 A, f64 gjgi, f64 lambda0, f64 lambdaMax)
        : Aji(A),
          gRatio(gjgi),
          lambda0(lambda0),
          lambdaMax(lambdaMax)
    {}
};

struct BackgroundAtom
{
    F64View2D n;
    F64View2D nStar;
    std::vector<BackgroundContinuum> continua;
    std::vector<ResonantRayleighLine> resonanceScatterers;
};

struct FastBackgroundContext
{
    int Nthreads;
    enki::TaskScheduler sched;
    FastBackgroundContext() : Nthreads(),
                              sched()
    {}

    ~FastBackgroundContext()
    {
        sched.WaitforAllAndShutdown();
    }

    void initialise(int numThreads)
    {
        Nthreads = numThreads;
        if (numThreads <= 1)
            return;
        sched.Initialize(Nthreads);
    }

    void basic_background(BackgroundData* bd, Atmosphere* atmosphere);
    void bf_opacities(BackgroundData* bd, std::vector<BackgroundAtom>* atoms,
                      Atmosphere* atmos);
    void rayleigh_scatter(BackgroundData* bd, std::vector<BackgroundAtom>* atoms,
                          Atmosphere* atmos);


};


#endif