#ifndef CMO_THREAD_STORAGE_HPP
#define CMO_THREAD_STORAGE_HPP

#include "Constants.hpp"
#include "CmoArray.hpp"
#include "JasPP.hpp"
#include "LwInternal.hpp"
#include "LwTransition.hpp"
#include "LwAtom.hpp"
#include "TaskScheduler.h"
#include <vector>
#include <memory>

struct DepthData;
struct Context;
struct Atmosphere;
struct Background;
struct Spectrum;

namespace LwInternal
{
struct TransitionStorage
{
    F64Arr Rij;
    F64Arr Rji;
    Transition trans;
};

struct TransitionStorageFactory
{
    Transition* trans;
    std::vector<std::unique_ptr<TransitionStorage>> tStorage;
    TransitionStorageFactory(Transition* t);
    Transition* copy_transition();
    void erase(Transition* t);
    void accumulate_rates();
    void accumulate_rates(const std::vector<size_t>& indices);
    void accumulate_prd_rates();
    void accumulate_prd_rates(const std::vector<size_t>& indices);
};

struct AtomStorage
{
    F64Arr3D Gamma;
    F64Arr eta;
    F64Arr2D gij;
    F64Arr2D wla;
    F64Arr2D U;
    F64Arr2D chi;
    Atom atom;
};

struct AtomStorageFactory
{
    Atom* atom;
    bool detailedStatic;
    std::vector<std::unique_ptr<AtomStorage>> aStorage;
    std::vector<TransitionStorageFactory> tStorage;
    AtomStorageFactory(Atom* a, bool detail);
    Atom* copy_atom();
    void erase(Atom* atom);
    void accumulate_Gamma_rates();
    void accumulate_Gamma_rates(const std::vector<size_t>& indices);
    void accumulate_prd_rates();
    void accumulate_prd_rates(const std::vector<size_t>& indices);
    void accumulate_Gamma_rates_parallel(scheduler* s);
    void accumulate_Gamma_rates_parallel(scheduler* s, const std::vector<size_t>& indices);
};

struct IntensityCoreStorage
{
    F64Arr I;
    F64Arr S;
    F64Arr JDag;
    F64Arr chiTot;
    F64Arr etaTot;
    F64Arr Uji;
    F64Arr Vij;
    F64Arr Vji;
    F64Arr Ieff;
    F64Arr PsiStar;
    std::vector<Atom*> activeAtoms;
    std::vector<Atom*> detailedAtoms;
    IntensityCoreData core;
    FormalData formal;

    IntensityCoreStorage(int Nspace)
        : I(F64Arr(0.0, Nspace)),
          S(F64Arr(0.0, Nspace)),
          JDag(F64Arr(0.0, Nspace)),
          chiTot(F64Arr(0.0, Nspace)),
          etaTot(F64Arr(0.0, Nspace)),
          Uji(F64Arr(0.0, Nspace)),
          Vij(F64Arr(0.0, Nspace)),
          Vji(F64Arr(0.0, Nspace)),
          Ieff(F64Arr(0.0, Nspace)),
          PsiStar(F64Arr(0.0, Nspace))
    {}
};

struct IntensityCoreFactory
{
    Atmosphere* atmos;
    Spectrum* spect;
    Background* background;
    DepthData* depthData;
    LwFsFn formal_solver;
    InterpFn interp;
    std::vector<AtomStorageFactory> activeAtoms;
    std::vector<AtomStorageFactory> detailedAtoms;
    std::vector<std::unique_ptr<IntensityCoreStorage>> arrayStorage;

    IntensityCoreFactory() : atmos(nullptr),
                             spect(nullptr),
                             background(nullptr),
                             depthData(nullptr)
    {}

    void initialise(Context* ctx);
    IntensityCoreData* new_intensity_core(bool psiOperator);
    void erase(IntensityCoreData* core);
    void accumulate_Gamma_rates();
    void accumulate_Gamma_rates(const std::vector<size_t>& indices);
    void accumulate_Gamma_rates(Context& ctx, const std::vector<size_t>& indices);
    void accumulate_prd_rates();
    void accumulate_prd_rates(const std::vector<size_t>& indices);
    void accumulate_Gamma_rates_parallel(Context& ctx);
    void accumulate_Gamma_rates_parallel(Context& ctx, const std::vector<size_t>& indices);
    void clear();
};

struct IterationCores
{
    IntensityCoreFactory* factory;
    std::vector<IntensityCoreData*> cores;
    std::vector<size_t> indices;
    IterationCores() : factory(nullptr),
                       cores()
    {};
    ~IterationCores();

    void initialise(IntensityCoreFactory* fac, int Nthreads);
    void accumulate_Gamma_rates();
    void accumulate_prd_rates();
    void accumulate_Gamma_rates_parallel(Context& ctx);
    void clear();
};


struct ThreadData
{
    IntensityCoreFactory threadDataFactory;
    IterationCores intensityCores;
    scheduler sched;
    void* schedMemory;

    ThreadData() : threadDataFactory(),
                   intensityCores(),
                   sched(),
                   schedMemory(nullptr)
    {}


    void initialise(Context* ctx);
    void clear(Context* ctx);

    ~ThreadData()
    {
        if (schedMemory)
        {
            scheduler_stop(&sched, 1);
            free(schedMemory);
            schedMemory = nullptr;
        }
    }
};

}
#else
#endif