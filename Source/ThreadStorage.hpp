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

struct PerTransFns
{
    AllocPerTransScratch alloc_per;
    FreePerTransScratch free_per;
};

struct PerAtomFns
{
    AllocPerAtomScratch alloc_per;
    FreePerAtomScratch free_per;
};

struct PerAtomTransFns
{
    PerAtomFns perAtom;
    PerTransFns perTrans;
};

struct TransitionStorage
{
    F64Arr Rij;
    F64Arr Rji;
    Transition trans;
    FreePerTransScratch free_method_scratch;

    TransitionStorage() = default;
    TransitionStorage(const TransitionStorage&) = delete;
    TransitionStorage(TransitionStorage&&) = delete;
    inline TransitionStorage& operator=(const TransitionStorage&) = delete;
    inline TransitionStorage& operator=(TransitionStorage&&) = delete;
    ~TransitionStorage()
    {
        if (free_method_scratch)
            free_method_scratch(&trans);
    }
};

struct TransitionStorageFactory
{
    Transition* trans;
    bool detailedStatic;
    std::vector<std::unique_ptr<TransitionStorage>> tStorage;
    PerTransFns methodFns;
    TransitionStorageFactory(Transition* t, PerTransFns perFns);
    Transition* copy_transition();
    void accumulate_rates();
    void accumulate_prd_rates();
};

struct AtomStorage
{
    F64Arr3D Gamma;
    Atom atom;
    FreePerAtomScratch free_method_scratch;

    AtomStorage() = default;
    AtomStorage(const AtomStorage&) = delete;
    AtomStorage(AtomStorage&&) = delete;
    inline AtomStorage& operator=(const AtomStorage&) = delete;
    inline AtomStorage& operator=(AtomStorage&&) = delete;
    ~AtomStorage()
    {
        if (free_method_scratch)
            free_method_scratch(&atom);
    }
};

struct AtomStorageFactory
{
    Atom* atom;
    bool detailedStatic;
    bool wlaGijStorage;
    bool defaultPerAtomStorage;
    int fsWidth;
    std::vector<std::unique_ptr<AtomStorage>> aStorage;
    std::vector<TransitionStorageFactory> tStorage;
    PerAtomFns methodFns;
    AtomStorageFactory(Atom* a, bool detail, bool wlaStorage,
                       bool defaultPerAtomStorage,
                       int fsWidth, PerAtomTransFns perFns);
    Atom* copy_atom();
    void accumulate_Gamma();
    void accumulate_Gamma_rates();
    void accumulate_prd_rates();
    void accumulate_rates();
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
    F64Arr2D JRest;
    std::vector<Atom*> activeAtoms;
    std::vector<Atom*> detailedAtoms;
    IntensityCoreData core;
    FormalData formal;

    IntensityCoreStorage(int Nspace, int NhPrd)
        : I(F64Arr(0.0, Nspace)),
          S(F64Arr(0.0, Nspace)),
          JDag(F64Arr(0.0, Nspace)),
          chiTot(F64Arr(0.0, Nspace)),
          etaTot(F64Arr(0.0, Nspace)),
          Uji(F64Arr(0.0, Nspace)),
          Vij(F64Arr(0.0, Nspace)),
          Vji(F64Arr(0.0, Nspace)),
          Ieff(F64Arr(0.0, Nspace)),
          PsiStar(F64Arr(0.0, Nspace)),
          JRest()
    {
        if (NhPrd > 0)
        {
            JRest = F64Arr2D(NhPrd, Nspace);
        }
    }
};

struct IntensityCoreFactory
{
    Atmosphere* atmos;
    Spectrum* spect;
    Background* background;
    DepthData* depthData;
    int fsWidth;
    LwFsFn formal_solver;
    InterpFn interp;
    std::vector<AtomStorageFactory> activeAtoms;
    std::vector<AtomStorageFactory> detailedAtoms;
    std::vector<std::unique_ptr<IntensityCoreStorage>> arrayStorage;

    IntensityCoreFactory() : atmos(nullptr),
                             spect(nullptr),
                             background(nullptr),
                             depthData(nullptr),
                             fsWidth(1),
                             formal_solver(),
                             interp(),
                             activeAtoms(),
                             detailedAtoms(),
                             arrayStorage()
    {}

    void initialise(Context* ctx);
    IntensityCoreData* new_intensity_core();
    IntensityCoreData* single_thread_intensity_core();
    void accumulate_JRest();
    void accumulate_Gamma_rates();
    void accumulate_prd_rates(bool includeDetailedAtoms);
    void accumulate_Gamma_rates_parallel(Context& ctx);
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
    void accumulate_prd_rates(bool includeDetailedAtoms);
    void accumulate_Gamma_rates_parallel(Context& ctx);
    void clear();
};


struct ThreadData
{
    IntensityCoreFactory threadDataFactory;
    IterationCores intensityCores;
    enki::TaskScheduler sched;
    std::function<void()> clear_global_scratch;

    ThreadData() : threadDataFactory(),
                   intensityCores(),
                   sched()
    {}


    void initialise(Context* ctx);
    void clear(Context* ctx);

    ~ThreadData()
    {
        sched.WaitforAllAndShutdown();
        if (clear_global_scratch)
            clear_global_scratch();
    }
    ThreadData(const ThreadData&) = delete;
    ThreadData(ThreadData&&) = delete;
    inline ThreadData& operator=(const ThreadData&) = delete;
    inline ThreadData& operator=(ThreadData&&) = delete;
};

}
#else
#endif