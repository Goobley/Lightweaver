#ifndef CMO_THREAD_STORAGE_HPP
#define CMO_THREAD_STORAGE_HPP

#include "Constants.hpp"
#include "CmoArray.hpp"
#include "JasPP.hpp"
#include "LwInternal.hpp"
#include "sched.h"
#include <vector>

struct Transition;
struct Atom;
struct Context;
struct Atmosphere;
struct Background;
struct Spectrum;
struct FormalData;
struct IntensityCoreData;

namespace LwInternal
{
struct TransitionStorage
{
    F64Arr Rij;
    F64Arr Rji;
};

struct TransitionStorageFactory
{
    Transition* trans;
    std::vector<Transition> tStorage;
    std::vector<TransitionStorage> arrayStorage;
    TransitionStorageFactory(Transition* t);
    Transition* copy_transition();
    void accumulate_rates();
};

struct AtomStorage
{
    F64Arr3D Gamma;
    F64Arr eta;
    F64Arr2D gij;
    F64Arr2D wla;
    F64Arr2D V;
    F64Arr2D U;
    F64Arr2D chi;
};

struct AtomStorageFactory
{
    Atom* atom;
    bool detailedStatic;
    std::vector<Atom> aStorage;
    std::vector<TransitionStorageFactory> tStorage;
    std::vector<AtomStorage> arrayStorage;
    AtomStorageFactory(Atom* a, bool detail);
    Atom* copy_atom();
    void accumulate_Gamma_rates();
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

    void set_Nspace(int Nspace)
    {
        I = F64Arr(0.0, Nspace);
        S = F64Arr(0.0, Nspace);
        JDag = F64Arr(0.0, Nspace);
        chiTot = F64Arr(0.0, Nspace);
        etaTot = F64Arr(0.0, Nspace);
        Uji = F64Arr(0.0, Nspace);
        Vij = F64Arr(0.0, Nspace);
        Vji = F64Arr(0.0, Nspace);
        Ieff = F64Arr(0.0, Nspace);
        PsiStar = F64Arr(0.0, Nspace);
    }
};

struct IntensityCoreFactory
{
    Context* ctx;
    Atmosphere* atmos;
    Spectrum* spect;
    Background* background;
    std::vector<AtomStorageFactory> activeAtoms;
    std::vector<AtomStorageFactory> detailedAtoms;
    std::vector<IntensityCoreStorage> arrayStorage;
    std::vector<FormalData> fdStorage;

    IntensityCoreFactory() : ctx(nullptr),
                             atmos(nullptr), 
                             spect(nullptr),
                             background(nullptr)
    {}

    void initialise(Context* context);
    IntensityCoreData new_intensity_core(bool psiOperator);
    void accumulate_Gamma_rates();
};


struct ThreadData
{
    IntensityCoreFactory threadDataFactory;
    std::vector<IntensityCoreData> intensityCores;
    scheduler sched;
    void* schedMemory;

    ThreadData() : threadDataFactory(),
                   sched(),
                   schedMemory(nullptr)
    {}

    ~ThreadData()
    {
        if (schedMemory)
        {
            scheduler_stop(&sched, 1);
            free(schedMemory);
        }
    }
};

}
#else
#endif