#ifndef CMO_LW_FORMAL_INTERFACE_HPP
#define CMO_LW_FORMAL_INTERFACE_HPP

#include "Constants.hpp"
#include "LwAtmosphere.hpp"
#include "LwIterationResult.hpp"
#include "LwExtraParams.hpp"

struct Context;
struct Atom;
struct Transition;
struct PlatformSharedLibrary;
namespace LwInternal
{
bool load_library(PlatformSharedLibrary* lib, const char* path);
template <typename F> F load_function(PlatformSharedLibrary lib, const char* name);
void close_library(PlatformSharedLibrary lib);
}

#ifdef _WIN32
    #include "LwFormalInterfaceWin.hpp"
#else
    #include "LwFormalInterfacePosix.hpp"
#endif

#include <vector>

namespace LwInternal
{
    struct FormalData;
}
struct Context;
struct NrTimeDependentData;

typedef void(*LwFsFn)(LwInternal::FormalData* fd, int la, int mu,
                      bool toObs, const F64View1D& wav);
struct FormalSolver
{
    LwFsFn solver;
    int Ndim;
    int width; // NOTE(cmo): For SIMD later on.
    const char* name;
};

typedef FormalSolver(*FsProvider)();

struct FormalSolverManager
{
    std::vector<FormalSolver> formalSolvers;
    std::vector<PlatformSharedLibrary> libs;

    FormalSolverManager();
    bool load_fs_from_path(const char* path);
};

typedef f64(*Interp2d)(const IntersectionData& grid, const IntersectionResult& loc,
                       const F64View2D& param);

struct InterpFn
{
    Interp2d interp_2d;
    int Ndim;
    const char* name;
    InterpFn() : interp_2d(nullptr),
                 Ndim(),
                 name("")
    {}
    InterpFn(int _Ndim, const char* _name, Interp2d interp) : interp_2d(interp),
                                                              Ndim(_Ndim),
                                                              name(_name)
    {}
};

typedef InterpFn(*InterpProvider)();

struct InterpFnManager
{
    std::vector<InterpFn> fns;
    std::vector<PlatformSharedLibrary> libs;

    InterpFnManager();
    bool load_fn_from_path(const char* path);
};

struct PrdIterData;
typedef IterationResult(*FormalSolIterFn)(Context& ctx, bool lambdaIterate, ExtraParams params);
typedef IterationResult(*SimpleFormalSol)(Context& ctx, bool upOnly, ExtraParams params);
typedef IterationResult(*FullStokesFormalSol)(Context& ctx, bool updateJ, bool upOnly,
                                              ExtraParams params);
typedef IterationResult(*RedistPrdLinesFn)(Context& ctx, int maxIter, f64 tol, ExtraParams params);
typedef void(*StatEqFn)(Atom* atom, ExtraParams params, int spaceStart, int spaceEnd);
typedef void(*TimeDepUpdateFn)(Atom* atomIn, F64View2D nOld, f64 dt, ExtraParams params,
                               int spaceStart, int spaceEnd);
typedef void(*NrPostUpdateFn)(Context& ctx, std::vector<Atom*>* atoms,
                              const std::vector<F64View3D>& dC,
                              F64View backgroundNe,
                              const NrTimeDependentData& timeDepData,
                              f64 crswVal,
                              ExtraParams params,
                              int spaceStart, int spaceEnd);

typedef void(*AllocPerAtomScratch)(Atom* atom, bool detailedStatic);
typedef void(*FreePerAtomScratch)(Atom* atom);
typedef void(*AllocPerTransScratch)(Transition* trans);
typedef void(*FreePerTransScratch)(Transition* trans);
typedef void(*AllocGlobalScratch)(Context* ctx);
typedef void(*FreeGlobalScratch)(Context* ctx);
typedef void(*AccumulateOverThreads)(Context* ctx);

struct FsIterationFns
{
    int Ndim;
    bool dimensionSpecific;
    bool respectsFormalSolver;
    bool defaultPerAtomStorage;
    bool defaultWlaGijStorage;
    const char* name;

    FormalSolIterFn fs_iter;
    SimpleFormalSol simple_fs;
    FullStokesFormalSol full_stokes_fs;
    RedistPrdLinesFn redistribute_prd;
    StatEqFn stat_eq;
    TimeDepUpdateFn time_dep_update;
    NrPostUpdateFn nr_post_update;

    AllocPerAtomScratch alloc_per_atom;
    FreePerAtomScratch free_per_atom;
    AllocPerTransScratch alloc_per_trans;
    FreePerTransScratch free_per_trans;
    AllocGlobalScratch alloc_global_scratch;
    FreeGlobalScratch free_global_scratch;
    AccumulateOverThreads accumulate_over_threads;
};

typedef FsIterationFns(*FsIterationFnsProvider)();

struct FsIterationFnsManager
{
    std::vector<FsIterationFns> fns;
    std::vector<PlatformSharedLibrary> libs;

    FsIterationFnsManager();
    bool load_fns_from_path(const char* path);
};

#else
#endif