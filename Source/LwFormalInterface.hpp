#ifndef CMO_LW_FORMAL_INTERFACE_HPP
#define CMO_LW_FORMAL_INTERFACE_HPP

#include "Constants.hpp"
#include "LwAtmosphere.hpp"

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

typedef void(*LwFsFn)(LwInternal::FormalData* fd, int la, int mu, bool toObs, f64 wav);
struct FormalSolver
{
    int Ndim;
    int width; // NOTE(cmo): For SIMD later on.
    const char* name;
    LwFsFn solver;
};

typedef FormalSolver(*FsProvider)();

struct FormalSolverManager
{
    std::vector<FormalSolver> formalSolvers;
    std::vector<PlatformSharedLibrary> libs;

    FormalSolverManager();
    ~FormalSolverManager();
    bool load_fs_from_path(const char* path);
};

typedef f64(*Interp2d)(const IntersectionData& grid, const IntersectionResult& loc,
                       const F64View2D& param);

struct InterpFn
{
    int Ndim;
    const char* name;
    Interp2d interp_2d;
    InterpFn() : Ndim(),
                 name(""),
                 interp_2d(nullptr)
    {}
    InterpFn(int _Ndim, const char* _name, Interp2d interp) : Ndim(_Ndim),
                                                              name(_name),
                                                              interp_2d(interp)
    {}
};

typedef InterpFn(*InterpProvider)();

struct InterpFnManager
{
    std::vector<InterpFn> fns;
    std::vector<PlatformSharedLibrary> libs;

    InterpFnManager();
    ~InterpFnManager();
    bool load_fn_from_path(const char* path);
};

#else
#endif