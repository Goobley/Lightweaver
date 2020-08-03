#ifndef CMO_LW_FORMAL_INTERFACE_HPP
#define CMO_LW_FORMAL_INTERFACE_HPP

#include "Constants.hpp"
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
    void load_fs_from_path(const char* path);
};

#else
#endif