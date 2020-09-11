#include "LwFormalInterface.hpp"
#include "LwInternal.hpp"
#include <cstring>
#include <cassert>

using namespace LwInternal;

bool FormalSolverManager::load_fs_from_path(const char* path)
{
    PlatformSharedLibrary lib{};
    if (!load_library(&lib, path))
    {
        printf("Fail1\n");
        return false;
    }

    libs.emplace_back(lib);

    FsProvider fs_provider = load_function<FsProvider>(lib, "fs_provider");
    if (!fs_provider)
    {
        printf("Fail2\n");
        return false;
    }

    FormalSolver fs = fs_provider();
    formalSolvers.emplace_back(fs);
    return true;
}

FormalSolverManager::FormalSolverManager()
{
    formalSolvers.emplace_back(FormalSolver{1, 1, "piecewise_linear_1d", piecewise_linear_1d});
    formalSolvers.emplace_back(FormalSolver{1, 1, "piecewise_bezier3_1d", piecewise_bezier3_1d});
    formalSolvers.emplace_back(FormalSolver{2, 1, "piecewise_linear_2d", piecewise_linear_2d});
    formalSolvers.emplace_back(FormalSolver{2, 1, "piecewise_besser_2d", piecewise_besser_2d});
}

FormalSolverManager::~FormalSolverManager()
{
}

InterpFnManager::InterpFnManager()
{
    fns.emplace_back(InterpFn(2, "interp_linear_2d", interp_linear_2d));
    fns.emplace_back(InterpFn(2, "interp_besser_2d", interp_besser_2d));
}

InterpFnManager::~InterpFnManager()
{
}

bool InterpFnManager::load_fn_from_path(const char* path)
{
    PlatformSharedLibrary lib{};
    if (!load_library(&lib, path))
        return false;

    libs.emplace_back(lib);

    InterpProvider interp_provider = load_function<InterpProvider>(lib, "interp_fn_provider");
    if (!interp_provider)
        return false;

    InterpFn interp = interp_provider();
    fns.emplace_back(interp);
    return true;
}