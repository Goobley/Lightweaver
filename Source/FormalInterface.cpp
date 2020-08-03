#include "LwFormalInterface.hpp"
#include "LwInternal.hpp"
#include <cstring>
#include <cassert>

using namespace LwInternal;

void FormalSolverManager::load_fs_from_path(const char* path)
{
    PlatformSharedLibrary lib{};
    if (!load_library(&lib, path))
        assert(false && "Failed to load shared library");

    libs.emplace_back(lib);

    FsProvider fs_provider = load_function<FsProvider>(lib, "fs_provider");
    if (!fs_provider)
    {
        assert(false && "Failed to load fs_provider from library");
    }

    FormalSolver fs = fs_provider();
    formalSolvers.emplace_back(fs);
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
    // for (const auto& lib : libs)
    // {
    //     close_library(lib);
    // }
}