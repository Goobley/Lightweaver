#include "LwFormalInterface.hpp"
#include "LwInternal.hpp"
#include "Lightweaver.hpp"
#include <cstring>
#include <cassert>

using namespace LwInternal;

bool FormalSolverManager::load_fs_from_path(const char* path)
{
    PlatformSharedLibrary lib{};
    if (!load_library(&lib, path))
    {
        return false;
    }

    libs.emplace_back(lib);

    FsProvider fs_provider = load_function<FsProvider>(lib, "fs_provider");
    if (!fs_provider)
    {
        return false;
    }

    FormalSolver fs = fs_provider();
    formalSolvers.emplace_back(fs);
    return true;
}

FormalSolverManager::FormalSolverManager()
{
    formalSolvers.emplace_back(FormalSolver{piecewise_linear_1d, 1, 1, "piecewise_linear_1d"});
    formalSolvers.emplace_back(FormalSolver{piecewise_besser_1d, 1, 1, "piecewise_besser_1d"});
    formalSolvers.emplace_back(FormalSolver{piecewise_bezier3_1d, 1, 1, "piecewise_bezier3_1d"});
    formalSolvers.emplace_back(FormalSolver{piecewise_linear_2d, 2, 1, "piecewise_linear_2d"});
    formalSolvers.emplace_back(FormalSolver{piecewise_besser_2d, 2, 1, "piecewise_besser_2d"});
}

InterpFnManager::InterpFnManager()
{
    fns.emplace_back(InterpFn(2, "interp_linear_2d", interp_linear_2d));
    fns.emplace_back(InterpFn(2, "interp_besser_2d", interp_besser_2d));
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

bool FsIterationFnsManager::load_fns_from_path(const char* path)
{
    PlatformSharedLibrary lib{};
    if (!load_library(&lib, path))
    {
        return false;
    }

    libs.emplace_back(lib);

    auto fs_provider = load_function<FsIterationFnsProvider>(lib, "fs_iteration_fns_provider");
    if (!fs_provider)
    {
        return false;
    }

    FsIterationFns fs = fs_provider();
    fns.emplace_back(fs);
    return true;
}

FsIterationFnsManager::FsIterationFnsManager()
{
    fns.emplace_back(FsIterationFns{
                        -1, false, true, true, true,
                        "mali_full_precond_scalar",
                        formal_sol_iteration_matrices_scalar,
                        formal_sol_scalar,
                        formal_sol_full_stokes_impl,
                        redistribute_prd_lines_scalar,
                        stat_eq_impl,
                        time_dependent_update_impl,
                        nr_post_update_impl
                        });
}