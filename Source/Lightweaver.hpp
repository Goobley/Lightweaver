#ifndef CMO_LIGHTWEAVER_HPP
#define CMO_LIGHTWEAVER_HPP

#include "LwMisc.hpp"
#include "LwAtmosphere.hpp"
#include "LwTransition.hpp"
#include "LwAtom.hpp"
#include "LwContext.hpp"
#include "LwIterationResult.hpp"
#include "LwExtraParams.hpp"

struct NrTimeDependentData
{
    f64 dt;
    std::vector<F64View2D> nPrev;
};

IterationResult formal_sol_gamma_matrices(Context& ctx, bool lambdaIterate=false, ExtraParams params=ExtraParams{});
IterationResult formal_sol_iteration_matrices_scalar(Context& ctx, bool lambdaIterate=false, ExtraParams params=ExtraParams{});
IterationResult formal_sol(Context& ctx, bool upOnly=true, ExtraParams params=ExtraParams{});
IterationResult formal_sol_scalar(Context& ctx, bool upOnly=true, ExtraParams params=ExtraParams{});
IterationResult formal_sol_full_stokes(Context& ctx, bool updateJ=false,
                                       bool upOnly=true,
                                       ExtraParams params=ExtraParams{});
IterationResult formal_sol_full_stokes_impl(Context& ctx, bool updateJ=false,
                                            bool upOnly=true,
                                            ExtraParams params=ExtraParams{});
IterationResult redistribute_prd_lines(Context& ctx, int maxIter, f64 tol, ExtraParams params=ExtraParams{});
IterationResult redistribute_prd_lines_scalar(Context& ctx, int maxIter, f64 tol, ExtraParams params=ExtraParams{});
void stat_eq(Context& ctx,  Atom* atom, ExtraParams params=ExtraParams{},
             int spaceStart=-1, int spaceEnd=-1);
void stat_eq_impl(Atom* atom, ExtraParams params=ExtraParams{},
                  int spaceStart=-1, int spaceEnd=-1);
void time_dependent_update(Context& ctx, Atom* atomIn, F64View2D nOld, f64 dt,
                           ExtraParams params=ExtraParams{},
                           int spaceStart=-1, int spaceEnd=-1);
void time_dependent_update_impl(Atom* atomIn, F64View2D nOld, f64 dt,
                                ExtraParams params=ExtraParams{},
                                int spaceStart=-1, int spaceEnd=-1);
void nr_post_update(Context& ctx, std::vector<Atom*>* atoms,
                    const std::vector<F64View3D>& dC,
                    F64View backgroundNe,
                    const NrTimeDependentData& timeDepData,
                    f64 crswVal,
                    ExtraParams params=ExtraParams{},
                    int spaceStart=-1, int spaceEnd=-1);
void nr_post_update_impl(Context& ctx, std::vector<Atom*>* atoms,
                         const std::vector<F64View3D>& dC,
                         F64View backgroundNe,
                         const NrTimeDependentData& timeDepData,
                         f64 crswVal,
                         ExtraParams params=ExtraParams{},
                         int spaceStart=-1, int spaceEnd=-1);
void configure_hprd_coeffs(Context& ctx, bool includeDetailedAtoms=false);

namespace EscapeProbability
{
void gamma_matrices_escape_prob(Atom* a, Background& background,
                                const Atmosphere& atmos);
}


#else
#endif