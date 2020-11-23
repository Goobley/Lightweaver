#ifndef CMO_LIGHTWEAVER_HPP
#define CMO_LIGHTWEAVER_HPP

#include "LwMisc.hpp"
#include "LwAtmosphere.hpp"
#include "LwTransition.hpp"
#include "LwAtom.hpp"
#include "LwContext.hpp"

// TODO(cmo): Do similar for standard iteration to get index of dJMax
struct PrdIterData
{
    int iter;
    f64 dRho;
};

struct NrTimeDependentData
{
    f64 dt;
    std::vector<F64View2D> nPrev;
};

f64 formal_sol_gamma_matrices(Context& ctx, bool lambdaIterate=false);
f64 formal_sol_update_rates(Context& ctx);
f64 formal_sol_update_rates_fixed_J(Context& ctx);
f64 formal_sol(Context& ctx);
f64 formal_sol_full_stokes(Context& ctx, bool updateJ=true);
PrdIterData redistribute_prd_lines(Context& ctx, int maxIter, f64 tol);
void stat_eq(Atom* atom, int spaceStart=-1, int spaceEnd=-1);
void time_dependent_update(Atom* atomIn, F64View2D nOld, f64 dt,
                           int spaceStart=-1, int spaceEnd=-1);
void parallel_stat_eq(Context* ctx, int chunkSize=20);
void parallel_time_dep_update(Context* ctx, const std::vector<F64View2D>& oldPops,
                              f64 dt, int chunkSize=20);
void nr_post_update(Context* ctx, std::vector<Atom*>* atoms,
                    const std::vector<F64View3D>& dC,
                    F64View backgroundNe,
                    const NrTimeDependentData& timeDepData,
                    f64 crswVal,
                    int spaceStart=-1, int spaceEnd=-1);
void parallel_nr_post_update(Context* ctx, std::vector<Atom*>* atoms,
                             const std::vector<F64View3D>& dC,
                             F64View backgroundNe,
                             const NrTimeDependentData& timeDepData,
                             f64 crswVal, int chunkSize=5);
void configure_hprd_coeffs(Context& ctx);

namespace EscapeProbability
{
void gamma_matrices_escape_prob(Atom* a, Background& background,
                                const Atmosphere& atmos);
}


#else
#endif