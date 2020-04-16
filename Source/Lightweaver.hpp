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

f64 formal_sol_gamma_matrices(Context& ctx, bool lambdaIterate=false);
f64 formal_sol_update_rates(Context& ctx);
f64 formal_sol_update_rates_fixed_J(Context& ctx);
f64 formal_sol(Context& ctx);
f64 formal_sol_full_stokes(Context& ctx, bool updateJ=true);
PrdIterData redistribute_prd_lines(Context& ctx, int maxIter, f64 tol);
void stat_eq(Atom* atom);
void time_dependent_update(Atom* atomIn, F64View2D nOld, f64 dt);
void planck_nu(long Nspace, double *T, double lambda, double *Bnu);
void configure_hprd_coeffs(Context& ctx);

namespace EscapeProbability
{
void gamma_matrices_escape_prob(Atom* a, Background& background, 
                                const Atmosphere& atmos);
}


#else
#endif