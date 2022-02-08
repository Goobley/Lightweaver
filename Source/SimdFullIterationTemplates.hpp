#ifndef CMO_SIMDFULLITERATIONTEMPLATES_HPP
#define CMO_SIMDFULLITERATIONTEMPLATES_HPP
#include "Constants.hpp"
#include "Simd.hpp"
#include "LwAtom.hpp"
#include "LwTransition.hpp"
#include "LwInternal.hpp"

#include <array>
#include <cstring>

namespace LwInternal
{
template <SimdType simd>
inline ForceInline
void setup_wavelength_opt(Atom* atom, int laIdx)
{
    return atom->setup_wavelength(laIdx);
}

template <SimdType type>
inline ForceInline void
uv_opt(Transition* t, int la, int mu, bool toObs, F64View Uji, F64View Vij, F64View Vji)
{
    return t->uv(la, mu, toObs, Uji, Vij, Vji);
}

struct CachedTrans
{
    int atomIdx;
    int kr;
};

template <int CacheSize=32>
struct CachedContributingTrans
{
    // NOTE(cmo): This struct is ~4kB at default size, which seems not too
    // unreasonable (same order as printf's typical buffers)
    bool continuaOnly;
    bool activeOverflow;
    bool staticOverflow;
    int Nactive;
    int Nstatic;
    std::array<CachedTrans, CacheSize> activeCache;
    std::array<CachedTrans, CacheSize> staticCache;
};

inline bool continua_only(const IntensityCoreData& data, int la)
{
    JasUnpack(*data, activeAtoms, detailedAtoms);
    bool continuaOnly = true;
    for (int a = 0; a < activeAtoms.size(); ++a)
    {
        auto& atom = *activeAtoms[a];
        for (int kr = 0; kr < atom.Ntrans; ++kr)
        {
            auto& t = *atom.trans[kr];
            if (!t.is_active(la))
                continue;
            continuaOnly = continuaOnly && (t.type == CONTINUUM);
        }
    }
    for (int a = 0; a < detailedAtoms.size(); ++a)
    {
        auto& atom = *detailedAtoms[a];
        for (int kr = 0; kr < atom.Ntrans; ++kr)
        {
            auto& t = *atom.trans[kr];
            if (!t.is_active(la))
                continue;
            continuaOnly = continuaOnly && (t.type == CONTINUUM);
        }
    }
    return continuaOnly;
}

inline CachedContributingTrans<>
cached_continua_only(const IntensityCoreData& data, int la)
{
    JasUnpack(*data, activeAtoms, detailedAtoms);
    CachedContributingTrans<> result{};
    JasUnpack(result, continuaOnly, Nactive,
              Nstatic, activeCache, staticCache);
    const int maxCache = activeCache.size();
    for (int a = 0; a < activeAtoms.size(); ++a)
    {
        auto& atom = *activeAtoms[a];
        for (int kr = 0; kr < atom.Ntrans; ++kr)
        {
            auto& t = *atom.trans[kr];
            if (!t.is_active(la))
                continue;
            continuaOnly = continuaOnly && (t.type == CONTINUUM);
            if (!result.activeOverflow)
            {
                activeCache[Nactive++] = { a, kr };
                if (Nactive == maxCache)
                    result.activeOverflow = true;
            }
        }
    }
    for (int a = 0; a < detailedAtoms.size(); ++a)
    {
        auto& atom = *detailedAtoms[a];
        for (int kr = 0; kr < atom.Ntrans; ++kr)
        {
            auto& t = *atom.trans[kr];
            if (!t.is_active(la))
                continue;
            continuaOnly = continuaOnly && (t.type == CONTINUUM);
            if (!result.staticOverflow)
            {
                staticCache[Nstatic++] = { a, kr };
                if (Nstatic == maxCache)
                    result.staticOverflow = true;
            }
        }
    }
    return result;
}

template <SimdType simd, bool iClean, bool jClean,
          bool FirstTrans, bool ComputeOperator,
          typename std::enable_if_t<simd == SimdType::Scalar, bool> = true>
inline ForceInline void
chi_eta_aux_accum(IntensityCoreData* data, Atom* atom, const Transition& t)
{
    JasUnpack(*(*data), activeAtoms, detailedAtoms);
    JasUnpack((*data), Uji, Vij, Vji, chiTot, etaTot);
    const int Nspace = data->atmos->Nspace;

    for (int k = 0; k < Nspace; ++k)
    {
        f64 chi = atom->n(t.i, k) * Vij(k) - atom->n(t.j, k) * Vji(k);
        f64 eta = atom->n(t.j, k) * Uji(k);

        if constexpr (ComputeOperator)
        {
            if constexpr (iClean)
            {
                atom->chi(t.i, k) += chi;
            }
            else
            {
                atom->chi(t.i, k) = chi;
                atom->U(t.i, k) = 0.0;
            }

            if constexpr (jClean)
            {
                atom->chi(t.j, k) -= chi;
                atom->U(t.j, k) += Uji(k);
            }
            else
            {
                atom->chi(t.j, k) = -chi;
                atom->U(t.j, k) = Uji(k);
            }

            if constexpr (FirstTrans)
            {
                atom->eta(k) = eta;
            }
            else
            {
                atom->eta(k) += eta;
            }
        }

        chiTot(k) += chi;
        etaTot(k) += eta;
    }
}

#include "Dispatch_chi_eta_aux_accum.ipp"

template <SimdType simd>
inline ForceInline void
gather_opacity_emissivity_opt(IntensityCoreData* data,
                              bool computeOperator, int la,
                              int mu, bool toObs)
{
    JasUnpack(*(*data), activeAtoms, detailedAtoms);
    JasUnpack((*data), Uji, Vij, Vji, chiTot, etaTot);
    const int Nspace = data->atmos->Nspace;
    for (int a = 0; a < activeAtoms.size(); ++a)
    {
        auto& atom = *activeAtoms[a];
        constexpr int StackAlloc = 128;
        bool ijCleanStore[StackAlloc] = { false };
        bool* ijClean = ijCleanStore;
        bool heapAlloc = false;
        bool firstTrans = true;
        if (atom.Nlevel > StackAlloc)
        {
            ijClean = (bool*)calloc(atom.Nlevel, 1);
            heapAlloc = true;
        }
        for (int kr = 0; kr < atom.Ntrans; ++kr)
        {
            auto& t = *atom.trans[kr];
            if (!t.is_active(la))
                continue;

            uv_opt<simd>(&t, la, mu, toObs, Uji, Vij, Vji);
            dispatch_chi_eta_aux_accum_<simd>(ijClean[t.i], ijClean[t.j],
                                            firstTrans, computeOperator,
                                            data, &atom, t);
            firstTrans = false;
            ijClean[t.i] = true;
            ijClean[t.j] = true;
        }
        if (heapAlloc)
        {
            free(ijClean);
        }
    }

    for (int a = 0; a < detailedAtoms.size(); ++a)
    {
        auto& atom = *detailedAtoms[a];
        for (int kr = 0; kr < atom.Ntrans; ++kr)
        {
            auto& t = *atom.trans[kr];
            if (!t.is_active(la))
                continue;

            uv_opt<simd>(&t, la, mu, toObs, Uji, Vij, Vji);
            chi_eta_aux_accum<simd, false, false, false, false, false>(data, &atom, t);
        }
    }
}

template <SimdType simd>
inline ForceInline void
compute_source_fn(F64View& S, const F64View& etaTot, const F64View& chiTot,
                  const F64View& sca, const F64View& JDag)
{
    const int Nspace = S.shape(0);
    for (int k = 0; k < Nspace; ++k)
    {
        S(k) = (etaTot(k) + sca(k) * JDag(k)) / chiTot(k);
    }
}

template <SimdType simd>
inline ForceInline void
compute_full_Ieff(F64View& I, F64View& PsiStar,
                  F64View& eta, F64View& Ieff)
{

    const int Nspace = I.shape(0);

    for (int k = 0; k < Nspace; ++k)
    {
        Ieff(k) = I(k) - PsiStar(k) * eta(k);
    }
}

template <SimdType simd, bool ComputeOperator, bool ComputeRates,
          typename std::enable_if_t<simd == SimdType::Scalar, bool> = true>
inline void
compute_full_operator_rates(Atom* a, int kr, f64 wmu,
                            IntensityCoreData* data)
{
    JasUnpack((*data), Uji, Vij, Vji, PsiStar, Ieff, I);
    const int Nspace = Uji.shape(0);
    auto& atom = *a;
    auto& t = *atom.trans[kr];
    for (int k = 0; k < Nspace; ++k)
    {
        const f64 wlamu = atom.wla(kr, k) * wmu;
        if constexpr (ComputeOperator)
        {
            f64 integrand = (Uji(k) + Vji(k) * Ieff(k)) - (PsiStar(k) * atom.chi(t.i, k) * atom.U(t.j, k));
            atom.Gamma(t.i, t.j, k) += integrand * wlamu;

            integrand = (Vij(k) * Ieff(k)) - (PsiStar(k) * atom.chi(t.j, k) * atom.U(t.i, k));
            atom.Gamma(t.j, t.i, k) += integrand * wlamu;
        }

        if constexpr (ComputeRates)
        {
            t.Rij(k) += I(k) * Vij(k) * wlamu;
            t.Rji(k) += (Uji(k) + I(k) * Vji(k)) * wlamu;
        }
    }
}

#include "Dispatch_compute_full_operator_rates.ipp"

template <SimdType simd, bool UpdateRates, bool PrdRatesOnly,
          bool ComputeOperator, bool StoreDepthData>
f64 intensity_core_opt(IntensityCoreData& data, int la, FsMode mode)
{
    JasUnpack(*data, atmos, spect, fd, background);
    JasUnpack(*data, activeAtoms, detailedAtoms, JDag);
    JasUnpack(data, chiTot, etaTot, Uji, Vij, Vji);
    JasUnpack(data, I, S, Ieff, PsiStar, JRest);
    const int Nspace = atmos.Nspace;
    const int Nrays = atmos.Nrays;
    const int Nspect = spect.wavelength.shape(0);
    const LwFsFn formal_solver = data.formal_solver;

    const bool updateJ = mode & FsMode::UpdateJ;
    const bool lambdaIterate = mode & FsMode::PureLambdaIteration;
    const bool upOnly = mode & FsMode::UpOnly;

    JDag = spect.J(la);
    F64View J = spect.J(la);
    if (updateJ)
        J.fill(0.0);

    for (int a = 0; a < activeAtoms.size(); ++a)
        setup_wavelength_opt<simd>(activeAtoms[a], la);
    for (int a = 0; a < detailedAtoms.size(); ++a)
        setup_wavelength_opt<simd>(detailedAtoms[a], la);

    // NOTE(cmo): If we only have continua then opacity is angle independent
    const auto transCache = cached_continua_only(data, la);
    const bool continuaOnly = transCache.continuaOnly;

    int toObsStart = 0;
    int toObsEnd = 2;
    if (upOnly)
        toObsStart = 1;

    for (int mu = 0; mu < Nrays; ++mu)
    {
        for (int toObsI = toObsStart; toObsI < toObsEnd; toObsI += 1)
        {
            bool toObs = (bool)toObsI;
            if (!continuaOnly || (continuaOnly && (mu == 0 && toObsI == toObsStart)))
            {

                memcpy(chiTot.data, &background.chi(la, 0), Nspace * sizeof(f64));
                memcpy(etaTot.data, &background.eta(la, 0), Nspace * sizeof(f64));
                // Gathers from all active non-background transitions
                // gather_opacity_emissivity(&data, ComputeOperator, la, mu, toObs);
                gather_opacity_emissivity_opt<simd>(&data, ComputeOperator, la, mu, toObs);
                compute_source_fn<simd>(S, etaTot, chiTot, background.sca(la), JDag);
                if constexpr (StoreDepthData)
                {
                    auto& depth = *data.depthData;
                    if (!continuaOnly)
                    {
                        memcpy(&depth.chi(la, mu, toObsI, 0),
                               chiTot.data, Nspace * sizeof(f64));
                        memcpy(&depth.eta(la, mu, toObsI, 0),
                               etaTot.data, Nspace * sizeof(f64));
                    }
                    else
                    {
                        memcpy(&depth.chi(la, 0, 0, 0),
                               chiTot.data,
                               Nrays * 2 * Nspace * sizeof(f64));
                        memcpy(&depth.eta(la, 0, 0, 0),
                               etaTot.data,
                               Nrays * 2 * Nspace * sizeof(f64));
                    }
                }
            }

            switch (atmos.Ndim)
            {
                case 1:
                {
                    formal_solver(&fd, la, mu, toObs, spect.wavelength);
                    spect.I(la, mu, 0) = I(0);

                } break;

                case 2:
                {
                    formal_solver(&fd, la, mu, toObs, spect.wavelength);
                    auto I2 = I.reshape(atmos.Nz, atmos.Nx);
                    for (int j = 0; j < atmos.Nx; ++j)
                        spect.I(la, mu, j) = I2(0, j);

                } break;

                default:
                    printf("Unexpected Ndim!\n");
                    assert(false);
            }

            if (updateJ)
            {
                // TODO(cmo): Break this out.
                for (int k = 0; k < Nspace; ++k)
                {
                    J(k) += 0.5 * atmos.wmu(mu) * I(k);
                }

                if (JRest && spect.hPrdActive && spect.hPrdActive(la))
                {
                    int hPrdLa = spect.la_to_hPrdLa(la);
                    for (int k = 0; k < Nspace; ++k)
                    {
                        const auto& coeffs = spect.JCoeffs(hPrdLa, mu, toObs, k);
                        for (const auto& c : coeffs)
                        {
                            JRest(c.idx, k) += 0.5 * atmos.wmu(mu) * c.frac * I(k);
                        }
                    }
                }
            }

            if constexpr (UpdateRates || ComputeOperator)
            {
                if (!transCache.activeOverflow)
                {
                    int prevAtomIdx = -1;
                    for (int cacheIdx = 0; cacheIdx < transCache.Nactive; ++cacheIdx)
                    {
                        auto cached = transCache.activeCache[cacheIdx];
                        auto& atom = *activeAtoms[cached.atomIdx];
                        auto& t = *atom.trans[cached.kr];

                        if constexpr (ComputeOperator)
                        {
                            if (cached.atomIdx != prevAtomIdx)
                            {
                                if (lambdaIterate)
                                    PsiStar.fill(0.0);

                                compute_full_Ieff<simd>(I, PsiStar, atom.eta, Ieff);
                                prevAtomIdx = cached.atomIdx;
                            }

                        }
                        const f64 wmu = 0.5 * atmos.wmu(mu);
                        uv_opt<simd>(&t, la, mu, toObs, Uji, Vij, Vji);

                        const bool computeRates = (UpdateRates && !PrdRatesOnly) ||
                                            (UpdateRates && PrdRatesOnly && t.rhoPrd);
                        dispatch_compute_full_operator_rates_<simd>(ComputeOperator,
                                                computeRates, &atom, cached.kr, wmu, &data);
                    }
                }
                else
                {
                    for (int a = 0; a < activeAtoms.size(); ++a)
                    {
                        auto& atom = *activeAtoms[a];
                        if constexpr (ComputeOperator)
                        {
                            if (lambdaIterate)
                                PsiStar.fill(0.0);

                            compute_full_Ieff<simd>(I, PsiStar, atom.eta, Ieff);
                        }

                        for (int kr = 0; kr < atom.Ntrans; ++kr)
                        {
                            auto& t = *atom.trans[kr];
                            if (!t.is_active(la))
                                continue;

                            const f64 wmu = 0.5 * atmos.wmu(mu);
                            uv_opt<simd>(&t, la, mu, toObs, Uji, Vij, Vji);

                            const bool computeRates = (UpdateRates && !PrdRatesOnly) ||
                                                (UpdateRates && PrdRatesOnly && t.rhoPrd);
                            dispatch_compute_full_operator_rates_<simd>(ComputeOperator,
                                                    computeRates, &atom, kr, wmu, &data);
                        }
                    }
                }
            }
            if constexpr (UpdateRates && !PrdRatesOnly)
            {
                for (int a = 0; a < detailedAtoms.size(); ++a)
                {
                    auto& atom = *detailedAtoms[a];

                    for (int kr = 0; kr < atom.Ntrans; ++kr)
                    {
                        auto& t = *atom.trans[kr];
                        if (!t.is_active(la))
                            continue;

                        const f64 wmu = 0.5 * atmos.wmu(mu);
                        uv_opt<simd>(&t, la, mu, toObs, Uji, Vij, Vji);

                        compute_full_operator_rates<simd, false, true>(&atom, kr,
                                                                       wmu, &data);
                    }
                }
            }
            if constexpr (StoreDepthData)
            {
                auto& depth = *data.depthData;
                for (int k = 0; k < Nspace; ++k)
                    depth.I(la, mu, toObsI, k) = I(k);
            }
        }
    }

    f64 dJMax = 0.0;
    if (updateJ)
    {
        for (int k = 0; k < Nspace; ++k)
        {
            f64 dJ = abs(1.0 - JDag(k) / J(k));
            dJMax = max(dJ, dJMax);
        }
    }
    return dJMax;
}

#include "Dispatch_intensity_core_opt.ipp"

template <SimdType simd>
void finalise_Gamma(Atom& atom)
{
    const int Nspace = atom.Gamma.shape(2);
    for (int k = 0; k < Nspace; ++k)
    {
        for (int i = 0; i < atom.Nlevel; ++i)
        {
            atom.Gamma(i, i, k) = 0.0;
            f64 gammaDiag = 0.0;
            for (int j = 0; j < atom.Nlevel; ++j)
            {
                gammaDiag += atom.Gamma(j, i, k);
            }
            atom.Gamma(i, i, k) = -gammaDiag;
        }
    }
}

inline bool should_parallelise_zeroing(const Context& ctx)
{
    if (ctx.Nthreads == 1)
        return false;

    // NOTE(cmo): This is quite empirical, but shouldn't matter in most instances.
    return ((((ctx.activeAtoms.size() + ctx.detailedAtoms.size()) > 8)
            && (ctx.atmos->Nspace >= 128))
           || ctx.atmos->Nspace >= 256);
}

inline void zero_Gamma_rates_JRest(Context* ctx)
{
    JasUnpack((*ctx), activeAtoms, detailedAtoms);
    auto& cores = ctx->threading.intensityCores;
    auto Nthreads = ctx->Nthreads;

    auto zero_JRest = [](decltype(cores.cores)& c)
    {
        for (auto& core : c)
        {
            core->JRest.fill(0.0);
        }
    };

    if (should_parallelise_zeroing(*ctx))
    {
        std::vector<Atom*> zeroTaskData;
        zeroTaskData.reserve(Nthreads * (activeAtoms.size() + detailedAtoms.size()));
        for (int c = 0; c < Nthreads; ++c)
        {
            for (auto* a : *cores.cores[c]->activeAtoms)
                zeroTaskData.emplace_back(a);
            for (auto* a : *cores.cores[c]->detailedAtoms)
                zeroTaskData.emplace_back(a);
        }
        auto zero_task = [](void* data, scheduler* s,
                            sched_task_partition p, sched_uint threadId)

        {
            auto& taskData = *(decltype(zeroTaskData)*)data;
            for (int i = p.start; i < p.end; ++i)
            {
                auto* atom = taskData[i];
                atom->zero_rates();
                if (atom->Gamma)
                    atom->zero_Gamma();
            }
        };
    {
        sched_task zeroing;
        scheduler_add(&ctx->threading.sched, &zeroing, zero_task,
                        (void*)(&zeroTaskData), zeroTaskData.size(), 1);
        if (ctx->spect->JRest)
            zero_JRest(cores.cores);
        scheduler_join(&ctx->threading.sched, &zeroing);
    }
    }
    else
    {
        for (auto& core : cores.cores)
        {
            for (auto& a : *core->activeAtoms)
            {
                a->zero_rates();
                a->zero_Gamma();
            }
            for (auto& a : *core->detailedAtoms)
            {
                a->zero_rates();
            }
        }
        if (ctx->spect->JRest)
            zero_JRest(cores.cores);
    }
}

template <SimdType simd>
f64 formal_sol_iteration_matrices_impl(Context& ctx, LwInternal::FsMode mode)
{
    JasUnpack(*ctx, atmos, spect, background, depthData);
    JasUnpack(ctx, activeAtoms, detailedAtoms);

    const int Nspace = atmos.Nspace;
    const int Nspect = spect.wavelength.shape(0);
    const bool lambdaIterate = mode & LwInternal::FsMode::PureLambdaIteration;

    if (ctx.Nthreads <= 1)
    {
        // NOTE(cmo): We're now creating a default core for single threaded work
        auto& iCore = *ctx.threading.intensityCores.cores[0];

        if (spect.JRest)
            spect.JRest.fill(0.0);

        for (auto& a : activeAtoms)
        {
            a->zero_rates();
        }
        for (auto& a : detailedAtoms)
        {
            a->zero_rates();
        }

        f64 dJMax = 0.0;
        FsMode mode = (FsMode::UpdateJ | FsMode::UpdateRates);
        if (lambdaIterate)
            mode = mode | FsMode::PureLambdaIteration;

        const bool storeDepthData = (ctx.depthData && ctx.depthData->fill);
        for (int la = 0; la < Nspect; ++la)
        {
            f64 dJ = dispatch_intensity_core_opt_<simd>(true, false, true,
                                                        storeDepthData,
                                                        iCore, la * ctx.formalSolver.width, mode);
            dJMax = max(dJ, dJMax);
        }
        for (int a = 0; a < activeAtoms.size(); ++a)
        {
            finalise_Gamma<simd>(*activeAtoms[a]);
        }
        return dJMax;
    }
    else
    {
        auto& cores = ctx.threading.intensityCores;

        zero_Gamma_rates_JRest(&ctx);

        int numFs = Nspect;
        if (ctx.formalSolver.width > 1)
            numFs = (Nspect + ctx.formalSolver.width - 1) / ctx.formalSolver.width;

        struct FsTaskData
        {
            IntensityCoreData* core;
            f64 dJ;
            i64 dJIdx;
            bool lambdaIterate;
            bool storeDepthData;
            int width;
        };
        const bool storeDepthData = (ctx.depthData && ctx.depthData->fill);
        std::vector<FsTaskData> taskData;
        taskData.reserve(ctx.Nthreads);
        for (int t = 0; t < ctx.Nthreads; ++t)
        {
            FsTaskData td;
            td.core = cores.cores[t];
            td.dJ = 0.0;
            td.dJIdx = 0;
            td.lambdaIterate = lambdaIterate;
            td.width = ctx.formalSolver.width;
            td.storeDepthData = storeDepthData;
            taskData.emplace_back(td);
        }

        auto fs_task = [](void* data, scheduler* s,
                          sched_task_partition p, sched_uint threadId)
        {
            auto& td = ((FsTaskData*)data)[threadId];
            FsMode mode = (FsMode::UpdateJ | FsMode::UpdateRates);
            if (td.lambdaIterate)
                mode = mode | FsMode::PureLambdaIteration;

            for (i64 la = p.start; la < p.end; ++la)
            {
                f64 dJ = dispatch_intensity_core_opt_
                            <simd>(true, false, true, td.storeDepthData,
                            *td.core, la * td.width, mode);
                td.dJ = max_idx(td.dJ, dJ, td.dJIdx, la);
            }
        };

        {
            sched_task formalSolutions;
            scheduler_add(&ctx.threading.sched, &formalSolutions,
                          fs_task, (void*)taskData.data(), numFs, 4);
            scheduler_join(&ctx.threading.sched, &formalSolutions);
        }

        f64 dJMax = 0.0;
        i64 maxIdx = 0;
        for (int t = 0; t < ctx.Nthreads; ++t)
            dJMax = max_idx(dJMax, taskData[t].dJ, maxIdx, taskData[t].dJIdx);


        ctx.threading.intensityCores.accumulate_Gamma_rates_parallel(ctx);

        for (int a = 0; a < activeAtoms.size(); ++a)
        {
            finalise_Gamma<simd>(*activeAtoms[a]);
        }
        return dJMax;
    }
}
}

#else
#endif