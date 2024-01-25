#ifndef CMO_SIMDFULLITERATIONTEMPLATES_HPP
#define CMO_SIMDFULLITERATIONTEMPLATES_HPP
#include "Constants.hpp"
#include "LwIterationResult.hpp"
#include "Simd.hpp"
#include "LwAtom.hpp"
#include "LwTransition.hpp"
#include "LwInternal.hpp"
#include "TaskSetWrapper.hpp"

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

template <SimdType simd, bool iClean, bool jClean,
          bool FirstTrans, bool ComputeOperator,
          typename std::enable_if_t<simd == SimdType::Scalar, bool> = true>
inline ForceInline void
chi_eta_aux_accum(IntensityCoreData* data, Atom* atom, const Transition& t)
{
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
    JasUnpack((*data), Uji, Vij, Vji);
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
compute_source_fn(F64View& S, F64View& etaTot, F64View& chiTot,
                  F64View& sca, F64View& JDag)
{
    const int Nspace = S.shape(0);
    for (int k = 0; k < Nspace; ++k)
    {
        S(k) = (etaTot(k) + sca(k) * JDag(k)) / chiTot(k);
    }
}

template <SimdType simd>
inline ForceInline void
accumulate_J(f64 halfwmu, F64View& J, F64View& I)
{
    const int Nspace = J.shape(0);
    for (int k = 0; k < Nspace; ++k)
    {
        J(k) += halfwmu * I(k);
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
f64 intensity_core_opt(IntensityCoreData& data, int la, FsMode mode, ExtraParams params)
{
    JasUnpack(*data, atmos, spect, fd, background);
    JasUnpack(*data, activeAtoms, detailedAtoms, JDag);
    JasUnpack(data, chiTot, etaTot, Uji, Vij, Vji);
    JasUnpack(data, I, S, Ieff, PsiStar, JRest);
    const int Nspace = atmos.Nspace;
    const int Nrays = atmos.Nrays;
    const LwFsFn formal_solver = data.formal_solver;

    const bool updateJ = mode & FsMode::UpdateJ;
    const bool upOnly = mode & FsMode::UpOnly;

    // NOTE(cmo): handle ZPlaneDecomposition
    const bool zPlaneDecomposition = params.contains("ZPlaneDecomposition");
    F64View2D zPlaneDown1D, zPlaneUp1D;
    F64View3D zPlaneDown2D, zPlaneUp2D;
    if (zPlaneDecomposition)
    {
        switch (atmos.Ndim)
        {
            case 1:
            {
                if (params.contains("ZPlaneDown"))
                    zPlaneDown1D = params.get_as<F64View2D>("ZPlaneDown");
                if (params.contains("ZPlaneUp"))
                    zPlaneUp1D = params.get_as<F64View2D>("ZPlaneUp");
            } break;

            case 2:
            {
                if (params.contains("ZPlaneDown"))
                    zPlaneDown2D = params.get_as<F64View3D>("ZPlaneDown");
                if (params.contains("ZPlaneUp"))
                    zPlaneUp2D = params.get_as<F64View3D>("ZPlaneUp");
            } break;

            default:
                printf("Unexpected Ndim!\n");
                assert(false);
        }
    }


    JDag = spect.J(la);
    F64View J = spect.J(la);
    if (updateJ)
        J.fill(0.0);

    for (int a = 0; a < activeAtoms.size(); ++a)
        setup_wavelength_opt<simd>(activeAtoms[a], la);
    for (int a = 0; a < detailedAtoms.size(); ++a)
        setup_wavelength_opt<simd>(detailedAtoms[a], la);

    // NOTE(cmo): If we only have continua then opacity is angle independent
    const bool continuaOnly = continua_only(data, la);

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
                gather_opacity_emissivity_opt<simd>(&data, ComputeOperator, la, mu, toObs);
                // NOTE(cmo): These are to keep the type checker happy, the
                // optimiser will likely elide.
                auto sca = background.sca(la);
                auto JView = JDag.slice(0, Nspace);
                compute_source_fn<simd>(S, etaTot, chiTot, sca, JView);
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
                        for (int mu = 0; mu < Nrays; ++mu)
                        {
                            for (int toObsI = 0; toObsI < toObsEnd; toObsI += 1)
                            {
                                memcpy(&depth.chi(la, mu, toObsI, 0),
                                       chiTot.data, Nspace * sizeof(f64));
                                memcpy(&depth.eta(la, mu, toObsI, 0),
                                       etaTot.data, Nspace * sizeof(f64));
                            }
                        }
                    }
                }
            }

            switch (atmos.Ndim)
            {
                case 1:
                {
                    formal_solver(&fd, la, mu, toObs, spect.wavelength);
                    spect.I(la, mu, 0) = I(0);

                    if (zPlaneDecomposition)
                    {
                        if (toObsI && zPlaneUp1D)
                        {
                            zPlaneUp1D(la, mu) = I(1);
                        }
                        else if (!toObsI && zPlaneDown1D)
                        {
                            zPlaneDown1D(la, mu) = I(atmos.Nz - 2);
                        }
                    }

                } break;

                case 2:
                {
                    formal_solver(&fd, la, mu, toObs, spect.wavelength);
                    auto I2 = I.reshape(atmos.Nz, atmos.Nx);
                    for (int j = 0; j < atmos.Nx; ++j)
                        spect.I(la, mu, j) = I2(0, j);

                    if (zPlaneDecomposition)
                    {
                        if (toObsI && zPlaneUp2D)
                        {
                            for (int j = 0; j < atmos.Nx; ++j)
                                zPlaneUp2D(la, mu, j) = I2(1, j);
                        }
                        else if (!toObsI && zPlaneDown2D)
                        {
                            for (int j = 0; j < atmos.Nx; ++j)
                                zPlaneDown2D(la, mu, j) = I2(atmos.Nz - 2, j);
                        }
                    }

                } break;

                default:
                    printf("Unexpected Ndim!\n");
                    assert(false);
            }

            if (updateJ)
            {
                accumulate_J<simd>(0.5 * atmos.wmu(mu), J, I);

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
                for (int a = 0; a < activeAtoms.size(); ++a)
                {
                    auto& atom = *activeAtoms[a];
                    if constexpr (ComputeOperator)
                    {
                        if (mode & FsMode::PureLambdaIteration)
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
            if constexpr (UpdateRates)
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

                        const bool computeRates = (UpdateRates && !PrdRatesOnly) ||
                                            (UpdateRates && PrdRatesOnly && t.rhoPrd);
                        dispatch_compute_full_operator_rates_<simd>(
                            false,
                            computeRates,
                            &atom,
                            kr,
                            wmu,
                            &data
                        );
                    }
                }
            }
            if constexpr (StoreDepthData)
            {
                auto& depth = *data.depthData;
                memcpy(&depth.I(la, mu, toObsI, 0),
                        I.data, Nspace * sizeof(f64));
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
        auto zero_task = [](void* data, enki::TaskScheduler* s,
                            enki::TaskSetPartition p, u32 threadId)

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
        enki::TaskScheduler* sched = &ctx->threading.sched;
        LwTaskSet zeroing(&zeroTaskData, sched, zeroTaskData.size(),
                          1, zero_task);
        sched->AddTaskSetToPipe(&zeroing);
        if (ctx->spect->JRest)
            zero_JRest(cores.cores);
        sched->WaitforTask(&zeroing);
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
IterationResult formal_sol_iteration_matrices_impl(Context& ctx, LwInternal::FsMode mode, ExtraParams params)
{
    JasUnpack(*ctx, spect);
    JasUnpack(ctx, activeAtoms, detailedAtoms);

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
        int maxIdx = 0;
        FsMode mode = (FsMode::UpdateJ | FsMode::UpdateRates);
        if (lambdaIterate)
            mode = mode | FsMode::PureLambdaIteration;

        const bool storeDepthData = (ctx.depthData && ctx.depthData->fill);
        for (int la = 0; la < Nspect; ++la)
        {
            f64 dJ = dispatch_intensity_core_opt_<simd>(true, false, true,
                                                        storeDepthData,
                                                        iCore, la * ctx.formalSolver.width,
                                                        mode, params);
            dJMax = max_idx(dJ, dJMax, maxIdx, la);
        }
        for (int a = 0; a < activeAtoms.size(); ++a)
        {
            finalise_Gamma<simd>(*activeAtoms[a]);
        }
        IterationResult result{};
        result.updatedJ = true;
        result.dJMax = dJMax;
        result.dJMaxIdx = maxIdx;
        return result;
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
            ExtraParams* params;
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
            td.params = &params;
            taskData.emplace_back(td);
        }

        auto fs_task = [](void* data, enki::TaskScheduler* s,
                          enki::TaskSetPartition p, u32 threadId)
        {
            auto& td = ((FsTaskData*)data)[threadId];
            FsMode mode = (FsMode::UpdateJ | FsMode::UpdateRates);
            if (td.lambdaIterate)
                mode = mode | FsMode::PureLambdaIteration;

            for (i64 la = p.start; la < p.end; ++la)
            {
                f64 dJ = dispatch_intensity_core_opt_
                            <simd>(true, false, true, td.storeDepthData,
                            *td.core, la * td.width, mode, *td.params);
                td.dJ = max_idx(td.dJ, dJ, td.dJIdx, la);
            }
        };

        {
            enki::TaskScheduler* sched = &ctx.threading.sched;
            const int taskSize = max(numFs / ctx.Nthreads / 16, 1);
            LwTaskSet formalSolutions(taskData.data(), sched, numFs, taskSize, fs_task);
            sched->AddTaskSetToPipe(&formalSolutions);
            sched->WaitforTask(&formalSolutions);
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

        IterationResult result{};
        result.updatedJ = true;
        result.dJMax = dJMax;
        result.dJMaxIdx = maxIdx;
        return result;
    }
}

template <SimdType simd>
IterationResult formal_sol_impl(Context& ctx, LwInternal::FsMode mode, ExtraParams params)
{
    JasUnpack(*ctx, spect);

    const int Nspect = spect.wavelength.shape(0);

    if (ctx.Nthreads <= 1)
    {
        // NOTE(cmo): We're now creating a default core for single threaded work
        auto& iCore = *ctx.threading.intensityCores.cores[0];

        for (int la = 0; la < Nspect; ++la)
        {
            intensity_core_opt<simd, false, false, false, false>(iCore, la, mode, params);
        }
        return IterationResult{};
    }
    else
    {
        auto& cores = ctx.threading.intensityCores;

        struct FsTaskData
        {
            IntensityCoreData* core;
            FsMode mode;
            ExtraParams* params;
        };
        std::vector<FsTaskData> taskData;
        taskData.reserve(ctx.Nthreads);
        for (int t = 0; t < ctx.Nthreads; ++t)
        {
            FsTaskData td;
            td.core = cores.cores[t];
            td.mode = mode;
            td.params = &params;
            taskData.emplace_back(td);
        }

        auto fs_task = [](void* data, enki::TaskScheduler* s,
                          enki::TaskSetPartition p, u32 threadId)
        {
            auto& td = ((FsTaskData*)data)[threadId];
            for (i64 la = p.start; la < p.end; ++la)
            {
                intensity_core_opt<simd, false, false, false, false>(*td.core, la, td.mode, *td.params);
            }
        };

        {
            enki::TaskScheduler* sched = &ctx.threading.sched;
            const int taskSize = max(Nspect / ctx.Nthreads / 16, 1);
            LwTaskSet formalSolutions(taskData.data(), sched, Nspect,
                                      taskSize, fs_task);
            sched->AddTaskSetToPipe(&formalSolutions);
            sched->WaitforTask(&formalSolutions);
        }

        return IterationResult{};
    }
}
}

#else
#endif