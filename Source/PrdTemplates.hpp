#ifndef CMO_LW_PRD_TEMPLATES_HPP
#define CMO_LW_PRD_TEMPLATES_HPP

#include "Lightweaver.hpp"
#include "SimdFullIterationTemplates.hpp"
#include "TaskSetWrapper.hpp"

namespace PrdCores
{
void total_depop_elastic_scattering_rate(const Transition* trans, const Atom& atom,
                                         F64View PjQj);

void prd_scatter(Transition* t, F64View PjQj, const Atom& atom,
                 const Atmosphere& atmos, const Spectrum& spect,
                 enki::TaskScheduler* sched);
}

template <SimdType simd>
f64 formal_sol_prd_update_rates(Context& ctx, ConstView<int> wavelengthIdxs)
{
    using namespace LwInternal;
    JasUnpack(*ctx, atmos, spect, background, depthData);
    JasUnpack(ctx, activeAtoms, detailedAtoms);

    const int Nspace = atmos.Nspace;

    if (ctx.Nthreads <= 1)
    {
        auto& iCore = *ctx.threading.intensityCores.cores[0];
        for (auto& a : activeAtoms)
        {
            for (auto& t : a->trans)
            {
                if (t->rhoPrd)
                {
                    t->zero_rates();
                }
            }
        }
        if (spect.JRest)
            spect.JRest.fill(0.0);

        f64 dJMax = 0.0;

        for (int i = 0; i < wavelengthIdxs.shape(0); ++i)
        {
            const f64 la = wavelengthIdxs(i);
            FsMode mode = (FsMode::UpdateJ | FsMode::UpdateRates | FsMode::PrdOnly);
            f64 dJ = intensity_core_opt<simd, true, true, false, false>(iCore, la, mode);
            dJMax = max(dJ, dJMax);
        }
        return dJMax;
    }
    else
    {
        auto& cores = ctx.threading.intensityCores;
        for (auto& core : cores.cores)
        {
            for (auto& a : *core->activeAtoms)
            {
                for (auto& t : a->trans)
                {
                    if (t->rhoPrd)
                    {
                        t->zero_rates();
                    }
                }
            }
            if (core->JRest)
                core->JRest.fill(0.0);
        }

        struct FsTaskData
        {
            IntensityCoreData* core;
            f64 dJ;
            i64 dJIdx;
            ConstView<int> idxs;
        };
        std::vector<FsTaskData> taskData;
        taskData.reserve(ctx.Nthreads);
        for (int t = 0; t < ctx.Nthreads; ++t)
        {
            FsTaskData td;
            td.core = cores.cores[t];
            td.dJ = 0.0;
            td.dJIdx = 0;
            td.idxs = wavelengthIdxs;
            taskData.emplace_back(td);
        }

        auto fs_task = [](void* data, enki::TaskScheduler* s,
                          enki::TaskSetPartition p, u32 threadId)
        {
            auto& td = ((FsTaskData*)data)[threadId];
            FsMode mode = (FsMode::UpdateJ | FsMode::UpdateRates
                           | FsMode::PrdOnly);
            for (i64 la = p.start; la < p.end; ++la)
            {
                f64 dJ = intensity_core_opt<SimdType::Scalar,
                                            true, true, false, false>
                                            (*td.core, td.idxs(la), mode);
                td.dJ = max_idx(td.dJ, dJ, td.dJIdx, la);
            }
        };

        {
            enki::TaskScheduler* sched = &ctx.threading.sched;
            const int Nla = wavelengthIdxs.shape(0);
            const int taskSize = max(Nla / ctx.Nthreads / 16, 1);
            LwTaskSet formalSolutions(taskData.data(), sched, wavelengthIdxs.shape(0),
                                      4, fs_task);
            sched->AddTaskSetToPipe(&formalSolutions);
            sched->WaitforTask(&formalSolutions);
        }

        f64 dJMax = 0.0;
        i64 maxIdx = 0;
        for (int t = 0; t < ctx.Nthreads; ++t)
            dJMax = max_idx(dJMax, taskData[t].dJ, maxIdx, taskData[t].dJIdx);


        ctx.threading.intensityCores.accumulate_prd_rates();
        return dJMax;
    }
}

template <SimdType simd>
f64 formal_sol_prd_update_rates(Context& ctx, const std::vector<int>& wavelengthIdxs)
{
    return formal_sol_prd_update_rates<simd>(ctx, ConstView<int>(wavelengthIdxs.data(), wavelengthIdxs.size()));
}

template <SimdType simd>
PrdIterData redistribute_prd_lines_template(Context& ctx, int maxIter, f64 tol)
{
    struct PrdData
    {
        Transition* line;
        const Atom& atom;
        Ng ng;

        PrdData(Transition* l, const Atom& a, Ng&& n)
            : line(l), atom(a), ng(n)
        {}
    };
    JasUnpack(*ctx, atmos, spect);
    JasUnpack(ctx, activeAtoms);
    std::vector<PrdData> prdLines;
    prdLines.reserve(10);
    for (auto& a : activeAtoms)
    {
        for (auto& t : a->trans)
        {
            if (t->rhoPrd)
            {
                prdLines.emplace_back(PrdData(t, *a, Ng(0, 0, 0, t->rhoPrd.flatten())));
            }
        }
    }
    auto JC = spect.JCoeffs.flatten();
    int maxC = 0;
    for (int i = 0; i < JC.shape(0); ++i)
    {
        if (JC(i).size() > maxC)
            maxC = JC(i).size();
    }
    printf("%d\n-----\n", maxC);

    if (prdLines.size() == 0)
        return {0, 0.0};

    const int Nspect = spect.wavelength.shape(0);
    auto& idxsForFs = spect.hPrdIdxs;
    std::vector<int> prdIdxs;
    if (spect.hPrdIdxs.size() == 0)
    {
        prdIdxs.reserve(Nspect);
        for (int la = 0; la < Nspect; ++la)
        {
            bool prdLinePresent = false;
            for (auto& p : prdLines)
                prdLinePresent = (p.line->active(la) || prdLinePresent);
            if (prdLinePresent)
                prdIdxs.emplace_back(la);
        }
        idxsForFs = prdIdxs;
    }

    int iter = 0;
    f64 dRho = 0.0;
    if (ctx.Nthreads <= 1)
    {
        F64Arr PjQj(atmos.Nspace);
        while (iter < maxIter)
        {
            ++iter;
            dRho = 0.0;
            for (auto& p : prdLines)
            {
                PrdCores::total_depop_elastic_scattering_rate(p.line, p.atom, PjQj);
                PrdCores::prd_scatter(p.line, PjQj, p.atom, atmos, spect, nullptr);
                p.ng.accelerate(p.line->rhoPrd.flatten());
                dRho = max(dRho, p.ng.max_change());
            }

            formal_sol_prd_update_rates<simd>(ctx, idxsForFs);

            if (dRho < tol)
                break;
        }
    }
    else
    {
        struct PrdTaskData
        {
            F64Arr PjQj;
            PrdData* line;
            f64 dRho;
            Atmosphere* atmos;
            Spectrum* spect;
        };
        auto taskData = std::vector<PrdTaskData>(prdLines.size());
        for (int i = 0; i < prdLines.size(); ++i)
        {
            auto& p = taskData[i];
            p.PjQj = F64Arr(atmos.Nspace);
            p.line = &prdLines[i];
            p.dRho = 0.0;
            p.atmos = &atmos;
            p.spect = &spect;
        }

        auto prd_task = [](void* data, enki::TaskScheduler* s,
                           enki::TaskSetPartition part, u32 threadId)
        {
            for (i64 lineIdx = part.start; lineIdx < part.end; ++lineIdx)
            {
                auto& td = ((PrdTaskData*)data)[lineIdx];
                auto& p = *td.line;
                PrdCores::total_depop_elastic_scattering_rate(p.line, p.atom, td.PjQj);
                PrdCores::prd_scatter(p.line, td.PjQj, p.atom, *td.atmos, *td.spect, s);
                p.ng.accelerate(p.line->rhoPrd.flatten());
                td.dRho = max(td.dRho, p.ng.max_change());
            }
        };

        while (iter < maxIter)
        {
            ++iter;
            dRho = 0.0;
            for (auto& p : taskData)
                p.dRho = 0.0;

            {
                enki::TaskScheduler* sched = &ctx.threading.sched;
                LwTaskSet prdScatter(taskData.data(), sched,
                                     prdLines.size(), 1, prd_task);
                sched->AddTaskSetToPipe(&prdScatter);
                sched->WaitforTask(&prdScatter);
            }
            formal_sol_prd_update_rates<simd>(ctx, idxsForFs);

            for (const auto& p : taskData)
            {
                dRho = max(dRho, p.dRho);
            }
            if (dRho < tol)
                break;

        }
    }

    return {iter, dRho};
}
#else
#endif