#include "Lightweaver.hpp"
#include "Utils.hpp"

#ifdef CMO_BASIC_PROFILE
#include <chrono>
using hrc = std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;
#endif

namespace PrdCores
{
void total_depop_elastic_scattering_rate(const Transition* trans, const Atom& atom, F64View PjQj)
{
    // NOTE(cmo): Contrary to first appearance when reading the RH paper, this
    // doesn't return Pj but instread Pj + Qj
    const int Nspace = trans->Rij.shape(0);

    for (int k = 0; k < Nspace; ++k)
    {
        PjQj(k) = trans->Qelast(k);
        for (int i = 0; i < atom.C.shape(0); ++i)
            PjQj(k) += atom.C(i, trans->j, k);

        for (auto& t : atom.trans)
        {
            if (t->j == trans->j)
                PjQj(k) += t->Rji(k);
            if (t->i == trans->j)
                PjQj(k) += t->Rij(k);
        }
    }
}


constexpr f64 PrdQWing = 4.0;
constexpr f64 PrdQCore = 4.0;
constexpr f64 PrdQSpread = 5.0;
constexpr f64 PrdDQ = 0.25;

/*
    * Gouttebroze's fast approximation for
    *  GII(q_abs, q_emit) = PII(q_abs, q_emit) / phi(q_emit)

    * See: P. Gouttebroze, 1986, A&A 160, 195
    *      H. Uitenbroek,  1989, A&A, 216, 310-314 (cross redistribution)
    */

inline f64 G_zero(f64 x)
{
    return 1.0 / (abs(x) + sqrt(square(x) + 1.273239545));
}

f64 GII(f64 adamp, f64 q_emit, f64 q_abs)
{
    constexpr f64 waveratio = 1.0;
    namespace C = Constants;
    f64 gii, pcore, aq_emit, umin, epsilon, giiwing, u1, phicore, phiwing;

    /* --- Symmetrize with respect to emission frequency --   --------- */

    if (q_emit < 0.0)
    {
        q_emit = -q_emit;
        q_abs = -q_abs;
    }
    pcore = 0.0;
    gii = 0.0;

    /* --- Core region --                                     --------- */

    if (q_emit < PrdQWing)
    {
        if ((q_abs < -PrdQWing) || (q_abs > q_emit + waveratio * PrdQSpread))
            return gii;
        if (abs(q_abs) <= q_emit)
            gii = G_zero(q_emit);
        else
            gii = exp(square(q_emit) - square(q_abs)) * G_zero(q_abs);

        if (q_emit >= PrdQCore)
        {
            phicore = exp(-square(q_emit));
            phiwing = adamp / (sqrt(C::Pi) * (square(adamp) + square(q_emit)));
            pcore = phicore / (phicore + phiwing);
        }
    }
    /* --- Wing region --                                     --------- */

    if (q_emit >= PrdQCore)
    {
        aq_emit = waveratio * q_emit;
        if (q_emit >= PrdQWing)
        {
            if (abs(q_abs - aq_emit) > waveratio * PrdQSpread)
                return gii;
            pcore = 0.0;
        }
        umin = abs((q_abs - aq_emit) / (1.0 + waveratio));
        giiwing = (1.0 + waveratio) * (1.0 - 2.0 * umin * G_zero(umin)) * exp(-square(umin));

        if (waveratio == 1.0)
        {
            epsilon = q_abs / aq_emit;
            giiwing *= (2.75 - (2.5 - 0.75 * epsilon) * epsilon);
        }
        else
        {
            u1 = abs((q_abs - aq_emit) / (waveratio - 1.0));
            giiwing -= abs(1.0 - waveratio) * (1.0 - 2.0 * u1 * G_zero(u1)) * exp(-square(u1));
        }
        /* --- Linear combination of core- and wing contributions ------- */

        giiwing = giiwing / (2.0 * waveratio * sqrt(C::Pi));
        gii = pcore * gii + (1.0 - pcore) * giiwing;
    }
    return gii;
}

constexpr int max_fine_grid_size()
{
    return max(3 * PrdQWing, 2 * PrdQSpread) / PrdDQ + 1;
}

void prd_scatter(Transition* t, F64View PjQj, const Atom& atom, const Atmosphere& atmos, const Spectrum& spect)
{
    auto& trans = *t;

    namespace C = Constants;
    const int Nlambda = trans.wavelength.shape(0);

    bool initialiseGii = (!trans.gII) || (trans.gII(0, 0, 0) < 0.0);
    constexpr int maxFineGrid = max_fine_grid_size();
    if (!trans.gII)
    {
        trans.prdStorage.gII = F64Arr3D(Nlambda, atmos.Nspace, maxFineGrid);
        trans.gII = trans.prdStorage.gII;
    }

    // Reset Rho
    trans.rhoPrd.fill(1.0);

    F64Arr Jk(Nlambda);
    F64Arr qAbs(Nlambda);
    F64Arr JFine(maxFineGrid);
    F64Arr qp(maxFineGrid);
    F64Arr wq(maxFineGrid);

    for (int k = 0; k < atmos.Nspace; ++k)
    {

        // NOTE(cmo): This isn't gamma as Pj / (Pj + Qj), but instead the whole
        // prefactor to the scattering integral prefactor for a particular line,
        // i.e. gamma * n_k B_{kj} / (n_j P_k)
        // This simplifies into the expression below, remembering that PjQj = Pj + Qj
        f64 gammaPrefactor = atom.n(trans.i, k) / atom.n(trans.j, k) * trans.Bij / PjQj(k);
        f64 Jbar = trans.Rij(k) / trans.Bij;

        // NOTE(cmo): Local mean intensity (in rest frame if using HPRD).
        if (spect.JRest)
        {
            for (int la = 0; la < Nlambda; ++la)
            {
                int prdLa = spect.la_to_prdLa(la + trans.Nblue);
                Jk(la) = spect.JRest(prdLa, k);
                // Jk(la) = spect.JRest(la + trans.Nblue, k);
            }
        }
        else
        {
            for (int la = 0; la < Nlambda; ++la)
            {
                Jk(la) = spect.J(la + trans.Nblue, k);
            }
        }
        // NOTE(cmo): Local wavelength in doppler units
        for (int la = 0; la < Nlambda; ++la)
        {
            qAbs(la) = (trans.wavelength(la) - trans.lambda0) * C::CLight / (trans.lambda0 * atom.vBroad(k));
        }

        for (int la = 0; la < Nlambda; ++la)
        {
            f64 qEmit = qAbs(la);

            // NOTE(cmo): Find integration range around qEmit for which GII is
            // non-zero. (Resonance PRD case only). Follows Uitenbroek 2001.
            int q0, qN;
            if (abs(qEmit) < PrdQCore)
            {
                q0 = -PrdQWing;
                qN = PrdQWing;
            }
            else if (abs(qEmit) < PrdQWing)
            {
                if (qEmit > 0.0)
                {
                    q0 = -PrdQWing;
                    qN = qEmit + PrdQSpread;
                }
                else
                {
                    q0 = qEmit - PrdQSpread;
                    qN = PrdQWing;
                }
            }
            else
            {
                q0 = qEmit - PrdQSpread;
                qN = qEmit + PrdQSpread;
            }
            // NOTE(cmo): Set up fine linear grid over this range.
            int Np = int((f64)(qN - q0) / PrdDQ) + 1;
            qp(0) = q0;
            for (int lap = 1; lap < Np; ++lap)
                qp(lap) = qp(lap - 1) + PrdDQ;

            // NOTE(cmo): Linearly interpolate mean intensity onto this grid.
            linear(qAbs, Jk, qp.slice(0, Np), JFine);

            if (initialiseGii)
            {
                // NOTE(cmo): Compute gII if needed.
                // Integration weights for general trapezoidal rule obtained
                // from averaging extended Simpson's rule with modified
                // Simpson's rule where both edge regions are treated with
                // trapezoid rule. Takes accuracy up to O(1/N^3). Explained in
                // Press et al, Num Rec Sec4.2.
                wq.fill(PrdDQ);
                wq(0) = 5.0 / 12.0 * PrdDQ;
                wq(1) = 13.0 / 12.0 * PrdDQ;
                wq(Np - 1) = 5.0 / 12.0 * PrdDQ;
                wq(Np - 2) = 13.0 / 12.0 * PrdDQ;
                for (int lap = 0; lap < Np; ++lap)
                    trans.gII(la, k, lap) = GII(trans.aDamp(k), qEmit, qp(lap)) * wq(lap);
            }
            F64View gII = trans.gII(la, k);

            // NOTE(cmo): Compute and normalise scattering integral.
            f64 gNorm = 0.0;
            f64 scatInt = 0.0;
            for (int lap = 0; lap < Np; ++lap)
            {
                // NOTE(cmo): Normalisation of the scattering integral is very
                // important, as discussed in HM2014 Sec 15.4. Whilst this
                // procedure may slightly distort the redistribution function,
                // it ensures that no photons are gained or lost in this
                // evaluation.
                gNorm += gII(lap);
                scatInt += JFine(lap) * gII(lap);
            }
            trans.rhoPrd(la, k) += gammaPrefactor * (scatInt / gNorm - Jbar);
        }
    }
}
}

f64 formal_sol_prd_update_rates(Context& ctx, ConstView<int> wavelengthIdxs)
{
    using namespace LwInternal;
    JasUnpack(*ctx, atmos, spect, background, depthData);
    JasUnpack(ctx, activeAtoms, detailedAtoms);

    const int Nspace = atmos.Nspace;

    if (ctx.Nthreads <= 1)
    {
        F64Arr chiTot = F64Arr(Nspace);
        F64Arr etaTot = F64Arr(Nspace);
        F64Arr S = F64Arr(Nspace);
        F64Arr Uji = F64Arr(Nspace);
        F64Arr Vij = F64Arr(Nspace);
        F64Arr Vji = F64Arr(Nspace);
        F64Arr I = F64Arr(Nspace);
        F64Arr Ieff = F64Arr(Nspace);
        F64Arr JDag = F64Arr(Nspace);
        FormalData fd;
        fd.atmos = &atmos;
        fd.chi = chiTot;
        fd.S = S;
        fd.I = I;
        fd.interp = ctx.interpFn.interp_2d;
        IntensityCoreData iCore;
        JasPackPtr(iCore, atmos, spect, fd, background, depthData);
        JasPackPtr(iCore, activeAtoms, detailedAtoms, JDag);
        JasPack(iCore, chiTot, etaTot, Uji, Vij, Vji);
        JasPack(iCore, I, S, Ieff);
        iCore.formal_solver = ctx.formalSolver.solver;

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
            f64 dJ = intensity_core(iCore, la, (FsMode::UpdateJ | FsMode::UpdateRates | FsMode::PrdOnly));
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
        }
        if (spect.JRest)
            spect.JRest.fill(0.0);

        struct FsTaskData
        {
            IntensityCoreData* core;
            f64 dJ;
            i64 dJIdx;
            ConstView<int> idxs;
        };
        FsTaskData* taskData = (FsTaskData*)malloc(ctx.Nthreads * sizeof(FsTaskData));
        for (int t = 0; t < ctx.Nthreads; ++t)
        {
            taskData[t].core = cores.cores[t];
            taskData[t].dJ = 0.0;
            taskData[t].dJIdx = 0;
            taskData[t].idxs = wavelengthIdxs;
        }

        auto fs_task = [](void* data, scheduler* s,
                          sched_task_partition p, sched_uint threadId)
        {
            auto& td = ((FsTaskData*)data)[threadId];
            FsMode mode = (FsMode::UpdateJ | FsMode::UpdateRates
                           | FsMode::PrdOnly);
            for (i64 la = p.start; la < p.end; ++la)
            {
                f64 dJ = intensity_core(*td.core, td.idxs(la), mode);
                td.dJ = max_idx(td.dJ, dJ, td.dJIdx, la);
            }
        };

        {
            sched_task formalSolutions;
            scheduler_add(&ctx.threading.sched, &formalSolutions,
                          fs_task, (void*)taskData, wavelengthIdxs.shape(0), 4);
            scheduler_join(&ctx.threading.sched, &formalSolutions);
        }

        f64 dJMax = 0.0;
        i64 maxIdx = 0;
        for (int t = 0; t < ctx.Nthreads; ++t)
            dJMax = max_idx(dJMax, taskData[t].dJ, maxIdx, taskData[t].dJIdx);


        ctx.threading.intensityCores.accumulate_prd_rates();
        return dJMax;
    }
}

f64 formal_sol_prd_update_rates(Context& ctx, const std::vector<int>& wavelengthIdxs)
{
    return formal_sol_prd_update_rates(ctx, ConstView<int>(wavelengthIdxs.data(), wavelengthIdxs.size()));
}

PrdIterData redistribute_prd_lines(Context& ctx, int maxIter, f64 tol)
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
                PrdCores::prd_scatter(p.line, PjQj, p.atom, atmos, spect);
                p.ng.accelerate(p.line->rhoPrd.flatten());
                dRho = max(dRho, p.ng.max_change());
            }

            formal_sol_prd_update_rates(ctx, idxsForFs);

            if (dRho < tol)
                break;
        }
    }
    else
    {
#ifdef CMO_BASIC_PROFILE
        hrc::time_point startTimes[maxIter+1];
        hrc::time_point midTimes[maxIter+1];
        hrc::time_point endTimes[maxIter+1];
#endif

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

        auto prd_task = [](void* data, scheduler* s,
                           sched_task_partition part, sched_uint threadId)
        {
            for (i64 lineIdx = part.start; lineIdx < part.end; ++lineIdx)
            {
                auto& td = ((PrdTaskData*)data)[lineIdx];
                auto& p = *td.line;
                PrdCores::total_depop_elastic_scattering_rate(p.line, p.atom, td.PjQj);
                PrdCores::prd_scatter(p.line, td.PjQj, p.atom, *td.atmos, *td.spect);
                p.ng.accelerate(p.line->rhoPrd.flatten());
                td.dRho = max(td.dRho, p.ng.max_change());
            }
        };

        while (iter < maxIter)
        {
            ++iter;
#ifdef CMO_BASIC_PROFILE
            startTimes[iter-1] = hrc::now();
#endif
            dRho = 0.0;
            for (auto& p : taskData)
                p.dRho = 0.0;

            {
                sched_task prdScatter;
                scheduler_add(&ctx.threading.sched, &prdScatter, prd_task, (void*)taskData.data(), prdLines.size(), 1);
                scheduler_join(&ctx.threading.sched, &prdScatter);
            }
#ifdef CMO_BASIC_PROFILE
            midTimes[iter-1] = hrc::now();
#endif

            formal_sol_prd_update_rates(ctx, idxsForFs);

            for (const auto& p : taskData)
            {
                dRho = max(dRho, p.dRho);
            }
#ifdef CMO_BASIC_PROFILE
            endTimes[iter-1] = hrc::now();
#endif

            if (dRho < tol)
                break;

        }

#ifdef CMO_BASIC_PROFILE
        for (int i = 0; i < iter+1; ++i)
        {
            int f = duration_cast<nanoseconds>(midTimes[i] - startTimes[i]).count();
            int s = duration_cast<nanoseconds>(endTimes[i] - midTimes[i]).count();
            printf("[PRD]  First: %d ns, Second: %d ns, Ratio: %.3e\n", f, s, (f64)f/(f64)s);
        }
#endif
    }

    return {iter, dRho};
}

void configure_hprd_coeffs(Context& ctx)
{
    namespace C = Constants;
    struct PrdData
    {
        Transition* line;
        const Atom& atom;

        PrdData(Transition* l, const Atom& a)
            : line(l), atom(a)
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
                prdLines.emplace_back(PrdData(t, *a));
            }
        }
    }

    if (prdLines.size() == 0)
        return;

    const int Nspect = spect.wavelength.shape(0);
    spect.prdActive = BoolArr(false, Nspect);
    spect.la_to_prdLa = I32Arr(0, Nspect);
    auto& prdLambdas = spect.prdIdxs;
    prdLambdas.clear();
    prdLambdas.reserve(Nspect);
    for (int la = 0; la < Nspect; ++la)
    {
        bool prdLinePresent = false;
        for (auto& p : prdLines)
            prdLinePresent = (p.line->active(la) || prdLinePresent);
        if (prdLinePresent)
        {
            prdLambdas.emplace_back(la);
            spect.prdActive(la) = true;
            spect.la_to_prdLa(la) = prdLambdas.size()-1;
        }
    }


    // NOTE(cmo): Rather than simply re-evaluating the rates over the wavelength
    // grids of the PRD lines, in cases of medium-high Doppler shifts it is
    // better to compute the FS over all wavelengths which scatter into the
    // "rest" PRD grid. This sometimes seems to converge significantly better
    // than the basic HPRD in these cases.
    auto check_lambda_scatter_into_prd_region = [&](int la)
    {
        constexpr f64 sign[] = {-1.0, 1.0};
        for (int mu = 0; mu < atmos.Nrays; ++mu)
        {
            for (int toObs = 0; toObs <= 1; ++toObs)
            {
                for (int k = 0; k < atmos.Nspace; ++k)
                {
                    const f64 s = sign[toObs];
                    const f64 fac = 1.0 + atmos.vlosMu(mu, k) * s / C::CLight;
                    int prevIndex = max(la-1, 0);
                    int nextIndex = min(la+1, (int)spect.wavelength.shape(0)-1);
                    const f64 prevLambda = spect.wavelength(prevIndex) * fac;
                    const f64 nextLambda = spect.wavelength(nextIndex) * fac;

                    int i = la;
                    for (; spect.wavelength(i) > prevLambda && i >= 0; --i);
                    for (; i < spect.wavelength.shape(0); ++i)
                    {
                        const f64 lambdaI = spect.wavelength(i);
                        if (spect.prdActive(i))
                            return true;
                        else if (lambdaI > nextLambda)
                            break;
                    }
                }
            }
        }
        return false;
    };

    auto& hPrdIdxs = spect.hPrdIdxs;
    hPrdIdxs.clear();
    hPrdIdxs.reserve(Nspect);
    spect.la_to_hPrdLa = I32Arr(0, Nspect);
    spect.hPrdActive = BoolArr(false, Nspect);
    for (int la = 0; la < Nspect; ++la)
    {
        if (check_lambda_scatter_into_prd_region(la))
        {
            hPrdIdxs.emplace_back(la);
            spect.hPrdActive(la) = true;
            spect.la_to_hPrdLa(la) = hPrdIdxs.size()-1;
        }
    }

    if (!spect.JRest ||
        !(spect.JRest.shape(0) == prdLambdas.size() && spect.JRest.shape(1) == atmos.Nspace)
       )
        spect.JRest = F64Arr2D(0.0, prdLambdas.size(), atmos.Nspace);
    spect.JCoeffs = Prd::JCoeffVec(hPrdIdxs.size(), atmos.Nrays, 2, atmos.Nspace);
    constexpr f64 sign[] = {-1.0, 1.0};

    for (auto idx : hPrdIdxs)
    {
        for (int mu = 0; mu < atmos.Nrays; ++mu)
        {
            for (int toObs = 0; toObs <= 1; ++toObs)
            {
                for (int k = 0; k < atmos.Nspace; ++k)
                {
                    int hPrdLa = spect.la_to_hPrdLa(idx);
                    auto coeffVec = spect.JCoeffs(hPrdLa, mu, toObs);
                    const f64 s = sign[toObs];

                    const f64 fac = 1.0 + atmos.vlosMu(mu, k) * s / C::CLight;
                    int prevIndex = max(idx-1, 0);
                    int nextIndex = min(idx+1, Nspect-1);
                    const f64 prevLambda = spect.wavelength(prevIndex) * fac;
                    const f64 lambdaRest = spect.wavelength(idx) * fac;
                    const f64 nextLambda = spect.wavelength(nextIndex) * fac;
                    bool doLowerHalf = true, doUpperHalf = true;
                    // These will only be equal on the ends. And we can't be at both ends at the same time.
                    if (prevIndex == idx)
                    {
                        doLowerHalf = false;
                        for (int i = 0; i < Nspect; ++i)
                        {
                            if (spect.wavelength(i) <= lambdaRest && spect.prdActive(i))
                                coeffVec(k).emplace_back(Prd::JInterpCoeffs{spect.la_to_prdLa(i), 1.0});
                            else
                                break;
                        }
                    }
                    else if (nextIndex == idx)
                    {
                        // NOTE(cmo): By doing this part here, there's a strong
                        // likelihood that the indices for the final point will
                        // not be monotonic, but the cost of this is probably
                        // siginificantly lower than sorting all of the arrays,
                        // especially as it is likely that this case won't be
                        // used as I don't expect a PRD line right on the edge
                        // of the wavelength window in most scenarios
                        doUpperHalf = false;
                        for (int i = Nspect-1; i >= 0; --i)
                        {
                            if (spect.wavelength(i) > lambdaRest && spect.prdActive(i))
                                coeffVec(k).emplace_back(Prd::JInterpCoeffs{spect.la_to_prdLa(i), 1.0});
                            else
                                break;
                        }
                    }

                    int i  = idx;
                    // NOTE(cmo): If the shift is s.t. spect.wavelength(idx) > prevLambda, then we need to roll back
                    for (; spect.wavelength(i) > prevLambda && i >= 0; --i);


                    // NOTE(cmo): Upper bound goes all the way to the top, but we will break out early when possible.
                    for (; i < Nspect; ++i)
                    {
                        const f64 lambdaI = spect.wavelength(i);
                        // NOTE(cmo): Early termination condition
                        if (lambdaI > nextLambda)
                            break;

                        // NOTE(cmo): Don't do these if this is an edge case and was previously handled with constant extrapolation
                        if (doLowerHalf && spect.prdActive(i) && lambdaI > prevLambda && lambdaI <= lambdaRest)
                        {
                            const f64 frac = (lambdaI - prevLambda) / (lambdaRest - prevLambda);
                            coeffVec(k).emplace_back(Prd::JInterpCoeffs{spect.la_to_prdLa(i), frac});
                        }
                        else if (doUpperHalf && spect.prdActive(i) && lambdaI > lambdaRest && lambdaI < nextLambda)
                        {
                            const f64 frac = (lambdaI - lambdaRest) / (nextLambda - lambdaRest);
                            coeffVec(k).emplace_back(Prd::JInterpCoeffs{spect.la_to_prdLa(i), 1.0 - frac});
                        }
                    }
                }
            }
        }
    }

    for (auto& p : prdLines)
    {
        auto& wavelength = p.line->wavelength;
        p.line->prdStorage.hPrdCoeffs = Prd::RhoCoeffVec(wavelength.shape(0), atmos.Nrays, 2, atmos.Nspace);
        p.line->hPrdCoeffs = p.line->prdStorage.hPrdCoeffs;
        auto& coeffs = p.line->hPrdCoeffs;
        for (int lt = 0; lt < wavelength.shape(0); ++lt)
        {
            for (int mu = 0; mu < atmos.Nrays; ++mu)
            {
                for (int toObs = 0; toObs <= 1; ++toObs)
                {
                    for (int k = 0; k < atmos.Nspace; ++k)
                    {
                        const f64 s = sign[toObs];
                        const f64 lambdaRest = wavelength(lt) * (1.0 + atmos.vlosMu(mu, k) * s / C::CLight);
                        auto& c = coeffs(lt, mu, toObs, k);
                        if (lambdaRest <= wavelength(0))
                        {
                            c.frac = 0.0;
                            c.i0 = 0;
                            c.i1 = 1;
                        }
                        else if (lambdaRest >= wavelength(wavelength.shape(0)-1))
                        {
                            c.frac = 1.0;
                            c.i0 = wavelength.shape(0) - 2;
                            c.i1 = wavelength.shape(0) - 1;
                        }
                        else
                        {
                            auto it = std::upper_bound(wavelength.data, wavelength.data + wavelength.shape(0), lambdaRest) - 1;
                            c.frac = (lambdaRest - *it) / (*(it+1) - *it);
                            c.i0 = it - wavelength.data;
                            c.i1 = c.i0 + 1;
                        }
                    }
                }
            }
        }
    }
}