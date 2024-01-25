#include "Lightweaver.hpp"
#include "Utils.hpp"
#include "SimdFullIterationTemplates.hpp"
#include "PrdTemplates.hpp"
#include "TaskSetWrapper.hpp"

namespace PrdCores
{
void total_depop_elastic_scattering_rate(const Transition* trans, const Atom& atom,
                                         F64View PjQj)
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
constexpr f64 PrdQCore = 2.0;
constexpr f64 PrdQSpread = 5.0;
constexpr f64 PrdDQ = 0.15;

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

f64 GII(f64 aDamp, f64 qEmit, f64 qAbs)
{
    // NOTE(cmo): qAbs: nu, qEmit: nu'
    // Specialised for non XRD case. Change waveratio to allow XRD.
    constexpr f64 waveratio = 1.0;
    namespace C = Constants;

    // The function is symmetric about nu' = 0
    if (qEmit < 0.0)
    {
        qEmit = -qEmit;
        qAbs = -qAbs;
    }


    f64 giiCore = 0.0;
    f64 coreFactor = 0.0;
    if (qEmit < PrdQWing)
    {
        // In core, or "transition", i.e. core value is needed.
        if ((qAbs < -PrdQWing) || (qAbs > qEmit + waveratio * PrdQSpread))
            // Outside the range of Rii
            return 0.0;

        if (abs(qAbs) <= qEmit)
            giiCore = G_zero(qEmit);
        else
            giiCore = exp(square(qEmit) - square(qAbs)) * G_zero(qAbs);

        if (qEmit >= PrdQCore && qEmit <= PrdQWing)
        {
            // "transition" regime
            f64 phiCore = exp(-square(qEmit));
            f64 phiWing = aDamp / (sqrt(C::Pi) * (square(aDamp) + square(qEmit)));
            coreFactor = phiCore / (phiCore + phiWing);
        }
        else
            return giiCore;
    }

    f64 gii = 0.0;
    if (qEmit >= PrdQCore)
    {
        // Wing value needed.
        f64 aqEmit = waveratio * qEmit;
        if ((qEmit >= PrdQWing) &&
            (abs(qAbs - aqEmit) > waveratio * PrdQSpread))
                // Outside the range of Rii
                return 0.0;

        f64 uMin = abs((qAbs - aqEmit) / (1.0 + waveratio));
        f64 giiWing = (1.0 + waveratio) * (1.0 - 2.0 * uMin * G_zero(uMin))
                        * exp(-square(uMin)) / (2.0 * waveratio * sqrt(C::Pi));

        if (waveratio == 1.0)
        {
            // Gouttebroze 1986 second order expansion for Rii.
            f64 ratio = qAbs / qEmit;
            giiWing *= (2.75 - (2.5 - 0.75 * ratio) * ratio);
        }
        else
        {
            // Uitenbroek 1989 general wing term.
            f64 u1 = abs((qAbs - aqEmit) / (waveratio - 1.0));
            giiWing -= abs(1.0 - waveratio) * (1.0 - 2.0 * u1 * G_zero(u1))
                        * exp(-square(u1));
        }

        // Compute the linear combination of core and wing terms (with
        // coreFactor = 0), if not in the "transition" range.
        gii = coreFactor * giiCore + (1.0 - coreFactor) * giiWing;
    }
    return gii;
}

constexpr int max_fine_grid_size()
{
    return max(2 * PrdQWing + PrdQSpread, 2 * PrdQSpread) / PrdDQ + 1;
}

void optimised_fine_linear(F64View xTable, F64View yTable, F64View x, F64View y)
{
    // NOTE(cmo): Here we are going to work on the assumption that x is
    // monotonic increasing, and varying slowly in comparison to xTable. This is the case
    // of the fine wavelength grid against the line's base grid. For that reason
    // we will find the initial index with upper bound, and then linearly search
    // upwards from there.

    const int N = x.shape(0);
    const int Ntable = xTable.shape(0);
    if (N < 1)
        return;

    // NOTE(cmo): Iter is a an iterator to the location in the table that upper
    // bounds our point.
    f64* iter;
    f64* start = xTable.data;
    f64* end = start + Ntable;
    if (x(0) <= xTable(0))
        iter = &xTable(0);
    else if (x(0) >= xTable(Ntable - 1))
        iter = &xTable(Ntable - 1);
    else
        iter = std::upper_bound(start, end, x(0));

    for (int i = 0; i < x.shape(0); ++i)
    {
        while (iter < end && *iter <= x(i))
            ++iter;

        if (iter == end)
        {
            y(i) = yTable(Ntable - 1);
            continue;
        }
        else if (iter == start)
        {
            y(i) = yTable(0);
            continue;
        }

        auto prev = iter - 1;
        auto xp = *prev;
        auto xn = *iter;
        f64 t = (x(i) - xp) / (xn - xp);
        y(i) = (1.0 - t) * yTable(prev - start) + t * yTable(iter - start);
    }
}

void optimised_fine_linear_fixed_spacing(F64View xTable, F64View yTable,
                                         f64 xStart, f64 xStep, int N, F64View y)
{
    // NOTE(cmo): Here we are going to work on the assumption that x is
    // monotonic increasing from xStart with N fixed steps of xStep, and typically
    // varying slowly in comparison to xTable. This is the case of the fine
    // wavelength grid against the line's base grid. For that reason we will
    // find the initial index with upper bound, and then linearly search upwards
    // from there.

    const int Ntable = xTable.shape(0);
    if (N < 1)
        return;

    // NOTE(cmo): Iter is a an iterator to the location in the table that upper
    // bounds our point.
    f64* iter;
    f64* start = xTable.data;
    f64* end = start + Ntable;
    f64 x = xStart;
    if (x <= xTable(0))
        iter = &xTable(0);
    else if (x >= xTable(Ntable - 1))
        iter = &xTable(Ntable - 1);
    else
        iter = std::upper_bound(start, end, x);

    for (int i = 0; i < N; ++i)
    {
        x = xStart + i * xStep;
        while (iter < end && *iter <= x)
            ++iter;

        if (iter == end)
        {
            y(i) = yTable(Ntable - 1);
            continue;
        }
        else if (iter == start)
        {
            y(i) = yTable(0);
            continue;
        }

        auto prev = iter - 1;
        auto xp = *prev;
        auto xn = *iter;
        f64 t = (x - xp) / (xn - xp);
        y(i) = (1.0 - t) * yTable(prev - start) + t * yTable(iter - start);
    }
}


std::pair<f64, f64>
scattering_int_range(f64 qEmit)
{
    // NOTE(cmo): Find integration range around qEmit for which GII is
    // non-zero. (Resonance PRD case only).
    f64 q0, qN;
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
    return { q0, qN };
}

std::pair<i32, i32>
fine_grid_idxs(f64 qEmit, const F64View& qFine)
{
    auto [q0, qN] = scattering_int_range(qEmit);
    auto start = qFine.data;
    const int NlambdaFine = qFine.shape(0);
    auto startIdx = std::upper_bound(start, start + NlambdaFine,
                                        q0) - start;
    if (startIdx != 0)
        startIdx -= 1;
    auto endIdx = std::upper_bound(start, start + NlambdaFine,
                                    qN) - start;
    return { startIdx, endIdx };
}


struct ThreadData
{
    Transition& trans;
    const Atom& atom;
    const Spectrum& spect;
    const Atmosphere& atmos;
    const F64View& PjQj;
    const bool computeGii;

    F64Arr Jk;
    F64Arr JFine;

    ThreadData(Transition& t, const Atom& a,
                const Spectrum& s, const Atmosphere& atmos,
                const F64View& Pj, bool initialiseG,
                int Nlambda, int maxFineGrid)
        : trans(t),
          atom(a),
          spect(s),
          atmos(atmos),
          PjQj(Pj),
          computeGii(initialiseG),
          Jk(Nlambda),
          JFine(maxFineGrid)
    {}
};


#if 0
struct PrdLineGrid
{
    F64Arr q;
    F64Arr dq;
};

PrdLineGrid compute_prd_line_grid(Transition* t, const F64View& qTrans)
{
    auto& trans = *t;
    const int Nlambda = trans.wavelength.shape(0);
    F64Arr q;
    auto& qp = q.dataStore;
    qp.reserve(Nlambda);
    qp.emplace_back(qTrans(0));

    int i = 1;
    while (i < Nlambda)
    {
        if (qTrans(i) > qp.back() + PrdDQ)
        {
            qp.emplace_back(qp.back() + PrdDQ);
        }
        else
        {
            qp.emplace_back(qTrans(i));
            i += 1;
        }
    }
    q.dim0 = qp.size();
    const int NlambdaFine = q.shape(0);

    assert(NlambdaFine > 3);
    F64Arr dq(NlambdaFine);
    dq(0) = 0.5 * (q(1) - q(0));
    for (int i = 1; i < NlambdaFine - 1; ++i)
    {
        dq(i) = 0.5 * (q(i+1) - q(i-1));
    }
    dq(NlambdaFine - 1) = 0.5 * (q(NlambdaFine - 1) - q(NlambdaFine - 2));
    return PrdLineGrid { q, dq };
}
void cmo_scattering_int(void* userdata, scheduler* s,
                        sched_task_partition p, sched_uint threadId)
{
    // NOTE(cmo): This function uses an approach that creates one fine
    // wavelength grid for each depth, and allows J to only be interpolated once
    // per depth. As it needs to contain at least the every wavelength in the
    // line's grid (so the integrals can be centred), and no gaps greater than
    // PrdDQ, we end up dramatically oversampling the integral in the line core
    // and it is a little slower than the other approach (for the case of
    // optimised linear interpolation). I am leaving the approach here, as it
    // may be valuable if a more accurate (and expensive) form of interpolation
    // were instead desired.
    // To use this approach, add
    /*
        Jasnah::Array2Own<i32> fineStart;
        Jasnah::Array2Own<i32> fineEnd;
        Jasnah::Array1Own<F64Arr> qFine;
        Jasnah::Array1Own<F64Arr> wq;
    */
    // into the PrdStorage struct, and change the call in prd_scatter
    namespace C = Constants;
    ThreadData& data = ((ThreadData*)userdata)[threadId];
    JasUnpack(data, trans, atom, spect, atmos, PjQj);
    JasUnpack(data, Jk);
    const bool computeGii = data.computeGii;
    const int Nlambda = trans.wavelength.shape(0);

    for (int k = p.start; k < p.end; ++k)
    {
        f64 gammaPrefactor = atom.n(trans.i, k) / atom.n(trans.j, k) * trans.Bij / PjQj(k);
        f64 Jbar = trans.Rij(k) / trans.Bij;

        // NOTE(cmo): Local mean intensity (in rest frame if using HPRD).
        if (spect.JRest)
        {
            for (int la = 0; la < Nlambda; ++la)
            {
                int prdLa = spect.la_to_prdLa(la + trans.Nblue);
                Jk(la) = spect.JRest(prdLa, k);
            }
        }
        else
        {
            for (int la = 0; la < Nlambda; ++la)
            {
                Jk(la) = spect.J(la + trans.Nblue, k);
            }
        }

        if (computeGii)
        {
            JasUnpack(trans.prdStorage, gII, qWave, qFine, wq);
            JasUnpack(trans.prdStorage, fineStart, fineEnd);
            auto qWavek = qWave(k);
            for (int la = 0; la < Nlambda; ++la)
            {
                f64 qEmit = (trans.wavelength(la) - trans.lambda0)
                        * C::CLight / (trans.lambda0 * atom.vBroad(k));
                qWavek(la) = qEmit;
            }
            auto grid = compute_prd_line_grid(&trans, qWavek);
            qFine(k) = grid.q;
            wq(k) = grid.dq;

            f64 aDamp = trans.aDamp(k);
            for (int la = 0; la < Nlambda; ++la)
            {
                f64 qEmit = qWavek(la);
                auto [startIdx, endIdx] = fine_grid_idxs(qEmit, qFine(k));
                fineStart(k, la) = startIdx;
                fineEnd(k, la) = endIdx;

                int len = endIdx - startIdx;
                gII(k, la) = F64Arr(len);
                auto& gIILine = gII(k, la);
                auto qFinek = qFine(k);
                for (int laFine = startIdx; laFine < endIdx; ++laFine)
                {
                    int lag = laFine - startIdx;
                    gIILine(lag) = GII(aDamp, qEmit, qFinek(laFine));
                }
            }
        }
        auto& coeffs = trans.prdStorage;
        F64Arr JFine(coeffs.qFine(k).shape(0));
        optimised_fine_linear(coeffs.qWave(k), Jk, coeffs.qFine(k), JFine);

        for (int la = 0; la < Nlambda; ++la)
        {
            f64 qEmit = coeffs.qWave(k, la);
            int startIdx = coeffs.fineStart(k, la);
            int endIdx = coeffs.fineEnd(k, la);

            F64View gII = coeffs.gII(k, la);
            F64View wq = coeffs.wq(k);

            // NOTE(cmo): Compute and normalise scattering integral.
            f64 gNorm = 0.0;
            f64 scatInt = 0.0;
            for (int laF = startIdx; laF < endIdx; ++laF)
            {
                // NOTE(cmo): Normalisation of the scattering integral is very
                // important, as discussed in HM2014 Sec 15.4. Whilst this
                // procedure may slightly distort the redistribution function,
                // it ensures that no photons are gained or lost in this
                // evaluation.
                f64 gii = gII(laF - startIdx);
                gii *= wq(laF);
                gNorm += gii;
                scatInt += JFine(laF) * gii;
            }
            trans.rhoPrd(la, k) += gammaPrefactor * (scatInt / gNorm - Jbar);
        }
    }
}
#endif

void scattering_int(ThreadData& data, int k)
{
    namespace C = Constants;
    JasUnpack(data, trans, atom, spect, PjQj);
    JasUnpack(data, Jk, JFine);
    const bool computeGii = data.computeGii;
    const int Nlambda = trans.wavelength.shape(0);

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
        }
    }
    else
    {
        for (int la = 0; la < Nlambda; ++la)
        {
            Jk(la) = spect.J(la + trans.Nblue, k);
        }
    }

    if (computeGii)
    {
        auto qWavek = trans.prdData->qWave(k);
        for (int la = 0; la < Nlambda; ++la)
        {
            f64 qEmit = (trans.wavelength(la) - trans.lambda0)
                    * C::CLight / (trans.lambda0 * atom.vBroad(k));
            qWavek(la) = qEmit;
        }
    }
    auto& coeffs = *trans.prdData;

    for (int la = 0; la < Nlambda; ++la)
    {
        f64 qEmit = coeffs.qWave(k, la);
        auto [q0, qN] = scattering_int_range(qEmit);

        // NOTE(cmo): Our grid is just linearly spaced in PrdDQ, so no need to
        // make it explicit, as we can adapt the surrounding functions.
        int Np = int((f64)(qN - q0) / PrdDQ) + 1;

        // NOTE(cmo): Linearly interpolate mean intensity onto this grid.
        optimised_fine_linear_fixed_spacing(coeffs.qWave(k), Jk, q0, PrdDQ, Np, JFine);

        if (computeGii)
        {
            // NOTE(cmo): Compute gII if needed.
            // Integration weights for general trapezoidal rule obtained
            // from averaging extended Simpson's rule with modified
            // Simpson's rule where both edge regions are treated with
            // trapezoid rule. Takes accuracy up to O(1/N^3). Explained in
            // Press et al, Num Rec Sec4.2.
            // NOTE(cmo): Avoid needing explicit storage for wq
            JasUnpack((*trans.prdData), gII, qWave);
            auto qWavek = qWave(k);
            f64 aDamp = trans.aDamp(k);
            f64 qEmit = qWavek(la);
            gII(k, la) = F64Arr(Np);
            auto& gIILine = gII(k, la);
            f64 qPrime = q0;
            gIILine(0) = GII(aDamp, qEmit, qPrime) * 5.0 / 12.0 * PrdDQ;
            qPrime += PrdDQ;
            gIILine(1) = GII(aDamp, qEmit, qPrime) * 13.0 / 12.0 * PrdDQ;
            for (int laFine = 2; laFine < Np - 2; ++laFine)
            {
                qPrime += PrdDQ;
                gIILine(laFine) = GII(aDamp, qEmit, qPrime) * PrdDQ;
            }
            qPrime += PrdDQ;
            gIILine(Np - 2) = GII(aDamp, qEmit, qPrime) * 13.0 / 12.0 * PrdDQ;
            qPrime += PrdDQ;
            gIILine(Np - 1) = GII(aDamp, qEmit, qPrime) * 5.0 / 12.0 * PrdDQ;
        }

        F64View gII = coeffs.gII(k, la);

        // NOTE(cmo): Compute and normalise scattering integral.
        f64 gNorm = 0.0;
        f64 scatInt = 0.0;
        for (int laF = 0; laF < Np; ++laF)
        {
            // NOTE(cmo): Normalisation of the scattering integral is very
            // important, as discussed in HM2014 Sec 15.4. Whilst this
            // procedure may slightly distort the redistribution function,
            // it ensures that no photons are gained or lost in this
            // evaluation.
            f64 gii = gII(laF);
            gNorm += gii;
            scatInt += JFine(laF) * gii;
        }
        trans.rhoPrd(la, k) += gammaPrefactor * (scatInt / gNorm - Jbar);
    }
}

void scattering_int_handler(void* userdata, enki::TaskScheduler* s,
                            enki::TaskSetPartition p, u32 threadId)
{
    ThreadData& data = ((ThreadData*)userdata)[threadId];
    for (int k = p.start; k < p.end; ++k)
    {
        scattering_int(data, k);
    }
}

void prd_scatter(Transition* t, F64View PjQj, const Atom& atom,
                 const Atmosphere& atmos, const Spectrum& spect,
                 enki::TaskScheduler* sched)
{
    auto& trans = *t;

    namespace C = Constants;
    const int Nlambda = trans.wavelength.shape(0);

    bool initialiseGii = !trans.prdData->upToDate;
    constexpr int maxFineGrid = max_fine_grid_size();
    if (initialiseGii)
    {
        JasUnpack((*trans.prdData), gII, qWave);
        auto& c = *trans.prdData;
        if (!gII)
        {
            gII = decltype(c.gII)(atmos.Nspace, Nlambda);
            qWave = decltype(c.qWave)(atmos.Nspace, Nlambda);
#if 0
            // NOTE(cmo): For cmo_scattering_int
            JasUnpack(trans.prdStorage, qFine, wq);
            JasUnpack(trans.prdStorage, fineStart, fineEnd);
            qFine = decltype(c.qFine)(atmos.Nspace);
            wq = decltype(c.wq)(atmos.Nspace);
            fineStart = decltype(c.fineStart)(atmos.Nspace, Nlambda);
            fineEnd = decltype(c.fineEnd)(atmos.Nspace, Nlambda);
#endif
        }
        trans.prdData->upToDate = true;
    }

    // NOTE(cmo): Reset Rho
    trans.rhoPrd.fill(1.0);

    if (!sched)
    {
        ThreadData data(trans, atom, spect, atmos,
                        PjQj, initialiseGii, Nlambda, maxFineGrid);
        for (int k = 0; k < atmos.Nspace; ++k)
        {
            scattering_int(data, k);
        }
    }
    else
    {
        std::vector<ThreadData> data;
        int Nthreads = sched->GetNumTaskThreads();
        data.reserve(Nthreads);
        for (int th = 0; th < Nthreads; ++th)
            data.emplace_back(ThreadData(trans, atom, spect, atmos, PjQj,
                                         initialiseGii, Nlambda, maxFineGrid));

        {
            const int taskSize = max(atmos.Nspace / Nthreads, 1);
            LwTaskSet scatteringInts(data.data(), sched, atmos.Nspace, taskSize,
                                     scattering_int_handler);
            sched->AddTaskSetToPipe(&scatteringInts);
            sched->WaitforTask(&scatteringInts);
        }
    }
}
}

IterationResult redistribute_prd_lines(Context& ctx, int maxIter, f64 tol, ExtraParams params)
{
    if (!ctx.iterFns.redistribute_prd)
        return redistribute_prd_lines_scalar(ctx, maxIter, tol, params);
    return ctx.iterFns.redistribute_prd(ctx, maxIter, tol, params);
}

IterationResult redistribute_prd_lines_scalar(Context& ctx, int maxIter, f64 tol, ExtraParams params)
{
    return redistribute_prd_lines_template<SimdType::Scalar>(ctx, maxIter, tol, params);
}

#if 0
void setup_hprd(Context& ctx)
{
    namespace C = Constants;
    JasUnpack(*ctx, atmos, spect);
    JasUnpack(ctx, activeAtoms);

    struct PrdData
    {
        Transition* line;
        const Atom& atom;

        PrdData(Transition* l, const Atom& a)
            : line(l), atom(a)
        {}
    };
    std::vector<PrdData> prdLines;
    prdLines.reserve(16);
    for (auto& a : activeAtoms)
    {
        for (auto& t : a->trans)
        {
            if (t->rhoPrd)
            {
                prdLines.emplace_back(PrdData { t, *a });
            }
        }
    }

    if (prdLines.size() == 0)
        return;


}
#endif

// TODO(cmo): This isn't super clear, rewrite it.
void configure_hprd_coeffs(Context& ctx, bool includeDetailedAtoms)
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
    JasUnpack(ctx, activeAtoms, detailedAtoms);
    std::vector<PrdData> prdLines;
    prdLines.reserve(16);
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
    if (includeDetailedAtoms)
    {
        for (auto& a : detailedAtoms)
        {
            for (auto& t : a->trans)
            {
                if (t->rhoPrd)
                {
                    prdLines.emplace_back(PrdData(t, *a));
                }
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
            prdLinePresent = (prdLinePresent || p.line->active(la));
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
                    for (; spect.wavelength(i) > prevLambda && i > 0; --i);
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
    for (auto& p : prdLines)
    {
        // NOTE(cmo): Recompute gII every time this is called because it's
        // likely that the atmosphere has changed if this is being called again.
        p.line->recompute_gII();
    }
}