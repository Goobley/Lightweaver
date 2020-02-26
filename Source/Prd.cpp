#include "Lightweaver.hpp"
#include "Utils.hpp"

namespace PrdCores
{
void total_depop_rate(const Transition* trans, const Atom& atom, F64View Pj)
{
    const int Nspace = trans->Rij.shape(0);

    for (int k = 0; k < Nspace; ++k)
    {
        Pj(k) = trans->Qelast(k);
        for (int i = 0; i < atom.C.shape(0); ++i)
            Pj(k) += atom.C(i, trans->j, k);

        for (auto& t : atom.trans)
        {
            if (t->j == trans->j)
                Pj(k) += t->Rji(k);
            if (t->i == trans->j)
                Pj(k) += t->Rij(k);
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

void prd_scatter(Transition* t, F64View Pj, const Atom& atom, const Atmosphere& atmos, const Spectrum& spect)
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
        f64 gamma = atom.n(trans.i, k) / atom.n(trans.j, k) * trans.Bij / Pj(k);
        f64 Jbar = trans.Rij(k) / trans.Bij;

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
        // Local mean intensity in doppler units
        for (int la = 0; la < Nlambda; ++la)
        {
            qAbs(la) = (trans.wavelength(la) - trans.lambda0) * C::CLight / (trans.lambda0 * atom.vBroad(k));
        }

        for (int la = 0; la < Nlambda; ++la)
        {
            f64 qEmit = qAbs(la);

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
            int Np = int((f64)(qN - q0) / PrdDQ) + 1;
            qp(0) = q0;
            for (int lap = 1; lap < Np; ++lap)
                qp(lap) = qp(lap - 1) + PrdDQ;

            linear(qAbs, Jk, qp.slice(0, Np), JFine);

            if (initialiseGii)
            {
                wq.fill(PrdDQ);
                wq(0) = 5.0 / 12.0 * PrdDQ;
                wq(1) = 13.0 / 12.0 * PrdDQ;
                wq(Np - 1) = 5.0 / 12.0 * PrdDQ;
                wq(Np - 2) = 13.0 / 12.0 * PrdDQ;
                for (int lap = 0; lap < Np; ++lap)
                    trans.gII(la, k, lap) = GII(trans.aDamp(k), qEmit, qp(lap)) * wq(lap);
            }
            F64View gII = trans.gII(la, k);

            f64 gNorm = 0.0;
            f64 scatInt = 0.0;
            for (int lap = 0; lap < Np; ++lap)
            {
                gNorm += gII(lap);
                scatInt += JFine(lap) * gII(lap);
            }
            trans.rhoPrd(la, k) += gamma * (scatInt / gNorm - Jbar);
        }
    }
}
}

f64 formal_sol_prd_update_rates(Context& ctx, ConstView<i32> wavelengthIdxs)
{
    using namespace LwInternal;
    JasUnpack(*ctx, atmos, spect, background);
    JasUnpack(ctx, activeAtoms, detailedAtoms);

    const int Nspace = atmos.Nspace;
    const int Nrays = atmos.Nrays;

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
    IntensityCoreData iCore;
    JasPackPtr(iCore, atmos, spect, fd, background);
    JasPackPtr(iCore, activeAtoms, detailedAtoms, JDag);
    JasPack(iCore, chiTot, etaTot, Uji, Vij, Vji);
    JasPack(iCore, I, S, Ieff);

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

f64 formal_sol_prd_update_rates(Context& ctx, const std::vector<int>& wavelengthIdxs)
{
    return formal_sol_prd_update_rates(ctx, ConstView<i32>(wavelengthIdxs.data(), wavelengthIdxs.size()));
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
                // t->zero_rates();
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
    F64Arr Pj(atmos.Nspace);
    while (iter < maxIter)
    {
        ++iter;
        dRho = 0.0;
        for (auto& p : prdLines)
        {
            PrdCores::total_depop_rate(p.line, p.atom, Pj);
            PrdCores::prd_scatter(p.line, Pj, p.atom, atmos, spect);
            p.ng.accelerate(p.line->rhoPrd.flatten());
            dRho = max(dRho, p.ng.max_change());
        }

        formal_sol_prd_update_rates(ctx, idxsForFs);

        if (dRho < tol)
            break;
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


    // NOTE(cmo): We can't simply store the prd wavelengths and then only
    // compute JRest from those. JRest can be only prdIdxs long, but we need to
    // compute all of the contributors to each of these prdIdx.
    // NOTE(cmo): This might be overcomplicating the problem. STiC is happy to
    // only compute these terms for the prd idxs. But I worry if a high Doppler
    // shift were to brind intensity into the prdLine region. I don't think it's
    // massively likely, but 500km/s is 0.8nm @ 500nm, and I don't want the line
    // wings missing Jbar that they should have.
    // NOTE(cmo): Okay, I need to think about this one some more (note that in
    // its current state, something is exploding, but that's neither here nor
    // there in realm of design). Currently when we compute the udpated the J
    // for use internal to the prd_scatter function, we only loop over the prd
    // wavelengths. Therefore it wouldn't be consistent to add in contributions
    // from outside that range. We would therefore have to extend the wavelength
    // range over which the FS was being calculated to continue using this wider
    // definition. I don't know if that's worthwhile, especially as, for the
    // most part, PRD is making lines narrower rather than wider. i.e.
    // conecentrating the intensity towards the core, which is where this method
    // will be fine anyway. This is already an approximation, so it's probably
    // best not to overcomplicate it

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