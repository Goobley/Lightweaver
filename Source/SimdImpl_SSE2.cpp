#include "Constants.hpp"
#include "Simd.hpp"
#include "LwInternal.hpp"
#include "Lightweaver.hpp"
#include "Bezier.hpp"
#include "JasPP.hpp"
#include "ThreadStorage.hpp"

#include <immintrin.h>
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>
#include <algorithm>
#include <chrono>

#include "SimdFullIterationTemplates.hpp"
#include "PrdTemplates.hpp"

namespace LwInternal
{
inline ForceInline __m128d fmadd_pd(__m128d a, __m128d b, __m128d c)
{
    return _mm_add_pd(_mm_mul_pd(a, b), c);
}

inline __m128d exp_pd_sse2(__m128d xIn)
{
    // NOTE(cmo): This is just a thin wrapper around the system exp, as I don't
    // think SSE2 will be commonly used, and the lack of fma + round isn't great
    // for performance here anyway.
    alignas(16) f64 xs[2];
    _mm_store_pd(xs, xIn);
    for (int i = 0; i < 2; ++i)
        xs[i] = exp(xs[i]);
    __m128d xOut = _mm_load_pd(xs);
    return xOut;
}

template <>
inline ForceInline
void setup_wavelength_opt<SimdType::SSE2>(Atom* atom, int laIdx)
{
    constexpr SimdType simd = SimdType::SSE2;
    constexpr int Stride = SimdWidth[(size_t)simd];
    const int Nspace = atom->atmos->Nspace;
    const int Nremainder = atom->atmos->Nspace % Stride;
    const int kMax = Nspace - Nremainder;

    namespace C = Constants;
    constexpr f64 pi4_h = 4.0 * C::Pi / C::HPlanck;
    constexpr f64 hc_4pi = 0.25 * C::HC / C::Pi;
    constexpr f64 pi4_hc = 1.0 / hc_4pi;
    constexpr f64 hc_k = C::HC / (C::KBoltzmann * C::NM_TO_M);


    for (int kr = 0; kr < atom->Ntrans; ++kr)
    {
        auto& t = *atom->trans[kr];
        if (!t.active(laIdx))
            continue;

        auto g = atom->gij(kr);
        auto w = atom->wla(kr);
        const int lt = t.lt_idx(laIdx);
        auto wlambda = t.wlambda(lt);
        if (t.type == TransitionType::LINE)
        {
            int k = 0;
            __m128d bRatio = _mm_set1_pd(t.Bji / t.Bij);
            __m128d wlaPi = _mm_set1_pd(wlambda * pi4_hc);
            for (; k < kMax; k += Stride)
            {
                _mm_storeu_pd(&g(k), bRatio);
                __m128d wphik = _mm_load_pd(&t.wphi(k));
                _mm_storeu_pd(&w(k), _mm_mul_pd(wlaPi, wphik));
            }
            for (; k < Nspace; ++k)
            {
                g(k) = t.Bji / t.Bij;
                w(k) = wlambda * t.wphi(k) * pi4_hc;
            }
        }
        else
        {
            const f64 hc_kl = hc_k / t.wavelength(t.lt_idx(laIdx));
            const f64 wlambda_lambda = wlambda / t.wavelength(t.lt_idx(laIdx));
            const auto& atmos = atom->atmos;
            auto& nStar = atom->nStar;
            __m128d mhc_kl4x = _mm_set1_pd(-hc_kl);
            __m128d wTerm = _mm_set1_pd(wlambda_lambda * pi4_h);
            int k = 0;
            for (; k < kMax; k += Stride)
            {
                __m128d nik = _mm_loadu_pd(&nStar(t.i, k));
                __m128d njk = _mm_loadu_pd(&nStar(t.j, k));
                __m128d temp = _mm_load_pd(&atmos->temperature(k));
                __m128d gk = _mm_mul_pd(_mm_div_pd(nik, njk),
                                           exp_pd_sse2(_mm_div_pd(mhc_kl4x, temp)));
                _mm_storeu_pd(&g(k), gk);
                _mm_storeu_pd(&w(k), wTerm);
            }
            for (; k < Nspace; ++k)
            {
                g(k) = nStar(t.i, k) / nStar(t.j, k) * exp(-hc_kl / atmos->temperature(k));
                w(k) = wlambda_lambda * pi4_h;
            }
        }

        // NOTE(cmo): We have to do a linear interpolation on rhoPrd in the
        // case of hybrid PRD, so we can't pre-multiply here in that
        // instance.
        if (t.rhoPrd && !t.hPrdCoeffs)
        {
            for (int k = 0; k < g.shape(0); ++k)
                g(k) *= t.rhoPrd(lt, k);

        }

        if (!t.gij)
            t.gij = g;
    }
}

template <>
inline ForceInline void
uv_opt<SimdType::SSE2>(Transition* t, int la, int mu, bool toObs,
                          F64View Uji, F64View Vij, F64View Vji)
{
    constexpr SimdType simd = SimdType::SSE2;
    constexpr int Stride = SimdWidth[(size_t)simd];
    const int Nspace = Vij.shape(0);
    const int Nremainder = Nspace % Stride;
    const int kMax = Nspace - Nremainder;
    namespace C = Constants;
    int lt = t->lt_idx(la);

    if (t->type == TransitionType::LINE)
    {
        constexpr f64 hc_4pi = 0.25 * C::HC / C::Pi;
        // NOTE(cmo): See note on change in LwTransition.hpp
        const f64 hnu_4pi = hc_4pi * (t->lambda0 / t->wavelength(lt));
        auto p = t->phi(lt, mu, (int)toObs);
        int k = 0;
        for (; k < kMax; k += Stride)
        {
            // Vij(k) = hc_4pi * t->Bij * p(k);
            __m128d phik = _mm_loadu_pd(&p(k));
            // __m128d Vijk = _mm_mul_pd(_mm_set1_pd(hc_4pi * t->Bij), phik);
            __m128d Vijk = _mm_mul_pd(_mm_set1_pd(hnu_4pi * t->Bij), phik);
            _mm_store_pd(&Vij(k), Vijk);

            // Vji(k) = t->gij(k) * Vij(k);
            __m128d gijk = _mm_loadu_pd(&t->gij(k));
            _mm_store_pd(&Vji(k), _mm_mul_pd(gijk, Vijk));
        }
        for (; k < Nspace; ++k)
        {
            Vij(k) = hc_4pi * t->Bij * p(k);
            Vji(k) = t->gij(k) * Vij(k);
        }
        // NOTE(cmo): Do the HPRD linear interpolation on rho here
        // As we make Uji, Vij, and Vji explicit, there shouldn't be any need for direct access to gij
        if (t->hPrdCoeffs)
        {
            for (int k = 0; k < Vij.shape(0); ++k)
            {
                const auto& coeffs = t->hPrdCoeffs(lt, mu, toObs, k);
                f64 rho = (1.0 - coeffs.frac) * t->rhoPrd(coeffs.i0, k)
                               + coeffs.frac * t->rhoPrd(coeffs.i1, k);
                Vji(k) *= rho;
            }
        }
        const f64 ABRatio = t->Aji / t->Bji;
        const __m128d ABRatio4x = _mm_set1_pd(ABRatio);
        k = 0;
        for (; k < kMax; k += Stride)
        {
            __m128d Vjik = _mm_load_pd(&Vji(k));
            _mm_store_pd(&Uji(k), _mm_mul_pd(ABRatio4x, Vjik));
        }
        for (; k < Nspace; ++k)
        {
            Uji(k) = ABRatio * Vji(k);
        }
    }
    else
    {
        constexpr f64 twoHc = 2.0 * C::HC / cube(C::NM_TO_M);
        const f64 hcl = twoHc / cube(t->wavelength(lt));
        const f64 a = t->alpha(lt);
        int k = 0;
        for (; k < kMax; k += Stride)
        {
            // Vij(k) = a;
            __m128d Vijk = _mm_set1_pd(a);
            _mm_store_pd(&Vij(k), Vijk);
            // Vji(k) = t->gij(k) * Vij(k);
            __m128d Vjik = _mm_mul_pd(_mm_loadu_pd(&t->gij(k)), Vijk);
            _mm_store_pd(&Vji(k), Vjik);
            // Uji(k) = hcl * Vji(k);
            _mm_store_pd(&Uji(k), _mm_mul_pd(_mm_set1_pd(hcl), Vjik));
        }
        for (; k < Nspace; ++k)
        {
            Vij(k) = a;
            Vji(k) = t->gij(k) * Vij(k);
            Uji(k) = hcl * Vji(k);
        }
    }
}

template <SimdType simd, bool iClean, bool jClean,
          bool FirstTrans, bool ComputeOperator,
          typename std::enable_if_t<simd == SimdType::SSE2, bool> = true>
inline void ForceInline
chi_eta_aux_accum(IntensityCoreData* data, Atom* atom, const Transition& t)
{
    constexpr int Stride = SimdWidth[(size_t)simd];
    JasUnpack((*data), Uji, Vij, Vji, chiTot, etaTot);
    const int Nspace = data->atmos->Nspace;
    const int kRemainder = Nspace % Stride;
    const int kMax = Nspace - kRemainder;

    __m128d zeroWide = _mm_setzero_pd();

    int k = 0;
    for (; k < kMax; k += Stride)
    {
        __m128d nik = _mm_loadu_pd(&atom->n(t.i, k));
        __m128d njk = _mm_loadu_pd(&atom->n(t.j, k));
        __m128d Vijk = _mm_load_pd(&Vij(k));
        __m128d Vjik = _mm_load_pd(&Vji(k));
        __m128d Ujik = _mm_load_pd(&Uji(k));
        // f64 chi = atom->n(t.i, k) * Vij(k) - atom->n(t.j, k) * Vji(k);
        __m128d chik = _mm_sub_pd(_mm_mul_pd(nik, Vijk), _mm_mul_pd(njk, Vjik));

        // f64 eta = atom->n(t.j, k) * Uji(k);
        __m128d etak = _mm_mul_pd(njk, Ujik);

        if constexpr (ComputeOperator)
        {
            if constexpr (iClean)
            {
                // atom->chi(t.i, k) += chi;
                __m128d chiic = _mm_loadu_pd(&atom->chi(t.i, k));
                _mm_storeu_pd(&atom->chi(t.i, k), _mm_add_pd(chiic, chik));
            }
            else
            {
                // atom->chi(t.i, k) = chi;
                _mm_storeu_pd(&atom->chi(t.i, k), chik);
                // atom->U(t.i, k) = 0.0;
                _mm_storeu_pd(&atom->U(t.i, k), zeroWide);
            }

            if constexpr (jClean)
            {
                // atom->chi(t.j, k) -= chi;
                __m128d chijc = _mm_loadu_pd(&atom->chi(t.j, k));
                _mm_storeu_pd(&atom->chi(t.j, k), _mm_sub_pd(chijc, chik));
                // atom->U(t.j, k) += Uji(k);
                __m128d Uc = _mm_loadu_pd(&atom->U(t.j, k));
                _mm_storeu_pd(&atom->U(t.j, k), _mm_add_pd(Uc, Ujik));
            }
            else
            {
                // atom->chi(t.j, k) = -chi;
                __m128d chim = _mm_xor_pd(chik, _mm_set1_pd(-0.0));
                _mm_storeu_pd(&atom->chi(t.j, k), chim);
                // atom->U(t.j, k) = Uji(k);
                _mm_storeu_pd(&atom->U(t.j, k), Ujik);
            }

            if constexpr (FirstTrans)
            {
                // atom->eta(k) = eta;
                _mm_store_pd(&atom->eta(k), etak);
            }
            else
            {
                // atom->eta(k) += eta;
                __m128d etakc = _mm_load_pd(&atom->eta(k));
                _mm_store_pd(&atom->eta(k), _mm_add_pd(etakc, etak));
            }
        }

        // chiTot(k) += chi;
        // etaTot(k) += eta;
        __m128d etaTotc = _mm_load_pd(&etaTot(k));
        __m128d chiTotc = _mm_load_pd(&chiTot(k));
        _mm_store_pd(&etaTot(k), _mm_add_pd(etaTotc, etak));
        _mm_store_pd(&chiTot(k), _mm_add_pd(chiTotc, chik));
    }
    for (; k < Nspace; ++k)
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

template <>
inline ForceInline void
compute_source_fn<SimdType::SSE2>(F64View& S, F64View& etaTot,
                                  F64View& chiTot, F64View& sca,
                                  F64View& JDag)
{
    constexpr SimdType simd = SimdType::SSE2;
    constexpr int Stride = SimdWidth[(size_t)simd];
    const int Nspace = S.shape(0);
    const int Nremainder = Nspace % Stride;
    const int kMax = Nspace - Nremainder;
    int k = 0;
    for (; k < kMax; k += Stride)
    {
        __m128d etak = _mm_load_pd(&etaTot(k));
        __m128d chik = _mm_load_pd(&chiTot(k));
        __m128d scak = _mm_loadu_pd(&sca(k));
        __m128d Jk = _mm_loadu_pd(&JDag(k));
        __m128d num = fmadd_pd(scak, Jk, etak);
        __m128d Sk = _mm_div_pd(num, chik);
        _mm_store_pd(&S(k), Sk);
    }
    for (; k < Nspace; ++k)
    {
        S(k) = (etaTot(k) + sca(k) * JDag(k)) / chiTot(k);
    }
}

template <>
inline ForceInline void
accumulate_J<SimdType::SSE2>(f64 halfwmu, F64View& J, F64View& I)
{
    constexpr SimdType simd = SimdType::SSE2;
    constexpr int Stride = SimdWidth[(size_t)simd];
    const int Nspace = I.shape(0);
    const int Nremainder = Nspace % Stride;
    const int kMax = Nspace - Nremainder;
    int k = 0;
    __m128d halfwmuWide = _mm_set1_pd(halfwmu);
    for (; k < kMax; k += Stride)
    {
        __m128d Jk = _mm_loadu_pd(&J(k));
        __m128d Ik = _mm_load_pd(&I(k));
        _mm_storeu_pd(&J(k), fmadd_pd(halfwmuWide, Ik, Jk));
    }
    for (; k < Nspace; ++k)
    {
        J(k) += halfwmu * I(k);
    }
}

template <>
inline ForceInline void
compute_full_Ieff<SimdType::SSE2>(F64View& I, F64View& PsiStar,
                                  F64View& eta, F64View& Ieff)
{

    constexpr SimdType simd = SimdType::SSE2;
    constexpr int Stride = SimdWidth[(size_t)simd];
    const int Nspace = I.shape(0);
    const int Nremainder = Nspace % Stride;
    const int kMax = Nspace - Nremainder;

    int k = 0;
    for (; k < kMax; k += Stride)
    {
        __m128d Ik = _mm_load_pd(&I(k));
        __m128d Psik = _mm_load_pd(&PsiStar(k));
        __m128d etak = _mm_load_pd(&eta(k));
        _mm_store_pd(&Ieff(k), _mm_sub_pd(Ik, _mm_mul_pd(Psik, etak)));
    }
    for (; k < Nspace; ++k)
    {
        Ieff(k) = I(k) - PsiStar(k) * eta(k);
    }
}

template <SimdType simd, bool ComputeOperator, bool ComputeRates,
          typename std::enable_if_t<simd == SimdType::SSE2, bool> = true>
inline void
compute_full_operator_rates(Atom* a, int kr, f64 wmu,
                            IntensityCoreData* data)
{
    JasUnpack((*data), Uji, Vij, Vji, PsiStar, Ieff, I);
    constexpr int Stride = SimdWidth[(size_t)simd];
    const int Nspace = Uji.shape(0);
    const int kRemainder = Nspace % Stride;
    const int kMax = Nspace - kRemainder;
    auto& atom = *a;
    auto& t = *atom.trans[kr];
    int k = 0;
    for (; k < kMax; k += Stride)
    {
        // const f64 wlamu = atom.wla(kr, k) * wmu;
        __m128d wlamuk = _mm_mul_pd(_mm_loadu_pd(&atom.wla(kr, k)),
                                       _mm_set1_pd(wmu));
        __m128d Ujik = _mm_load_pd(&Uji(k));
        __m128d Vjik = _mm_load_pd(&Vji(k));
        __m128d Vijk = _mm_load_pd(&Vij(k));
        if constexpr (ComputeOperator)
        {
            __m128d Ieffk = _mm_load_pd(&Ieff(k));
            __m128d PsiStark = _mm_load_pd(&PsiStar(k));
            __m128d atomChiik = _mm_loadu_pd(&atom.chi(t.i, k));
            __m128d atomChijk = _mm_loadu_pd(&atom.chi(t.j, k));
            __m128d atomUik = _mm_loadu_pd(&atom.U(t.i, k));
            __m128d atomUjk = _mm_loadu_pd(&atom.U(t.j, k));

            // f64 integrand = (Uji(k) + Vji(k) * Ieff(k)) - (PsiStar(k) * atom.chi(t.i, k) * atom.U(t.j, k));
            __m128d term1 = _mm_add_pd(_mm_mul_pd(Vjik, Ieffk), Ujik);
            __m128d chiU = _mm_mul_pd(atomChiik, atomUjk);
            // __m128d integrand = _mm_fnmadd_pd(PsiStark, chiU, term1);
            __m128d integrand = _mm_sub_pd(term1, _mm_mul_pd(PsiStark, chiU));
            // atom.Gamma(t.i, t.j, k) += integrand * wlamu;
            __m128d currentIntegral = _mm_loadu_pd(&atom.Gamma(t.i, t.j, k));
            __m128d integralChunk = _mm_add_pd(_mm_mul_pd(integrand, wlamuk),
                                               currentIntegral);
            _mm_storeu_pd(&atom.Gamma(t.i, t.j, k), integralChunk);

            // integrand = (Vij(k) * Ieff(k)) - (PsiStar(k) * atom.chi(t.j, k) * atom.U(t.i, k));
            chiU = _mm_mul_pd(atomChijk, atomUik);
            __m128d term2 = _mm_mul_pd(PsiStark, chiU);
            integrand = _mm_sub_pd(_mm_mul_pd(Vijk, Ieffk), term2);
            // atom.Gamma(t.j, t.i, k) += integrand * wlamu;
            currentIntegral = _mm_loadu_pd(&atom.Gamma(t.j, t.i, k));
            integralChunk = _mm_add_pd(_mm_mul_pd(integrand, wlamuk), currentIntegral);
            _mm_storeu_pd(&atom.Gamma(t.j, t.i, k), integralChunk);
        }

        if constexpr (ComputeRates)
        {
            __m128d Ik = _mm_load_pd(&I(k));
            // t.Rij(k) += I(k) * Vij(k) * wlamu;
            __m128d Rijk = _mm_loadu_pd(&t.Rij(k));
            __m128d integrand = _mm_mul_pd(Ik, Vijk);
            __m128d integralChunk = _mm_add_pd(_mm_mul_pd(integrand, wlamuk), Rijk);
            _mm_storeu_pd(&t.Rij(k), integralChunk);

            // t.Rji(k) += (Uji(k) + I(k) * Vji(k)) * wlamu;
            __m128d Rjik = _mm_loadu_pd(&t.Rji(k));
            integrand = _mm_add_pd(_mm_mul_pd(Ik, Vjik), Ujik);
            integralChunk = _mm_add_pd(_mm_mul_pd(integrand, wlamuk), Rjik);
            _mm_storeu_pd(&t.Rji(k), integralChunk);
        }
    }
    for (; k < Nspace; ++k)
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
}

using LwInternal::FsMode;

IterationResult formal_sol_iteration_matrices_SSE2(Context& ctx, bool lambdaIterate, ExtraParams params)
{
    if constexpr (SSE2_available())
    {
        FsMode mode = (FsMode::UpdateJ | FsMode::UpdateRates);
        if (lambdaIterate)
            mode = mode | FsMode::PureLambdaIteration;

        return LwInternal::formal_sol_iteration_matrices_impl<SimdType::SSE2>(ctx, mode, params);
    }
    else
    {
        fprintf(stderr, "Attempted to call %s, but instruction set not available.\nThis message shouldn't appear, please open an issue.\n", __func__);
    }
}

IterationResult formal_sol_SSE2(Context& ctx, bool upOnly, ExtraParams params)
{
    FsMode mode = FsMode::FsOnly;
    if (upOnly)
        mode = mode | FsMode::UpOnly;
    return LwInternal::formal_sol_impl<SimdType::SSE2>(ctx, mode, params);
}

IterationResult redistribute_prd_lines_SSE2(Context& ctx, int maxIter, f64 tol, ExtraParams params)
{
    return redistribute_prd_lines_template<SimdType::SSE2>(ctx, maxIter, tol, params);
}

extern "C"
{
    FsIterationFns fs_iteration_fns_provider()
    {
        return FsIterationFns {
            -1, false, true, true, true,
            "mali_full_precond_SSE2",
            formal_sol_iteration_matrices_SSE2,
            formal_sol_SSE2,
            formal_sol_full_stokes_impl,
            redistribute_prd_lines_SSE2,
            stat_eq_impl,
            time_dependent_update_impl,
            nr_post_update_impl
        };
    }
}