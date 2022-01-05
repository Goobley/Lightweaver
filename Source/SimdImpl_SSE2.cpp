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

namespace LwInternal
{
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
        auto p = t->phi(lt, mu, (int)toObs);
        int k = 0;
        for (; k < kMax; k += Stride)
        {
            // Vij(k) = hc_4pi * t->Bij * p(k);
            __m128d phik = _mm_loadu_pd(&p(k));
            __m128d Vijk = _mm_mul_pd(_mm_set1_pd(hc_4pi * t->Bij), phik);
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
    JasUnpack(*(*data), activeAtoms, detailedAtoms);
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
            __m128d Rijk = _mm_load_pd(&t.Rij(k));
            __m128d integrand = _mm_mul_pd(Ik, Vijk);
            __m128d integralChunk = _mm_add_pd(_mm_mul_pd(integrand, wlamuk), Rijk);
            _mm_store_pd(&t.Rij(k), integralChunk);

            // t.Rji(k) += (Uji(k) + I(k) * Vji(k)) * wlamu;
            __m128d Rjik = _mm_load_pd(&t.Rji(k));
            integrand = _mm_add_pd(_mm_mul_pd(Ik, Vjik), Ujik);
            integralChunk = _mm_add_pd(_mm_mul_pd(integrand, wlamuk), Rjik);
            _mm_store_pd(&t.Rji(k), integralChunk);
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
