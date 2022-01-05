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
uv_opt<SimdType::AVX2FMA>(Transition* t, int la, int mu, bool toObs,
                          F64View Uji, F64View Vij, F64View Vji)
{
    constexpr SimdType simd = SimdType::AVX2FMA;
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
            __m256d phik = _mm256_loadu_pd(&p(k));
            __m256d Vijk = _mm256_mul_pd(_mm256_set1_pd(hc_4pi * t->Bij), phik);
            _mm256_store_pd(&Vij(k), Vijk);

            // Vji(k) = t->gij(k) * Vij(k);
            __m256d gijk = _mm256_loadu_pd(&t->gij(k));
            _mm256_store_pd(&Vji(k), _mm256_mul_pd(gijk, Vijk));
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
        const __m256d ABRatio4x = _mm256_set1_pd(ABRatio);
        k = 0;
        for (; k < kMax; k += Stride)
        {
            __m256d Vjik = _mm256_load_pd(&Vji(k));
            _mm256_store_pd(&Uji(k), _mm256_mul_pd(ABRatio4x, Vjik));
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
            __m256d Vijk = _mm256_set1_pd(a);
            _mm256_store_pd(&Vij(k), Vijk);
            // Vji(k) = t->gij(k) * Vij(k);
            __m256d Vjik = _mm256_mul_pd(_mm256_loadu_pd(&t->gij(k)), Vijk);
            _mm256_store_pd(&Vji(k), Vjik);
            // Uji(k) = hcl * Vji(k);
            _mm256_store_pd(&Uji(k), _mm256_mul_pd(_mm256_set1_pd(hcl), Vjik));
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
          typename std::enable_if_t<simd == SimdType::AVX2FMA, bool> = true>
inline void ForceInline
chi_eta_aux_accum(IntensityCoreData* data, Atom* atom, const Transition& t)
{
    constexpr int Stride = SimdWidth[(size_t)simd];
    JasUnpack(*(*data), activeAtoms, detailedAtoms);
    JasUnpack((*data), Uji, Vij, Vji, chiTot, etaTot);
    const int Nspace = data->atmos->Nspace;
    const int kRemainder = Nspace % Stride;
    const int kMax = Nspace - kRemainder;

    __m256d zeroWide = _mm256_setzero_pd();

    int k = 0;
    for (; k < kMax; k += Stride)
    {
        __m256d nik = _mm256_loadu_pd(&atom->n(t.i, k));
        __m256d njk = _mm256_loadu_pd(&atom->n(t.j, k));
        __m256d Vijk = _mm256_load_pd(&Vij(k));
        __m256d Vjik = _mm256_load_pd(&Vji(k));
        __m256d Ujik = _mm256_load_pd(&Uji(k));
        // f64 chi = atom->n(t.i, k) * Vij(k) - atom->n(t.j, k) * Vji(k);
        __m256d chik = _mm256_fmsub_pd(nik, Vijk, _mm256_mul_pd(njk, Vjik));

        // f64 eta = atom->n(t.j, k) * Uji(k);
        __m256d etak = _mm256_mul_pd(njk, Ujik);

        if constexpr (ComputeOperator)
        {
            if constexpr (iClean)
            {
                // atom->chi(t.i, k) += chi;
                __m256d chiic = _mm256_loadu_pd(&atom->chi(t.i, k));
                _mm256_storeu_pd(&atom->chi(t.i, k), _mm256_add_pd(chiic, chik));
            }
            else
            {
                // atom->chi(t.i, k) = chi;
                _mm256_storeu_pd(&atom->chi(t.i, k), chik);
                // atom->U(t.i, k) = 0.0;
                _mm256_storeu_pd(&atom->U(t.i, k), zeroWide);
            }

            if constexpr (jClean)
            {
                // atom->chi(t.j, k) -= chi;
                __m256d chijc = _mm256_loadu_pd(&atom->chi(t.j, k));
                _mm256_storeu_pd(&atom->chi(t.j, k), _mm256_sub_pd(chijc, chik));
                // atom->U(t.j, k) += Uji(k);
                __m256d Uc = _mm256_loadu_pd(&atom->U(t.j, k));
                _mm256_storeu_pd(&atom->U(t.j, k), _mm256_add_pd(Uc, Ujik));
            }
            else
            {
                // atom->chi(t.j, k) = -chi;
                __m256d chim = _mm256_xor_pd(chik, _mm256_set1_pd(-0.0));
                _mm256_storeu_pd(&atom->chi(t.j, k), chim);
                // atom->U(t.j, k) = Uji(k);
                _mm256_storeu_pd(&atom->U(t.j, k), Ujik);
            }

            if constexpr (FirstTrans)
            {
                // atom->eta(k) = eta;
                _mm256_store_pd(&atom->eta(k), etak);
            }
            else
            {
                // atom->eta(k) += eta;
                __m256d etakc = _mm256_load_pd(&atom->eta(k));
                _mm256_store_pd(&atom->eta(k), _mm256_add_pd(etakc, etak));
            }
        }

        // chiTot(k) += chi;
        // etaTot(k) += eta;
        __m256d etaTotc = _mm256_load_pd(&etaTot(k));
        __m256d chiTotc = _mm256_load_pd(&chiTot(k));
        _mm256_store_pd(&etaTot(k), _mm256_add_pd(etaTotc, etak));
        _mm256_store_pd(&chiTot(k), _mm256_add_pd(chiTotc, chik));
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
compute_full_Ieff<SimdType::AVX2FMA>(F64View& I, F64View& PsiStar,
                                     F64View& eta, F64View& Ieff)
{

    constexpr SimdType simd = SimdType::AVX2FMA;
    constexpr int Stride = SimdWidth[(size_t)simd];
    const int Nspace = I.shape(0);
    const int Nremainder = Nspace % Stride;
    const int kMax = Nspace - Nremainder;

    int k = 0;
    for (; k < kMax; k += Stride)
    {
        __m256d Ik = _mm256_load_pd(&I(k));
        __m256d Psik = _mm256_load_pd(&PsiStar(k));
        __m256d etak = _mm256_load_pd(&eta(k));
        _mm256_store_pd(&Ieff(k), _mm256_fnmadd_pd(Psik, etak, Ik));
    }
    for (; k < Nspace; ++k)
    {
        Ieff(k) = I(k) - PsiStar(k) * eta(k);
    }
}

template <SimdType simd, bool ComputeOperator, bool ComputeRates,
          typename std::enable_if_t<simd == SimdType::AVX2FMA, bool> = true>
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
        __m256d wlamuk = _mm256_mul_pd(_mm256_loadu_pd(&atom.wla(kr, k)),
                                       _mm256_set1_pd(wmu));
        __m256d Ujik = _mm256_load_pd(&Uji(k));
        __m256d Vjik = _mm256_load_pd(&Vji(k));
        __m256d Vijk = _mm256_load_pd(&Vij(k));
        if constexpr (ComputeOperator)
        {
            __m256d Ieffk = _mm256_load_pd(&Ieff(k));
            __m256d PsiStark = _mm256_load_pd(&PsiStar(k));
            __m256d atomChiik = _mm256_loadu_pd(&atom.chi(t.i, k));
            __m256d atomChijk = _mm256_loadu_pd(&atom.chi(t.j, k));
            __m256d atomUik = _mm256_loadu_pd(&atom.U(t.i, k));
            __m256d atomUjk = _mm256_loadu_pd(&atom.U(t.j, k));

            // f64 integrand = (Uji(k) + Vji(k) * Ieff(k)) - (PsiStar(k) * atom.chi(t.i, k) * atom.U(t.j, k));
            __m256d term1 = _mm256_fmadd_pd(Vjik, Ieffk, Ujik);
            __m256d chiU = _mm256_mul_pd(atomChiik, atomUjk);
            __m256d integrand = _mm256_fnmadd_pd(PsiStark, chiU, term1);
            // atom.Gamma(t.i, t.j, k) += integrand * wlamu;
            __m256d currentIntegral = _mm256_loadu_pd(&atom.Gamma(t.i, t.j, k));
            __m256d integralChunk = _mm256_fmadd_pd(integrand, wlamuk, currentIntegral);
            _mm256_storeu_pd(&atom.Gamma(t.i, t.j, k), integralChunk);

            // integrand = (Vij(k) * Ieff(k)) - (PsiStar(k) * atom.chi(t.j, k) * atom.U(t.i, k));
            chiU = _mm256_mul_pd(atomChijk, atomUik);
            __m256d term2 = _mm256_mul_pd(PsiStark, chiU);
            integrand = _mm256_fmsub_pd(Vijk, Ieffk, term2);
            // atom.Gamma(t.j, t.i, k) += integrand * wlamu;
            currentIntegral = _mm256_loadu_pd(&atom.Gamma(t.j, t.i, k));
            integralChunk = _mm256_fmadd_pd(integrand, wlamuk, currentIntegral);
            _mm256_storeu_pd(&atom.Gamma(t.j, t.i, k), integralChunk);
        }

        if constexpr (ComputeRates)
        {
            __m256d Ik = _mm256_load_pd(&I(k));
            // t.Rij(k) += I(k) * Vij(k) * wlamu;
            __m256d Rijk = _mm256_load_pd(&t.Rij(k));
            __m256d integrand = _mm256_mul_pd(Ik, Vijk);
            __m256d integralChunk = _mm256_fmadd_pd(integrand, wlamuk, Rijk);
            _mm256_store_pd(&t.Rij(k), integralChunk);

            // t.Rji(k) += (Uji(k) + I(k) * Vji(k)) * wlamu;
            __m256d Rjik = _mm256_load_pd(&t.Rji(k));
            integrand = _mm256_fmadd_pd(Ik, Vjik, Ujik);
            integralChunk = _mm256_fmadd_pd(integrand, wlamuk, Rjik);
            _mm256_store_pd(&t.Rji(k), integralChunk);
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
