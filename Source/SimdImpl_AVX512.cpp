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
uv_opt<SimdType::AVX512>(Transition* t, int la, int mu, bool toObs,
                          F64View Uji, F64View Vij, F64View Vji)
{
    constexpr SimdType simd = SimdType::AVX512;
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
            __m512d phik = _mm512_loadu_pd(&p(k));
            __m512d Vijk = _mm512_mul_pd(_mm512_set1_pd(hc_4pi * t->Bij), phik);
            _mm512_store_pd(&Vij(k), Vijk);

            // Vji(k) = t->gij(k) * Vij(k);
            __m512d gijk = _mm512_loadu_pd(&t->gij(k));
            _mm512_store_pd(&Vji(k), _mm512_mul_pd(gijk, Vijk));
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
        const __m512d ABRatio4x = _mm512_set1_pd(ABRatio);
        k = 0;
        for (; k < kMax; k += Stride)
        {
            __m512d Vjik = _mm512_load_pd(&Vji(k));
            _mm512_store_pd(&Uji(k), _mm512_mul_pd(ABRatio4x, Vjik));
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
            __m512d Vijk = _mm512_set1_pd(a);
            _mm512_store_pd(&Vij(k), Vijk);
            // Vji(k) = t->gij(k) * Vij(k);
            __m512d Vjik = _mm512_mul_pd(_mm512_loadu_pd(&t->gij(k)), Vijk);
            _mm512_store_pd(&Vji(k), Vjik);
            // Uji(k) = hcl * Vji(k);
            _mm512_store_pd(&Uji(k), _mm512_mul_pd(_mm512_set1_pd(hcl), Vjik));
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
          typename std::enable_if_t<simd == SimdType::AVX512, bool> = true>
inline void ForceInline
chi_eta_aux_accum(IntensityCoreData* data, Atom* atom, const Transition& t)
{
    constexpr int Stride = SimdWidth[(size_t)simd];
    JasUnpack(*(*data), activeAtoms, detailedAtoms);
    JasUnpack((*data), Uji, Vij, Vji, chiTot, etaTot);
    const int Nspace = data->atmos->Nspace;
    const int kRemainder = Nspace % Stride;
    const int kMax = Nspace - kRemainder;

    __m512d zeroWide = _mm512_setzero_pd();

    int k = 0;
    for (; k < kMax; k += Stride)
    {
        __m512d nik = _mm512_loadu_pd(&atom->n(t.i, k));
        __m512d njk = _mm512_loadu_pd(&atom->n(t.j, k));
        __m512d Vijk = _mm512_load_pd(&Vij(k));
        __m512d Vjik = _mm512_load_pd(&Vji(k));
        __m512d Ujik = _mm512_load_pd(&Uji(k));
        // f64 chi = atom->n(t.i, k) * Vij(k) - atom->n(t.j, k) * Vji(k);
        __m512d chik = _mm512_fmsub_pd(nik, Vijk, _mm512_mul_pd(njk, Vjik));

        // f64 eta = atom->n(t.j, k) * Uji(k);
        __m512d etak = _mm512_mul_pd(njk, Ujik);

        if constexpr (ComputeOperator)
        {
            if constexpr (iClean)
            {
                // atom->chi(t.i, k) += chi;
                __m512d chiic = _mm512_loadu_pd(&atom->chi(t.i, k));
                _mm512_storeu_pd(&atom->chi(t.i, k), _mm512_add_pd(chiic, chik));
            }
            else
            {
                // atom->chi(t.i, k) = chi;
                _mm512_storeu_pd(&atom->chi(t.i, k), chik);
                // atom->U(t.i, k) = 0.0;
                _mm512_storeu_pd(&atom->U(t.i, k), zeroWide);
            }

            if constexpr (jClean)
            {
                // atom->chi(t.j, k) -= chi;
                __m512d chijc = _mm512_loadu_pd(&atom->chi(t.j, k));
                _mm512_storeu_pd(&atom->chi(t.j, k), _mm512_sub_pd(chijc, chik));
                // atom->U(t.j, k) += Uji(k);
                __m512d Uc = _mm512_loadu_pd(&atom->U(t.j, k));
                _mm512_storeu_pd(&atom->U(t.j, k), _mm512_add_pd(Uc, Ujik));
            }
            else
            {
                // atom->chi(t.j, k) = -chi;
                __m512d chim = _mm512_xor_pd(chik, _mm512_set1_pd(-0.0));
                _mm512_storeu_pd(&atom->chi(t.j, k), chim);
                // atom->U(t.j, k) = Uji(k);
                _mm512_storeu_pd(&atom->U(t.j, k), Ujik);
            }

            if constexpr (FirstTrans)
            {
                // atom->eta(k) = eta;
                _mm512_store_pd(&atom->eta(k), etak);
            }
            else
            {
                // atom->eta(k) += eta;
                __m512d etakc = _mm512_load_pd(&atom->eta(k));
                _mm512_store_pd(&atom->eta(k), _mm512_add_pd(etakc, etak));
            }
        }

        // chiTot(k) += chi;
        // etaTot(k) += eta;
        __m512d etaTotc = _mm512_load_pd(&etaTot(k));
        __m512d chiTotc = _mm512_load_pd(&chiTot(k));
        _mm512_store_pd(&etaTot(k), _mm512_add_pd(etaTotc, etak));
        _mm512_store_pd(&chiTot(k), _mm512_add_pd(chiTotc, chik));
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
compute_full_Ieff<SimdType::AVX512>(F64View& I, F64View& PsiStar,
                                     F64View& eta, F64View& Ieff)
{

    constexpr SimdType simd = SimdType::AVX512;
    constexpr int Stride = SimdWidth[(size_t)simd];
    const int Nspace = I.shape(0);
    const int Nremainder = Nspace % Stride;
    const int kMax = Nspace - Nremainder;

    int k = 0;
    for (; k < kMax; k += Stride)
    {
        __m512d Ik = _mm512_load_pd(&I(k));
        __m512d Psik = _mm512_load_pd(&PsiStar(k));
        __m512d etak = _mm512_load_pd(&eta(k));
        _mm512_store_pd(&Ieff(k), _mm512_fnmadd_pd(Psik, etak, Ik));
    }
    for (; k < Nspace; ++k)
    {
        Ieff(k) = I(k) - PsiStar(k) * eta(k);
    }
}

template <SimdType simd, bool ComputeOperator, bool ComputeRates,
          typename std::enable_if_t<simd == SimdType::AVX512, bool> = true>
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
        __m512d wlamuk = _mm512_mul_pd(_mm512_loadu_pd(&atom.wla(kr, k)),
                                       _mm512_set1_pd(wmu));
        __m512d Ujik = _mm512_load_pd(&Uji(k));
        __m512d Vjik = _mm512_load_pd(&Vji(k));
        __m512d Vijk = _mm512_load_pd(&Vij(k));
        if constexpr (ComputeOperator)
        {
            __m512d Ieffk = _mm512_load_pd(&Ieff(k));
            __m512d PsiStark = _mm512_load_pd(&PsiStar(k));
            __m512d atomChiik = _mm512_loadu_pd(&atom.chi(t.i, k));
            __m512d atomChijk = _mm512_loadu_pd(&atom.chi(t.j, k));
            __m512d atomUik = _mm512_loadu_pd(&atom.U(t.i, k));
            __m512d atomUjk = _mm512_loadu_pd(&atom.U(t.j, k));

            // f64 integrand = (Uji(k) + Vji(k) * Ieff(k)) - (PsiStar(k) * atom.chi(t.i, k) * atom.U(t.j, k));
            __m512d term1 = _mm512_fmadd_pd(Vjik, Ieffk, Ujik);
            __m512d chiU = _mm512_mul_pd(atomChiik, atomUjk);
            __m512d integrand = _mm512_fnmadd_pd(PsiStark, chiU, term1);
            // atom.Gamma(t.i, t.j, k) += integrand * wlamu;
            __m512d currentIntegral = _mm512_loadu_pd(&atom.Gamma(t.i, t.j, k));
            __m512d integralChunk = _mm512_fmadd_pd(integrand, wlamuk, currentIntegral);
            _mm512_storeu_pd(&atom.Gamma(t.i, t.j, k), integralChunk);

            // integrand = (Vij(k) * Ieff(k)) - (PsiStar(k) * atom.chi(t.j, k) * atom.U(t.i, k));
            chiU = _mm512_mul_pd(atomChijk, atomUik);
            __m512d term2 = _mm512_mul_pd(PsiStark, chiU);
            integrand = _mm512_fmsub_pd(Vijk, Ieffk, term2);
            // atom.Gamma(t.j, t.i, k) += integrand * wlamu;
            currentIntegral = _mm512_loadu_pd(&atom.Gamma(t.j, t.i, k));
            integralChunk = _mm512_fmadd_pd(integrand, wlamuk, currentIntegral);
            _mm512_storeu_pd(&atom.Gamma(t.j, t.i, k), integralChunk);
        }

        if constexpr (ComputeRates)
        {
            __m512d Ik = _mm512_load_pd(&I(k));
            // t.Rij(k) += I(k) * Vij(k) * wlamu;
            __m512d Rijk = _mm512_load_pd(&t.Rij(k));
            __m512d integrand = _mm512_mul_pd(Ik, Vijk);
            __m512d integralChunk = _mm512_fmadd_pd(integrand, wlamuk, Rijk);
            _mm512_store_pd(&t.Rij(k), integralChunk);

            // t.Rji(k) += (Uji(k) + I(k) * Vji(k)) * wlamu;
            __m512d Rjik = _mm512_load_pd(&t.Rji(k));
            integrand = _mm512_fmadd_pd(Ik, Vjik, Ujik);
            integralChunk = _mm512_fmadd_pd(integrand, wlamuk, Rjik);
            _mm512_store_pd(&t.Rji(k), integralChunk);
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
