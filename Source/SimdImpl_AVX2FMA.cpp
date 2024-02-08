#include "Lightweaver.hpp"
#include "Constants.hpp"
#include "Simd.hpp"
#include "LwInternal.hpp"
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
inline __m256d polynomial_13_lin(__m256d x, f64 c2, f64 c3, f64 c4,
    f64 c5, f64 c6, f64 c7, f64 c8,
    f64 c9, f64 c10, f64 c11, f64 c12, f64 c13)
{
    // c13*x^13 + ... + c2*x^2 + x + 0

    __m256d x2 = _mm256_mul_pd(x, x);
    __m256d x4 = _mm256_mul_pd(x2, x2);
    __m256d x8 = _mm256_mul_pd(x4, x4);

    // NOTE(cmo): Do all the inner powers first
    __m256d t3 = _mm256_fmadd_pd(_mm256_set1_pd(c3), x, _mm256_set1_pd(c2));
    __m256d t5 = _mm256_fmadd_pd(_mm256_set1_pd(c5), x, _mm256_set1_pd(c4));
    __m256d t7 = _mm256_fmadd_pd(_mm256_set1_pd(c7), x, _mm256_set1_pd(c6));
    __m256d t9 = _mm256_fmadd_pd(_mm256_set1_pd(c9), x, _mm256_set1_pd(c8));
    __m256d t11 = _mm256_fmadd_pd(_mm256_set1_pd(c11), x, _mm256_set1_pd(c10));
    __m256d t13 = _mm256_fmadd_pd(_mm256_set1_pd(c13), x, _mm256_set1_pd(c12));

    // NOTE(cmo): Next layer
    __m256d tt3 = _mm256_fmadd_pd(t3, x2, x);
    __m256d tt7 = _mm256_fmadd_pd(t7, x2, t5);
    __m256d tt11 = _mm256_fmadd_pd(t11, x2, t9);

    __m256d ttt7 = _mm256_fmadd_pd(tt7, x4, tt3);
    __m256d ttt13 = _mm256_fmadd_pd(t13, x4, tt11);

    return _mm256_fmadd_pd(ttt13, x8, ttt7);
}

inline __m256d pow2n(const __m256d n) {
    const __m256d pow2_52 = _mm256_set1_pd(4503599627370496.0);   // 2^52
    const __m256d bias = _mm256_set1_pd(1023.0);                  // bias in exponent
    __m256d a = _mm256_add_pd(n, _mm256_add_pd(bias, pow2_52));   // put n + bias in least significant bits
    __m256i b = _mm256_castpd_si256(a);  // bit-cast to integer
    __m256i c = _mm256_slli_epi64(b, 52); // shift left 52 places to get value into exponent field
    __m256d d = _mm256_castsi256_pd(c);   // bit-cast back to double
    return d;
}

inline __m256d abs_pd_avx(__m256d xIn)
{
    unsigned int data[2] = {0xFFFFFFFFu, 0x7FFFFFFFu};
    __m256d mask = _mm256_broadcast_sd((double*)data);
    return _mm256_and_pd(xIn, mask);
}

inline __m256i bitwise_not(__m256i x)
{
    // NOTE(cmo): Sometime in 2023/2024 MSVC has stopped supporting ~ for __m256i
    // Still seems to be fine applying it to __mmask8 for AVX512.
    // This is equivalent to an xor with fff...
    // MSVC doesn't seem great at optimising this to the cmp_eq form that
    // gcc/clang use (cmp_eq x, x == fff...). So just write that explicitly.
    return _mm256_xor_si256(x, _mm256_cmpeq_epi32(x, x));
}

inline __m256i finite_mask_avx(__m256d x)
{
    __m256i i = _mm256_castpd_si256(x);
    __m256i iShift = _mm256_sll_epi64(i, _mm_cvtsi64_si128(1));
    __m256i exp_val = _mm256_set1_epi64x(0xFFE0000000000000);
    __m256i result = bitwise_not(_mm256_cmpeq_epi64(_mm256_and_si256(iShift, exp_val), exp_val));
    return result;
}


// NOTE(cmo): AVX impl of exp_pd, based on Agner Fog's vector class
// https://github.com/vectorclass/version2/blob/master/vectormath_exp.h
// The implementation here, based on a classic Taylor series, rather than a
// minimax function makes sense for our case, as we primarily value precision
// close to 0. i.e. we know the behaviour outwith this.
// Original under Apache v2 license.
inline __m256d exp_pd_avx(__m256d xIn)
{
    constexpr f64 p2 = 1.0 / 2.0;
    constexpr f64 p3 = 1.0 / 6.0;
    constexpr f64 p4 = 1.0 / 24.0;
    constexpr f64 p5 = 1.0 / 120.0;
    constexpr f64 p6 = 1.0 / 720.0;
    constexpr f64 p7 = 1.0 / 5040.0;
    constexpr f64 p8 = 1.0 / 40320.0;
    constexpr f64 p9 = 1.0 / 362880.0;
    constexpr f64 p10 = 1.0 / 3628800.0;
    constexpr f64 p11 = 1.0 / 39916800.0;
    constexpr f64 p12 = 1.0 / 479001600.0;
    constexpr f64 p13 = 1.0 / 6227020800.0;

    constexpr f64 log2e = 1.44269504088896340736;

    constexpr f64 xMax = 708.39;
    constexpr f64 ln2dHi = 0.693145751953125;
    constexpr f64 ln2dLo = 1.42860682030941723212e-6;

    __m256d x = xIn;

    __m256d r = _mm256_round_pd(_mm256_mul_pd(xIn, _mm256_set1_pd(log2e)),
        _MM_FROUND_TO_NEAREST_INT);
    // nmul_add(a, b, c) -> -(a * b) + c i.e. fnmadd
    // x = x0 - r * ln2Lo - r * ln2Hi
    x = _mm256_fnmadd_pd(r, _mm256_set1_pd(ln2dHi), x);
    x = _mm256_fnmadd_pd(r, _mm256_set1_pd(ln2dLo), x);

    __m256d z = polynomial_13_lin(x, p2, p3, p4, p5, p6, p7, p8,
        p9, p10, p11, p12, p13);
    __m256d n2 = pow2n(r);
    z = _mm256_mul_pd(_mm256_add_pd(z, _mm256_set1_pd(1.0)), n2);

    // NOTE(cmo): Error/edge-case handling code. The previous warning was prophetic.
    // abs(xIn) < xMax
    __m256i mask1 = _mm256_castpd_si256(_mm256_cmp_pd(abs_pd_avx(xIn), _mm256_set1_pd(xMax), 1));
    __m256i mask2 = finite_mask_avx(xIn);
    __m256i mask = _mm256_and_si256(mask1, mask2);
    // if all mask is set, then exit normally.
    if (_mm256_testc_si256(mask, _mm256_set1_epi64x(-1)) != 0)
        return z;

    __m256d maskd = _mm256_castsi256_pd(mask);
    __m256d inputSign = _mm256_and_pd(xIn, _mm256_set1_pd(-0.0));
    __m256d inf = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FF0000000000000));
    r = _mm256_blendv_pd(inf, _mm256_set1_pd(0.0), inputSign); // values for over/underflow/inf
    z = _mm256_blendv_pd(r, z, maskd); // +/- underflow

    __m256d nan_mask = _mm256_cmp_pd(xIn, xIn, 3); // check for unordered comparison, i.e. a value is nan
    z = _mm256_blendv_pd(z, xIn, nan_mask); // set output to nan if input is nan

    return z;
}

template <>
inline ForceInline
void setup_wavelength_opt<SimdType::AVX2FMA>(Atom* atom, int laIdx)
{
    constexpr SimdType simd = SimdType::AVX2FMA;
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
            __m256d bRatio = _mm256_set1_pd(t.Bji / t.Bij);
            __m256d wlaPi = _mm256_set1_pd(wlambda * pi4_hc);
            for (; k < kMax; k += Stride)
            {
                _mm256_storeu_pd(&g(k), bRatio);
                __m256d wphik = _mm256_loadu_pd(&t.wphi(k));
                _mm256_storeu_pd(&w(k), _mm256_mul_pd(wlaPi, wphik));
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
            __m256d mhc_kl4x = _mm256_set1_pd(-hc_kl);
            __m256d wTerm = _mm256_set1_pd(wlambda_lambda * pi4_h);
            int k = 0;
            for (; k < kMax; k += Stride)
            {
                __m256d nik = _mm256_loadu_pd(&nStar(t.i, k));
                __m256d njk = _mm256_loadu_pd(&nStar(t.j, k));
                __m256d temp = _mm256_loadu_pd(&atmos->temperature(k));
                __m256d gk = _mm256_mul_pd(_mm256_div_pd(nik, njk),
                                           exp_pd_avx(_mm256_div_pd(mhc_kl4x, temp)));
                _mm256_storeu_pd(&g(k), gk);
                _mm256_storeu_pd(&w(k), wTerm);
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
        // NOTE(cmo): See note on change in LwTransition.hpp
        const f64 hnu_4pi = hc_4pi * (t->lambda0 / t->wavelength(lt));
        auto p = t->phi(lt, mu, (int)toObs);
        int k = 0;
        for (; k < kMax; k += Stride)
        {
            // Vij(k) = hc_4pi * t->Bij * p(k);
            __m256d phik = _mm256_loadu_pd(&p(k));
            // __m256d Vijk = _mm256_mul_pd(_mm256_set1_pd(hc_4pi * t->Bij), phik);
            __m256d Vijk = _mm256_mul_pd(_mm256_set1_pd(hnu_4pi * t->Bij), phik);
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
compute_source_fn<SimdType::AVX2FMA>(F64View& S, F64View& etaTot,
                                     F64View& chiTot, F64View& sca,
                                     F64View& JDag)
{
    constexpr SimdType simd = SimdType::AVX2FMA;
    constexpr int Stride = SimdWidth[(size_t)simd];
    const int Nspace = S.shape(0);
    const int Nremainder = Nspace % Stride;
    const int kMax = Nspace - Nremainder;
    int k = 0;
    for (; k < kMax; k += Stride)
    {
        __m256d etak = _mm256_load_pd(&etaTot(k));
        __m256d chik = _mm256_load_pd(&chiTot(k));
        __m256d scak = _mm256_loadu_pd(&sca(k));
        __m256d Jk = _mm256_loadu_pd(&JDag(k));
        __m256d num = _mm256_fmadd_pd(scak, Jk, etak);
        __m256d Sk = _mm256_div_pd(num, chik);
        _mm256_store_pd(&S(k), Sk);
    }
    for (; k < Nspace; ++k)
    {
        S(k) = (etaTot(k) + sca(k) * JDag(k)) / chiTot(k);
    }
}

template <>
inline ForceInline void
accumulate_J<SimdType::AVX2FMA>(f64 halfwmu, F64View& J, F64View& I)
{
    constexpr SimdType simd = SimdType::AVX2FMA;
    constexpr int Stride = SimdWidth[(size_t)simd];
    const int Nspace = I.shape(0);
    const int Nremainder = Nspace % Stride;
    const int kMax = Nspace - Nremainder;
    int k = 0;
    __m256d halfwmuWide = _mm256_set1_pd(halfwmu);
    for (; k < kMax; k += Stride)
    {
        __m256d Jk = _mm256_loadu_pd(&J(k));
        __m256d Ik = _mm256_load_pd(&I(k));
        _mm256_storeu_pd(&J(k), _mm256_fmadd_pd(halfwmuWide, Ik, Jk));
    }
    for (; k < Nspace; ++k)
    {
        J(k) += halfwmu * I(k);
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
            __m256d Rijk = _mm256_loadu_pd(&t.Rij(k));
            __m256d integrand = _mm256_mul_pd(Ik, Vijk);
            __m256d integralChunk = _mm256_fmadd_pd(integrand, wlamuk, Rijk);
            _mm256_storeu_pd(&t.Rij(k), integralChunk);

            // t.Rji(k) += (Uji(k) + I(k) * Vji(k)) * wlamu;
            __m256d Rjik = _mm256_loadu_pd(&t.Rji(k));
            integrand = _mm256_fmadd_pd(Ik, Vjik, Ujik);
            integralChunk = _mm256_fmadd_pd(integrand, wlamuk, Rjik);
            _mm256_storeu_pd(&t.Rji(k), integralChunk);
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

IterationResult formal_sol_iteration_matrices_AVX2FMA(Context& ctx, bool lambdaIterate, ExtraParams params)
{
    if constexpr (AVX2FMA_available())
    {
        FsMode mode = (FsMode::UpdateJ | FsMode::UpdateRates);
        if (lambdaIterate)
            mode = mode | FsMode::PureLambdaIteration;

        return LwInternal::formal_sol_iteration_matrices_impl<SimdType::AVX2FMA>(ctx, mode, params);
    }
    else
    {
        fprintf(stderr, "Attempted to call %s, but instruction set not available.\nThis message shouldn't appear, please open an issue.\n", __func__);
    }
}

IterationResult formal_sol_AVX2FMA(Context& ctx, bool upOnly, ExtraParams params)
{
    FsMode mode = FsMode::FsOnly;
    if (upOnly)
        mode = mode | FsMode::UpOnly;
    return LwInternal::formal_sol_impl<SimdType::AVX2FMA>(ctx, mode, params);
}

IterationResult redistribute_prd_lines_AVX2FMA(Context& ctx, int maxIter, f64 tol, ExtraParams params)
{
    return redistribute_prd_lines_template<SimdType::AVX2FMA>(ctx, maxIter, tol, params);
}

extern "C"
{
    FsIterationFns fs_iteration_fns_provider()
    {
        return FsIterationFns {
            -1, false, true, true, true,
            "mali_full_precond_AVX2FMA",
            formal_sol_iteration_matrices_AVX2FMA,
            formal_sol_AVX2FMA,
            formal_sol_full_stokes_impl,
            redistribute_prd_lines_AVX2FMA,
            stat_eq_impl,
            time_dependent_update_impl,
            nr_post_update_impl
        };
    }
}