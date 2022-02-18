#include "Lightweaver.hpp"
#include "LwInternal.hpp"
#include "Simd.hpp"

#include <immintrin.h>

using namespace LwInternal;

namespace Linear
{
inline __m256d polynomial_7_full(__m256d x, f64 c0, f64 c1, f64 c2,
                                 f64 c3, f64 c4, f64 c5, f64 c6, f64 c7)
{

    // c7*x^7 + ... + c2*x^2 + c1*x + c0

    __m256d x2 = _mm256_mul_pd(x, x);
    __m256d x4 = _mm256_mul_pd(x2, x2);

    // NOTE(cmo): Do all the inner powers first
    __m256d t1 = _mm256_fmadd_pd(_mm256_set1_pd(c1), x, _mm256_set1_pd(c0));
    __m256d t3 = _mm256_fmadd_pd(_mm256_set1_pd(c3), x, _mm256_set1_pd(c2));
    __m256d t5 = _mm256_fmadd_pd(_mm256_set1_pd(c5), x, _mm256_set1_pd(c4));
    __m256d t7 = _mm256_fmadd_pd(_mm256_set1_pd(c7), x, _mm256_set1_pd(c6));

    // NOTE(cmo): Next layer
    __m256d tt3 = _mm256_fmadd_pd(t3, x2, t1);
    __m256d tt7 = _mm256_fmadd_pd(t7, x2, t5);
    __m256d ttt7 = _mm256_fmadd_pd(tt7, x4, tt3);

    return ttt7;
}

inline __m256d pow2n (const __m256d n) {
    const __m256d pow2_52 = _mm256_set1_pd(4503599627370496.0);   // 2^52
    const __m256d bias = _mm256_set1_pd(1023.0);                  // bias in exponent
    __m256d a = _mm256_add_pd(n, _mm256_add_pd(bias, pow2_52));   // put n + bias in least significant bits
    __m256i b = _mm256_castpd_si256(a);  // bit-cast to integer
    __m256i c = _mm256_slli_epi64(b,52); // shift left 52 places to get value into exponent field
    __m256d d = _mm256_castsi256_pd(c);   // bit-cast back to double
    return d;
}

struct alignas(32) w4x
{
    __m256d w0;
    __m256d w1;
};

inline w4x w01_expansion(__m256d xIn)
{
    constexpr f64 w0_0  = 0.0;
    constexpr f64 w0_1  = -1.0;
    constexpr f64 w0_2  = -1.0/2.0;
    constexpr f64 w0_3  = -1.0/6.0;
    constexpr f64 w0_4  = -1.0/24.0;
    constexpr f64 w0_5  = -1.0/120.0;
    constexpr f64 w0_6  = -1.0/720.0;
    constexpr f64 w0_7  = -1.0/5040.0;
    constexpr f64 w0_8  = -1.0/40320.0;
    constexpr f64 w0_9  = -1.0/362880.0;
    constexpr f64 w0_10 = -1.0/3628800.0;
    constexpr f64 w0_11 = -1.0/39916800.0;
    constexpr f64 w0_12 = -1.0/479001600.0;
    constexpr f64 w0_13 = -1.0/6227020800.0;

    constexpr f64 w1_0  = 0.0;
    constexpr f64 w1_1  = 0.0;
    constexpr f64 w1_2  = (2.0 - 1.0)/2.0;
    constexpr f64 w1_3  = (3.0 - 1.0)/6.0;
    constexpr f64 w1_4  = (4.0 - 1.0)/24.0;
    constexpr f64 w1_5  = (5.0 - 1.0)/120.0;
    constexpr f64 w1_6  = (6.0 - 1.0)/720.0;
    constexpr f64 w1_7  = (7.0 - 1.0)/5040.0;
    constexpr f64 w1_8  = (8.0 - 1.0)/40320.0;
    constexpr f64 w1_9  = (9.0 - 1.0)/362880.0;
    constexpr f64 w1_10 = (10.0 - 1.0)/3628800.0;
    constexpr f64 w1_11 = (11.0 - 1.0)/39916800.0;
    constexpr f64 w1_12 = (12.0 - 1.0)/479001600.0;
    constexpr f64 w1_13 = (13.0 - 1.0)/6227020800.0;


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

    // __m256d z0 = polynomial_13_full(x, w0_0, w0_1, w0_2, w0_3, w0_4, w0_5,
    //                                    w0_6, w0_7, w0_8, w0_9, w0_10, w0_11,
    //                                    w0_12, w0_13);
    __m256d z0 = polynomial_7_full(x, w0_0, w0_1, w0_2, w0_3, w0_4, w0_5,
                                      w0_6, w0_7);

    __m256d n2 = pow2n(r);

    const __m256d One = _mm256_set1_pd(1.0);
    w4x ws;
    // 1-e^x = (z0 * n2) - (n2 - 1)
    // This series is good enough for any x < 0 we care about
    ws.w0 = _mm256_fmsub_pd(z0, n2, _mm256_sub_pd(n2, One));
    // 1 - (1 - x) e^x
    // w0 + x e^x
    // w0 - x * (z0 - 1) * n2
    // fmnadd(mul(sub(z0, One), n2), x, w0)
    ws.w1 = _mm256_fnmadd_pd(_mm256_mul_pd(_mm256_sub_pd(z0, One), n2),
                             xIn, ws.w0);
    __m256d w1Lim = _mm256_set1_pd(-1.0e-3);
    __m256d w1Cond = _mm256_cmp_pd(xIn, w1Lim, _CMP_GT_OS);
    int mask = _mm256_movemask_pd(w1Cond);
    if (mask) // If any need expansion treatment
    {
        // This series is not good for larger x, hence the condition
        // __m256d z1 = polynomial_13_full(x, w1_0, w1_1, w1_2, w1_3, w1_4, w1_5,
        //                                 w1_6, w1_7, w1_8, w1_9, w1_10, w1_11,
        //                                 w1_12, w1_13);
        __m256d z1 = polynomial_7_full(x, w1_0, w1_1, w1_2, w1_3, w1_4, w1_5,
                                          w1_6, w1_7);
                            //  direct,  expansion,             if need expansion
        ws.w1 = _mm256_blendv_pd(ws.w1, _mm256_mul_pd(z1, n2), w1Cond);
    }

    // TODO(cmo): Probably should have some of the nan/inf error handling code.
    return ws;
}

w4x w2_4x(__m256d dtau)
{
    const __m256d bigTauLim = _mm256_set1_pd(50.0);
    const __m256d bigTauCond = _mm256_cmp_pd(dtau, bigTauLim, _CMP_GT_OS);
    const __m256d One = _mm256_set1_pd(1.0);
    const __m256d Zero = _mm256_setzero_pd();
    w4x result;
    result.w0 = One;
    result.w1 = One;
    // If they are not all over the limit
    if (_mm256_movemask_pd(bigTauCond) != 0xf)
    {
        __m256d  mdtau = _mm256_sub_pd(Zero, dtau);
        w4x smallerTerms = w01_expansion(mdtau);
                                     // small,        big,       if big
        result.w0 = _mm256_blendv_pd(smallerTerms.w0, result.w0, bigTauCond);
        result.w1 = _mm256_blendv_pd(smallerTerms.w1, result.w1, bigTauCond);
    }

    return result;
}

// NOTE(cmo): Abs impl based on https://stackoverflow.com/a/32422471/3847013
__m256d abs_mask()
{
    // NOTE(cmo): All 1s except sign bit
    __m256i m1 = _mm256_set1_epi64x(-1);
    return _mm256_castsi256_pd(_mm256_srli_epi64(m1, 1));
}

__m256d mm256_abs_pd(__m256d x)
{
    return _mm256_and_pd(abs_mask(), x);
}

template <bool ComputeOperator>
void piecewise_linear_1d_impl(FormalData* fd, f64 zmu, bool toObs, f64 Istart)
{
    JasUnpack((*fd), chi, S, Psi, I, atmos);
    constexpr SimdType simd = SimdType::AVX2FMA;
    constexpr int Stride = SimdWidth[(size_t)simd];
    auto& height = atmos->height;
    const int Ndep = atmos->Nspace;
    const int Ninteg = Ndep - 1;
    const int Nremainder = Ninteg % Stride;
    const int kMax = Ndep - Nremainder - 1;

    int dk = -1;
    int k_start = Ndep - 1;
    int k_simdStart = k_start - (Stride - 1);
    int k_end = 0;
    int k_simdEnd = Nremainder;
    if (!toObs)
    {
        dk = 1;
        k_start = 0;
        k_simdStart = 0;
        k_end = Ndep - 1;
        k_simdEnd = kMax;
    }

    // if (Nremainder != 0)
    // {
    //     printf("%d, %d, %d, %d, %d, %d\n", Ndep, Nremainder, k_start,
    //                                        k_end, k_simdStart, k_simdEnd);
    // }

    auto simd_end_cond = [toObs, Nremainder, kMax](int k)
    {
        if (toObs)
            return k > Nremainder;
        else
            return k != kMax;
    };

    // if (Nremainder == 0)
    //     k_simdEnd -= dk * Stride;

    // f64 dtau_uw = zmu * (chi(k_start) + chi(k_start + dk)) * abs(height(k_start) - height(k_start + dk));
    // f64 dS_uw = (S(k_start) - S(k_start + dk)) / dtau_uw;

    f64 I_upw = Istart;

    I(k_start) = I_upw;
    if constexpr (ComputeOperator)
        Psi(k_start) = 0.0;


    // f64 dtau_uw[Stride] = {};
    // f64 dS_uw[Stride] = {};
    // f64 w2s[Stride][2] = {};
    // f64 sourceTerms[Stride] = {};
    __m256d One = _mm256_set1_pd(1.0);
    __m256d zmu4x = _mm256_set1_pd(zmu);
    alignas(32) f64 edt[Stride] = {};
    alignas(32) f64 source[Stride] = {};
    alignas(32) f64 intens[Stride] = {};

    int k = k_simdStart;
    bool first = true;
    for (; simd_end_cond(k); k += (dk * Stride))
    {
        // NOTE(cmo): There's a lot of issues here for the toObs case.
        // Nremainder needs to be worked out much more correctly.  e.g. for a
        // toObs case with Ndep = 300, this will arrive here with k = 0 and dk =
        // -1, thus loading from before the start of an array.  Clearly this is
        // already bad, but due to how we write I and Psi, they are writing a
        // value into I[-1], which I believe is where malloc and co store
        // important info.
        // Verified that plausible sized doubles are being written into this
        // space with the debugger.
        __m256d chik = _mm256_loadu_pd(&chi(k));
        __m256d chikdk = _mm256_loadu_pd(&chi(k + dk));
        __m256d Sk = _mm256_loadu_pd(&S(k));
        __m256d Skdk = _mm256_loadu_pd(&S(k + dk));
        __m256d heightk = _mm256_loadu_pd(&height(k));
        __m256d heightkdk = _mm256_loadu_pd(&height(k + dk));
        __m256d dtau_uw = _mm256_mul_pd(zmu4x,
                                        _mm256_mul_pd(_mm256_add_pd(chik, chikdk),
                                                      mm256_abs_pd(_mm256_sub_pd(
                                                            heightk, heightkdk))));
        __m256d rcpDtau_uw = _mm256_div_pd(One, dtau_uw);
        __m256d dS_uw = _mm256_mul_pd(_mm256_sub_pd(Sk, Skdk), rcpDtau_uw);

        // for (int i = 0; i < Stride; ++i)
        // {
        //     dtau_uw[i] = zmu * (chi(k + dk*i) + chi(k + dk*(i+1)))
        //                      * abs(height(k + dk*i) - height(k + dk*(i+1)));
        //     dS_uw[i] = (S(k + dk*i) - S(k + dk*(i+1))) / dtau_uw[i];
        // }

        auto w2s = w2_4x(dtau_uw);

        // for (int i = 0; i < Stride; ++i)
        // {
        //     w2(dtau_uw[i], w2s[i]);
        // }

        __m256d sourceTerms = _mm256_fmadd_pd(w2s.w0, Skdk, _mm256_mul_pd(w2s.w1, dS_uw));

        // for (int i = 0; i < Stride; ++i)
        // {
        //     sourceTerms[i] = w2s[i][0] * S(k + dk*(i+1)) + w2s[i][1] * dS_uw[i];
        // }

        _mm256_store_pd(source, sourceTerms);
        _mm256_store_pd(edt, _mm256_sub_pd(One, w2s.w0));
        int iStart = 0;
        int iEnd = Stride;
        int dI = 1;
        if (toObs)
        {
            iStart = Stride - 1;
            iEnd = -1;
            dI = -1;
        }
        for (int i = iStart; i != iEnd; i += dI)
        {
            I_upw = edt[i] * I_upw + source[i];
	    // if (k + dk + i >= kMax)
        //         continue;
            // __m128d Iu = _mm_load_sd(&I_upw);
            // __m128d e = _mm_load_sd(&edt[i]);
            // __m128d s = _mm_load_sd(&source[i]);
            // Iu = _mm_fmadd_sd(e, Iu, s);
            // _mm_store_sd(&I_upw, Iu);
            // intens[i] = I_upw;
            I(k + dk + i) = I_upw;
        }
        // _mm256_storeu_pd(&I(k+dk), _mm256_load_pd(intens));

        // for (int i = 0; i < Stride; ++i)
        // {
        //     I_upw = (1.0 - w2s[i][0]) * I_upw + sourceTerms[i];
        //     I(k + dk*(i+1)) = I_upw;
        // }

        // I(k + dk) = (1.0 - w[0]) * I_upw + w[0] * S(k + dk) + w[1] * dS_uw;

        if constexpr (ComputeOperator)
        {
            __m256d Psikdk = _mm256_div_pd(_mm256_sub_pd(w2s.w0,
                                                         _mm256_mul_pd(w2s.w1, rcpDtau_uw)),
                                                         chikdk);
            _mm256_storeu_pd(&Psi(k+dk), Psikdk);
        }
	// if (first)
    //          printf("k+dk, %d\n", k+dk);
	// first = false;
        // if constexpr (ComputeOperator)
        // {
        //     for (int i = 0; i < Stride; ++i)
        //         Psi(k + dk*(i+1)) = w2s[i][0] - w2s[i][1] / dtau_uw[i];
        // }

        // I_upw = I(k + dk);
    }
    if (Nremainder != 0)
    {
        // if (toObs)
        //     k -= (dk * (Stride-1));
        k = k_simdEnd;
        for (; k != k_end; k += dk)
        {
            f64 dtau_uw = zmu * (chi(k) + chi(k + dk)) * abs(height(k) - height(k + dk));
            f64 rcpDtau_uw = 1.0 / dtau_uw;
            f64 dS_uw = (S(k) - S(k + dk)) * rcpDtau_uw;
            f64 w[2];
            w2(dtau_uw, w);

            I(k + dk) = (1.0 - w[0]) * I_upw + w[0] * S(k + dk) + w[1] * dS_uw;

            if constexpr (ComputeOperator)
                Psi(k + dk) = (w[0] - w[1] * rcpDtau_uw) / chi(k+dk);

            I_upw = I(k + dk);
        }
    }

    // f64 dtau_uw = zmu * (chi(k_end - dk) + chi(k_end))
    //                   * abs(height(k_end - dk) - height(k_end));
    // f64 dS_uw = (S(k_end - dk) - S(k_end)) / dtau_uw;
    // f64 w[2];
    // w2(dtau_uw, w);
    // I(k_end) = (1.0 - w[0]) * I_upw + w[0] * S(k_end) + w[1] * dS_uw;
    // if constexpr (ComputeOperator)
    // {
    //     // Psi(k_end) = w[0] - w[1] / dtau_uw;
    //     for (int k = 0; k < Psi.shape(0); ++k)
    //         Psi(k) /= chi(k);
    // }
}

void piecewise_linear_1d_AVX2FMA(FormalData* fd, int la, int mu,
                                 bool toObs, const F64View1D& wave)
{
    const f64 wav = wave(la);
    JasUnpack((*fd), atmos, I, chi);
    f64 zmu = 0.5 / atmos->muz(mu);
    auto height = atmos->height;

    int dk = -1;
    int kStart = atmos->Nspace - 1;
    int kEnd = 0;
    if (!toObs)
    {
        dk = 1;
        kStart = 0;
        kEnd = atmos->Nspace - 1;
    }
    f64 dtau_uw = zmu * (chi(kStart) + chi(kStart + dk)) * abs(height(kStart) - height(kStart + dk));

    f64 Iupw = 0.0;
    if (toObs)
    {
        if (atmos->zLowerBc.type == THERMALISED)
        {
            f64 Bnu[2];
            int Nspace = atmos->Nspace;
            planck_nu(2, &atmos->temperature(Nspace - 2), wav, Bnu);
            Iupw = Bnu[1] - (Bnu[0] - Bnu[1]) / dtau_uw;
        }
        else if (atmos->zLowerBc.type == CALLABLE)
        {
            int muIdx = atmos->zLowerBc.idxs(mu, int(toObs));
            if (muIdx == -1)
            {
                // NOTE(cmo): This shouldn't be possible, so I won't try to
                // recover.
                printf("Error in boundary condition indexing\n");
                assert(false);
            }
            Iupw = atmos->zLowerBc.bcData(la, muIdx, 0);
        }
    }
    else
    {
        if (atmos->zUpperBc.type == THERMALISED)
        {
            f64 Bnu[2];
            planck_nu(2, &atmos->temperature(0), wav, Bnu);
            Iupw = Bnu[0] - (Bnu[1] - Bnu[0]) / dtau_uw;
        }
        else if (atmos->zUpperBc.type == CALLABLE)
        {
            int muIdx = atmos->zUpperBc.idxs(mu, int(toObs));
            if (muIdx == -1)
            {
                printf("Error in boundary condition indexing\n");
                assert(false);
                Iupw = 0.0;
            }
            else
                Iupw = atmos->zUpperBc.bcData(la, muIdx, 0);
        }
    }

    bool computeOperator = bool(fd->Psi);
    if (computeOperator)
        piecewise_linear_1d_impl<true>(fd, zmu, toObs, Iupw);
    else
        piecewise_linear_1d_impl<false>(fd, zmu, toObs, Iupw);
}
}

extern "C"
{
FormalSolver fs_provider()
{
    return FormalSolver { Linear::piecewise_linear_1d_AVX2FMA, 1, 1,
                          "piecewise_linear_1d_AVX2FMA" };
}
}
