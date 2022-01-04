#ifndef CMO_SIMD_HPP
#define CMO_SIMD_HPP

#include "CmoArray.hpp"

enum class SimdType
{
    Scalar,
    SSE2,
    AVX2FMA,
    AVX512,
    NEON,
    SIMD_TYPE_COUNT
};

constexpr int SimdWidth[(size_t)SimdType::SIMD_TYPE_COUNT] = {1, 2, 4, 8, 2};

#ifdef _WIN32
#define ForceInline __forceinline
#else
#define ForceInline __attribute__((always_inline))
#endif

struct Atom;
struct Transition;

namespace LwInternal
{
struct IntensityCoreData;

template <SimdType type>
inline void
uv_opt(Transition* t, int la, int mu, bool toObs, F64View Uji, F64View Vij, F64View Vji);

template <SimdType simd, bool iClean, bool jClean,
          bool FirstTrans, bool ComputeOperator,
          typename SFINAE>
inline void
chi_eta_aux_accum(IntensityCoreData* data, Atom* atom, const Transition& t);

template <SimdType simd>
inline void
compute_full_Ieff(F64View& I, F64View& PsiStar,
                  F64View& eta, F64View& Ieff);

template <SimdType simd, bool ComputeOperator, bool ComputeRates,
          typename SFINAE>
inline void
compute_full_operator_rates(Atom* a, int kr, f64 wmu, IntensityCoreData* data);
}

#else
#endif