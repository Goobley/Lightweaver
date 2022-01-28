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

constexpr inline
bool SSE2_available()
{
// NOTE(cmo) This first attempt doesn't seem to work reliably for the 64-bit
// compiler.
// #if (defined(_WIN32) && _M_IX86_FP==2) || defined(__SSE2__)
#if (defined(_WIN32) && (defined(_M_AMD64) || defined(_M_X64))) \
  || defined(__SSE2__)
    return true;
#else
    return false;
#endif
}

constexpr inline
bool AVX2FMA_available()
{
#if (defined(_WIN32) && defined(__AVX2__)) \
    || (defined(__AVX2__) && defined(__FMA__))
    return true;
#else
    return false;
#endif
}

constexpr inline
bool AVX512_available()
{
#if defined(__AVX512F__) && defined(__AVX512DQ__)
// NOTE(cmo): We use instructions from both AVX512F and AVX512DQ, I believe
// where one is available, the other is too.
    return true;
#else
    return false;
#endif
}

constexpr int SimdWidth[(size_t)SimdType::SIMD_TYPE_COUNT] = {1, 2, 4, 8, 2};

#ifdef _WIN32
#define ForceInline __forceinline
#else
#define ForceInline __attribute__((always_inline))
#endif

#else
#endif