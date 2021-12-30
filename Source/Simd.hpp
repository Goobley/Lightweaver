#ifndef CMO_SIMD_HPP
#define CMO_SIMD_HPP

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

#else
#endif