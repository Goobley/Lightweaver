#include <utility>
// NOTE(cmo): Machine-Generated!
template <SimdType simd, typename ...Args>
inline auto dispatch_compute_full_operator_rates_(bool first, bool second, Args&& ...args)
-> decltype(compute_full_operator_rates<simd, false, false>(std::forward<Args>(args)...))
{
    u32 dispatcher__ = first;
    dispatcher__ += second << 1;
    

    switch (dispatcher__)
    {
    case 0:
    {
        return compute_full_operator_rates<simd, false, false>(std::forward<Args>(args)...);
    } break;
    case 1:
    {
        return compute_full_operator_rates<simd, true, false>(std::forward<Args>(args)...);
    } break;
    case 2:
    {
        return compute_full_operator_rates<simd, false, true>(std::forward<Args>(args)...);
    } break;
    case 3:
    {
        return compute_full_operator_rates<simd, true, true>(std::forward<Args>(args)...);
    } break;
    
    default:
    {
        assert(false);
    } break;
    }
}
