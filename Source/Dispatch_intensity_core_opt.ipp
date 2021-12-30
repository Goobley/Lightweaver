#include <utility>
// NOTE(cmo): Machine-Generated!
template <SimdType simd, typename ...Args>
inline auto dispatch_intensity_core_opt_(bool first, bool second, bool third, bool fourth, Args&& ...args)
-> decltype(intensity_core_opt<simd, false, false, false, false>(std::forward<Args>(args)...))
{
    u32 dispatcher__ = first;
    dispatcher__ += second << 1;
    dispatcher__ += third << 2;
    dispatcher__ += fourth << 3;
    

    switch (dispatcher__)
    {
    case 0:
    {
        return intensity_core_opt<simd, false, false, false, false>(std::forward<Args>(args)...);
    } break;
    case 1:
    {
        return intensity_core_opt<simd, true, false, false, false>(std::forward<Args>(args)...);
    } break;
    case 2:
    {
        return intensity_core_opt<simd, false, true, false, false>(std::forward<Args>(args)...);
    } break;
    case 3:
    {
        return intensity_core_opt<simd, true, true, false, false>(std::forward<Args>(args)...);
    } break;
    case 4:
    {
        return intensity_core_opt<simd, false, false, true, false>(std::forward<Args>(args)...);
    } break;
    case 5:
    {
        return intensity_core_opt<simd, true, false, true, false>(std::forward<Args>(args)...);
    } break;
    case 6:
    {
        return intensity_core_opt<simd, false, true, true, false>(std::forward<Args>(args)...);
    } break;
    case 7:
    {
        return intensity_core_opt<simd, true, true, true, false>(std::forward<Args>(args)...);
    } break;
    case 8:
    {
        return intensity_core_opt<simd, false, false, false, true>(std::forward<Args>(args)...);
    } break;
    case 9:
    {
        return intensity_core_opt<simd, true, false, false, true>(std::forward<Args>(args)...);
    } break;
    case 10:
    {
        return intensity_core_opt<simd, false, true, false, true>(std::forward<Args>(args)...);
    } break;
    case 11:
    {
        return intensity_core_opt<simd, true, true, false, true>(std::forward<Args>(args)...);
    } break;
    case 12:
    {
        return intensity_core_opt<simd, false, false, true, true>(std::forward<Args>(args)...);
    } break;
    case 13:
    {
        return intensity_core_opt<simd, true, false, true, true>(std::forward<Args>(args)...);
    } break;
    case 14:
    {
        return intensity_core_opt<simd, false, true, true, true>(std::forward<Args>(args)...);
    } break;
    case 15:
    {
        return intensity_core_opt<simd, true, true, true, true>(std::forward<Args>(args)...);
    } break;
    
    default:
    {
        assert(false);
    } break;
    }
}
