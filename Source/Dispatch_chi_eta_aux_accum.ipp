#include <utility>
// NOTE(cmo): Machine-Generated!
template <SimdType simd, typename ...Args>
inline auto dispatch_chi_eta_aux_accum_(bool first, bool second, bool third, bool fourth, Args&& ...args)
-> decltype(chi_eta_aux_accum<simd, false, false, false, false>(std::forward<Args>(args)...))
{
    u32 dispatcher__ = first;
    dispatcher__ += second << 1;
    dispatcher__ += third << 2;
    dispatcher__ += fourth << 3;
    

    switch (dispatcher__)
    {
    case 0:
    {
        return chi_eta_aux_accum<simd, false, false, false, false>(std::forward<Args>(args)...);
    } break;
    case 1:
    {
        return chi_eta_aux_accum<simd, true, false, false, false>(std::forward<Args>(args)...);
    } break;
    case 2:
    {
        return chi_eta_aux_accum<simd, false, true, false, false>(std::forward<Args>(args)...);
    } break;
    case 3:
    {
        return chi_eta_aux_accum<simd, true, true, false, false>(std::forward<Args>(args)...);
    } break;
    case 4:
    {
        return chi_eta_aux_accum<simd, false, false, true, false>(std::forward<Args>(args)...);
    } break;
    case 5:
    {
        return chi_eta_aux_accum<simd, true, false, true, false>(std::forward<Args>(args)...);
    } break;
    case 6:
    {
        return chi_eta_aux_accum<simd, false, true, true, false>(std::forward<Args>(args)...);
    } break;
    case 7:
    {
        return chi_eta_aux_accum<simd, true, true, true, false>(std::forward<Args>(args)...);
    } break;
    case 8:
    {
        return chi_eta_aux_accum<simd, false, false, false, true>(std::forward<Args>(args)...);
    } break;
    case 9:
    {
        return chi_eta_aux_accum<simd, true, false, false, true>(std::forward<Args>(args)...);
    } break;
    case 10:
    {
        return chi_eta_aux_accum<simd, false, true, false, true>(std::forward<Args>(args)...);
    } break;
    case 11:
    {
        return chi_eta_aux_accum<simd, true, true, false, true>(std::forward<Args>(args)...);
    } break;
    case 12:
    {
        return chi_eta_aux_accum<simd, false, false, true, true>(std::forward<Args>(args)...);
    } break;
    case 13:
    {
        return chi_eta_aux_accum<simd, true, false, true, true>(std::forward<Args>(args)...);
    } break;
    case 14:
    {
        return chi_eta_aux_accum<simd, false, true, true, true>(std::forward<Args>(args)...);
    } break;
    case 15:
    {
        return chi_eta_aux_accum<simd, true, true, true, true>(std::forward<Args>(args)...);
    } break;
    
    default:
    {
        assert(false);
    } break;
    }
}
