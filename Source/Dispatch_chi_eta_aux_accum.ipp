#include <utility>
// NOTE(cmo): Machine-Generated!
template <SimdType simd, typename ...Args>
auto dispatch_chi_eta_aux_accum_(bool first, bool second, bool third, bool fourth, bool fifth, Args&& ...args)
-> decltype(chi_eta_aux_accum<simd, false, false, false, false, false>(std::forward<Args>(args)...))
{
    u32 dispatcher__ = first;
    dispatcher__ += second << 1;
    dispatcher__ += third << 2;
    dispatcher__ += fourth << 3;
    dispatcher__ += fifth << 4;
    

    switch (dispatcher__)
    {
    case 0:
    {
        return chi_eta_aux_accum<simd, false, false, false, false, false>(std::forward<Args>(args)...);
    } break;
    case 1:
    {
        return chi_eta_aux_accum<simd, true, false, false, false, false>(std::forward<Args>(args)...);
    } break;
    case 2:
    {
        return chi_eta_aux_accum<simd, false, true, false, false, false>(std::forward<Args>(args)...);
    } break;
    case 3:
    {
        return chi_eta_aux_accum<simd, true, true, false, false, false>(std::forward<Args>(args)...);
    } break;
    case 4:
    {
        return chi_eta_aux_accum<simd, false, false, true, false, false>(std::forward<Args>(args)...);
    } break;
    case 5:
    {
        return chi_eta_aux_accum<simd, true, false, true, false, false>(std::forward<Args>(args)...);
    } break;
    case 6:
    {
        return chi_eta_aux_accum<simd, false, true, true, false, false>(std::forward<Args>(args)...);
    } break;
    case 7:
    {
        return chi_eta_aux_accum<simd, true, true, true, false, false>(std::forward<Args>(args)...);
    } break;
    case 8:
    {
        return chi_eta_aux_accum<simd, false, false, false, true, false>(std::forward<Args>(args)...);
    } break;
    case 9:
    {
        return chi_eta_aux_accum<simd, true, false, false, true, false>(std::forward<Args>(args)...);
    } break;
    case 10:
    {
        return chi_eta_aux_accum<simd, false, true, false, true, false>(std::forward<Args>(args)...);
    } break;
    case 11:
    {
        return chi_eta_aux_accum<simd, true, true, false, true, false>(std::forward<Args>(args)...);
    } break;
    case 12:
    {
        return chi_eta_aux_accum<simd, false, false, true, true, false>(std::forward<Args>(args)...);
    } break;
    case 13:
    {
        return chi_eta_aux_accum<simd, true, false, true, true, false>(std::forward<Args>(args)...);
    } break;
    case 14:
    {
        return chi_eta_aux_accum<simd, false, true, true, true, false>(std::forward<Args>(args)...);
    } break;
    case 15:
    {
        return chi_eta_aux_accum<simd, true, true, true, true, false>(std::forward<Args>(args)...);
    } break;
    case 16:
    {
        return chi_eta_aux_accum<simd, false, false, false, false, true>(std::forward<Args>(args)...);
    } break;
    case 17:
    {
        return chi_eta_aux_accum<simd, true, false, false, false, true>(std::forward<Args>(args)...);
    } break;
    case 18:
    {
        return chi_eta_aux_accum<simd, false, true, false, false, true>(std::forward<Args>(args)...);
    } break;
    case 19:
    {
        return chi_eta_aux_accum<simd, true, true, false, false, true>(std::forward<Args>(args)...);
    } break;
    case 20:
    {
        return chi_eta_aux_accum<simd, false, false, true, false, true>(std::forward<Args>(args)...);
    } break;
    case 21:
    {
        return chi_eta_aux_accum<simd, true, false, true, false, true>(std::forward<Args>(args)...);
    } break;
    case 22:
    {
        return chi_eta_aux_accum<simd, false, true, true, false, true>(std::forward<Args>(args)...);
    } break;
    case 23:
    {
        return chi_eta_aux_accum<simd, true, true, true, false, true>(std::forward<Args>(args)...);
    } break;
    case 24:
    {
        return chi_eta_aux_accum<simd, false, false, false, true, true>(std::forward<Args>(args)...);
    } break;
    case 25:
    {
        return chi_eta_aux_accum<simd, true, false, false, true, true>(std::forward<Args>(args)...);
    } break;
    case 26:
    {
        return chi_eta_aux_accum<simd, false, true, false, true, true>(std::forward<Args>(args)...);
    } break;
    case 27:
    {
        return chi_eta_aux_accum<simd, true, true, false, true, true>(std::forward<Args>(args)...);
    } break;
    case 28:
    {
        return chi_eta_aux_accum<simd, false, false, true, true, true>(std::forward<Args>(args)...);
    } break;
    case 29:
    {
        return chi_eta_aux_accum<simd, true, false, true, true, true>(std::forward<Args>(args)...);
    } break;
    case 30:
    {
        return chi_eta_aux_accum<simd, false, true, true, true, true>(std::forward<Args>(args)...);
    } break;
    case 31:
    {
        return chi_eta_aux_accum<simd, true, true, true, true, true>(std::forward<Args>(args)...);
    } break;
    
    default:
    {
        assert(false);
    } break;
    }
}
