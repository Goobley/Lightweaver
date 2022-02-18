// -*- c++ -*-
#if !defined(JASPP_H)
/* ==========================================================================
   $File: JasPP.hpp $
   $Version: 1.0 $
   $Notice: (C) Copyright 2015-2019 Chris Osborne. All Rights Reserved. $
   $License: MIT: http://opensource.org/licenses/MIT $
   ========================================================================== */

#define JASPP_H

#ifndef JASNAH_NO_FIF
    #define If ((
    #define Then )?(
    #define Else ):(
    #define End ))
#endif

#define JAS_STRINGIFY_IMPL(x) #x
#define JasStringify(x) JAS_STRINGIFY_IMPL(x)

#define JAS_CONCAT_IMPL(A, B) A##B
#define JasConcat(A, B) JAS_CONCAT_IMPL(A, B)

#define JAS_EXPAND(x) x

#define JAS_UNPACK1(_1)
#define JAS_UNPACK2(OBJ, NAME1) auto& NAME1(OBJ.NAME1) // Unpacks 1 param
#define JAS_UNPACK3(OBJ, NAME1, NAME2) auto& NAME1(OBJ.NAME1); auto& NAME2(OBJ.NAME2) // Unpacks 2 params
#define JAS_UNPACK4(OBJ, NAME1, NAME2, NAME3) auto& NAME1(OBJ.NAME1); auto& NAME2(OBJ.NAME2); \
    auto& NAME3 = OBJ.NAME3
#define JAS_UNPACK5(OBJ, NAME1, NAME2, NAME3, NAME4) auto& NAME1(OBJ.NAME1); auto& NAME2(OBJ.NAME2); \
    auto& NAME3(OBJ.NAME3); auto& NAME4(OBJ.NAME4)
#define JAS_UNPACK6(OBJ, NAME1, NAME2, NAME3, NAME4, NAME5) auto& NAME1(OBJ.NAME1); auto& NAME2(OBJ.NAME2); \
    auto& NAME3(OBJ.NAME3); auto& NAME4(OBJ.NAME4); auto& NAME5(OBJ.NAME5)
#define JAS_UNPACK7(OBJ, NAME1, NAME2, NAME3, NAME4, NAME5, NAME6) auto& NAME1(OBJ.NAME1); auto& NAME2(OBJ.NAME2); \
    auto& NAME3(OBJ.NAME3); auto& NAME4(OBJ.NAME4); auto& NAME5(OBJ.NAME5); auto& NAME6(OBJ.NAME6)
#define JAS_UNPACK8(OBJ, NAME1, NAME2, NAME3, NAME4, NAME5, NAME6, NAME7) auto& NAME1(OBJ.NAME1); auto& NAME2(OBJ.NAME2); \
    auto& NAME3(OBJ.NAME3); auto& NAME4(OBJ.NAME4); auto& NAME5(OBJ.NAME5); auto& NAME6(OBJ.NAME6); auto& NAME7(OBJ.NAME7)

#define JAS_VA_NARGS_IMPL(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, N, ...) N
#define JAS_VA_NARGS(...) JAS_EXPAND(JAS_VA_NARGS_IMPL(__VA_ARGS__, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1))


#define JAS_UNPACK_IMPL2(Length, ...) JAS_EXPAND(JAS_UNPACK##Length(__VA_ARGS__))
#define JAS_UNPACK_IMPL(Length, ...) JAS_EXPAND(JAS_UNPACK_IMPL2(Length, __VA_ARGS__))
#define JasUnpack(...) JAS_EXPAND(JAS_UNPACK_IMPL(JAS_VA_NARGS(__VA_ARGS__), __VA_ARGS__))

#define JAS_PACK1(_1)
#define JAS_PACK2(OBJ, L1) OBJ.L1 = L1
#define JAS_PACK3(OBJ, L1, L2) OBJ.L1 = L1; OBJ.L2 = L2
#define JAS_PACK4(OBJ, L1, L2, L3) OBJ.L1 = L1; OBJ.L2 = L2; OBJ.L3 = L3
#define JAS_PACK5(OBJ, L1, L2, L3, L4) OBJ.L1 = L1; OBJ.L2 = L2; OBJ.L3 = L3; OBJ.L4 = L4
#define JAS_PACK6(OBJ, L1, L2, L3, L4, L5) OBJ.L1 = L1; OBJ.L2 = L2; OBJ.L3 = L3; OBJ.L4 = L4; OBJ.L5 = L5
#define JAS_PACK7(OBJ, L1, L2, L3, L4, L5, L6) OBJ.L1 = L1; OBJ.L2 = L2; OBJ.L3 = L3; OBJ.L4 = L4; OBJ.L5 = L5; OBJ.L6 = L6
#define JAS_PACK8(OBJ, L1, L2, L3, L4, L5, L6, L7) OBJ.L1 = L1; OBJ.L2 = L2; OBJ.L3 = L3; OBJ.L4 = L4; OBJ.L5 = L5; OBJ.L6 = L6; OBJ.L7 = L7

#define JAS_PACK_IMPL2(Length, ...) JAS_EXPAND(JAS_PACK##Length(__VA_ARGS__))
#define JAS_PACK_IMPL(Length, ...) JAS_EXPAND(JAS_PACK_IMPL2(Length, __VA_ARGS__))
#define JasPack(...) JAS_EXPAND(JAS_PACK_IMPL(JAS_VA_NARGS(__VA_ARGS__), __VA_ARGS__))

#define JAS_PACK_PTR1(_1)
#define JAS_PACK_PTR2(OBJ, L1) OBJ.L1 = &L1
#define JAS_PACK_PTR3(OBJ, L1, L2) OBJ.L1 = &L1; OBJ.L2 = &L2
#define JAS_PACK_PTR4(OBJ, L1, L2, L3) OBJ.L1 = &L1; OBJ.L2 = &L2; OBJ.L3 = &L3
#define JAS_PACK_PTR5(OBJ, L1, L2, L3, L4) OBJ.L1 = &L1; OBJ.L2 = &L2; OBJ.L3 = &L3; OBJ.L4 = &L4
#define JAS_PACK_PTR6(OBJ, L1, L2, L3, L4, L5) OBJ.L1 = &L1; OBJ.L2 = &L2; OBJ.L3 = &L3; OBJ.L4 = &L4; OBJ.L5 = &L5
#define JAS_PACK_PTR7(OBJ, L1, L2, L3, L4, L5, L6) OBJ.L1 = &L1; OBJ.L2 = &L2; OBJ.L3 = &L3; OBJ.L4 = &L4; OBJ.L5 = &L5; OBJ.L6 = &L6
#define JAS_PACK_PTR8(OBJ, L1, L2, L3, L4, L5, L6, L7) OBJ.L1 = &L1; OBJ.L2 = &L2; OBJ.L3 = &L3; OBJ.L4 = &L4; OBJ.L5 = &L5; OBJ.L6 = &L6; OBJ.L7 = &L7

#define JAS_PACK_PTR_IMPL2(Length, ...) JAS_EXPAND(JAS_PACK_PTR##Length(__VA_ARGS__))
#define JAS_PACK_PTR_IMPL(Length, ...) JAS_EXPAND(JAS_PACK_PTR_IMPL2(Length, __VA_ARGS__))
#define JasPackPtr(...) JAS_EXPAND(JAS_PACK_PTR_IMPL(JAS_VA_NARGS(__VA_ARGS__), __VA_ARGS__))

#define InWhich(BODY, x, y)                     \
[=](auto x)                                     \
BODY                                            \
(y)

#define InWhichFn(xin, BODY, x, y)              \
[=](auto xin)                                   \
{                                               \
    return [=](auto x)                          \
           BODY                                 \
           (y);                                 \
}

#define LetWhere(STMT, x, y)                    \
[=](auto x)                                     \
{                                               \
    return STMT;                                \
}                                               \
(y)

#define JAS_DO_WHERE1(_1)
#define JAS_DO_WHERE2(_1, _2)
#define JAS_DO_WHERE3(EXPR, IN1, IN1EXPR)          \
[=](auto IN1)                                   \
{                                               \
    return EXPR;                                \
}                                               \
(IN1EXPR)
#define JAS_DO_WHERE4(_1, _2, _3, _4)
#define JAS_DO_WHERE6(_1, _2, _3, _4, _5, _6)
#define JAS_DO_WHERE8(_1, _2, _3, _4, _5, _6, _7, _8)
#define JAS_DO_WHERE10(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10)

#define JAS_DO_WHERE5(EXPR, IN1, IN1EXPR, IN2, IN2EXPR)    \
[=](auto IN1, auto IN2)                             \
{                                                   \
    return EXPR;                                    \
}                                                   \
(IN1EXPR, IN2EXPR)
#define JAS_DO_WHERE7(EXPR, IN1, IN1EXPR, IN2, IN2EXPR, IN3, IN3EXPR)  \
[=](auto IN1, auto IN2, auto IN3)                                        \
{                                                   \
    return EXPR;                                    \
}                                                   \
(IN1EXPR, IN2EXPR, IN3EXPR)

#define JAS_DO_WHERE9(EXPR, IN1, IN1EXPR, IN2, IN2EXPR, IN3, IN3EXPR, IN4, IN4EXPR) \
[=](auto IN1, auto IN2, auto IN3, auto IN4)                         \
{                                                                   \
    return EXPR;                                                    \
}                                                                   \
(IN1EXPR, IN2EXPR, IN3EXPR, IN4EXPR)

#define JAS_DO_WHERE11(EXPR, IN1, IN1EXPR, IN2, IN2EXPR, IN3, IN3EXPR, IN4, IN4EXPR, IN5, IN5EXPR) \
[=](auto IN1, auto IN2, auto IN3, auto IN4, auto IN5)               \
{                                                                   \
    return EXPR;                                                    \
}                                                                   \
(IN1EXPR, IN2EXPR, IN3EXPR, IN4EXPR, IN5EXPR)


#define JAS_DO_WHERE_IMPL2(Length, ...) JAS_DO_WHERE##Length(__VA_ARGS__)
#define JAS_DO_WHERE_IMPL(Length, ...) JAS_DO_WHERE_IMPL2(Length, __VA_ARGS__)
#define ExprWhere(...) JAS_DO_WHERE_IMPL(JAS_VA_NARGS(__VA_ARGS__), __VA_ARGS__)

// Add recursive let-in definition, maybe we can sugar ExprWhere to give something to this effect


#endif
