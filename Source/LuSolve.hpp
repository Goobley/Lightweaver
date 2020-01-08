#ifndef CMO_LUSOLVE_HPP
#define CMO_LUSOLVE_HPP

#include "CmoArray.hpp"

void solve_lin_eq(F64View2D A, F64View b, bool improve=true);

#endif