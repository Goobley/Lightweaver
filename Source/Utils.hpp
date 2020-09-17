#ifndef CMO_LW_UTILS_HPP
#define CMO_LW_UTILS_HPP
#include "Constants.hpp"
#include "CmoArray.hpp"
#include "JasPP.hpp"
#include <algorithm>

template <typename T, typename U>
inline int hunt(int len, T first, U val)
{
    auto last = first + len;
    auto it = std::upper_bound(first, last, val) - 1;
    return it - first;
}

inline int hunt(F64View x, f64 val)
{
    return hunt(x.dim0, x.data, val);
}

inline void linear(F64View xTable, F64View yTable, F64View x, F64View y)
{
    const int Ntable = xTable.shape(0);
    const int N = x.shape(0);
    bool ascend = xTable(1) > xTable(0);
    const f64 xMin = If ascend Then xTable(0) Else xTable(Ntable-1) End;
    const f64 xMax = If ascend Then xTable(Ntable-1) Else xTable(0) End;

    for (int n = 0; n < N; ++n)
    {
        if (x(n) <= xMin)
            y(n) = If ascend Then yTable(0) Else yTable(Ntable-1) End;
        else if (x(n) >= xMax)
            y(n) = If ascend Then yTable(Ntable-1) Else yTable(0) End;
        else
        {
            int j = hunt(xTable, x(n));

            f64 fx = (xTable(j+1) - x(n)) / (xTable(j+1) - xTable(j));
            y(n) = fx * yTable(j) + (1 - fx) * yTable(j+1);
        }
    }
}

inline f64 linear(F64View xTable, F64View yTable, f64 x)
{
    const int Ntable = xTable.shape(0);
    bool ascend = xTable(1) > xTable(0);
    const f64 xMin = If ascend Then xTable(0) Else xTable(Ntable-1) End;
    const f64 xMax = If ascend Then xTable(Ntable-1) Else xTable(0) End;

    if (x <= xMin)
        return If ascend Then yTable(0) Else yTable(Ntable-1) End;

    if (x >= xMax)
        return If ascend Then yTable(Ntable-1) Else yTable(0) End;

    int j = hunt(xTable, x);

    f64 fx = (xTable(j+1) - x) / (xTable(j+1) - xTable(j));
    return fx * yTable(j) + (1 - fx) * yTable(j+1);
}

inline f64 bilinear(int Ncol, int Nrow, const f64 *f, f64 x, f64 y) {
  int i, j, i1, j1;
  double fx, fy;

  /* --- Bilinear interpolation of the function f on the fractional
         indices x and y --                            -------------- */

  i = (int)x;
  fx = x - i;
  if (i == Ncol - 1)
    i1 = i;
  else
    i1 = i + 1;
  j = (int)y;
  fy = y - j;
  if (j == Nrow - 1)
    j1 = j;
  else
    j1 = j + 1;

  return (1.0 - fx) * (1.0 - fy) * f[j * Ncol + i] +
         fx * (1.0 - fy) * f[j * Ncol + i1] +
         (1.0 - fx) * fy * f[j1 * Ncol + i] + fx * fy * f[j1 * Ncol + i1];
}


#else
#endif