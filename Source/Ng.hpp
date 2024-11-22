#ifndef CMO_NG_HPP
#define CMO_NG_HPP

#include "Constants.hpp"
#include "CmoArray.hpp"
#include "LuSolve.hpp"
#include <cmath>
#include <tuple>

struct NgChange
{
    f64 dMax;
    i64 dMaxIdx;
};

struct NgArgs {
    int nOrder;
    int nPeriod;
    int nDelay;
    f64 threshold;
    f64 lowerThreshold;
};

struct Ng
{
    int len;
    int Norder;
    int Nperiod;
    int Ndelay;
    f64 threshold;
    f64 lowerThreshold;
    F64Arr2D previous;
    F64Arr1D prevRelChange;
    NgChange prevNgChange;
    int count;
    bool init;

    Ng() : len(0), Norder(0), Nperiod(0), Ndelay(0),
           threshold(0.0), lowerThreshold(0.0),
           previous{}, prevRelChange{}, prevNgChange{},
           count(0), init(false)
    {}

    Ng(const NgArgs& args, F64View sol)
       : len(sol.shape(0)), Norder(args.nOrder), Nperiod(args.nPeriod),
         Ndelay(max(args.nDelay, args.nPeriod+2)),
         threshold(args.threshold), lowerThreshold(args.lowerThreshold),
         previous(0.0, Norder+2, len), prevRelChange(10.0, Norder+2),
         prevNgChange{}, count(0), init(true)
    {
        auto storage = previous(count);
        for (int k = 0; k < len; ++k)
            storage(k) = sol(k);
        count += 1;
    }

    Ng(const Ng& other) = default;
    Ng(Ng&& other) = default;
    Ng& operator=(const Ng& other) = default;
    Ng& operator=(Ng&& other) = default;

    inline int storage_index(int cnt)
    {
        return cnt % (Norder + 2);
    }

    bool accelerate(F64View sol)
    {
        if (!init)
        {
            // NOTE(cmo): If we got to here without being initialised then we're just being used for max_change
            len = sol.shape(0);
            previous = F64Arr2D(0.0, 2, len);
            prevRelChange = F64Arr1D(1.0, 2);
            init = true;
        }

        auto idx = storage_index(count);
        for (int k = 0; k < len; ++k)
            previous(idx, k) = sol(k);
        count += 1;
        compute_max_change();
        prevRelChange(idx) = prevNgChange.dMax;

        if (!((Norder > 0)
              && (count >= Ndelay)
              && ((count - Ndelay) % Nperiod) == 0)
           )
            return false;

        if (prevNgChange.dMax < lowerThreshold)
            return false;

        for (int i = 0; i < prevRelChange.shape(0); ++i)
        {
            if (prevRelChange(i) > threshold) {
                return false;
            }
        }

        auto Delta = F64Arr2D(Norder + 1, len);
        auto weight = F64Arr(len);

        for (int i = 0; i <= Norder; ++i)
        {
            int ip = storage_index(count - i - 1);
            int ipp = storage_index(count - i - 2);
            for (int k = 0; k < len; ++k)
                Delta(i, k) = previous(ip, k) - previous(ipp, k);
        }
        for (int k = 0; k < len; ++k)
            weight(k) = 1.0 / abs(sol(k));

        auto A = F64Arr2D(0.0, Norder, Norder);
        auto b = F64Arr1D(0.0, Norder);
        for (int j = 0; j < Norder; ++j)
        {
            for (int k = 0; k < len; ++k)
                b(j) += weight(k)
                        * Delta(0, k)
                        * (Delta(0, k) - Delta(j+1, k));

            for (int i = 0; i < Norder; ++i)
                for (int k = 0; k < len; ++k)
                    A(i,j) += weight(k)
                              * (Delta(j+1, k) - Delta(0, k))
                              * (Delta(i+1, k) - Delta(0, k));
        }
        solve_lin_eq(A, b);

        int i0 = storage_index(count - 1);
        for (int i = 0; i < Norder; ++i)
        {
            int ip = storage_index(count - i - 2);
            for (int k = 0; k < len; ++k)
                sol(k) += b(i) * (previous(ip, k) - previous(i0, k));
        }
        for (int k = 0; k < len; ++k)
            previous(i0, k) = sol(k);

        return true;
    }

    inline NgChange relative_change_from_prev(F64View newSol)
    {
        if (!init || count < 1)
            return { 0.0, 0 };

        auto sol = previous(storage_index(count-1));
        if (newSol.shape(0) != len)
            return { 0.0, 0 };

        f64 dMax = 0.0;
        int maxIdx = 0;
        for (int k = 0; k < len; ++k)
        {
            if (newSol(k) != 0.0)
            {
                f64 change = abs((newSol(k) - sol(k)) / newSol(k));
                dMax = max_idx(dMax, change, maxIdx, k);
            }
        }
        return { dMax, maxIdx };
    }

    inline NgChange compute_max_change()
    {
        if (!init || count < 2)
            return { 0.0, 0 };

        auto old = previous(storage_index(count-2));
        auto current = previous(storage_index(count-1));
        f64 dMax = 0.0;
        int maxIdx = 0;
        for (int k = 0; k < len; ++k)
        {
            if (current(k) != 0.0)
            {
                f64 change = abs((current(k) - old(k)) / current(k));
                dMax = max_idx(dMax, change, maxIdx, k);
            }
        }
        prevNgChange = { dMax, maxIdx };
        return { dMax, maxIdx };
    }

    inline NgChange max_change()
    {
        return prevNgChange;
    }

    inline void clear()
    {
        previous.fill(0);
        count = 0;
    }
};

#endif