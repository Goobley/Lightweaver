#ifndef CMO_LW_EXTRA_PARAMS_HPP
#define CMO_LW_EXTRA_PARAMS_HPP
#include "CmoArray.hpp"
#include "Constants.hpp"

#include <string>
#include <unordered_map>
#include <variant>

typedef std::variant<std::monostate, // empty state
                     std::string, bool, i64, f64,
                     Jasnah::Array1NonOwn<i64>,
                     Jasnah::Array2NonOwn<i64>,
                     Jasnah::Array3NonOwn<i64>,
                     Jasnah::Array4NonOwn<i64>,
                     Jasnah::Array5NonOwn<i64>,
                     F64View1D,
                     F64View2D,
                     F64View3D,
                     F64View4D,
                     F64View5D>
        Variant;

struct ExtraParams
{
    std::unordered_map<std::string, Variant> map;

    bool contains(const std::string& key) const
    {
        auto iter = map.find(key);
        return iter != map.end();
    }

    template <typename T>
    void insert(const std::string& key, T value)
    {
        map.insert_or_assign(key, value);
    }

    template <typename T>
    T& get_as(const std::string& key)
    {
        // NOTE(cmo): This can throw from either step here.
        auto& var = map.at(key);
        return std::get<T>(var);
    }

    Variant get(const std::string& key)
    {
        auto iter = map.find(key);
        if (iter == map.end())
            return Variant{}; // returns an "empty" Variant

        return iter->second;
    }
};
#else
#endif