#ifndef CMO_LW_FORMAL_INTERFACE_WIN_HPP
#define CMO_LW_FORMAL_INTERFACE_WIN_HPP
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

struct PlatformSharedLibrary
{
    HMODULE handle;
};

#ifdef CMO_FORMAL_INTERFACE_IMPL
namespace LwInternal
{
bool load_library(PlatformSharedLibrary* lib, const char* path)
{
    std::vector<WCHAR> wPath;
    int wPathLen = MultiByteToWideChar(CP_UTF8, MB_PRECOMPOSED, path, -1, wPath.data(), 0);
    wPath.resize(wPathLen);
    MultiByteToWideChar(CP_UTF8, MB_PRECOMPOSED, path, -1, wPath.data(), wPathLen);
    int absPathLen = GetFullPathNameW(wPath.data(), 0, nullptr, nullptr);
    std::vector<WCHAR> absPath(absPathLen);
    GetFullPathNameW(wPath.data(), absPathLen, absPath.data(), nullptr);
    HMODULE handle = LoadLibraryW(absPath.data());
    if (!handle)
    {
        lib->handle = nullptr;
        return false;
    }
    lib->handle = handle;
    return true;
}

template <typename F>
F load_function(PlatformSharedLibrary lib, const char* name)
{
    F f = (F)GetProcAddress(lib.handle, name);

    return f;
}

void close_library(PlatformSharedLibrary lib)
{
    FreeLibrary(lib.handle);
}
}
#endif
#else
#endif
