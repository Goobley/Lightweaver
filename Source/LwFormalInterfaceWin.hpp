#ifndef CMO_LW_FORMAL_INTERFACE_WIN_HPP
#define CMO_LW_FORMAL_INTERFACE_WIN_HPP
#define WIN32_LEAN_AND_MEAN
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
    printf("Loading from path (converted) %ls\n", wPath.data());
    int absPathLen = GetFullPathNameW(wPath.data(), 0, nullptr, nullptr);
    std::vector<WCHAR> absPath(absPathLen);
    GetFullPathNameW(wPath.data(), absPathLen, absPath.data(), nullptr);
    printf("Abs path (converted) %ls\n", absPath.data());
    HMODULE handle = LoadLibraryW(absPath.data());
    if (!handle)
    {
        printf("Couldn't load DLL\n");
        lib->handle = nullptr;
        return false;
    }
    lib->handle = handle;
    printf("loaded DLL\n");
    return true;
}

template <typename F>
F load_function(PlatformSharedLibrary lib, const char* name)
{
    F f = (F)GetProcAddress(lib.handle, name);
    printf("Attempted function load\n");

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
