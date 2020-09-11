#ifndef CMO_LW_FORMAL_INTERFACE_POSIX_HPP
#define CMO_LW_FORMAL_INTERFACE_POSIX_HPP

struct PlatformSharedLibrary
{
    void* handle;
};

#ifdef CMO_FORMAL_INTERFACE_IMPL
#include <dlfcn.h>

namespace LwInternal
{
bool load_library(PlatformSharedLibrary* lib, const char* path)
{
    void* handle = dlopen(path, RTLD_LAZY);
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
    F f = (F)dlsym(lib.handle, name);

    return f;
}

void close_library(PlatformSharedLibrary lib)
{
    dlclose(lib.handle);
}
}
#endif
#else
#endif
