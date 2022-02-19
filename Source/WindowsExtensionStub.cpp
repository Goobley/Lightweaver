#include "JasPP.hpp"

#ifndef LW_MODULE_STUB_NAME
#error "This file must be included with the value of LW_MODULE_STUB_NAME defined to be the module name (for the Windows linker)"
#else

extern "C"
{
#include "Python.h"

PyMODINIT_FUNC JasConcat(PyInit_, LW_MODULE_STUB_NAME)()
{
    return nullptr;
}

}
#endif