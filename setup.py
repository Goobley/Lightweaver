from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(name="CAtmosphere", ext_modules=cythonize([Extension(
    "CAtmosphere",
    ["CAtmosphere.pyx", "Atmosphere.cpp", "Formal.cpp", "Faddeeva.cc", "LuSolve.cpp"],
    include_dirs=[np.get_include()],
    extra_compile_args=["-std=c++17", "-Wno-sign-compare", "-march=native", "-mavx2", "-funroll-loops"],
    language="c++"
), 
Extension(
    "witt_cmo",
    ["witt_cmo.pyx"],
    include_dirs=[np.get_include()]
)], language_level=3))