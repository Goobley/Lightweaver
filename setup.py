from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(name="CAtmosphere", ext_modules=cythonize([
    Extension("CmoArrayHelper", ["CmoArrayHelper.pyx"],
    include_dirs=[np.get_include()], 
    extra_compile_args=["-std=c++17"], language="c++"),
    Extension(
    "CAtmosphere",
    ["CAtmosphere.pyx", "Background.cpp", "Atmosphere.cpp", "Formal.cpp", "Faddeeva.cc", "LuSolve.cpp"],
    include_dirs=[np.get_include()],
    extra_compile_args=["-std=c++17", "-Wno-sign-compare", "-march=native", "-mavx2", "-funroll-loops"],
    language="c++"
)], language_level=3))