from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(name='lightweaver', version='0.1.0', packages=['lightweaver'],
 ext_modules=cythonize([
    Extension(
        "lightweaver.LwCompiled",
        ["Source/LightweaverAmalgamated.cpp", "Source/CAtmosphere.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-std=c++17", "-Wno-sign-compare", "-march=native", "-mavx2", "-funroll-loops"],
        language="c++"
)], language_level=3))

# set -x CXX "/home/osborne/gcc-8/bin/g++8 -pthread"; set -x CC "/home/osborne/gcc-8/bin/g++8 -pthread"; set -x LDSHARED "/home/osborne/gcc-8/bin/g++8 -pthread -shared"; set -x DISTUTILS_DEBUG 1; and sudo -HE python3.7 -m pip install -vvv -e .

# set -x CC "/home/osborne/gcc-8/bin/gcc8 -pthread"; set -x CXX "/home/osborne/gcc-8/bin/g++8  -pthread"; set -x LDSHARED "/home/osborne/gcc-8/bin/gcc8 -pthread -shared"; python3.7 -m pip --proxy $http_proxy install -vvv -e .
