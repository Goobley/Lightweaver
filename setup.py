from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import platform
import os

posixArgs = ["-std=c++17", "-Wno-sign-compare", "-march=native", "-mavx2", "-funroll-loops"]
posixLibs = ['dl']
msvcArgs = ["/std:c++17"]
msvcLibs = []

if platform.system() == 'Windows':
    compileArgs = msvcArgs
    libs = msvcLibs
else:
    compileArgs = posixArgs
    libs = posixLibs

setup(
    name='lightweaver',
    version='0.1.0',
    packages=['lightweaver'],
    ext_modules=cythonize([
        Extension(
            'lightweaver.LwCompiled',
            ['Source/LightweaverAmalgamated.cpp', 'Source/CAtmosphere.pyx'],
            # depends=['Source/'+ f for f in os.listdir('Source/')],
            include_dirs=[np.get_include()],
            libraries=libs,
            extra_compile_args=compileArgs,
            language="c++")],
        language_level=3),
    author='Chris Osborne',
    include_package_data=True
    )

# set -x CXX "/home/osborne/gcc-8/bin/g++8 -pthread"; set -x CC "/home/osborne/gcc-8/bin/g++8 -pthread"; set -x LDSHARED "/home/osborne/gcc-8/bin/g++8 -pthread -shared"; set -x DISTUTILS_DEBUG 1; and sudo -HE python3.7 -m pip install -vvv -e .

# set -x CC "/home/osborne/gcc-8/bin/gcc8 -pthread"; set -x CXX "/home/osborne/gcc-8/bin/g++8  -pthread"; set -x LDSHARED "/home/osborne/gcc-8/bin/gcc8 -pthread -shared"; python3.7 -m pip --proxy $http_proxy install -vvv -e .

# cl /Ox /std:c++17 -nologo -WL /MD /GL Parabolic2d.cpp -LD /link /DLL -incremental:no /LTCG -EXPORT:fs_provider /OUT:Parabolic2d.dll
# NOTE(cmo): in the x86_64 cross tools command prompt