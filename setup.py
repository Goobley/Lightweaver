from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import platform
import os

def readme():
    with open('README.md', 'r') as f:
        return f.read()

posixArgs = ["-std=c++17", "-Wno-sign-compare", "-funroll-loops"]
if 'LW_CI_BUILD'  in os.environ:
    # NOTE(cmo): Compile for sandy bridge or newer when building on CI
    posixArgs += ["-march=corei7-avx", "-mtune=corei7-avx"]
else:
    # NOTE(cmo): Local compile
    posixArgs += ["-march=native", "-mtune=native"]

# TODO(cmo): Find similar architecture args for MSVC

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
    setup_requires=['setuptools_scm'],
    use_scm_version=True,
    packages=['lightweaver'],
    ext_modules=cythonize([
        Extension(
            'lightweaver.LwCompiled',
            ['Source/LightweaverAmalgamated.cpp', 'Source/LwMiddleLayer.pyx'],
            include_dirs=[np.get_include()],
            libraries=libs,
            extra_compile_args=compileArgs,
            language="c++")],
        language_level=3),
    install_requires=['numpy', 'scipy', 'matplotlib', 'numba', 'parse',
                      'specutils', 'tqdm', 'weno4'],
    author='Chris Osborne',
    author_email='c.osborne.1@research.gla.ac.uk',
    license='MIT',
    url='http://github.com/Goobley/Lightweaver',
    description='Non-LTE Radiative Transfer Framework in Python',
    include_package_data=True,
    long_description=readme(),
    long_description_content_type='text/markdown',
    python_requires='>=3.8'
    )

# set -x CXX "/home/osborne/gcc-8/bin/g++8 -pthread"; set -x CC "/home/osborne/gcc-8/bin/g++8 -pthread"; set -x LDSHARED "/home/osborne/gcc-8/bin/g++8 -pthread -shared"; set -x DISTUTILS_DEBUG 1; and sudo -HE python3.7 -m pip install -vvv -e .

# set -x CC "/home/osborne/gcc-8/bin/gcc8 -pthread"; set -x CXX "/home/osborne/gcc-8/bin/g++8  -pthread"; set -x LDSHARED "/home/osborne/gcc-8/bin/gcc8 -pthread -shared"; python3.7 -m pip --proxy $http_proxy install -vvv -e .

# cl /Ox /std:c++17 -nologo -WL /MD /GL Parabolic2d.cpp -LD /link /DLL -incremental:no /LTCG -EXPORT:fs_provider /OUT:Parabolic2d.dll
# NOTE(cmo): in the x86_64 cross tools command prompt