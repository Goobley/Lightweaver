from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np
import os
import os.path as path
import platform
from copy import copy

# NOTE(cmo): There's an implicit (not great) assumption in the following that a
# Windows compile will always be done with MSVC. Whilst this is true for me, and
# on the CI build scripts, this may not be true in general (cygwin-etc). Some
# modification may be needed for that eventuality to use GNUish flags.

# We slightly abuse the setuptools eco-system to build the SIMD implementations,
# primarily on Windows. These are not technically Python extension modules, but
# we do want them to be built easily, using the same toolchain as Lightweaver
# itself. To this end, we do pretend that they are extension modules. For
# Windows this means including a stub function PyInit_{ModuleName} that can be
# exported by the linker, or DLL construction fails. This is handled through
# compiling and linking against Source/WindowsExtensionStub.cpp whilst defining
# LW_MODULE_STUB_NAME to be the name of the module to be "exported".

def readme():
    with open('README.md', 'r') as f:
        return f.read()

posixCiArgs = ['-march=corei7-avx', '-mtune=corei7-avx']
posixLocalArgs = ['-march=native', '-mtune=native']
posixArgs = {
   'baseCompileArgs': ['-std=c++17', '-Wno-sign-compare']
                      + (posixCiArgs if ('LW_CI_BUILD' in os.environ
                                        and platform.system() != 'Darwin')
                         else posixLocalArgs),
   'SSE2Args': ['-msse2'],
   'AVX2FMAArgs': ['-mavx2', '-mfma'],
   'AVX512Args': ['-mavx512f', '-mavx512dq', '-mfma'],
   'libs': ['dl'],
   'linkArgs': [],
   'stubDefinePrefix': '-DLW_MODULE_STUB_NAME=',
   'lwCoreDefine': ['-DLW_CORE_LIB'],
   'fsIterExtensionExports': [],
}
msvcArgs = {
   'baseCompileArgs': ['/std:c++17', '/Z7'],
   'SSE2Args': [],
   'AVX2FMAArgs': ['/arch:AVX2'],
   'AVX512Args': ['/arch:AVX512'],
   'libs': [],
   'linkArgs': ['/DEBUG:FULL'],
   'stubDefinePrefix': '/DLW_MODULE_STUB_NAME=',
   'lwCoreDefine': ['/DLW_CORE_LIB'],
   'fsIterExtensionExports': ['fs_iteration_fns_provider'],
}

def prepend_source_dir(x):
    return [path.join('Source', y) for y in x]

coreSource = prepend_source_dir(['LightweaverAmalgamated.cpp'])
coreDepends = ['Atmosphere.cpp', 'Background.cpp', 'Background.hpp', 'Bezier.hpp',
               'CmoArray.hpp', 'Constants.hpp', 'EscapeProbability.cpp', 'Faddeeva.cc',
               'Faddeeva.hh', 'FastBackground.cpp', 'FastBackground.hpp',
               'FormalInterface.cpp', 'FormalScalar.cpp', 'FormalScalar2d.cpp',
               'FormalStokes.cpp', 'LuSolve.cpp', 'LuSolve.hpp', 'LwAtmosphere.hpp',
               'LwAtom.hpp', 'LwContext.hpp', 'LwFormalInterface.hpp',
               'LwFormalInterfacePosix.hpp', 'LwFormalInterfaceWin.hpp',
               'LwInternal.hpp', 'LwMisc.hpp', 'LwTransition.hpp', 'Ng.hpp', 'Prd.cpp',
               'Simd.hpp', 'SimdFullIterationTemplates.hpp', 'TaskScheduler.h',
               'TaskStorage.cpp', 'TaskStorage.hpp', 'UpdatePopulations.cpp', 'Utils.hpp']
coreDepends = prepend_source_dir(coreDepends)
stubSource = []
if platform.system() == 'Windows':
    stubSource.append('WindowsExtensionStub.cpp')
stubSource = prepend_source_dir(stubSource)

if platform.system() == 'Windows':
    buildArgs = msvcArgs
else:
    buildArgs = posixArgs

SimdImpls = ['SSE2', 'AVX2FMA', 'AVX512']
for simd in SimdImpls:
    if f'LW_NO_{simd}_LIB' in os.environ:
        SimdImpls.remove(simd)

simdImplDepends = {impl: coreDepends + prepend_source_dir([f'SimdImpl_{impl}.cpp'])
                   for impl in SimdImpls}

def extension_list(args):
    lwExts = []
    lwExts.append(Extension('lightweaver.LwCompiled',
                  sources=[path.join('Source', 'LwMiddleLayer.pyx')] + coreSource,
                  depends=coreDepends,
                  include_dirs=[np.get_include()],
                  language='c++',
                  extra_compile_args=args['baseCompileArgs'] + args['lwCoreDefine'],
                  extra_link_args=args['linkArgs']))
    lwExts = cythonize(lwExts, language_level=3)
    for simdImpl in SimdImpls:
        lwExts.append(Extension(f'lightweaver.DefaultIterSchemes.SimdImpl_{simdImpl}',
                                sources=[path.join('Source', f'SimdImpl_{simdImpl}.cpp')] +
                                        coreSource + stubSource,
                                depends=simdImplDepends[simdImpl],
                                language='c++',
                                extra_compile_args=args['baseCompileArgs'] +
                                                   args[f'{simdImpl}Args'] +
                                                   [f'{args["stubDefinePrefix"]}SimdImpl_{simdImpl}'],
                                extra_link_args=args['linkArgs'],
                                # NOTE(cmo): There seems to be a bug with
                                # export_symbols affecting its arguments, so we
                                # submit a copy.
                                export_symbols=copy(args['fsIterExtensionExports']),
                                optional=True))
    return lwExts

setup(name='lightweaver',
      setup_requires=['setuptools_scm'],
      use_scm_version=True,
      install_requires=['numpy<1.22,>=1.19', 'scipy', 'matplotlib', 'numba>=0.55',
                        'parse', 'specutils', 'tqdm', 'weno4', 'adjustText', 'pyyaml'],
      author='Chris Osborne',
      author_email='c.osborne.1@research.gla.ac.uk',
      license='MIT',
      url='http://github.com/Goobley/Lightweaver',
      description='Non-LTE Radiative Transfer Framework in Python',
      ext_modules=extension_list(buildArgs),
      include_package_data=True,
      long_description=readme(),
      long_description_content_type='text/markdown',
      python_requires='>=3.8')