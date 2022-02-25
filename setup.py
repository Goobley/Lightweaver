import os
import shutil
import sys
import warnings
from copy import copy
from distutils.file_util import copy_file
from distutils.sysconfig import get_config_var
from os import path
from typing import Dict, List, Union

import numpy as np
from Cython.Build import cythonize
from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.extension import Extension

# NOTE(cmo): There's an implicit (not great) assumption in the following that a
# Windows compile will always be done with MSVC. Whilst this is true for me, and
# on the CI build scripts, this may not be true in general (cygwin-etc). Some
# modification may be needed for that eventuality to use GNUish flags.
# As we are now using sys.platform rather than platform.system(), there is an
# option to distinguish between win32 and cygwin, but I don't have a cygwin
# environment.

# We slightly abuse the setuptools eco-system to build the SIMD implementations,
# primarily on Windows. These are not technically Python extension modules, but
# we do want them to be built easily, using the same toolchain as Lightweaver
# itself. To this end, we do pretend that they are extension modules. For
# Windows this means including a stub function PyInit_{ModuleName} that can be
# exported by the linker, or DLL construction fails. This is handled through
# compiling and linking against Source/WindowsExtensionStub.cpp whilst defining
# LW_MODULE_STUB_NAME to be the name of the module to be "exported".
# This is also done for enkiTS. However, now that we're defining our own
# build_ext, we can check for our own library type and not add the `PyInit`
# symbol when `build_ext.get_export_symbols` is called.

BuildDir = 'LwBuild'
CI_BUILD = 'LW_CI_BUILD' in os.environ

class LwSharedLibraryNoExtension(Extension):
    pass

# NOTE(cmo): Based on https://stackoverflow.com/a/60285245/3847013 , but
# modified for current setuptools, and to catch only the necessary library
class LwBuildExt(build_ext):
    def get_ext_filename(self, fullname):
        filename = super().get_ext_filename(fullname)
        so_ext = os.getenv('SETUPTOOLS_EXT_SUFFIX')
        if not so_ext:
            so_ext = get_config_var('EXT_SUFFIX')

        if fullname in self.ext_map:
            extension = self.ext_map[fullname]
            if isinstance(extension, LwSharedLibraryNoExtension):
                base_ext = path.splitext(filename)[1]
                filename = filename.replace(so_ext, "") + base_ext
        return filename

    def run(self):
        if sys.platform == 'win32':
            # NOTE(cmo): For any of our LwSharedLibraryNoExtension on Windows,
            # make sure the necessary /IMPLIB is put in the right place (we need
            # it), so add an /IMPLIB call too
            for ext in self.extensions:
                if isinstance(ext, LwSharedLibraryNoExtension):
                    fullname = self.get_ext_fullname(ext.name)
                    filename = self.get_ext_filename(fullname)
                    output = path.splitext(path.join(self.build_lib, filename))[0]
                    output += '.lib'
                    extras = [f'/IMPLIB:{output}']
                    # NOTE(cmo): Don't clobber anyone else holding a reference to this list.
                    ext.extra_link_args = ext.extra_link_args + extras
        elif sys.platform == 'darwin':
            # NOTE(cmo): On macOS we need to set the install_name of these
            # libraries to minimise the amount of rpath shenanigans.
            for ext in self.extensions:
                if isinstance(ext, LwSharedLibraryNoExtension):
                    fullname = self.get_ext_fullname(ext.name)
                    filename = self.get_ext_filename(fullname)
                    lib_name = path.split(filename)[1]
                    if any(arg == '-install_name' for arg in ext.extra_link_args):
                        continue
                    install_name = ['-install_name', f'@rpath/{lib_name}']
                    ext.extra_link_args = ext.extra_link_args + install_name
        super().run()

    def finalize_options(self):
        super().finalize_options()
        if sys.platform != 'darwin':
            return

        lw_shlibs = [ext for ext in self.extensions
                     if isinstance(ext, LwSharedLibraryNoExtension)]
        if lw_shlibs:
            self.setup_shlib_compiler()

    def build_extension(self, ext):
        if sys.platform != 'darwin':
            return super().build_extension(ext)

        if not isinstance(ext, LwSharedLibraryNoExtension):
            return super().build_extension(ext)

        # NOTE(cmo): Based on build_extension in setuptools, on macOS we need
        # the shlib_compiler, to output a dynamic library, rather than a bundle
        # as is normally used.
        try:
            _compiler = self.compiler
            self.compiler = self.shlib_compiler
            super().build_extension(ext)
        finally:
            self.compiler = _compiler

    def copy_extensions_to_source(self):
        super().copy_extensions_to_source()
        if sys.platform != 'win32':
            return

        if not self.inplace:
            warnings.warn('This block was only anticipated to run on an inplace (development) build, results may be not as expected.')

        build_py = self.get_finalized_command('build_py')
        for ext in self.extensions:
            if isinstance(ext, LwSharedLibraryNoExtension):
                fullname = self.get_ext_fullname(ext.name)
                modpath = fullname.split('.')
                package = '.'.join(modpath[:-1])
                package_dir = build_py.get_package_dir(package)

                filename = self.get_ext_filename(fullname)
                base_file = path.splitext(filename)[0]
                for file_ext in ['.exp', '.lib']:
                    extra_file_name = base_file + file_ext
                    dest_filename = os.path.join(package_dir,
                                                os.path.basename(extra_file_name))
                    src_filename = os.path.join(self.build_lib, extra_file_name)

                    copy_file(
                        src_filename, dest_filename, verbose=self.verbose,
                        dry_run=self.dry_run
                    )


def readme():
    with open('README.md', 'r') as f:
        return f.read()

posixCiArgs : Dict[str, List[str]] = {
    'linux': ['-march=corei7-avx', '-mtune=corei7-avx'],
    'darwin': [],
    'win32': [],
    'cygwin': [],
    'aix': []
}
posixLinkerArgs = {
    'linux': ['-Wl,-rpath,$ORIGIN', '-Wl,-rpath,$ORIGIN/..', '-Wl,-zlazy'],
    'darwin': ['-Wl,-rpath,@loader_path', '-Wl,-rpath,@loader_path/..'],
    'win32': [],
    'cygwin': [],
    'aix': []
}
posixLocalArgs = ['-march=native', '-mtune=native']
posixArgs : Dict[str, Union[str, List[str]]] = {
   'baseCompileArgs': ['-std=c++17', '-Wno-sign-compare'],
   'coreCompileArgs': (posixCiArgs[sys.platform] if CI_BUILD
                                                 else posixLocalArgs),
   'SSE2Args': ['-msse2'],
   'AVX2FMAArgs': ['-mavx2', '-mfma'],
   'AVX512Args': ['-mavx512f', '-mavx512dq', '-mfma'],
   'libs': ['dl', 'enkiTS'],
   'libDirs': [path.join(BuildDir, 'lightweaver')],
   'linkArgs': posixLinkerArgs[sys.platform],
   'stubDefinePrefix': '-DLW_MODULE_STUB_NAME=',
   'lwCoreDefine': ['-DLW_CORE_LIB'],
   'enkiTSBuild': ['-DENKITS_BUILD_DLL'],
   'fsIterExtensionExports': [],
}
msvcArgs : Dict[str, Union[str, List[str]]] = {
   # NOTE(cmo): The last three of these disable some of the narrowing/sign
   # compare warnings.  Whilst these might very occasionally be useful, they
   # make too much noise.
   'baseCompileArgs': ['/std:c++17', '/Z7', '/DENKITS_DLL',
                       '/wd4244', '/wd4267', '/wd4018'],
   'coreCompileArgs': [],
   'SSE2Args': [],
   'AVX2FMAArgs': ['/arch:AVX2'],
   'AVX512Args': ['/arch:AVX512'],
   'libs': ['libenkiTS'],
   'libDirs': [path.join(BuildDir, 'lightweaver')],
   'linkArgs': ['/DEBUG:FULL'],
   'stubDefinePrefix': '/DLW_MODULE_STUB_NAME=',
   'lwCoreDefine': ['/DLW_CORE_LIB'],
   'enkiTSBuild': ['/DENKITS_BUILD_DLL'],
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
if sys.platform == 'win32':
    stubSource.append('WindowsExtensionStub.cpp')
stubSource = prepend_source_dir(stubSource)

if sys.platform == 'win32':
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
    lwExts.append(LwSharedLibraryNoExtension('lightweaver.libenkiTS',
                  sources=[path.join('Source', 'TaskScheduler.cpp')] + stubSource,
                  extra_compile_args=args['baseCompileArgs']
                                     + [f'{args["stubDefinePrefix"]}libenkiTS']
                                     + args['enkiTSBuild'],
                  language='c++'))
    lwExts.append(Extension('lightweaver.LwCompiled',
                  sources=[path.join('Source', 'LwMiddleLayer.pyx')] + coreSource,
                  depends=coreDepends,
                  include_dirs=[np.get_include()],
                  language='c++',
                  libraries=args['libs'],
                  library_dirs=args['libDirs'],
                  extra_compile_args=args['baseCompileArgs'] + args['coreCompileArgs']
                                     + args['lwCoreDefine'],
                  extra_link_args=args['linkArgs']))
    lwExts = cythonize(lwExts, language_level=3)
    for simdImpl in SimdImpls:
        lwExts.append(Extension(f'lightweaver.DefaultIterSchemes.SimdImpl_{simdImpl}',
                                sources=[path.join('Source', f'SimdImpl_{simdImpl}.cpp')] +
                                        coreSource + stubSource,
                                depends=simdImplDepends[simdImpl],
                                language='c++',
                                libraries=args['libs'],
                                library_dirs=args['libDirs'],
                                extra_compile_args=args['baseCompileArgs'] +
                                                   args[f'{simdImpl}Args'] +
                                                   [f'{args["stubDefinePrefix"]}SimdImpl_{simdImpl}'],
                                extra_link_args=args['linkArgs'],
                                # NOTE(cmo): There is a bug with
                                # export_symbols affecting its arguments, so we
                                # submit a copy. See setuptools #3058
                                export_symbols=copy(args['fsIterExtensionExports']),
                                optional=True))
    return lwExts

# NOTE(cmo): Delete pre-existing build directory if present, otherwise building
# multiple wheels results in all of the libraries (for different python versions
# being copied into them...)
if CI_BUILD and path.exists(BuildDir) and path.isdir(BuildDir):
    shutil.rmtree(BuildDir)

setup(name='lightweaver',
      setup_requires=['setuptools_scm'],
      use_scm_version=True,
      packages=['lightweaver'],
      install_requires=['numpy<1.22,>=1.19', 'scipy', 'matplotlib', 'numba>=0.55',
                        'parse', 'specutils', 'tqdm', 'weno4', 'pyyaml'],
      author='Chris Osborne',
      author_email='c.osborne.1@research.gla.ac.uk',
      license='MIT',
      url='http://github.com/Goobley/Lightweaver',
      description='Non-LTE Radiative Transfer Framework in Python',
      ext_modules=extension_list(buildArgs),
      cmdclass={'build_ext': LwBuildExt },
      include_package_data=True,
      long_description=readme(),
      long_description_content_type='text/markdown',
      python_requires='>=3.8',
      options={
          'build': {
              'build_lib': BuildDir
          }
      })
