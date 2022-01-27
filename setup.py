from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration
from Cython.Build import cythonize
import numpy as np
import os
import platform

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
   'SSE2Args': ['-msse4.2'],
   'AVX2FMAArgs': ['-mavx2', '-mfma'],
   'AVX512Args': ['-mavx512f', '-mavx512dq', '-mfma'],
   'libs': ['dl'],
   'linkArgs': None,
}
msvcArgs = {
   'baseCompileArgs': ['/std:c++17', '/Z7'],
   'SSE2Args': [],
   'AVX2FMAArgs': ['/arch:AVX2'],
   'AVX512Args': ['/arch:AVX512'],
   'libs': None,
   'linkArgs': ['DEBUG:FULL'],
}

if platform.system() == 'Windows':
    args = msvcArgs
else:
    args = posixArgs

SimdImpls = ['SSE2', 'AVX2FMA', 'AVX512']
for simd in SimdImpls:
    if f'LW_NO_{simd}_LIB' in os.environ:
        SimdImpls.remove(simd)

def make_config(args):
    lwConf = Configuration('lightweaver')
    lwConf.add_installed_library('LightweaverCore',
                                 sources=['Source/LightweaverAmalgamated.cpp'],
                                 install_dir='lightweaver',
                                 build_info={
                                     'extra_compiler_args': args['baseCompileArgs']
                                 })
    lwConf.add_extension('LwCompiled',
                         sources=['Source/LwMiddleLayer.pyx'],
                         include_dirs=[np.get_include()],
                         language='c++',
                         libraries=['LightweaverCore'],
                         extra_compile_args=args['baseCompileArgs'],
                         extra_link_args=args['linkArgs'])
    for simdImpl in SimdImpls:
        lwConf.add_extension(f'DefaultIterSchemes.SimdImpl_{simdImpl}',
                            sources=[f'Source/SimdImpl_{simdImpl}.cpp'],
                            libraries=['LightweaverCore'],
                            language='c++',
                            extra_compile_args=args['baseCompileArgs'] +
                                                args[f'{simdImpl}Args'],
                            extra_link_args=args['linkArgs'])
    lwConf.ext_modules = cythonize(lwConf.ext_modules, language_level=3)
    return lwConf

setup(setup_requires=['setuptools_scm'],
      use_scm_version=True,
      install_requires=['numpy<1.22,>=1.19', 'scipy', 'matplotlib', 'numba>=0.55',
                        'parse', 'specutils', 'tqdm', 'weno4', 'adjustText', 'pyyaml'],
      author='Chris Osborne',
      author_email='c.osborne.1@research.gla.ac.uk',
      license='MIT',
      url='http://github.com/Goobley/Lightweaver',
      description='Non-LTE Radiative Transfer Framework in Python',
      include_package_data=True,
      long_description=readme(),
      long_description_content_type='text/markdown',
      python_requires='>=3.8',
      **make_config(args).todict())