from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np
import os
import os.path as path
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
   'linkArgs': [],
   'stubDefinePrefix': '-DLW_MODULE_STUB_NAME=',
   'fsIterExtensionExports': [],
}
msvcArgs = {
   'baseCompileArgs': ['/std:c++17', '/Z7'],
   'SSE2Args': [],
   'AVX2FMAArgs': ['/arch:AVX2'],
   'AVX512Args': ['/arch:AVX512'],
   'libs': None,
   'linkArgs': ['/DEBUG:FULL'],
   'stubDefinePrefix': '/DLW_MODULE_STUB_NAME=',
   'fsIterExtensionExports': ['/EXPORT:fs_iteration_fns_provider'],
}

coreSource = [path.join('Source', 'LightweaverAmalgamated.cpp')]
stubSource = []
if platform.system() == 'Windows':
    stubSource.append(path.join('Source', 'WindowsExtensionStub.cpp'))

if platform.system() == 'Windows':
    buildArgs = msvcArgs
else:
    buildArgs = posixArgs

SimdImpls = ['SSE2', 'AVX2FMA', 'AVX512']
for simd in SimdImpls:
    if f'LW_NO_{simd}_LIB' in os.environ:
        SimdImpls.remove(simd)

def extension_list(args):
    lwExts = []
    lwExts.append(Extension('lightweaver.LwCompiled',
                  sources=[path.join('Source', 'LwMiddleLayer.pyx')] + coreSource,
                  include_dirs=[np.get_include()],
                  language='c++',
                  extra_compile_args=args['baseCompileArgs'],
                  extra_link_args=args['linkArgs']))
    for simdImpl in SimdImpls:
        lwExts.append(Extension(f'lightweaver.DefaultIterSchemes.SimdImpl_{simdImpl}',
                                sources=[path.join('Source', f'SimdImpl_{simdImpl}.cpp')] +
                                        coreSource + stubSource,
                                language='c++',
                                extra_compile_args=args['baseCompileArgs'] +
                                                args[f'{simdImpl}Args'] +
                                                [f'{args["stubDefinePrefix"]}SimdImpl_{simdImpl}'],
                                extra_link_args=args['linkArgs'] + args['fsIterExtensionExports'],
                                optional=True))
    lwExts = cythonize(lwExts, language_level=3)
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