from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

# extensions = [
#     Extension("Elements", ["Elements.pyx"],
#         include_dirs=[np.get_include()],
#         libraries=['Abund'],
#         library_dirs=['.'])]
#     # Everything but primes.pyx is included here.
#     # Extension("*", ["*.pyx"],
#         # include_dirs=[...],
#         # libraries=[...],
#         # library_dirs=[...]),

# setup(
#     name="Elements",
#     ext_modules=cythonize(extensions),
# )

extensions = [
    Extension("CAtomicTable", ["CAtomicTable.pyx"],
        include_dirs=[np.get_include()],
        libraries=[],
        library_dirs=[]),
    Extension("ComputationalStructures", ["ComputationalStructures.pyx"],
        include_dirs=[np.get_include()],
        libraries=[],
        library_dirs=[])]
    # Everything but primes.pyx is included here.
    # Extension("*", ["*.pyx"],
        # include_dirs=[...],
        # libraries=[...],
        # library_dirs=[...]),

setup(
    name="CAtomicTable",
    ext_modules=cythonize(extensions),
)
