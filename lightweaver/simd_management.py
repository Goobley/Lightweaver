from os import path
from typing import List

from numpy.core._multiarray_umath import __cpu_features__

# NOTE(cmo): These are added in reverse order of preference (due to width), i.e.
# try to use the key furthest down the list.
LwSimdImplsAndFlags = {
    'SSE2': ['SSE2'],
    'AVX2FMA': ['AVX2', 'FMA3'],
    'AVX512': ['AVX512F', 'AVX512DQ']
}

def get_available_simd_suffixes() -> List[str]:
    '''
    Verifies the necessary flags against the features NumPy indicates are
    available, and returns a list of available LightweaverSimdImpls
    '''
    validExts = []
    for impl, flags in LwSimdImplsAndFlags.items():
        if all(__cpu_features__[flag] for flag in flags):
            validExts.append(impl)
    return validExts

def filter_usable_simd_impls(implLibs: List[str]) -> List[str]:
    '''
    Filter a list of SimdImpl library names, returning those that are usable on
    the current machine.
    '''
    usableImpls = get_available_simd_suffixes()
    result = []
    for lib in implLibs:
        _, libName = path.split(lib)
        # NOTE(cmo): A lib name is expected to be of the form
        # SimdImpl_{SimdType}.{pep3149}.so. So we split at the underscore and
        # check what the name starts with.
        nameEnd = libName.split('_')[1]
        for simdType in usableImpls:
            if nameEnd.startswith(simdType):
                result.append(lib)
                break
    return result
