import Constants as C
from copy import copy, deepcopy

def gaunt_bf(wvl, nEff, charge) -> float:
    # /* --- M. J. Seaton (1960), Rep. Prog. Phys. 23, 313 -- ----------- */
    # Copied from RH, ensuring vectorisation support 
    x = C.HC / (wvl * C.NM_TO_M) / (C.E_RYDBERG * charge**2)
    x3 = x**(1.0/3.0)
    nsqx = 1.0 / (nEff**2 *x)

    return 1.0 + 0.1728 * x3 * (1.0 - 2.0 * nsqx) - 0.0496 * x3**2 \
            * (1.0 - (1.0 - nsqx) * (2.0 / 3.0) * nsqx)

class ConvergenceError(Exception):
    pass

# def copy_state_dict_preserve_atoms(state):
#     from AtomicModel import *
#     from AtomicSet import SpeciesStateTable, RadiativeSet

#     memo = {}
#     s = copy(state)
#     s['arguments'] = copy(s['arguments'])
#     args = s['arguments']
#     args['atmos'] = deepcopy(args['atmos'], memo)
#     spect = copy(args['spect'])
#     memo[id(args['spect'])] = spect
#     args['spect'] = spect
#     memo[id(spect.radSet)] = spect.radSet
#     memo[id(spect.wavelength)] = spect.wavelength
#     memo[id(spect.models)] = spect.models
