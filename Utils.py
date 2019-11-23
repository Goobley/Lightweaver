import Constants as C

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
