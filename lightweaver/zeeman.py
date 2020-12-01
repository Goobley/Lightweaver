import numpy as np
from typing import Optional, cast, Iterator, TYPE_CHECKING
from fractions import Fraction
from dataclasses import dataclass

if TYPE_CHECKING:
    from .atomic_model import AtomicLine

def fraction_range(start: Fraction, stop: Fraction,
                   step: Fraction=Fraction(1,1)) -> Iterator[Fraction]:
    '''
    Works like range, but with Fractions. Does no checking, so best to make
    sure the range you're asking for is sane and divides down properly.
    '''
    while start < stop:
        yield start
        start += step

@dataclass
class ZeemanComponents:
    '''
    Storage for communicating the Zeeman components between functions, also
    shared with the backend, giving a slightly tighter contract than usual:
    all arrays must be contiguous and alpha must be of dtype np.int32.
    '''
    alpha: np.ndarray
    strength: np.ndarray
    shift: np.ndarray

def zeeman_strength(Ju: Fraction, Mu: Fraction, Jl: Fraction, Ml: Fraction) -> float:
    '''
    Computes the strength of a Zeeman component, following del Toro Iniesta
    (p. 137) albeit larger by a factor of 2 which is corrected by
    normalisation.
    Takes J upper and lower (u and l respectively), and M upper and lower.
    '''
    alpha  = int(Ml - Mu)
    dJ = int(Ju - Jl)

    # These parameters are x2 those in del Toro Iniesta (p. 137), but we normalise after the fact, so it's fine

    if dJ == 0: # jMin = ju = jl
        if alpha == 0: # pi trainsitions
            s = 2.0 * Mu**2
        elif alpha == -1: # sigma_b transitions
            s = (Ju + Mu) * (Ju - Mu + 1.0)
        elif alpha == 1: # sigma_r transitions
            s = (Ju - Mu) * (Ju + Mu + 1.0)
    elif dJ == 1: # jMin = jl, Mi = Ml
        if alpha == 0: # pi trainsitions
            s = 2.0 * ((Jl + 1)**2 - Ml**2)
        elif alpha == -1: # sigma_b transitions
            s = (Jl + Ml + 1) * (Jl + Ml + 2.0)
        elif alpha == 1: # sigma_r transitions
            s = (Jl - Ml + 1.0) * (Jl - Ml + 2.0)
    elif dJ == -1: # jMin = ju, Mi = Mu
        if alpha == 0: # pi trainsitions
            s = 2.0 * ((Ju + 1)**2 - Mu**2)
        elif alpha == -1: # sigma_b transitions
            s = (Ju - Mu + 1) * (Ju - Mu + 2.0)
        elif alpha == 1: # sigma_r transitions
            s = (Ju + Mu + 1.0) * (Ju + Mu + 2.0)
    else:
        raise ValueError('Invalid dJ: %d' % dJ)

    return float(s)

def lande_factor(J: Fraction, L: int, S: Fraction) -> float:
    '''
    Computes the Lande g-factor for an atomic level from the J, L, and S
    quantum numbers.
    '''
    if J == 0.0:
        return 0.0
    return float(1.5 + (S * (S + 1.0) - L * (L + 1)) / (2.0 * J * (J + 1.0)))

def effective_lande(line: 'AtomicLine'):
    '''
    Computes the effective Lande g-factor for an atomic line.
    '''
    if line.gLandeEff is not None:
        return line.gLandeEff

    i = line.iLevel
    j = line.jLevel
    if any(x is None for x in [i.J, i.L, i.S, j.J, j.L, j.S]):
        raise ValueError('Cannot compute gLandeEff as gLandeEff not set and some of J, L and S None for line %s'%repr(line))
    gL = lande_factor(i.J, i.L, i.S) # type: ignore
    gU = lande_factor(j.J, j.L, j.S) # type: ignore

    return 0.5 * (gU + gL) + \
           0.25 * (gU - gL) * (j.J * (j.J + 1.0) - i.J * (i.J + 1.0)) # type: ignore

def compute_zeeman_components(line: 'AtomicLine') -> Optional[ZeemanComponents]:
    '''
    Computes, if possible, the set of Zeeman components for an atomic line.

    If gLandeEff is specified on the line, then basic three-component Zeeman
    splitting will be computed directly.
    Otherwise, if both the lower and upper levels of the line support
    LS-coupling (i.e. J, L, and S all specified, and J <= L + S), then the
    LS-coupling formalism is applied to compute the components of "anomalous"
    Zeeman splitting.
    If neither of these cases are fulfilled, then None is returned.

    Parameters
    ----------
    line : AtomicLine
        The line to attempt to compute the Zeeman components from.

    Returns
    -------
    components : ZeemanComponents or None
        The Zeeman splitting components, if possible.
    '''
    # NOTE(cmo): Just do basic three-component Zeeman splitting if an effective
    # Lande g-factor is specified on the line.
    if line.gLandeEff is not None:
        alpha = np.array([-1, 0, 1], dtype=np.int32)
        strength = np.ones(3)
        shift = alpha * line.gLandeEff
        return ZeemanComponents(alpha, strength, shift)

    # NOTE(cmo): Do LS coupling ("anomalous" Zeeman splitting)
    if line.iLevel.lsCoupling and line.jLevel.lsCoupling:
        # Mypy... you're a pain sometimes... (even if you are technically correct)
        Jl = cast(Fraction, line.iLevel.J)
        Ll = cast(int, line.iLevel.L)
        Sl = cast(Fraction, line.iLevel.S)
        Ju = cast(Fraction, line.jLevel.J)
        Lu = cast(int, line.jLevel.L)
        Su = cast(Fraction, line.jLevel.S)

        gLl = lande_factor(Jl, Ll, Sl)
        gLu = lande_factor(Ju, Lu, Su)
        alpha = []
        strength = []
        shift = []
        norm = np.zeros(3)

        for ml in fraction_range(-Jl, Jl+1):
            for mu in fraction_range(-Ju, Ju+1):
                if abs(ml - mu) <= 1.0:
                    alpha.append(int(ml - mu))
                    shift.append(gLl*ml - gLu*mu)
                    strength.append(zeeman_strength(Ju, mu, Jl, ml))
                    norm[alpha[-1]+1] += strength[-1]
        alpha = np.array(alpha, dtype=np.int32)
        strength = np.array(strength)
        shift = np.array(shift)
        strength /= norm[alpha + 1]

        return ZeemanComponents(alpha, strength, shift)
    return None