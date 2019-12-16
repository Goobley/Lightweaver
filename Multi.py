import Constants as C
from copy import copy, deepcopy
import numpy as np
import re
from typing import Tuple
from dataclasses import dataclass
from Atmosphere import Atmosphere, ScaleType

@dataclass
class MultiMetadata:
    name: str
    logG: float

def read_multi_atmos(filename: str) -> Tuple[MultiMetadata, Atmosphere]:
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        raise ValueError('Atmosphere file not found (%s)' % filename)

    def get_line(commentPattern='^\s*\*'):
        while len(lines) > 0:
            line = lines.pop(0)
            if not re.match(commentPattern, line):
                return line.strip()
        return None

    atmosName = get_line()

    scaleStr = get_line()
    logG = float(get_line()) - 2 # For conversion to log[m.s^-2]
    Nspace = int(get_line())

    dscale = np.zeros(Nspace)
    temp = np.zeros(Nspace)
    ne = np.zeros(Nspace)
    vlos = np.zeros(Nspace)
    vturb = np.zeros(Nspace)
    for k in range(Nspace):
        vals = get_line().split()
        vals = [float(v) for v in vals]
        dscale[k] = vals[0]
        temp[k] = vals[1]
        ne[k] = vals[2]
        vlos[k] = vals[3]
        vturb[k] = vals[4]

    scaleMode = scaleStr[0].upper()
    if scaleMode == 'M':
        scaleType = ScaleType.ColumnMass
        dscale = 10**dscale * (C.G_TO_KG / C.CM_TO_M**2)
    elif scaleMode == 'T':
        scaleType = ScaleType.Tau500
        dscale = 10**dscale
    elif scaleMode == 'H':
        scaleType = ScaleType.Geometric
        dscale *= C.KM_TO_M
    else:
        raise ValueError('Unknown scale type: %s (expected M, T, or H)' % scaleStr)

    vlos *= C.KM_TO_M
    vturb *= C.KM_TO_M
    ne /= C.CM_TO_M**3

    if len(lines) <= Nspace:
        raise ValueError('Hydrogen populations not supplied!')

    hPops = np.zeros((6, Nspace))
    for k in range(Nspace):
        vals = get_line().split()
        vals = [float(v) for v in vals]
        hPops[:, k] = vals

    hPops /= C.CM_TO_M**3

    meta = MultiMetadata(atmosName, logG)
    atmos = Atmosphere(scale=scaleType,
                       depthScale=dscale,
                       temperature=temp,
                       vlos=vlos,
                       vturb=vturb,
                       ne=ne,
                       hydrogenPops=hPops)

    return (meta, atmos)

    


    



