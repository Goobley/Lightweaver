from lightweaver.atomic_model import *
from lightweaver.collisional_rates import *
from lightweaver.broadening import *
from lightweaver.atomic_table import PeriodicTable
from lightweaver.barklem import BarklemCrossSectionError
from typing import List
from parse import parse
import os
from dataclasses import dataclass

import re
from fractions import Fraction
import colorama
from colorama import Fore, Style

# https://stackoverflow.com/a/3303361
def clean(s):
    # Replace '.' with '_'
    s = re.sub('[.]', '_', s)
    # Remove invalid characters
    s = re.sub('[^0-9a-zA-Z_]', '', s)
    # Remove leading characters until we find a letter or underscore
    s = re.sub('^[^a-zA-Z_]+', '', s)
    return s

@dataclass
class PrincipalQuantum:
    J: Fraction
    L: int
    S: Fraction

class CompositeLevelError(Exception):
    pass

def get_oribital_number(orbit: str) -> int:
    orbits = ['S', 'P', 'D', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'Q', 'R', 'T', 'U', 'V', 'W', 'X']
    return orbits.index(orbit)


def determinate(level: AtomicLevel) -> PrincipalQuantum:
    endIdx = [level.label.upper().rfind(x) for x in ['E', 'O']]
    maxIdx = max(endIdx)
    if maxIdx == -1:
        raise ValueError("Unable to determine parity of level %s" % (repr(level)))
    label = level.label[:maxIdx+1].upper()
    words: List[str] = label.split()

    # _, multiplicity, orbit = parse('{}{:d}{!s}', words[-1])
    match = re.match('[\S-]*(\d)(\S)[EO]$', words[-1])
    if match is None:
        raise ValueError('Unable to parse level label: %s' % level.label)
    else:
        multiplicity = int(match.group(1))
        orbit = match.group(2)
    S = Fraction(int(multiplicity - 1), 2)
    L = get_oribital_number(orbit)
    J = Fraction(int(level.g - 1.0), 2)

    # if J > L + S:
    #     raise CompositeLevelError('J (%f) > L (%d) + S (%f): %s' %(J, L, S, repr(level)))

    return PrincipalQuantum(J=J, L=L, S=S)

def check_barklem_compatible(vals: List[float],
                             iLev: AtomicLevel, jLev: AtomicLevel) -> bool:

    if vals[0] >= 20.0:
        return True

    if iLev.stage > 0:
        return False

    lowerNum = iLev.L
    upperNum = jLev.L
    if upperNum is None or lowerNum is None:
        return False

    if not ((abs(upperNum - lowerNum) == 1)
            and (max(upperNum, lowerNum) <= 3)):
        return False

    # NOTE(cmo): We're not checking the table bounds here, but that should be fine.

    return True

def getNextLine(data):
    if len(data) == 0:
        return None
    for i, d in enumerate(data):
        if d.strip().startswith('#') or d.strip() == '':
            # print('Skipping %s' % d)
            continue
        # print('Accepting %s' % d)
        break
    d = data[i]
    if i == len(data) - 1:
        data[:] = []
        return d.strip()
    data[:] = data[i+1:]
    return d.strip()

def maybe_int(s):
    try:
        v = int(s)
    except:
        v = None
    return v

def conv_atom(inFile):
    with open(inFile, 'r') as fi:
        data = fi.readlines()

    ID = getNextLine(data)
    element = PeriodicTable[ID]
    print(Fore.GREEN + '='*40 + '\n' + 'Reading model atom %s from file %s' % (ID, inFile) + Style.RESET_ALL)
    Ns = [maybe_int(d) for d in getNextLine(data).split()]
    Nlevel = Ns[0]
    Nline = Ns[1]
    Ncont = Ns[2]
    Nfixed = Ns[3]

    if Nfixed != 0:
        raise ValueError("Fixed transitions are not supported")

    levels = []
    # levelNos: List[int] = []
    for n in range(Nlevel):
        line = getNextLine(data)

        res = parse('{:f}{}{:f}{}\'{}\'{}{:d}{}{:d}', line.strip())
        # print(res)
        # print(line)
        E = res[0]
        g = res[2]
        label = res[4].strip()
        stage = res[6]
        # levelNo = int(res[8])
        # if n > 0:
        #     if levelNo < levelNos[-1]:
        #         raise ValueError('Levels are not monotonically increasing (%f < %f)' % (levelNo, levelNos[-1]))
        # levelNos.append(levelNo)
        levels.append(AtomicLevel(E=E, g=g, label=label, stage=stage))
        try:
            qNos = determinate(levels[-1])
            levels[-1].J = qNos.J
            levels[-1].L = qNos.L
            levels[-1].S = qNos.S
        except Exception as e:
            print(Fore.BLUE + 'Unable to determine quantum numbers for %s' % repr(levels[-1]))
            print('\t %s' % (repr(e)) + Style.RESET_ALL)


    lines = []
    lineNLambdas = []
    for n in range(Nline):
        line = getNextLine(data)
        line = line.split()

        j = int(line[0])
        i = int(line[1])
        f = float(line[2])
        typ = line[3]
        Nlambda = int(line[4])
        sym = line[5]
        qCore = float(line[6])
        qWing = float(line[7])
        vdw = line[8]
        vdwParams = [float(x) for x in line[9:13]]
        gRad = float(line[13])
        stark = float(line[14])
        if len(line) > 15:
            gLande = float(line[15])
        else:
            gLande = None

        if typ.upper() == 'PRD':
            lineType = LineType.PRD
        elif typ.upper() == 'VOIGT':
            lineType = LineType.CRD
        else:
            raise ValueError('Only PRD and VOIGT lines are supported, found type %s' % typ)

        # if sym.upper() != 'ASYMM':
        #     print('Only Asymmetric lines are supported, doubling Nlambda')
        #     Nlambda *= 2

        if vdw.upper() == 'PARAMTR':
            vdwApprox: VdwApprox = VdwRidderRensbergen(vdwParams)
        elif vdw.upper() == 'UNSOLD':
            vdwParams = [vdwParams[0], vdwParams[2]]
            vdwApprox = VdwUnsold(vdwParams)
        elif vdw.upper() == 'BARKLEM':
            vdwParams = [vdwParams[0], vdwParams[2]]
            if check_barklem_compatible(vdwParams, levels[i], levels[j]):
                vdwApprox = VdwBarklem(vdwParams)
            else:
                vdwApprox = VdwUnsold(vdwParams)
        else:
            raise ValueError('Unknown vdw type %s' % vdw)

        if stark <= 0:
            starkBroaden = MultiplicativeStarkBroadening(abs(stark))
        else:
            starkBroaden = QuadraticStarkBroadening(stark)

        broadening = LineBroadening(natural=[RadiativeBroadening(gRad)], elastic=[vdwApprox, starkBroaden])
        if element == PeriodicTable[1]:
            broadening.elastic.append(HydrogenLinearStarkBroadening())

        quadrature = LinearCoreExpWings(qCore=qCore, qWing=qWing, Nlambda=Nlambda)
        lines.append(VoigtLine(j=j, i=i, f=f, type=lineType, quadrature=quadrature, broadening=broadening, gLandeEff=gLande))
        lineNLambdas.append(Nlambda)


    continua: List[AtomicContinuum] = []
    for n in range(Ncont):
        line = getNextLine(data)
        line = line.split()
        j = int(line[0])
        i = int(line[1])
        alpha0 = float(line[2])
        Nlambda = int(line[3])
        wavelengthDep = line[4]
        minLambda = float(line[5])


        if wavelengthDep.upper() == 'EXPLICIT':
            wavelengths = []
            alphas = []
            for _ in range(Nlambda):
                l = getNextLine(data)
                l = l.split()
                wavelengths.append(float(l[0]))
                alphas.append(float(l[1]))
            wavelengthGrid = wavelengths[::-1]
            alphaGrid = alphas[::-1]
            continua.append(ExplicitContinuum(j=j, i=i, wavelengthGrid=wavelengthGrid, alphaGrid=alphaGrid))
        elif wavelengthDep.upper() == 'HYDROGENIC':
            continua.append(HydrogenicContinuum(j=j, i=i, alpha0=alpha0, minWavelength=minLambda, NlambdaGen=Nlambda))
        else:
            raise ValueError('Unknown Continuum type %s' % wavelengthDep)

    collisions: List[CollisionalRates] = []
    while True:
        line = getNextLine(data)
        if line == 'END' or line is None:
            break

        line = line.split()
        if line[0].upper() == 'TEMP':
            Ntemp = int(line[1])
            temperatureGrid = []
            for i in range(Ntemp):
                temperatureGrid.append(float(line[i+2]))
        elif line[0].upper() == 'OMEGA':
            i1 = int(line[1])
            i2 = int(line[2])
            j = max(i1, i2)
            i = min(i1, i2)
            rates = []
            for nt in range(Ntemp):
                rates.append(float(line[nt+3]))
            collisions.append(Omega(j=j, i=i, temperature=temperatureGrid, rates=rates))
        elif line[0].upper() == 'CI':
            i1 = int(line[1])
            i2 = int(line[2])
            j = max(i1, i2)
            i = min(i1, i2)
            rates = []
            for nt in range(Ntemp):
                rates.append(float(line[nt+3]))
            collisions.append(CI(j=j, i=i, temperature=temperatureGrid, rates=rates))
        elif line[0].upper() == 'CE':
            i1 = int(line[1])
            i2 = int(line[2])
            j = max(i1, i2)
            i = min(i1, i2)
            rates = []
            for nt in range(Ntemp):
                rates.append(float(line[nt+3]))
            collisions.append(CE(j=j, i=i, temperature=temperatureGrid, rates=rates))
        elif line[0].upper() == 'CP':
            i1 = int(line[1])
            i2 = int(line[2])
            j = max(i1, i2)
            i = min(i1, i2)
            rates = []
            for nt in range(Ntemp):
                rates.append(float(line[nt+3]))
            collisions.append(CP(j=j, i=i, temperature=temperatureGrid, rates=rates))
        elif line[0].upper() == 'CH':
            i1 = int(line[1])
            i2 = int(line[2])
            j = max(i1, i2)
            i = min(i1, i2)
            rates = []
            for nt in range(Ntemp):
                rates.append(float(line[nt+3]))
            collisions.append(CH(j=j, i=i, temperature=temperatureGrid, rates=rates))
        elif line[0].upper() == 'CH0':
            i1 = int(line[1])
            i2 = int(line[2])
            j = max(i1, i2)
            i = min(i1, i2)
            rates = []
            for nt in range(Ntemp):
                rates.append(float(line[nt+3]))
            collisions.append(ChargeExchangeNeutralH(j=j, i=i, temperature=temperatureGrid, rates=rates))
        elif line[0].upper() == 'CH+':
            i1 = int(line[1])
            i2 = int(line[2])
            j = max(i1, i2)
            i = min(i1, i2)
            rates = []
            for nt in range(Ntemp):
                rates.append(float(line[nt+3]))
            collisions.append(ChargeExchangeProton(j=j, i=i, temperature=temperatureGrid, rates=rates))
        elif line[0].upper() == 'AR85-CDI':
            i1 = int(line[1])
            i2 = int(line[2])
            Nrow = int(line[3])
            Mshell = 5
            j = max(i1, i2)
            i = min(i1, i2)
            cdi = []
            for n in range(Nrow):
                line = getNextLine(data)
                line = line.split()
                cdi.append([])
                for m in range(Mshell):
                    cdi[n].append(float(line[m]))
            collisions.append(Ar85Cdi(j=j, i=i, cdi=cdi))
        elif line[0].upper() == 'BURGESS':
            i1 = int(line[1])
            i2 = int(line[2])
            j = max(i1, i2)
            i = min(i1, i2)
            fudge = float(line[3])
            collisions.append(Burgess(j=j, i=i, fudge=fudge))
        else:
            print(Fore.YELLOW + "Ignoring unknown collisional string %s" % line[0].upper() + Style.RESET_ALL)

    atom = AtomicModel(element=element, levels=levels, lines=lines, continua=continua, collisions=collisions)

    # for i, l in enumerate(atom.lines):
    #     l.Nlambda = lineNLambdas[i]
    return repr(atom)

colorama.init()
fails = open('Fails.txt', 'w')
path = './Atoms/'
excludeFiles = ['FeII_big.atom', 'He_9_incorrect_translation.atom',
                'C_I+II_9.atom', 'LiI.atom']
baseFiles = sorted([f for f in os.listdir(path) if f.endswith('.atom') and f not in excludeFiles])
# baseFiles = ['He_9.atom']
files = [path+f for f in baseFiles]
atoms = []
doneFiles = []
for i, f in enumerate(files):
    try:
        atoms.append(conv_atom(f))
        doneFiles.append(baseFiles[i])
    except Exception as e:
        print(Fore.RED +  'Failed: ' + Style.RESET_ALL, f)
        print(Fore.BLUE + '->' + repr(e) + Style.RESET_ALL)
        print('-'*40)
        fails.write('Failed: %s\n' % f)
        fails.write('->%s\n' % repr(e))
        fails.write('-'*40 + '\n')

with open('rh_atoms.py', 'w') as fi:
    fi.write('from lightweaver.atomic_model import *\n')
    fi.write('from lightweaver.collisional_rates import *\n')
    fi.write('from lightweaver.broadening import *\n')
    fi.write('from lightweaver.atomic_table import Element\n')
    for i, a in enumerate(atoms):
        s = clean(doneFiles[i]) + ' = lambda: \\\n'
        s += a
        s += '\n'
        fi.write(s)
