from AtomicModel import *
from parse import parse
inFile = '../Atoms/H_6.atom'

with open(inFile, 'r') as fi:
    data = fi.readlines()

def getNextLine(data):
    if len(data) == 0:
        return None
    for i, d in enumerate(data):
        if d.strip().startswith('#') or d.strip() == '':
            print('Skipping %s' % d)
            continue
        print('Accepting %s' % d)
        break
    d = data[i]
    if i == len(data) - 1:
        data[:] = []
        return d.strip()
    data[:] = data[i+1:]
    return d.strip()

ID = getNextLine(data)
Nlevel, Nline, Ncont, Nfixed = [int(d) for d in getNextLine(data).split()]

if Nfixed != 0:
    raise ValueError("Fixed transitions are not supported")

levels = []
levelNos: List[int] = []
for n in range(Nlevel):
    line = getNextLine(data)

    res = parse('{:f}{}{:f}{}\'{}\'{}{:d}{}{:d}', line.strip())
    print(res)
    print(line)
    E = res[0]
    g = res[2]
    label = res[4].strip()
    stage = res[6]
    levelNo = int(res[8])
    if n > 0:
        if levelNo < levelNos[-1]:
            raise ValueError('Levels are not monotonically increasing')
    levelNos.append(levelNo)
    levels.append(AtomicLevel(E=E, g=g, label=label, stage=stage))



lines = []
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
        gLande = 0.0

    if typ.upper() == 'PRD':
        lineType = LineType.PRD
    elif typ.upper() == 'VOIGT':
        lineType = LineType.CRD
    else:
        raise ValueError('Only PRD and VOIGT lines are supported, found type %s' % typ)
    
    if sym.upper() != 'ASYMM':
        print('Only Asymmetric lines are supported, doubling Nlambda')
        Nlambda *= 2

    if vdw.upper() == 'PARAMTR':
        vdwApprox: VdwApprox = VdwRidderRensbergen(vdwParams)
    elif vdw.upper() == 'UNSOLD':
        vdwParams = [vdwParams[0], vdwParams[2]]
        vdwApprox = VdwUnsold(vdwParams)
    elif vdw.upper() == 'BARKLEM':
        vdwParams = [vdwParams[0], vdwParams[2]]
        vdwApprox = VdwBarklem(vdwParams)
    else:
        raise ValueError('Unknown vdw type %s' % vdw)

    lines.append(VoigtLine(j=j, i=i, f=f, type=lineType, Nlambda=Nlambda, qCore=qCore, qWing=qWing, vdw=vdwApprox, gRad=gRad, stark=stark, gLandeEff=gLande))

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
        alphaGrid = [list(x) for x in list(zip(wavelengths, alphas))]
        continua.append(ExplicitContinuum(j=j, i=i, alphaGrid=alphaGrid))
    elif wavelengthDep.upper() == 'HYDROGENIC':
        continua.append(HydrogenicContinuum(j=j, i=i, alpha0=alpha0, minLambda=minLambda, Nlambda=Nlambda))
    else:
        raise ValueError('Unknown Continuum type %s' % wavelengthDep)

collisions: List[CollisionalRates] = []
while True:
    line = getNextLine(data)
    if line == 'END':
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

print(levels)
print(lines)
print(continua)
print(collisions)

atom = AtomicModel(name=ID, levels=levels, lines=lines, continua=continua, collisions=collisions)

