# Parsing abundance data from LaTeX tables found in Asplund et al 2009, from
# source downloaded from ArXiv 0909.0948

from dataclasses import dataclass, asdict
from typing import List, Iterable
import pickle
from itertools import chain
flatten_list = chain.from_iterable

def expected_mass_from_str(s: str):
    # NOTE(cmo): Handle [x,y] range by returning mean
    if s.startswith('[') and ',' in s:
        vals = tuple(map(float, s[1:-1].split(',')))
        return 0.5 * (vals[0] + vals[1])

    # NOTE(cmo): Handle [x] by returning x
    if s.startswith('['):
        return float(s[1:-1])

    # NOTE(cmo): Handle x.....(y) by returning x
    if (idx := s.find('(')) != -1:
        return float(s[:idx])

    # NOTE(cmo): if it's none of these then try to convert to float
    return float(s)

def parse_nist_mass_blocks(linesIn):
    lines = linesIn
    lines = [l.replace('\n', '') for l in lines]

    Z = 0
    massData = {}
    nameMapping = {}
    while len(lines) > 7:
        while (line := lines[0]) == '' or line.startswith('#'):
            lines = lines[1:]

        Z = int(lines[0].split('=')[1])
        if Z > 92:
            break
        name = lines[1].split('=')[1].strip()
        # NOTE(cmo): Ignore the special case names for Deuterium and Tritium for now
        name = 'H' if (name == 'D' or name == 'T') else name
        N = int(lines[2].split('=')[1])
        isoMass = expected_mass_from_str(lines[3].split('=')[1].strip())
        aMass = expected_mass_from_str(lines[5].split('=')[1].strip())

        lines = lines[7:]

        if Z not in massData:
            massData[Z] = aMass
            nameMapping[Z] = name
            nameMapping[name] = Z

        massData[(N, Z)] = isoMass
        nameMapping[(N, Z)] = name
    return massData, nameMapping

# NOTE(cmo): Parse all the masses from the NIST data
# File isn't a great format.
with open('NistMasses.txt', 'r') as f:
    lines = f.readlines()

massData, nameMapping = parse_nist_mass_blocks(lines)
with open('AtomicMassesNames.pickle', 'wb') as pkl:
    pickle.dump((massData, nameMapping), pkl)

@dataclass
class Element:
    Z: int
    name: str
    mass: float

    def __lt__(self, other):
        return self.Z < other.Z

@dataclass
class ElementalAbundance:
    elem: Element
    abundance: float
    solarData: bool

    @property
    def Z(self):
        return self.elem.Z

    @property
    def name(self):
        return self.elem.name

    def __lt__(self, other):
        return self.Z < other.Z

@dataclass
class IsotopeProportion:
    N: int
    mass: float
    proportion: float

@dataclass
class ElementalDistribution:
    elem: ElementalAbundance
    isotopes: List[IsotopeProportion]

    @property
    def Z(self):
        return self.elem.Z

    @property
    def name(self):
        return self.elem.name

    def __lt__(self, other):
        return self.Z < other.Z

abundanceStr = """\
1  & H   & $12.00$            &  $8.22 \pm 0.04$        & 44 & Ru  &  $1.75 \pm 0.08$   &  $1.76 \pm 0.03$  \\
2  & He  & $[10.93 \pm 0.01]$ &  $1.29$                 & 45 & Rh  &  $0.91 \pm 0.10$   &  $1.06 \pm 0.04$  \\
3  & Li  &  $1.05 \pm 0.10$   &  $3.26 \pm 0.05$        & 46 & Pd  &  $1.57 \pm 0.10$   &  $1.65 \pm 0.02$  \\
4  & Be  &  $1.38 \pm 0.09$   &  $1.30 \pm 0.03$        & 47 & Ag  &  $0.94 \pm 0.10$   &  $1.20 \pm 0.02$  \\
5  & B   &  $2.70 \pm 0.20$   &  $2.79 \pm 0.04$        & 48 & Cd  &                    &  $1.71 \pm 0.03$  \\
6  & C   &  $8.43 \pm 0.05$   &  $7.39 \pm 0.04$        & 49 & In  &  $0.80 \pm 0.20$   &  $0.76 \pm 0.03$  \\
7  & N   &  $7.83 \pm 0.05$   &  $6.26 \pm 0.06$        & 50 & Sn  &  $2.04 \pm 0.10$   &  $2.07 \pm 0.06$  \\
8  & O   &  $8.69 \pm 0.05$   &  $8.40 \pm 0.04$        & 51 & Sb  &  $$                &  $1.01 \pm 0.06$  \\
9  & F   &  $4.56 \pm 0.30$   &  $4.42 \pm 0.06$        & 52 & Te  &                    &  $2.18 \pm 0.03$  \\
10 & Ne  &  $[7.93 \pm 0.10]$ &  $-1.12$                & 53 & I   &                    &  $1.55 \pm 0.08$  \\
11 & Na  &  $6.24 \pm 0.04$   &  $6.27 \pm 0.02$        & 54 & Xe  &  $[2.24 \pm 0.06]$ &  $-1.95$  \\
12 & Mg  &  $7.60 \pm 0.04$   &  $7.53 \pm 0.01$        & 55 & Cs  &                    &  $1.08 \pm 0.02$  \\
13 & Al  &  $6.45 \pm 0.03$   &  $6.43 \pm 0.01$        & 56 & Ba  &  $2.18 \pm 0.09$   &  $2.18 \pm 0.03$  \\
14 & Si  &  $7.51 \pm 0.03$   &  $7.51 \pm 0.01$        & 57 & La  &  $1.10 \pm 0.04$   &  $1.17 \pm 0.02$  \\
15 & P   &  $5.41 \pm 0.03$   &  $5.43 \pm 0.04$        & 58 & Ce  &  $1.58 \pm 0.04$   &  $1.58 \pm 0.02$  \\
16 & S   &  $7.12 \pm 0.03$   &  $7.15 \pm 0.02$        & 59 & Pr  &  $0.72 \pm 0.04$   &  $0.76 \pm 0.03$  \\
17 & Cl  &  $5.50 \pm 0.30$   &  $5.23 \pm 0.06$        & 60 & Nd  &  $1.42 \pm 0.04$   &  $1.45 \pm 0.02$  \\
18 & Ar  &  $[6.40 \pm 0.13]$   &  $-0.50$              & 62 & Sm  &  $0.96 \pm 0.04$   &  $0.94 \pm 0.02$  \\
19 & K   &  $5.03 \pm 0.09$   &  $5.08 \pm 0.02$        & 63 & Eu  &  $0.52 \pm 0.04$   &  $0.51 \pm 0.02$  \\
20 & Ca  &  $6.34 \pm 0.04$   &  $6.29 \pm 0.02$        & 64 & Gd  &  $1.07 \pm 0.04$   &  $1.05 \pm 0.02$  \\
21 & Sc  &  $3.15 \pm 0.04$   &  $3.05 \pm 0.02$        & 65 & Tb  &  $0.30 \pm 0.10$   &  $0.32 \pm 0.03$  \\
22 & Ti  &  $4.95 \pm 0.05$   &  $4.91 \pm 0.03$        & 66 & Dy  &  $1.10 \pm 0.04$   &  $1.13 \pm 0.02$  \\
23 & V   &  $3.93 \pm 0.08$   &  $3.96 \pm 0.02$        & 67 & Ho  &  $0.48 \pm 0.11$   &  $0.47 \pm 0.03$  \\
24 & Cr  &  $5.64 \pm 0.04$   &  $5.64 \pm 0.01$        & 68 & Er  &  $0.92 \pm 0.05$   &  $0.92 \pm 0.02$  \\
25 & Mn  &  $5.43 \pm 0.05$   &  $5.48 \pm 0.01$        & 69 & Tm  &  $0.10 \pm 0.04$   &  $0.12 \pm 0.03$  \\
26 & Fe  &  $7.50 \pm 0.04$   &  $7.45 \pm 0.01$        & 70 & Yb  &  $0.84 \pm 0.11$   &  $0.92 \pm 0.02$  \\
27 & Co  &  $4.99 \pm 0.07$   &  $4.87 \pm 0.01$        & 71 & Lu  &  $0.10 \pm 0.09$   &  $0.09 \pm 0.02$  \\
28 & Ni  &  $6.22 \pm 0.04$   &  $6.20 \pm 0.01$        & 72 & Hf  &  $0.85 \pm 0.04$   &  $0.71 \pm 0.02$  \\
29 & Cu  &  $4.19 \pm 0.04$   &  $4.25 \pm 0.04$        & 73 & Ta  &                    &  -$0.12 \pm 0.04$ \\
30 & Zn  &  $4.56 \pm 0.05$   &  $4.63 \pm 0.04$        & 74 & W   &  $0.85 \pm 0.12$   &  $0.65 \pm 0.04$  \\
31 & Ga  &  $3.04 \pm 0.09$   &  $3.08 \pm 0.02$        & 75 & Re  &                    &  $0.26 \pm 0.04$  \\
32 & Ge  &  $3.65 \pm 0.10$   &  $3.58 \pm 0.04$        & 76 & Os  &  $1.40 \pm 0.08$   &  $1.35 \pm 0.03$  \\
33 & As  &                    &  $2.30 \pm 0.04$        & 77 & Ir  &  $1.38 \pm 0.07$   &  $1.32 \pm 0.02$  \\
34 & Se  &                    &  $3.34 \pm 0.03$        & 78 & Pt  &                    &  $1.62 \pm 0.03$  \\
35 & Br  &                    &  $2.54 \pm 0.06$        & 79 & Au  &  $0.92 \pm 0.10$   &  $0.80 \pm 0.04$  \\
36 & Kr  &  $[3.25 \pm 0.06]$   &  $-2.27$              & 80 & Hg  &                    &  $1.17 \pm 0.08$  \\
37 & Rb  &  $2.52 \pm 0.10$   &  $2.36 \pm 0.03$        & 81 & Tl  &  $0.90 \pm 0.20$   &  $0.77 \pm 0.03$  \\
38 & Sr  &  $2.87 \pm 0.07$   &  $2.88 \pm 0.03$        & 82 & Pb  &  $1.75 \pm 0.10$   &  $2.04 \pm 0.03$  \\
39 & Y   &  $2.21 \pm 0.05$   &  $2.17 \pm 0.04$        & 83 & Bi  &  $$                &  $0.65 \pm 0.04$  \\
40 & Zr  &  $2.58 \pm 0.04$   &  $2.53 \pm 0.04$        & 90 & Th  &  $0.02 \pm 0.10$   &  $0.06 \pm 0.03$  \\
41 & Nb  &  $1.46 \pm 0.04$   &  $1.41 \pm 0.04$        & 92 & U   &                    &  -$0.54 \pm 0.03$  \\
42 & Mo  &  $1.88 \pm 0.08$   &  $1.94 \pm 0.04$        & $$\\
"""

# NOTE(cmo): Parse the abundance table into a spreadsheet of cells
replaceChars = ['$', '[', ']', '\\']
for char in replaceChars:
    abundanceStr = abundanceStr.replace(char, '')

abundanceRows : List[str] = abundanceStr.split('\n')

splitStrs = ['&']
abundanceCells = []
# NOTE(cmo): Also trim off last empty line here with choice of iterable for loop
for r in abundanceRows[:-1]:
    row : Iterable[str] = [r]
    for s in splitStrs:
        row = flatten_list([x.split(s) for x in row])
    row = [x.split('pm')[0].strip() if 'pm' in x else x.strip() for x in row]
    abundanceCells.append(row)

FullRow = len(abundanceCells[0])
elements = []
for row in abundanceCells:
    abund, solar = (float(row[2]), True) if row[2] != '' else (float(row[3]), False)
    Z = int(row[0])
    e = ElementalAbundance(elem=Element(Z=Z, name=row[1], mass=massData[Z]),
                           abundance=abund, solarData=solar)
    elements.append(e)

    if len(row) < FullRow:
        continue

    abund, solar = (float(row[-2]), True) if row[-2] != '' else (float(row[-1]), False)
    Z = int(row[4])
    e = ElementalAbundance(elem=Element(Z=Z, name=row[5], mass=massData[Z]),
                           abundance=abund, solarData=solar)
    elements.append(e)

elements = sorted(elements)

isotopeStr = """\
H  & 1   & $99.998$  & S  & 32 & $94.93$   & Fe & 57 & $2.119$   & Kr & 82  & $11.655$ & Pd & 105 & $22.33$\\
   & 2   & $ 0.002$  &    & 33 & $0.76$    &    & 58 & $0.282$   &    & 83  & $11.546$ &    & 106 & $27.33$\\
   &     &           &    & 34 & $4.29$    &    &    &           &    & 84  & $56.903$ &    & 108 & $26.46$\\
He & 3   & $0.0166$  &    & 36 & $0.02$    & Co & 59 & $100.0$   &    & 86  & $17.208$ &    & 110 & $11.72$\\
   & 4   & $99.9834$ &    &    &           &    &    &           &    &     &          &    &     &        \\
   &     &           & Cl & 35 & $75.78$   & Ni & 58 & $68.0769$ & Rb & 85  & $70.844$ & Ag & 107 & $51.839$\\
Li & 6   & $7.59$    &    & 37 & $24.22$   &    & 60 & $26.2231$ &    & 87  & $29.156$ &    & 109 & $48.161$\\
   & 7   & $92.41$   &    &    &           &    & 61 & $1.1399$  &    &     &          &    &     &        \\
   &     &           & Ar & 36 & $84.5946$ &    & 62 & $3.6345$  & Sr & 84  & $0.5580$ & Cd & 106 & $1.25$\\
Be & 9   & $100.0$   &    & 38 & $15.3808$ &    & 64 & $0.9256$  &    & 86  & $9.8678$ &    & 108 & $0.89$\\
   &     &           &    & 40 & $0.0246$  &    &    &           &    & 87  & $6.8961$ &    & 110 & $12.49$\\
B  & 10  & $19.9$    &    &    &           & Cu & 63 & $69.17$   &    & 88  & $82.6781$&    & 111 & $12.80$\\
   & 11  & $80.1$    & K  & 39 & $93.132$  &    & 65 & $30.83$   &    &     &          &    & 112 & $24.13$\\
   &     &           &    & 40 & $0.147$   &    &    &           & Y  & 89  & $100.0$  &    & 113 & $12.22$\\
C  & 12  & $98.8938$ &    & 41 & $6.721$   & Zn & 64 & $48.63$   &    &     &          &    & 114 & $28.73$\\
   & 13  & $1.1062$  &    &    &           &    & 66 & $27.90$   & Zr & 90  & $51.45$  &    & 116 & $7.49$\\
   &     &           & Ca & 40 & $96.941$  &    & 67 & $4.10$    &    & 91  & $11.22$  &    &     &        \\
N  & 14  & $99.771$  &    & 42 & $0.647$   &    & 68 & $18.75$   &    & 92  & $17.15$  & In & 113 & $4.29$\\
   & 15  & $0.229$   &    & 43 & $0.135$   &    & 70 & $0.62$    &    & 94  & $17.38$  &    & 115 & $95.71$\\
   &     &           &    & 44 & $2.086$   &    &    &           &    & 96  & $2.80$   &    &     &        \\
O  & 16  & $99.7621$ &    & 46 & $0.004$   & Ga & 69 & $60.108$  &    &     &          & Sn & 112 & $0.97$\\
   & 17  & $0.0379$  &    & 48 & $0.187$   &    & 71 & $39.892$  & Nb & 93  & $100.0$  &    & 114 & $0.66$\\
   & 18  & $0.2000$  &    &    &           &    &    &           &    &     &          &    & 115 & $0.34$\\
   &     &           & Sc & 45 & $100.0$   & Ge & 70 & $20.84$   & Mo & 92  & $14.525$ &    & 116 & $14.54$\\
F  & 19  & $100.0$   &    &    &           &    & 72 & $27.54$   &    & 94  & $9.151$  &    & 117 & $7.68$\\
   &     &           & Ti & 46 & $8.25$    &    & 73 & $7.73$    &    & 95  & $15.838$ &    & 118 & $24.22$\\
Ne & 20  & $92.9431$ &    & 47 & $7.44$    &    & 74 & $36.28$   &    & 96  & $16.672$ &    & 119 & $8.59$\\
   & 21  & $0.2228$  &    & 48 & $73.72$   &    & 76 & $7.61$    &    & 97  & $9.599$  &    & 120 & $32.58$\\
   & 22  & $6.8341$  &    & 49 & $5.41$    &    &    &           &    & 98  & $24.391$ &    & 122 & $4.63$\\
   &     &           &    & 50 & $5.18$    & As & 75 & $100.0$   &    & 100 & $9.824$   &    & 124 & $5.79$\\
Na & 23  & $100.0$   &    &    &           &    &    &           &    &     &          &    &     &        \\
   &     &           & V  & 50 & $0.250$   & Se & 74 & $0.89$    & Ru & 96  & $5.54$   & Sb & 121 & $57.21$\\
Mg & 24  & $78.99$   &    & 51 & $99.750$  &    & 76 & $9.37$    &    & 98  & $1.87$   &    & 123 & $42.79$\\
   & 25  & $10.00$   &    &    &           &    & 77 & $7.63$    &    & 99  & $12.76$  &    &     &        \\
   & 26  & $11.01$   & Cr & 50 & $4.345$   &    & 78 & $23.77$   &    & 100 & $12.60$  & Te & 120 & $0.09$\\
   &     &           &    & 52 & $83.789$  &    & 80 & $49.61$   &    & 101 & $17.06$  &    & 122 & $2.55$\\
Al & 27  & $100.0$   &    & 53 & $9.501$   &    & 82 & $8.73$    &    & 102 & $31.55$  &    & 123 & $0.89$\\
   &     &           &    & 54 & $2.365$   &    &    &           &    & 104 & $18.62$  &    & 124 & $4.74$\\
Si & 28  & $92.2297$ &    &    &           & Br & 79 & $50.69$   &    &     &          &    & 125 & $7.07$\\
   & 29  & $4.6832$  & Mn & 55 & $100.0$   &    & 81 & $49.31$   & Rh & 103 & $100.0$  &    & 126 & $18.84$\\
   & 30  & $3.0872$  &    &   &            &    &    &           &    &     &          &    & 128 & $31.74$\\
   &     &           & Fe & 54 & $5.845$   & Kr & 78 & $0.362$   & Pd & 102 & $1.02$   &    & 130 & $34.08$\\
P  & 31  & $100.0$   &    & 56 & $91.754$  &    & 80 & $2.326$   &    & 104 & $11.14$  &    &     &        \\
I  & 127 & $100.0$  & Nd & 142 & $27.044$& Dy & 160 & $2.329$   & Hf & 178 & $27.297$ & Pt & 196 & $25.242$ \\
   &     &          &    & 143 & $12.023$&    & 161 & $18.889$  &    & 179 & $13.629$ &    & 198 & $7.163$  \\
Xe & 124 & $0.122$  &    & 144 & $23.729$&    & 162 & $25.475$  &    & 180 & $35.100$ &    &     &          \\
   & 126 & $0.108$  &    & 145 & $8.763$ &    & 163 & $24.896$  &    &     &          & Au & 197 & $100.0$  \\
   & 128 & $2.188$  &    & 146 & $17.130$&    & 164 & $28.260$  & Ta & 180 & $0.012$  &    &     &          \\
   & 129 & $27.255$ &    & 148 & $5.716$ &    &     &           &    & 181 & $99.988$ & Hg & 196 & $0.15$   \\
   & 130 & $4.376$  &    & 150 & $5.596$ & Ho & 165 & $100.0$   &    &     &          &    & 198 & $9.97$   \\
   & 131 & $21.693$ &    &     &         &    &     &           & W  & 180 & $0.12$   &    & 199 & $16.87$  \\
   & 132 & $26.514$ & Sm & 144 & $3.07$  & Er & 162 & $0.139$   &    & 182 & $26.50$  &    & 200 & $23.10$  \\
   & 134 & $9.790$  &    & 147 & $14.99$ &    & 164 & $1.601$   &    & 183 & $14.31$  &    & 201 & $13.18$  \\
   & 136 & $7.954$  &    & 148 & $11.24$ &    & 166 & $33.503$  &    & 184 & $30.64$  &    & 202 & $29.86$  \\
   &     &          &    & 149 & $13.82$ &    & 167 & $22.869$  &    & 186 & $28.43$  &    & 204 & $6.87$   \\
Cs & 133 & $100.0$  &    & 150 & $7.38$  &    & 168 & $26.978$  &    &     &          &    &     &          \\
   &     &          &    & 152 & $26.75$ &    & 170 & $14.910$  & Re & 185 & $35.662$ & Tl & 203 & $29.524$ \\
Ba & 130 & $0.106$  &    & 154 & $22.75$ &    &     &           &    & 187 & $64.338$ &    & 205 & $70.476$ \\
   & 132 & $0.101$  &    &     &         & Tm & 169 & $100.0$   &    &     &          &    &     &          \\
   & 134 & $2.417$  & Eu & 151 & $47.81$ &    &     &           & Os & 184 & $0.020$  & Pb & 204 & $1.997$  \\
   & 135 & $6.592$  &    & 153 & $52.19$ & Yb & 168 & $0.12$    &    & 186 & $1.598$  &    & 206 & $18.582$ \\
   & 136 & $7.854$  &    &     &         &    & 170 & $2.98$    &    & 187 & $1.271$  &    & 207 & $20.563$ \\
   & 137 & $11.232$ & Gd & 152 & $0.20$  &    & 171 & $14.09$   &    & 188 & $13.337$ &    & 208 & $58.858$ \\
   & 138 & $71.698$ &    & 154 & $2.18$  &    & 172 & $21.69$   &    & 189 & $16.261$ &    &     &          \\
   &     &          &    & 155 & $14.80$ &    & 173 & $16.10$   &    & 190 & $26.444$ & Bi & 209 & $100.0$  \\
La & 138 & $0.091$  &    & 156 & $20.47$ &    & 174 & $32.03$   &    & 192 & $41.070$ &    &     &          \\
   & 139 & $99.909$ &    & 157 & $15.65$ &    & 176 & $13.00$   &    &     &          & Th & 232 & $100.0$  \\
   &     &          &    & 158 & $24.84$ &    &     &           & Ir & 191 & $37.3$   &    &     &          \\
Ce & 136 & $0.185$  &    & 160 & $21.86$ & Lu & 175 & $97.1795$ &    & 193 & $62.7$   & U  & 234 & $0.002$  \\
   & 138 & $0.251$  &    &     &         &    & 176 & $2.8205$  &    &     &          &    & 235 & $24.286$ \\
   & 140 & $88.450$ & Tb & 159 & $100.0$ &    &     &           & Pt & 190 & $0.014$  &    & 238 & $75.712$ \\
   & 142 & $11.114$ &    &     &         & Hf & 174 & $0.162$   &    & 192 & $0.782$  &    &     &          \\
   &     &          & Dy & 156 & $0.056$ &    & 176 & $5.206$   &    & 194 & $32.967$ &    &     &          \\
Pr & 141 & $100.0$  &    & 158 & $0.095$ &    & 177 & $18.606$  &    & 195  & $33.832$ &       &      &    \\
"""

for char in replaceChars:
    isotopeStr = isotopeStr.replace(char, '')

isotopeRows = isotopeStr.split('\n')
isotopeCells = [[y.strip() for y in x.split('&')] for x in isotopeRows[:-1]]
# NOTE(cmo): Each element takes up 3 columns of the row it's in, and is spread
# over multiple rows, so rearrange into table with 3 columns keeping the
# elements contiguous in rows.
isotopeReshape = []

for elemStart in range(0, len(isotopeCells[0]), 3):
    elemRange = slice(elemStart, elemStart+3)
    for row in isotopeCells:
        selection = row[elemRange]
        if not all([s == '' for s in selection]):
            isotopeReshape.append(row[elemRange])

dist = []
for row in isotopeReshape:
    if row[0] != '':
        try:
            elem = [e for e in elements if e.name == row[0]][0]
        except:
            raise ValueError('Unable to find element for name %s' % row[0])
        dist.append(ElementalDistribution(elem, []))
    N = int(row[1])
    iso = IsotopeProportion(N=N, proportion=(float(row[2]) / 100),
                            mass=massData[(N, elem.Z)])
    dist[-1].isotopes.append(iso)

# NOTE(cmo): Ensure normalisation to machine precision
for ele in dist:
    totalAbund = 0.0
    for iso in ele.isotopes:
        totalAbund += iso.proportion
    for iso in ele.isotopes:
        iso.proportion /= totalAbund

distDict = [asdict(d) for d in dist]
with open('AbundancesAsplund09.pickle', 'wb') as pkl:
    pickle.dump(distDict, pkl)
