import warnings

import astropy.units as u
import crtaf
from fractions import Fraction

from lightweaver.atomic_model import AtomicModel, AtomicLevel, LineType, LinearQuadrature, TabulatedQuadrature, LinearCoreExpWings, VoigtLine, HydrogenicContinuum, ExplicitContinuum
from lightweaver.broadening import LineBroadening, RadiativeBroadening, HydrogenLinearStarkBroadening, MultiplicativeStarkBroadening, QuadraticStarkBroadening, VdwUnsold, ScaledExponentBroadening
from lightweaver.collisional_rates import Omega, CE, CI, CH, CP, ChargeExchangeProton, ChargeExchangeNeutralH
from lightweaver.atomic_table import PeriodicTable

def from_crtaf(model: crtaf.Atom) -> AtomicModel:
    crtaf_labels = []
    levels = {}
    for label, level in model.levels.items():
        crtaf_labels.append(label)
        J = None
        if level.J is not None:
            J = Fraction(level.J.numerator, level.J.denominator)
        L = None
        if level.L is not None:
            L = level.L
        S = None
        if level.S is not None:
            S = Fraction(level.S.numerator, level.S.denominator)
        lw_level = AtomicLevel(
            E=level.energy.to("cm-1", equivalencies=u.spectral()).value.item(),
            g=level.g,
            label=level.label if level.label is not None else "",
            stage=level.stage-1,
            J=J,
            L=L,
            S=S,
        )
        levels[label] = lw_level

    crtaf_labels = sorted(crtaf_labels, key=lambda x: levels[x].E)
    levels = [levels[l] for l in crtaf_labels]
    level_conversion_dict = {label: idx for idx, label in enumerate(crtaf_labels)}

    lines = []
    for line in model.lines:
        if not isinstance(line, crtaf.VoigtBoundBound):
            raise ValueError(f"Unexpected line type encountered {line!r}, can only handle Voigt/PRD-Voigt.")

        ty = LineType.CRD
        if isinstance(line, crtaf.PrdVoigtBoundBound):
            ty = LineType.PRD

        natural_broadening = []
        elastic_broadening = []
        for b in line.broadening:
            if isinstance(b, crtaf.NaturalBroadening):
                natural_broadening.append(RadiativeBroadening(b.value.to("s-1").value))
            elif isinstance(b, crtaf.StarkLinearSutton):
                elastic_broadening.append(HydrogenLinearStarkBroadening())
            elif isinstance(b, crtaf.StarkMultiplicative):
                elastic_broadening.append(MultiplicativeStarkBroadening(b.C_4.value))
            elif isinstance(b, crtaf.StarkQuadratic):
                elastic_broadening.append(QuadraticStarkBroadening(b.scaling))
            elif isinstance(b, crtaf.VdWUnsold):
                elastic_broadening.append(VdwUnsold(vals=[b.H_scaling, b.He_scaling]))
            elif isinstance(b, crtaf.ScaledExponents):
                lw_b = ScaledExponentBroadening(
                    scaling=b.scaling,
                    temperatureExp=b.temperature_exponent,
                    hydrogenExp=b.hydrogen_exponent,
                    electronExp=b.electron_exponent,
                )
                if b.elastic:
                    elastic_broadening.append(lw_b)
                else:
                    natural_broadening.append(lw_b)
            else:
                raise ValueError(f"Unexpected broadening ({b}), can only handle core CRTAF types.")
        broadening = LineBroadening(natural=natural_broadening, elastic=elastic_broadening)

        q = line.wavelength_grid
        if isinstance(q, crtaf.LinearGrid):
            grid = LinearQuadrature(
                Nlambda=q.n_lambda,
                deltaLambda=q.delta_lambda.to(u.nm).value.item(),
            )
        elif isinstance(q, crtaf.TabulatedGrid):
            grid = TabulatedQuadrature(
                wavelengthGrid=q.wavelengths.to(u.nm).value.tolist(),
            )
        elif isinstance(q, crtaf.LinearCoreExpWings):
            grid = LinearCoreExpWings(
                qCore=q.q_core,
                qWing=q.q_wing,
                Nlambda=q.n_lambda,
            )
        else:
            raise ValueError(f"Unexpected WavelengthGrid ({q}), can only handle core CRTAF types and LinearCoreExpWings.")

        lw_line = VoigtLine(
            j=level_conversion_dict[line.transition[0]],
            i=level_conversion_dict[line.transition[1]],
            f=line.f_value,
            type=ty,
            quadrature=grid,
            broadening=broadening,
        )
        lines.append(lw_line)


    continua = []
    for cont in model.continua:
        if isinstance(cont, crtaf.HydrogenicBoundFree):
            lw_cont = HydrogenicContinuum(
                j=level_conversion_dict[cont.transition[0]],
                i=level_conversion_dict[cont.transition[1]],
                NlambdaGen=cont.n_lambda,
                alpha0=cont.sigma_peak.to("m2").value,
                minWavelength=cont.lambda_min.to(u.nm).value,
            )
        elif isinstance(cont, crtaf.TabulatedBoundFree):
            lw_cont = ExplicitContinuum(
                j=level_conversion_dict[cont.transition[0]],
                i=level_conversion_dict[cont.transition[1]],
                wavelengthGrid=cont.wavelengths.to(u.nm).value,
                alphaGrid=cont.sigma.to("m2").value,
            )
        else:
            raise ValueError(f"Unexpected continuum ({cont}), can only handle Hydrogenic and Tabulated.")
        continua.append(lw_cont)

    collisions = []
    for coll in model.collisions:
        j = level_conversion_dict[coll.transition[0]]
        i = level_conversion_dict[coll.transition[1]]
        for process in coll.data:
            if isinstance(process, crtaf.OmegaRate):
                lw_coll = Omega(
                    j=j,
                    i=i,
                    temperature=process.temperature.to(u.K).value.tolist(),
                    rates=process.data.value.tolist(),
                )
            elif isinstance(process, crtaf.CIRate):
                lw_coll = CI(
                    j=j,
                    i=i,
                    temperature=process.temperature.to(u.K).value.tolist(),
                    rates=process.data.to("m3 s-1 K(-1/2)").value.tolist(),
                )
            elif isinstance(process, crtaf.CERate):
                lw_coll = CE(
                    j=j,
                    i=i,
                    temperature=process.temperature.to(u.K).value.tolist(),
                    rates=process.data.to("m3 s-1 K(-1/2)").value.tolist(),
                )
            elif isinstance(process, crtaf.CHRate):
                lw_coll = CH(
                    j=j,
                    i=i,
                    temperature=process.temperature.to(u.K).value.tolist(),
                    rates=process.data.to("m3 s-1").value.tolist(),
                )
            elif isinstance(process, crtaf.CPRate):
                lw_coll = CP(
                    j=j,
                    i=i,
                    temperature=process.temperature.to(u.K).value.tolist(),
                    rates=process.data.to("m3 s-1").value.tolist(),
                )
            elif isinstance(process, crtaf.ChargeExcHRate):
                lw_coll = ChargeExchangeNeutralH(
                    j=j,
                    i=i,
                    temperature=process.temperature.to(u.K).value.tolist(),
                    rates=process.data.to("m3 s-1").value.tolist(),
                )
            elif isinstance(process, crtaf.ChargeExcPRate):
                lw_coll = ChargeExchangeProton(
                    j=j,
                    i=i,
                    temperature=process.temperature.to(u.K).value.tolist(),
                    rates=process.data.to("m3 s-1").value.tolist(),
                )
            else:
                raise ValueError(f"Unexpected collisional rate encountered ({coll}), expected one of [Omega, CI, CE, CH, CP, ChargeExcH, ChargeExcP].")
            collisions.append(lw_coll)

    if model.element.N is not None:
        warnings.warn("N provided. Whilst Lightweaver has the ability to handle isotopes, the CRTAF parser currently does not.")
    lw_model = AtomicModel(
        element=PeriodicTable[model.element.symbol],
        levels=levels,
        lines=lines,
        continua=continua,
        collisions=collisions,
    )
    return lw_model

