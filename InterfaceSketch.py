import lightweaver as lw

atoms = lw.default_atoms()
atoms.set_active('H', 'CA')
atmos = lw.Atmosphere(...)

molecules = lw.default_molecules()

ltePops = lw.AtomicTable().lte_populations(atmos)
icePops = molecules.compute_ice(atmos, ltePops)

rhCtx = lw.Context(atoms, atmos, ltePops, icePops, nRays=5)
try:
    rhCtx.iterate_stat_eq()
except Exception as e:
    print(e)

rhVertical = rhCtx.new_context_with(nRays=1, mu=[1.0])
rhVertical.solve_spectrum()

wavelength, spectrum = rhVertical.spectrum.wavelength, rhVertical.spectrum.spectrum

# Do a plot


