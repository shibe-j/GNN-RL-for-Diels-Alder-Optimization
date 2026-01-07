from chemistrylab import material

class Butadiene(material.Material):
    def __init__(self, mol=0):
        super().__init__(
            mol=mol,
            name="Butadiene",
            # CORRECTED: Gas density is ~0.0024 g/mL (2.4 g/L). 0.62 is for liquid.
            density={"s": None, "l": 0.62, "g": 0.0024},
            polarity=0.0,
            temperature=298,
            pressure=101.325,
            phase="g",
            charge=0.0,
            molar_mass=54.09,
            color=0.0,
            solute=True,
            solvent=False,
            boiling_point=268.7,  # Refined from NIST
            melting_point=164.3,
            specific_heat=1.46,   # CORRECTED: ~79 J/molK -> 1.46 J/gK (Gas)
            enthalpy_fusion=7984, # CORRECTED: MJ/mol -> J/mol (7.98 kJ/mol)
            enthalpy_vapor=22600, # J/mol
            spectra_overlap=None,
            spectra_no_overlap=None,
            index=None            
        )

class Acrylonitrile(material.Material):
    def __init__(self, mol=0):
        super().__init__(
            mol=mol,
            name="Acrylonitrile",
            density={"s": None, "l": 0.806, "g": None},
            polarity=3.88,
            temperature=298,
            pressure=101.325,
            phase="l",
            charge=0.0,
            molar_mass=53.06,
            color=0.0,
            solute=False,
            solvent=True,
            boiling_point=350.5,
            melting_point=189.6,
            specific_heat=2.09,   # J/gK
            enthalpy_fusion=6610, # CORRECTED: MJ/mol -> J/mol (6.61 kJ/mol)
            enthalpy_vapor=33000, # CORRECTED: 0.188 -> 33000 J/mol (33 kJ/mol)
            spectra_overlap=None,
            spectra_no_overlap=None,
            index=None
        )

class Cyanocyclohexene(material.Material):
    # Specifically 3-cyclohexene-1-carbonitrile (Diels-Alder adduct)
    def __init__(self, mol=0):
        super().__init__(
            mol=mol,
            name="Cyanocyclohexene",
            density={"s": None, "l": 0.96, "g": None},
            polarity=3.5, 
            temperature=298,
            pressure=101.325,
            phase="l",
            charge=0.0,
            molar_mass=107.15,
            color=0.0,
            solute=False,
            solvent=False,
            boiling_point=474.0,  # CORRECTED: 358K is at 17mmHg. Standard BP is ~200.9C (474K).
            specific_heat=1.82,   # CORRECTED: 195.24 J/molK -> 1.82 J/gK
            spectra_overlap=None,
            spectra_no_overlap=None,
            index=None
        )

material.register(Butadiene, Acrylonitrile, Cyanocyclohexene)