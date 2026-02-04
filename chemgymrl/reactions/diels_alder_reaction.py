"""
Utilities for the Butadiene + Acrylonitrile Diels–Alder reaction.

Important: this module should be import-safe (no file writes / prints at import),
since the gym environment imports reaction definitions during registration.
"""

from __future__ import annotations

import numpy as np

from chemistrylab.reactions.reaction_info import ReactInfo


def build_diels_alder_react_info() -> ReactInfo:
    """
    Returns the `ReactInfo` for:
      Butadiene + Acrylonitrile -> Cyanocyclohexene

    Shapes follow ChemGymRL docs:
      - `stoich_coeff_arr`: [reactions, materials]
      - `conc_coeff_arr`:  [materials, reactions]
    """

    materials = ["Butadiene", "Acrylonitrile", "Cyanocyclohexene"]

    return ReactInfo(
        name="but+acr diels alder",
        REACTANTS=["Butadiene", "Acrylonitrile"],
        PRODUCTS=["Cyanocyclohexene"],
        SOLVENTS=["Acrylonitrile"],
        MATERIALS=materials,
        pre_exp_arr=np.array([1.0e10], dtype=np.float64),
        activ_energy_arr=np.array([9.0e4], dtype=np.float64),
        stoich_coeff_arr=np.array([[-1.0, -1.0, 1.0]], dtype=np.float32),
        conc_coeff_arr=np.array([[-1.0], [-1.0], [1.0]], dtype=np.float32),
    )


if __name__ == "__main__":
    # Optional helper for regenerating the json in the right format.
    import os

    info = build_diels_alder_react_info()
    out_path = os.path.join(
        os.path.dirname(__file__),
        "available_reactions",
        "diels_alder.json",
    )
    info.dump_to_json(out_path)
    print(f"Saved reaction info to {out_path}")