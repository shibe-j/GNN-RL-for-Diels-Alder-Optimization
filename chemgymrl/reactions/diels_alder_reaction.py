from chemistrylab.reactions.reaction_info import ReactInfo
from chemistrylab.reactions.reaction import Reaction
from chemistrylab import material, vessel
import numpy as np
import json
from chemgymrl.materials.but_and_acr_materials import Butadiene, Acrylonitrile, Cyanocyclohexene


r_info = ReactInfo(
    name="but+acr diels alder",
    REACTANTS=["Butadiene", "Acrylonitrile"],
    PRODUCTS=["Cyanocyclohexene"],
    SOLVENTS=["Acrylonitrile"],
    MATERIALS=["Butadiene", "Acrylonitrile", "Cyanocyclohexene"],
    pre_exp_arr=np.array([1.0e7]),
    activ_energy_arr=np.array([9.0e4]),
    stoich_coeff_arr=np.array([
        [-1.0, -1.0, 1.0]
    ]),
    conc_coeff_arr=np.array([
        [-1.0, -1.0, 1.0]
    ]).astype(np.float32),
)

# Custom JSON dump to preserve arrays
output_path = 'chemgymrl/reactions/available_reactions/diels_alder.json'
data = {
    "name": r_info.name,
    "REACTANTS": r_info.REACTANTS,
    "PRODUCTS": r_info.PRODUCTS,
    "SOLVENTS": r_info.SOLVENTS,
    "MATERIALS": r_info.MATERIALS,
    "pre_exp_arr": r_info.pre_exp_arr.tolist(),  # Convert to list to preserve array structure
    "activ_energy_arr": r_info.activ_energy_arr.tolist(),
    "stoich_coeff_arr": r_info.stoich_coeff_arr.tolist(),
    "conc_coeff_arr": r_info.conc_coeff_arr.tolist(),
}

with open(output_path, 'w') as f:
    json.dump(data, f, indent=2)

print(f"Saved reaction info to {output_path}")

# Rest of your testing code
reaction = Reaction(r_info)
v = vessel.Vessel("Diels Alder vessel")

butadiene = Butadiene(mol=1)
acrylonitrile = Acrylonitrile(mol=1)
cyanocyclohexene = Cyanocyclohexene(mol=0)
v.material_dict={butadiene._name:butadiene, acrylonitrile._name:acrylonitrile, cyanocyclohexene._name:cyanocyclohexene}
v.default_dt=0.01

print(v.get_material_dataframe())

for _ in range(10):
    reaction.update_concentrations(v)
    print(v.get_material_dataframe())