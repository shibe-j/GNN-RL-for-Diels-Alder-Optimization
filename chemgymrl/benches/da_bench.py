import numpy as np
from chemistrylab.util.reward import RewardGenerator
from chemistrylab import material, vessel
from chemistrylab.benches.general_bench import *
from chemistrylab.reactions.reaction_info import ReactInfo, REACTION_PATH
from chemistrylab.reactions.reaction import Reaction
from chemistrylab.lab.shelf import Shelf

from ..reactions import REACTION_PATH
from ..materials.but_and_acr_materials import Butadiene, Acrylonitrile, Cyanocyclohexene


def get_mat(mat, amount, name=None):
    "Makes a Vessel with a single material"

    my_vessel = vessel.Vessel(
        label=f'{mat} Vessel' if name is None else name,
        ignore_layout=True
    )
    # create the material dictionary for the vessel
    matclass = material.REGISTRY[mat]()
    matclass.mol = amount
    material_dict = {mat: matclass}
    # instruct the vessel to update its material dictionary

    my_vessel.material_dict = material_dict
    my_vessel.validate_solvents()
    my_vessel.validate_solutes()
    my_vessel.default_dt = 0.01

    return my_vessel

class DielsAlderReact_v0(GenBench):
    """
    A bench environment for performing the Diels-Alder reaction between
    Butadiene and Acrylonitrile to produce Cyanocyclohexene.
    """

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 10,
    }

    def __init__(self):
        # Base reward: count moles of product
        base_rew = RewardGenerator(
            use_purity=False,
            exclude_solvents=True,  # Changed: Exclude acrylonitrile from reward
            include_dissolved=False
        )

        #If you want to add a solvent put it into the default reaction vessel
        #in general and add reactants into it.

        shelf = Shelf([
            get_mat("Butadiene", 1, "Reaction Vessel"),
            get_mat("Acrylonitrile", 1,)
        ])

        actions = [
            Action([0], [ContinuousParam(298, 450, 0, (300,))], 'heat contact', [0], 0.01, False),
            Action([1], [ContinuousParam(0, 1, 1e-3, ())], 'pour by percent', [0], 0.01, False),
        ]

        react_info = ReactInfo.from_json(REACTION_PATH+"/diels_alder_reaction.py")
        print("="*70)
        print("LOADED REACTION PARAMETERS:")
        print(f"Activation Energy: {float(react_info.activ_energy_arr):.1f} J/mol")
        print(f"Pre-exponential Factor: {float(react_info.pre_exp_arr):.2e}")
        print("="*70)
        
        super(DielsAlderReact_v0, self).__init__(
            shelf,
            actions,
            ["PVT", "spectra", "targets"],
            targets=react_info.PRODUCTS,
            default_events=(Event("react", (Reaction(react_info),), None),),
            reward_function=base_rew,
            discrete=False,
            max_steps=20
        )

        def get_keys_to_action(self):
            # Control with the numpad or number keys.
            keys = dict()
            for i in range(5):
                arr = np.zeros(5)
                arr[i] = 1
                keys[str(i + 1)] = arr
            keys[()] = np.zeros(5)
            return keys