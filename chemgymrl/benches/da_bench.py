import numpy as np
from chemistrylab.util.reward import RewardGenerator
from chemistrylab import material, vessel
from chemistrylab.benches.general_bench import *
from chemistrylab.reactions.reaction_info import ReactInfo
from chemistrylab.reactions.reaction import Reaction
from chemistrylab.lab.shelf import Shelf

from chemgymrl.reactions import REACTION_PATH
from chemgymrl.materials import but_and_acr_materials  # noqa: F401  (registers materials)

# --- debug instrumentation (python -> NDJSON file) ---
import json as _json
import time as _time
from pathlib import Path as _Path

_DEBUG_LOG_PATH = _Path(__file__).resolve().parents[2] / ".cursor" / "debug.log"

def _dbg_log(*, runId: str, hypothesisId: str, location: str, message: str, data: dict):
    # #region agent log
    try:
        _DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "sessionId": "debug-session",
            "runId": runId,
            "hypothesisId": hypothesisId,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(_time.time() * 1000),
        }
        with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(_json.dumps(payload) + "\n")
    except Exception:
        pass
    # #endregion


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
            get_mat("Acrylonitrile", 1, "Acrylonitrile Vessel")
        ])

        actions=[
            Action([0], [ContinuousParam(298, 450, 0, (300,))], 'heat contact', [0], 0.01, False),
            Action([1], [ContinuousParam(0, 1, 1e-3, ())], 'pour by percent', [0], 0.01, False),
        ]
        """
        actions = [
            Action([0], [ContinuousParam(298, 450, 0, (300,))], 'heat contact', [0], 0.01, False),
            Action([1], [ContinuousParam(0, 1, 1e-3, ())], 'pour by percent', [0], 0.01, False),
        ]
        """

        react_info = ReactInfo.from_json(REACTION_PATH + "/diels_alder.json")
        _dbg_log(
            runId="pre-fix",
            hypothesisId="H2",
            location="chemgymrl/benches/da_bench.py:__init__",
            message="Loaded ReactInfo",
            data={
                "name": getattr(react_info, "name", None),
                "REACTANTS": getattr(react_info, "REACTANTS", None),
                "PRODUCTS": getattr(react_info, "PRODUCTS", None),
                "MATERIALS": getattr(react_info, "MATERIALS", None),
                "pre_exp_shape": getattr(getattr(react_info, "pre_exp_arr", None), "shape", None),
                "Ea_shape": getattr(getattr(react_info, "activ_energy_arr", None), "shape", None),
                "stoich_shape": getattr(getattr(react_info, "stoich_coeff_arr", None), "shape", None),
                "conc_shape": getattr(getattr(react_info, "conc_coeff_arr", None), "shape", None),
            },
        )
        
        super(DielsAlderReact_v0, self).__init__(
            shelf,
            actions,
            ["PVT", "spectra", "targets"],
            targets=react_info.PRODUCTS,
            default_events=(Event("react", (Reaction(react_info),), None),),
            reward_function=base_rew,
            discrete=False,
            #discrete=True,
            max_steps=20
        )
        _dbg_log(
            runId="pre-fix",
            hypothesisId="H1",
            location="chemgymrl/benches/da_bench.py:__init__",
            message="Bench initialized",
            data={
                "actions_n": len(actions),
                "targets": react_info.PRODUCTS,
                "max_steps": 20,
                "default_events": ["react"],
            },
        )

    def reset(self, **kwargs):
        out = super().reset(**kwargs)
        # Try to snapshot starting moles in reaction vessel
        rv = None
        try:
            rv = self.shelf[0]
        except Exception:
            pass
        mats = {}
        if rv is not None:
            try:
                for k, m in rv.material_dict.items():
                    mats[k] = float(getattr(m, "mol", 0.0))
            except Exception:
                mats = {"_err": "could_not_read_materials"}
        _dbg_log(
            runId="pre-fix",
            hypothesisId="H3",
            location="chemgymrl/benches/da_bench.py:reset",
            message="Reset snapshot",
            data={"reaction_vessel_mol": mats},
        )
        # Track product moles to provide dense reward signal.
        # This avoids the common issue where the built-in reward only appears at episode end.
        try:
            self._prev_product_mol = float(mats.get("Cyanocyclohexene", 0.0))  # type: ignore[attr-defined]
        except Exception:
            self._prev_product_mol = 0.0  # type: ignore[attr-defined]
        return out

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        # Snapshot key quantities after step
        rv = None
        try:
            rv = self.shelf[0]
        except Exception:
            pass
        mats = {}
        if rv is not None:
            try:
                for k, m in rv.material_dict.items():
                    mats[k] = float(getattr(m, "mol", 0.0))
            except Exception:
                mats = {"_err": "could_not_read_materials"}

        # Dense reward: positive delta in product moles each step.
        # Keep the original reward in info for debugging/compare.
        dense_reward = reward
        try:
            current_prod = float(mats.get("Cyanocyclohexene", 0.0))
            prev_prod = float(getattr(self, "_prev_product_mol", 0.0))
            delta = current_prod - prev_prod
            if delta < 0:
                delta = 0.0
            dense_reward = float(delta)
            self._prev_product_mol = current_prod  # type: ignore[attr-defined]
            if isinstance(info, dict):
                info = dict(info)
                info["orig_reward"] = float(reward) if np.isscalar(reward) else str(reward)
                info["product_mol"] = current_prod
                info["product_delta"] = dense_reward
        except Exception:
            pass
        _dbg_log(
            runId="post-fix",
            hypothesisId="H1",
            location="chemgymrl/benches/da_bench.py:step",
            message="Step snapshot",
            data={
                "action": np.asarray(action).tolist() if hasattr(action, "__len__") else action,
                "reward": float(dense_reward) if np.isscalar(dense_reward) else str(dense_reward),
                "orig_reward": float(reward) if np.isscalar(reward) else str(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "reaction_vessel_mol": mats,
                "info_keys": list(info.keys()) if isinstance(info, dict) else str(type(info)),
            },
        )
        return obs, dense_reward, terminated, truncated, info

    def get_keys_to_action(self):
        # Control with the numpad or number keys.
        keys = dict()
        for i in range(5):
            arr = np.zeros(5)
            arr[i] = 1
            keys[str(i + 1)] = arr
        keys[()] = np.zeros(5)
        return keys