import gymnasium as gym
from gymnasium import spaces
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import rdMolDescriptors

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')


class DielsAlderSelectivityEnv(gym.Env):
    """
    Custom Gymnasium environment for optimizing *one fixed* Diels–Alder reaction's yield.
    
    This is a lightweight surrogate model (not quantum chemistry / kinetics).
    The agent adjusts reaction "conditions" and gets rewarded for increasing
    a yield proxy driven by:
    - conversion (rate proxy via exp(-ΔG‡/RT))
    - selectivity (endo/regio probabilities)
    - side reactions/decomposition (increase with harsher conditions)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        seed: int | None = None,
        max_steps: int = 20,
        temp_step: int = 5,
        temp_c_min: int = 0,
        temp_c_max: int = 200,
        step_minutes: float = 10.0,
        reaction_smiles: str | None = None,
        diene_smiles: str = "C=CC=C",          # 1,3-butadiene
        dienophile_smiles: str = "C=CC#N",     # acrylonitrile
    ):
        super().__init__()

        # Spaces
        # One reaction is fixed; actions only modify conditions.
        # 0-2: choose solvent
        # 3-5: choose catalyst
        # 6: temperature up
        # 7: temperature down
        self.action_space = spaces.Discrete(8)

        self.fp_size = 2048 * 2
        # Extra scalars: gap proxy, solvent polarity, EWG, temperature, catalyst strength,
        # conversion, side_fraction, bias
        self.obs_size = self.fp_size + 8
        # Morgan bits are 0/1; extra scalars are normalized/clipped to [0, 1] except bias=1.
        low = np.zeros((self.obs_size,), dtype=np.float32)
        high = np.ones((self.obs_size,), dtype=np.float32)
        low[-1] = 1.0
        high[-1] = 1.0
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Fixed reaction identity (2 reactants)
        self.reaction_smiles = reaction_smiles
        if reaction_smiles is not None:
            diene_smiles, dienophile_smiles = self._parse_reaction_smiles(reaction_smiles)

        self.diene_smiles = diene_smiles
        self.dienophile_smiles = dienophile_smiles

        self.solvents = ['hexane', 'DCM', 'acetonitrile']
        self.solvent_polarity = {
            'hexane': 0.1,
            'DCM': 0.4,
            'acetonitrile': 0.8
        }

        # Catalyst choices (coarse proxy). "strength" influences rate/selectivity but increases side rxns/cost.
        self.catalysts = ['none', 'lewis_acid', 'organocatalyst']
        self.catalyst_strength = {
            'none': 0.0,
            'lewis_acid': 0.8,
            'organocatalyst': 0.5,
        }

        self.temperature = None
        self.diene = None
        self.dienophile = None
        self.solvent = None
        self.catalyst = None
        self.state = None

        # Action granularity
        # Temperature adjustment per action (smaller => finer control)
        self.temp_step = int(temp_step)
        self.temp_c_min = int(temp_c_min)
        self.temp_c_max = int(temp_c_max)
        self.step_minutes = float(step_minutes)

        # Episode "process" state (yield proxy components)
        self.conversion = 0.0
        self.side_fraction = 0.0
        self.prev_yield = 0.0

        # Precompute fixed reactant fingerprints for efficiency
        self._diene_fp = self._featurize_molecule(self.diene_smiles)
        self._dienophile_fp = self._featurize_molecule(self.dienophile_smiles)
        self._ewg_strength = self._compute_ewg_strength(self.dienophile_smiles)

        self.step_count = 0
        self.max_steps = max_steps

        # Initialize RNG (Gymnasium will also reseed on reset(seed=...))
        super().reset(seed=seed)

    def _parse_reaction_smiles(self, reaction_smiles: str) -> tuple[str, str]:
        """
        Parse a simple reaction SMILES like 'diene.dienophile>>product'.
        Heuristically identify the conjugated diene vs the dienophile.
        """
        left = reaction_smiles.split(">>", 1)[0]
        reactants = [r for r in left.split(".") if r]
        if len(reactants) < 2:
            raise ValueError(f"reaction_smiles must contain 2+ reactants: {reaction_smiles!r}")

        scored: list[tuple[int, str]] = []
        for smi in reactants:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(f"Invalid reactant SMILES in reaction_smiles: {smi!r}")
            n_double = sum(
                1 for b in mol.GetBonds()
                if b.GetBondType() == Chem.rdchem.BondType.DOUBLE
            )
            scored.append((n_double, smi))

        # Conjugated dienes typically have >=2 double bonds; pick the max as diene.
        scored.sort(reverse=True, key=lambda x: x[0])
        diene = scored[0][1]
        dienophile = scored[1][1]
        return diene, dienophile

    def _featurize_molecule(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES for dienophile: {smiles!r}")
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        arr = np.zeros((2048,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    def _compute_ewg_strength(self, smiles: str) -> float:
        """
        Very rough EWG heuristic in [0, 1] from functional group counts.
        This is just a surrogate feature, not a physical descriptor.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0
        nitrile = len(mol.GetSubstructMatches(Chem.MolFromSmarts("C#N")))
        carbonyl = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3]=[OX1]")))
        sulfonyl = len(mol.GetSubstructMatches(Chem.MolFromSmarts("S(=O)(=O)")))
        halogen = sum(1 for a in mol.GetAtoms() if a.GetSymbol() in {"F", "Cl", "Br", "I"})
        score = 0.35 * nitrile + 0.25 * carbonyl + 0.25 * sulfonyl + 0.05 * halogen
        return float(np.clip(score, 0.0, 1.0))

    def _temp_to_norm(self, temp_c: int) -> float:
        # Normalize temperature to [0, 1] based on allowed range
        denom = max(1, (self.temp_c_max - self.temp_c_min))
        return float(np.clip((temp_c - self.temp_c_min) / denom, 0.0, 1.0))

    def _update_state(self):
        polarity = self.solvent_polarity[self.solvent]
        cat_strength = self.catalyst_strength[self.catalyst]

        self.state = np.concatenate([
            self._diene_fp,
            self._dienophile_fp,
            np.array([
                0.5,                      # placeholder "gap proxy" (kept but normalized)
                float(np.clip(polarity, 0.0, 1.0)),
                float(np.clip(self._ewg_strength, 0.0, 1.0)),
                self._temp_to_norm(int(self.temperature)),
                float(np.clip(cat_strength, 0.0, 1.0)),
                float(np.clip(self.conversion, 0.0, 1.0)),
                float(np.clip(self.side_fraction, 0.0, 1.0)),
                1.0                       # bias (constant)
            ], dtype=np.float32)
        ])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Temperature in °C for interpretability (typical DA can range widely depending on system)
        self.temperature = int(self.np_random.integers(self.temp_c_min, self.temp_c_max + 1))
        self.solvent = self.np_random.choice(self.solvents)
        self.catalyst = self.np_random.choice(self.catalysts)

        # Fixed reactants for this environment instance
        self.diene = self.diene_smiles
        self.dienophile = self.dienophile_smiles

        self.conversion = 0.0
        self.side_fraction = 0.0
        self.prev_yield = 0.0

        self.step_count = 0
        self._update_state()

        return self.state.astype(np.float32), {}

    def step(self, action):
        assert self.action_space.contains(action)

        if action < 3:
            self.solvent = self.solvents[action]
        elif action < 6:
            self.catalyst = self.catalysts[action - 3]
        elif action == 6:
            self.temperature = min(self.temperature + self.temp_step, self.temp_c_max)
        elif action == 7:
            self.temperature = max(self.temperature - self.temp_step, self.temp_c_min)

        # ============================================
        # SURROGATE CHEMISTRY MODEL (yield-oriented)
        # ============================================
        
        # Catalyst effects (coarse proxies)
        cat_strength = self.catalyst_strength[self.catalyst]

        # 1. Endo selectivity proxy
        # Baseline: polar solvents modestly favor endo; Lewis acids can enhance endo in many systems.
        if self.solvent == 'acetonitrile':
            endo_prob = 0.80
        elif self.solvent == 'DCM':
            endo_prob = 0.72
        else:  # hexane
            endo_prob = 0.55

        endo_prob = float(np.clip(endo_prob + 0.15 * cat_strength, 0.05, 0.95))
        
        endo = int(self.np_random.random() < endo_prob)

        # 2. Regioselectivity proxy (depends on dienophile; catalysts can help a bit)
        if self.dienophile == 'C=CC#N':
            regio_prob = 0.90
        elif self.dienophile == 'CC=CC#N':
            regio_prob = 0.80
        elif self.dienophile == 'C(C#N)=CC#N':
            regio_prob = 0.72
        else:  # 'C(C(=O)N)=CC#N'
            regio_prob = 0.82

        regio_prob = float(np.clip(regio_prob + 0.05 * cat_strength, 0.05, 0.97))
        regio_correct = int(self.np_random.random() < regio_prob)

        # 3. Activation free energy proxy ΔG‡ (kcal/mol)
        # For a *fixed* reaction we use a baseline barrier and modulate by solvent/catalyst/EWG.
        # (Still a surrogate; units are chosen for reasonable dynamics.)
        base_dg_dagger = 19.0
        base_dg_dagger -= 2.0 * self._ewg_strength
        base_dg_dagger -= 0.5 * self.solvent_polarity[self.solvent]

        # Catalyst lowers ΔG‡ (proxy)
        base_dg_dagger = float(base_dg_dagger - 2.5 * cat_strength)

        # Add modest stochasticity (solvent/impurities/etc.)
        dg_dagger = base_dg_dagger + float(self.np_random.uniform(-1.0, 1.0))

        # Convert temperature to K for rate proxy
        temp_k = float(self.temperature) + 273.15
        R = 0.0019872041  # kcal/(mol*K)
        # Relative rate proxy (dimensionless). Clamp to avoid numerical extremes.
        log_k_rel = float(np.clip(-dg_dagger / (R * temp_k), -50.0, 50.0))

        # 4. Conversion dynamics (simple first-order in "unreacted fraction")
        # dt in arbitrary units; scale k to get reasonable conversion per step.
        dt = self.step_minutes
        k = float(np.exp(log_k_rel) * 1e-3)  # scaled dimensionless
        frac_new = float(1.0 - np.exp(-k * dt))
        frac_new = float(np.clip(frac_new, 0.0, 1.0))
        self.conversion = float(self.conversion + (1.0 - self.conversion) * frac_new)

        # 5. Side reactions / decomposition (grow with harsher conditions)
        # Penalize high temperature and strong catalyst; polar solvents can slightly reduce.
        harsh_temp = max(0.0, (self.temperature - 120.0) / 80.0)  # starts above ~120°C
        solvent_protect = 0.1 if self.solvent in ['DCM', 'acetonitrile'] else 0.0
        side_step = 0.01 + 0.04 * harsh_temp + 0.03 * cat_strength - solvent_protect
        side_step = float(np.clip(side_step, 0.0, 0.08))
        self.side_fraction = float(np.clip(self.side_fraction + side_step, 0.0, 0.9))

        # 6. Yield proxy: conversion * selectivity * (1 - side)
        # Selectivity factor uses the realized endo/regio outcomes to keep stochasticity.
        selectivity_factor = 0.2 + 0.4 * endo + 0.4 * regio_correct  # in [0.2, 1.0]
        yield_proxy = float(np.clip(self.conversion * selectivity_factor * (1.0 - self.side_fraction), 0.0, 1.0))

        # ============================================
        # REWARD: maximize yield increase per step (with costs)
        # ============================================
        
        reward = yield_proxy - float(self.prev_yield)
        self.prev_yield = yield_proxy

        # Practical costs/penalties
        # - catalyst cost/complexity
        reward -= 0.03 * cat_strength
        # - high temperature operational penalty (soft)
        reward -= 0.0008 * max(0.0, self.temperature - 100.0)
        
        # Small noise to encourage exploration
        reward += self.np_random.normal(0.0, 0.05)

        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.max_steps

        info = {
            "endo": endo,
            "regio_correct": regio_correct,
            "dg_dagger_kcal_mol": float(dg_dagger),
            "log_k_rel": float(log_k_rel),
            "temperature_c": int(self.temperature),
            "temperature_k": float(temp_k),
            "solvent": self.solvent,
            "diene_smiles": self.diene_smiles,
            "dienophile_smiles": self.dienophile_smiles,
            "reaction_smiles": self.reaction_smiles,
            "catalyst": self.catalyst,
            "conversion": float(self.conversion),
            "side_fraction": float(self.side_fraction),
            "yield_proxy": float(yield_proxy),
            "endo_prob": float(endo_prob),
            "regio_prob": float(regio_prob),
        }

        self._update_state()
        return self.state.astype(np.float32), float(reward), terminated, truncated, info
    