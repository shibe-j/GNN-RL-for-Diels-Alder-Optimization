import gymnasium as gym
from gymnasium import spaces
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')


class DielsAlderSelectivityEnv(gym.Env):
    """
    Custom Gymnasium environment for Diels–Alder selectivity.
    """

    metadata = {"render_modes": []}

    def __init__(self, seed=None, max_steps=10):
        super().__init__()

        # RNG
        self.np_random = None
        self.seed(seed)

        # Spaces
        self.action_space = spaces.Discrete(9)

        self.fp_size = 2048
        self.obs_size = self.fp_size + 5
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(self.obs_size,),
            dtype=np.float32
        )

        # Chemistry
        self.substituents = [
            'C=CC#N',
            'CC=CC#N',
            'C(C#N)=CC#N',
            'C(C(=O)N)=CC#N'
        ]

        self.solvents = ['hexane', 'DCM', 'acetonitrile']
        self.solvent_polarity = {
            'hexane': 0.1,
            'DCM': 0.4,
            'acetonitrile': 0.8
        }

        self.temperature = None
        self.dienophile = None
        self.solvent = None
        self.state = None

        self.step_count = 0
        self.max_steps = max_steps

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _featurize_molecule(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        arr = np.zeros((2048,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    def _update_state(self):
        fp = self._featurize_molecule(self.dienophile)

        ewg = float(self.dienophile in [
            'C(C#N)=CC#N',
            'C(C(=O)N)=CC#N'
        ])

        polarity = self.solvent_polarity[self.solvent]

        self.state = np.concatenate([
            fp,
            np.array([
                1.0,                      # gap proxy
                polarity,
                ewg,
                self.temperature / 500.0,
                1.0                       # bias
            ], dtype=np.float32)
        ])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.temperature = int(self.np_random.integers(280, 351))
        self.solvent = self.np_random.choice(self.solvents)
        self.dienophile = self.np_random.choice(self.substituents)

        self.step_count = 0
        self._update_state()

        return self.state.astype(np.float32), {}

    def step(self, action):
        assert self.action_space.contains(action)

        if action < 4:
            self.dienophile = self.substituents[action]
        elif action < 7:
            self.solvent = self.solvents[action - 4]
        elif action == 7:
            self.temperature = min(self.temperature + 10, 400)
        elif action == 8:
            self.temperature = max(self.temperature - 10, 250)

        self._update_state()

        # Stochastic chemistry
        endo_prob = 0.8 if self.solvent in ['DCM', 'acetonitrile'] else 0.3
        endo = int(self.np_random.random() < endo_prob)

        if self.dienophile == 'C=CC#N':
            regio_prob = 0.9
        elif self.dienophile == 'CC=CC#N':
            regio_prob = 0.8
        else:
            regio_prob = 0.5

        regio_correct = int(self.np_random.random() < regio_prob)

        ewg_bonus = float(self.dienophile in [
            'C(C#N)=CC#N',
            'C(C(=O)N)=CC#N'
        ])

        deltaG = 20.0 - 5.0 * self.np_random.random() * ewg_bonus

        reward = (
            -0.1 * deltaG +
            1.0 * endo +
            1.0 * regio_correct +
            self.np_random.normal(0.0, 0.05)
        )

        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.max_steps

        info = {
            "endo": endo,
            "regio_correct": regio_correct
        }

        return self.state.astype(np.float32), float(reward), terminated, truncated, info
