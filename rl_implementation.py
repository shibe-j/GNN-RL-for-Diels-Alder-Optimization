import gymnasium as gym
from gymnasium import spaces
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')


class DielsAlderSelectivityEnv(gym.Env):
    """
    Improved Custom Gymnasium environment for Diels–Alder selectivity.
    
    Key improvements:
    - Better reward structure with positive rewards for good outcomes
    - More predictable chemistry with clearer cause-effect
    - Intermediate shaping rewards
    - Longer episodes for better optimization
    """

    metadata = {"render_modes": []}

    def __init__(self, seed=None, max_steps=20):
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
            'C=CC#N',           # Simple EWG
            'CC=CC#N',          # Methyl + EWG
            'C(C#N)=CC#N',      # Double EWG (stronger)
            'C(C(=O)N)=CC#N'    # Amide + EWG (strongest)
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

        # EWG strength (0 = weak, 1 = strong)
        if self.dienophile == 'C(C(=O)N)=CC#N':
            ewg = 1.0  # Strongest
        elif self.dienophile == 'C(C#N)=CC#N':
            ewg = 0.8  # Strong
        elif self.dienophile == 'C=CC#N':
            ewg = 0.5  # Moderate
        else:
            ewg = 0.3  # Weak

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

        # ============================================
        # IMPROVED CHEMISTRY MODEL
        # ============================================
        
        # 1. Endo selectivity (polar solvents favor endo)
        if self.solvent == 'acetonitrile':
            endo_prob = 0.95
        elif self.solvent == 'DCM':
            endo_prob = 0.85
        else:  # hexane
            endo_prob = 0.25
        
        endo = int(self.np_random.random() < endo_prob)

        # 2. Regioselectivity (depends on dienophile)
        if self.dienophile == 'C=CC#N':
            regio_prob = 0.95  # Simple, very selective
        elif self.dienophile == 'CC=CC#N':
            regio_prob = 0.85  # Methyl causes some issues
        elif self.dienophile == 'C(C#N)=CC#N':
            regio_prob = 0.70  # Two EWGs = moderate
        else:  # 'C(C(=O)N)=CC#N'
            regio_prob = 0.80  # Strong directing effect

        regio_correct = int(self.np_random.random() < regio_prob)

        # 3. Activation barrier (deltaG in kcal/mol)
        # Strong EWGs lower the barrier
        if self.dienophile == 'C(C(=O)N)=CC#N':
            base_deltaG = 14.0
        elif self.dienophile == 'C(C#N)=CC#N':
            base_deltaG = 16.0
        elif self.dienophile == 'C=CC#N':
            base_deltaG = 18.0
        else:  # 'CC=CC#N'
            base_deltaG = 19.0

        # Temperature effect (higher temp helps overcome barrier)
        temp_factor = (self.temperature - 250) / 150.0  # 0 to 1
        deltaG = base_deltaG * (1.0 - 0.2 * temp_factor)
        
        # Add small randomness
        deltaG += self.np_random.uniform(-1.0, 1.0)

        # ============================================
        # IMPROVED REWARD STRUCTURE
        # ============================================
        
        reward = 0.0
        
        # Major rewards for selectivity (most important)
        reward += 3.0 * endo             # Endo product is highly desirable
        reward += 2.0 * regio_correct    # Regioselectivity is important
        
        # Barrier penalty (want low barrier for faster reaction)
        # Normalize so typical barriers give -0.5 to -1.5 penalty
        reward -= 0.1 * deltaG
        
        # Bonus for using strong EWG dienophiles (good chemistry practice)
        if self.dienophile in ['C(C#N)=CC#N', 'C(C(=O)N)=CC#N']:
            reward += 0.5
        
        # Shaping reward: polar solvent + strong EWG is a smart combo
        if self.solvent in ['DCM', 'acetonitrile'] and \
           self.dienophile in ['C(C#N)=CC#N', 'C(C(=O)N)=CC#N']:
            reward += 0.3
        
        # Small penalty for extreme temperatures (simulate practicality)
        if self.temperature > 380 or self.temperature < 260:
            reward -= 0.2
        
        # Small noise to encourage exploration
        reward += self.np_random.normal(0.0, 0.05)

        self.step_count += 1
        terminated = False
        truncated = self.step_count >= self.max_steps

        info = {
            "endo": endo,
            "regio_correct": regio_correct,
            "deltaG": float(deltaG),
            "temperature": self.temperature,
            "solvent": self.solvent,
            "dienophile": self.dienophile
        }

        return self.state.astype(np.float32), float(reward), terminated, truncated, info
    