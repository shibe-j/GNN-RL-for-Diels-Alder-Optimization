from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
from functools import lru_cache


class MorganFingerprintGenerator:
    """
    Robust Morgan fingerprint generator with caching.
    """

    def __init__(self, radius=2, n_bits=2048, use_counts=False):
        self.radius = radius
        self.n_bits = n_bits
        self.use_counts = use_counts

    @lru_cache(maxsize=512)
    def _mol_from_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        return mol

    @lru_cache(maxsize=512)
    def _fp_from_smiles(self, smiles):
        mol = self._mol_from_smiles(smiles)

        if self.use_counts:
            fp = AllChem.GetHashedMorganFingerprint(
                mol, self.radius, nBits=self.n_bits
            )
            arr = np.zeros((self.n_bits,), dtype=np.float32)
            for idx, value in fp.GetNonzeroElements().items():
                arr[idx % self.n_bits] += float(value)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, self.radius, nBits=self.n_bits
            )
            arr = np.zeros((self.n_bits,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)

        return arr.astype(np.float32)

    def featurize(self, smiles: str) -> np.ndarray:
        """
        Returns a (n_bits,) float32 numpy array.
        """
        return self._fp_from_smiles(smiles).copy()
