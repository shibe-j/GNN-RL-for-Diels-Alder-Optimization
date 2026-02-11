"""
Convert data_mingap.csv to AQME-ready input CSV(s).

Reads data_mingap.csv from the parent folder (deepchem/), maps diene/dienophile
IDs to SMILES via the same logic as reaction_optimizer.id_to_smiles(), and writes
aqme_input.csv in this folder (deepchem/aqme/).

Click Run to use the defaults below; data_mingap.csv is not modified.
"""

from pathlib import Path

import pandas as pd
from rdkit import Chem

# Script lives in deepchem/aqme/; dataset is in deepchem/
AQME_DIR = Path(__file__).resolve().parent
DEEPCHEM_DIR = AQME_DIR.parent

# -----------------------------------------------------------------------------
# Default usage (click Run) — edit these if needed
# -----------------------------------------------------------------------------
DEFAULT_DATA_FILE = "data_mingap.csv"   # name in deepchem/ (parent folder)
DEFAULT_OUTPUT_FILE = "aqme_input.csv"   # written in this folder (aqme/)
INCLUDE_REACTANTS = True                 # include unique Diene_* and Dienophile_* rows
# -----------------------------------------------------------------------------

# Replicate id_to_smiles so this script can run standalone (no torch/gymnasium)
def id_to_smiles(id_str: str, is_diene: bool = True, mapped: bool = False) -> str:
    lookup = {
        "1": "F", "2": "C#N", "3": "OC", "4": "C", "5": "C(C)(C)C",
        "6": "", "7": "c1ccccc1", "8": "C(=O)OC", "9": "C=O",
    }
    parts = str(id_str).split("_")
    groups = [lookup.get(p, "") for p in parts]
    if is_diene:
        while len(groups) < 4:
            groups.append("")
        groups = groups[:4]
        g1, g2, g3, g4 = groups
        c1, c2, c3, c4 = (f"({g})" if g else "" for g in (g1, g2, g3, g4))
        smiles = f"[C:1]{c1}=[C:2]{c2}[C:3]{c3}=[C:4]{c4}" if mapped else f"C{c1}=C{c2}C{c3}=C{c4}"
    else:
        while len(groups) < 2:
            groups.append("")
        groups = groups[:2]
        g1, g2 = groups
        c1, c2 = (f"({g})" if g else "" for g in (g1, g2))
        smiles = f"[C:5]{c1}=[C:6]{c2}" if mapped else f"C{c1}=C{c2}"
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else smiles


def get_ts_constraints_dist(combined_mapped_smiles: str, bond_length: float = 2.35) -> str:
    """
    Get constraints_dist string for Diels-Alder TS: two forming bonds
    C2-C5 and C3-C6 (map numbers). Returns 1-based atom indices for AQME/CREST.
    """
    mol = Chem.MolFromSmiles(combined_mapped_smiles)
    if mol is None:
        return ""
    map_to_idx = {}
    for atom in mol.GetAtoms():
        num = atom.GetAtomMapNum()
        if num:
            map_to_idx[num] = atom.GetIdx()
    # Forming bonds: diene C2–C5 and C3–C6
    for k in (2, 3, 5, 6):
        if k not in map_to_idx:
            return ""
    # AQME/CREST use 1-based atom indices in the constraint format
    i2, i3 = map_to_idx[2], map_to_idx[3]
    i5, i6 = map_to_idx[5], map_to_idx[6]
    return f"[[{i2 + 1},{i5 + 1},{bond_length}],[{i3 + 1},{i6 + 1},{bond_length}]]"


def build_aqme_input_csv(
    data_path: Path,
    out_path: Path,
    include_reactants: bool = True,
) -> None:
    df = pd.read_csv(data_path)
    if "dienophile" not in df.columns or "diene" not in df.columns:
        raise ValueError(f"Expected columns 'dienophile' and 'diene' in {data_path}")

    rows = []

    if include_reactants:
        seen_diene = set()
        seen_dienophile = set()
        for _, row in df.iterrows():
            di, dph = row["diene"], row["dienophile"]
            if di not in seen_diene:
                seen_diene.add(di)
                smi = id_to_smiles(di, is_diene=True, mapped=False)
                name = "Diene_" + str(di).replace("_", "-")
                rows.append({"SMILES": smi, "code_name": name, "constraints_dist": ""})
            if dph not in seen_dienophile:
                seen_dienophile.add(dph)
                smi = id_to_smiles(dph, is_diene=False, mapped=False)
                name = "Dienophile_" + str(dph).replace("_", "-")
                rows.append({"SMILES": smi, "code_name": name, "constraints_dist": ""})

    for _, row in df.iterrows():
        di, dph = row["diene"], row["dienophile"]
        diene_smi = id_to_smiles(di, is_diene=True, mapped=True)
        dienophile_smi = id_to_smiles(dph, is_diene=False, mapped=True)
        combined = f"{diene_smi}.{dienophile_smi}"
        constraints = get_ts_constraints_dist(combined)
        name = f"TS_{str(dph).replace('_', '-')}_{str(di).replace('_', '-')}"
        rows.append({"SMILES": combined, "code_name": name, "constraints_dist": constraints})

    out_df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} rows to {out_path}")


def main():
    data_path = DEEPCHEM_DIR / DEFAULT_DATA_FILE
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    out_path = AQME_DIR / DEFAULT_OUTPUT_FILE
    build_aqme_input_csv(data_path, out_path, include_reactants=INCLUDE_REACTANTS)
    print("Done. Use run_aqme_pipeline.py in this folder next (gas and/or toluene).")


if __name__ == "__main__":
    main()
