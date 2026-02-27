import argparse
import json
import os
from pathlib import Path
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.nn import LayerNorm, TransformerConv, global_max_pool, global_mean_pool
from torch.nn import Dropout, Linear, ReLU, Sequential

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

TARGET_COL = "Min_TSـall"
DEFAULT_ROW_IDX = 0
DEFAULT_DATA_FILE = "data/evaluation_data.csv"

# Optional: stable-baselines3 algorithms for RL optimization (continuous action space)
try:
    from stable_baselines3 import A2C, PPO, SAC, TD3
    _SB3_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    PPO = A2C = SAC = TD3 = None
    _SB3_AVAILABLE = False

# Algorithm registry: name -> (AlgoClass, default_extra_kwargs for .learn() or constructor)
# All use MlpPolicy; env has Box action (0,1)^6 and Box observation.
RL_ALGORITHMS = {}
if _SB3_AVAILABLE:
    RL_ALGORITHMS["ppo"] = (PPO, {"policy": "MlpPolicy", "verbose": 1})
    RL_ALGORITHMS["a2c"] = (A2C, {"policy": "MlpPolicy", "verbose": 1})
    RL_ALGORITHMS["sac"] = (
        SAC,
        {
            "policy": "MlpPolicy",
            "verbose": 1,
            "buffer_size": 50_000,
            "learning_starts": 1000,
        },
    )
    RL_ALGORITHMS["td3"] = (
        TD3,
        {
            "policy": "MlpPolicy",
            "verbose": 1,
            "buffer_size": 50_000,
            "learning_starts": 1000,
        },
    )

# Set which RL algorithm to use here: "ppo", "a2c", "sac", "td3"
# (CLI --algorithm overrides this when provided.)
RL_ALGORITHM = "td3"

# Training length per algorithm (built-in; no CLI). PPO needs more steps (n_steps=2048).
RL_TIMESTEPS = {"ppo": 100_000, "a2c": 20_000, "sac": 10_000, "td3": 10_000}


def id_to_smiles(id_str, is_diene: bool = True, mapped: bool = False) -> str:
    lookup = {
        "1": "F",
        "2": "C#N",
        "3": "OC",
        "4": "C",
        "5": "C(C)(C)C",
        "6": "",
        "7": "c1ccccc1",
        "8": "C(=O)OC",
        "9": "C=O",
    }

    parts = str(id_str).split("_")
    groups = [lookup.get(p, "") for p in parts]

    if is_diene:
        g1, g2, g3, g4 = groups
        c1 = f"({g1})" if g1 else ""
        c2 = f"({g2})" if g2 else ""
        c3 = f"({g3})" if g3 else ""
        c4 = f"({g4})" if g4 else ""
        smiles = f"[C:1]{c1}=[C:2]{c2}[C:3]{c3}=[C:4]{c4}" if mapped else f"C{c1}=C{c2}C{c3}=C{c4}"
    else:
        g1, g2 = groups
        c1 = f"({g1})" if g1 else ""
        c2 = f"({g2})" if g2 else ""
        smiles = f"[C:5]{c1}=[C:6]{c2}" if mapped else f"C{c1}=C{c2}"

    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Chem.MolToSmiles(mol)
    return smiles


def _normalize_list(values, means, stds):
    values = np.array(values, dtype=float)
    means = np.asarray(means)
    stds = np.asarray(stds)
    stds = np.where(stds == 0, 1.0, stds)
    return ((values - means) / stds).tolist()


def _hybridization_onehot(atom):
    hyb = atom.GetHybridization()
    if hyb == Chem.rdchem.HybridizationType.SP:
        return [1, 0, 0, 0]
    if hyb == Chem.rdchem.HybridizationType.SP2:
        return [0, 1, 0, 0]
    if hyb == Chem.rdchem.HybridizationType.SP3:
        return [0, 0, 1, 0]
    return [0, 0, 0, 1]


def mol_to_pyg_data(smiles, pz_pops, nbo_charges, volumes) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    num_atoms = mol.GetNumAtoms()

    frags = Chem.GetMolFrags(mol)
    frag_id = np.zeros(num_atoms, dtype=int)
    for frag_idx, atom_idxs in enumerate(frags):
        for atom_idx in atom_idxs:
            frag_id[atom_idx] = frag_idx

    pz = np.zeros(num_atoms, dtype=float)
    nbo = np.zeros(num_atoms, dtype=float)
    vol = np.zeros(num_atoms, dtype=float)

    if len(frags) > 0:
        diene_atoms = list(frags[0])
        for i, atom_idx in enumerate(diene_atoms[:4]):
            if i < len(pz_pops):
                pz[atom_idx] = pz_pops[i]
            if i < len(nbo_charges):
                nbo[atom_idx] = nbo_charges[i]
            if i < len(volumes):
                vol[atom_idx] = volumes[i]

    if len(frags) > 1:
        dienophile_atoms = list(frags[1])
        for j, atom_idx in enumerate(dienophile_atoms[:2]):
            src = 4 + j
            if src < len(pz_pops):
                pz[atom_idx] = pz_pops[src]
            if src < len(nbo_charges):
                nbo[atom_idx] = nbo_charges[src]
            if src < len(volumes):
                vol[atom_idx] = volumes[src]

    node_features = []
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        atomic_num = atom.GetAtomicNum() / 20.0
        node_features.append(
            [
                atomic_num,
                float(pz[atom_idx]),
                float(nbo[atom_idx]),
                float(vol[atom_idx]),
                0.0 if frag_id[atom_idx] == 0 else 1.0,
            ]
        )

    x = torch.tensor(node_features, dtype=torch.float)

    edge_indices = []
    edge_attrs = []
    bond_type_map = {
        Chem.rdchem.BondType.SINGLE: [1, 0, 0],
        Chem.rdchem.BondType.DOUBLE: [0, 1, 0],
        Chem.rdchem.BondType.AROMATIC: [0, 0, 1],
    }
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        b_type = bond_type_map.get(bond.GetBondType(), [0, 0, 0])
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [b_type, b_type]

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def create_reaction_graph_from_row(row, feature_stats) -> Data:
    diene_smi = id_to_smiles(row["diene"], is_diene=True)
    dienophile_smi = id_to_smiles(row["dienophile"], is_diene=False)
    combined_smi = f"{diene_smi}.{dienophile_smi}"

    pz_list = [
        row["pz_pop_C1_D"],
        row["pz_pop_C2_D"],
        row["pz_pop_C3_D"],
        row["pz_pop_C4_D"],
        row["pz_pop_C1_dPh"],
        row["pz_pop_C2_dPh"],
    ]
    nbo_list = [
        row["NBO_1_D"],
        row["NBO_2_D"],
        row["NBO_3_D"],
        row["NBO_4_D"],
        row["NBO_1dPh"],
        row["NBO_2_dPh"],
    ]
    vol_list = [
        row["Volume_D_1"],
        row["Volume_D_2"],
        row["Volume_D_3"],
        row["Volume_D_4"],
        row["Volume_dPh_1"],
        row["Volume_dPh_2"],
    ]

    pz_list = _normalize_list(pz_list, feature_stats["pz_means"], feature_stats["pz_stds"])
    nbo_list = _normalize_list(nbo_list, feature_stats["nbo_means"], feature_stats["nbo_stds"])
    vol_list = _normalize_list(vol_list, feature_stats["vol_means"], feature_stats["vol_stds"])

    graph = mol_to_pyg_data(combined_smi, pz_list, nbo_list, vol_list)
    return graph


def mol_to_pyg_data_impl2(smiles, pz_map, nbo_map, vol_map) -> Data:
    """Build PyG Data with 15 node dims, 5 edge dims (impl2 format)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    frags = Chem.GetMolFrags(mol, asMols=False)
    atom_to_frag = {}
    for frag_id, atom_ids in enumerate(frags):
        for atom_id in atom_ids:
            atom_to_frag[atom_id] = frag_id

    node_features = []
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        atom_map = atom.GetAtomMapNum()
        frag_id = atom_to_frag.get(atom_idx, 0)

        atomic_num = atom.GetAtomicNum() / 20.0
        degree = atom.GetDegree() / 4.0
        formal_charge = float(atom.GetFormalCharge())
        aromatic = 1.0 if atom.GetIsAromatic() else 0.0
        ring = 1.0 if atom.IsInRing() else 0.0
        num_hs = atom.GetTotalNumHs() / 4.0
        mass = atom.GetMass() / 200.0
        hyb = _hybridization_onehot(atom)
        pz = float(pz_map.get(atom_map, 0.0))
        nbo = float(nbo_map.get(atom_map, 0.0))
        vol = float(vol_map.get(atom_map, 0.0))
        frag_flag = 0.0 if frag_id == 0 else 1.0

        node_features.append([
            atomic_num, degree, formal_charge, aromatic, ring, num_hs, mass,
            *hyb, pz, nbo, vol, frag_flag,
        ])

    x = torch.tensor(node_features, dtype=torch.float)

    edge_indices = []
    edge_attrs = []
    bond_type_map = {
        Chem.rdchem.BondType.SINGLE: [1, 0, 0],
        Chem.rdchem.BondType.DOUBLE: [0, 1, 0],
        Chem.rdchem.BondType.AROMATIC: [0, 0, 1],
    }
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        b_type = bond_type_map.get(bond.GetBondType(), [0, 0, 0])
        conj = 1.0 if bond.GetIsConjugated() else 0.0
        ring = 1.0 if bond.IsInRing() else 0.0
        b_feat = b_type + [conj, ring]
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [b_feat, b_feat]

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def create_reaction_graph_from_row_impl2(row, feature_stats) -> Data:
    """Build reaction graph in impl2 format (15 node, 5 edge) for condition optimizer."""
    diene_smi = id_to_smiles(row["diene"], is_diene=True, mapped=True)
    dienophile_smi = id_to_smiles(row["dienophile"], is_diene=False, mapped=True)
    combined_smi = f"{diene_smi}.{dienophile_smi}"

    pz_list = [
        row["pz_pop_C1_D"], row["pz_pop_C2_D"], row["pz_pop_C3_D"], row["pz_pop_C4_D"],
        row["pz_pop_C1_dPh"], row["pz_pop_C2_dPh"],
    ]
    nbo_list = [
        row["NBO_1_D"], row["NBO_2_D"], row["NBO_3_D"], row["NBO_4_D"],
        row["NBO_1dPh"], row["NBO_2_dPh"],
    ]
    vol_list = [
        row["Volume_D_1"], row["Volume_D_2"], row["Volume_D_3"], row["Volume_D_4"],
        row["Volume_dPh_1"], row["Volume_dPh_2"],
    ]

    pz_list = _normalize_list(pz_list, feature_stats["pz_means"], feature_stats["pz_stds"])
    nbo_list = _normalize_list(nbo_list, feature_stats["nbo_means"], feature_stats["nbo_stds"])
    vol_list = _normalize_list(vol_list, feature_stats["vol_means"], feature_stats["vol_stds"])

    pz_map = {1: pz_list[0], 2: pz_list[1], 3: pz_list[2], 4: pz_list[3], 5: pz_list[4], 6: pz_list[5]}
    nbo_map = {1: nbo_list[0], 2: nbo_list[1], 3: nbo_list[2], 4: nbo_list[3], 5: nbo_list[4], 6: nbo_list[5]}
    vol_map = {1: vol_list[0], 2: vol_list[1], 3: vol_list[2], 4: vol_list[3], 5: vol_list[4], 6: vol_list[5]}

    return mol_to_pyg_data_impl2(combined_smi, pz_map, nbo_map, vol_map)


def build_feature_stats(train_df) -> dict:
    pz_cols = [
        "pz_pop_C1_D",
        "pz_pop_C2_D",
        "pz_pop_C3_D",
        "pz_pop_C4_D",
        "pz_pop_C1_dPh",
        "pz_pop_C2_dPh",
    ]
    nbo_cols = [
        "NBO_1_D",
        "NBO_2_D",
        "NBO_3_D",
        "NBO_4_D",
        "NBO_1dPh",
        "NBO_2_dPh",
    ]
    vol_cols = [
        "Volume_D_1",
        "Volume_D_2",
        "Volume_D_3",
        "Volume_D_4",
        "Volume_dPh_1",
        "Volume_dPh_2",
    ]

    return {
        "pz_means": train_df[pz_cols].mean().values,
        "pz_stds": train_df[pz_cols].std().replace(0, 1.0).values,
        "nbo_means": train_df[nbo_cols].mean().values,
        "nbo_stds": train_df[nbo_cols].std().replace(0, 1.0).values,
        "vol_means": train_df[vol_cols].mean().values,
        "vol_stds": train_df[vol_cols].std().replace(0, 1.0).values,
    }


class DielsAlderTransformer(torch.nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim=96, heads=4, dropout=0.1):
        super().__init__()

        self.conv1 = TransformerConv(
            input_dim, hidden_dim, heads=heads, edge_dim=edge_dim, dropout=dropout
        )
        self.ln1 = LayerNorm(hidden_dim * heads)

        self.conv2 = TransformerConv(
            hidden_dim * heads, hidden_dim, heads=heads, edge_dim=edge_dim, dropout=dropout
        )
        self.ln2 = LayerNorm(hidden_dim * heads)

        self.conv3 = TransformerConv(
            hidden_dim * heads, hidden_dim, heads=heads, edge_dim=edge_dim, dropout=dropout
        )
        self.ln3 = LayerNorm(hidden_dim * heads)

        self.dropout = Dropout(dropout)
        self.mlp = Sequential(
            Linear(hidden_dim * heads * 2, hidden_dim),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_dim, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        x1 = self.conv1(x, edge_index, edge_attr)
        x1 = self.dropout(torch.relu(self.ln1(x1)))

        x2 = self.conv2(x1, edge_index, edge_attr)
        x2 = self.dropout(torch.relu(self.ln2(x2 + x1)))

        x3 = self.conv3(x2, edge_index, edge_attr)
        x3 = self.dropout(torch.relu(self.ln3(x3 + x2)))

        x_mean = global_mean_pool(x3, batch)
        x_max = global_max_pool(x3, batch)
        x = torch.cat([x_mean, x_max], dim=-1)

        return self.mlp(x)


class ConditionAwareTransformer(torch.nn.Module):
    """Wraps DielsAlderTransformer and injects reaction conditions.
    Conditions: T_norm, conc_diene_norm, conc_dienophile_norm, residence_time_norm, lewis_acid_norm.
    """

    def __init__(self, base_model: DielsAlderTransformer, condition_dim: int = 5):
        super().__init__()
        self.base = base_model

        base_in = self.base.mlp[0].in_features
        base_hidden = self.base.mlp[0].out_features
        dropout = self.base.mlp[2].p if hasattr(self.base.mlp[2], "p") else 0.1

        self.conditioned_mlp = torch.nn.Sequential(
            torch.nn.Linear(base_in + condition_dim, base_hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(base_hidden, 1),
        )

    def _pooled_graph_features(self, data):
        if not hasattr(data, "batch") or data.batch is None:
            data.batch = torch.zeros(
                data.num_nodes, dtype=torch.long, device=data.x.device
            )
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        x1 = self.base.conv1(x, edge_index, edge_attr)
        x1 = self.base.dropout(torch.relu(self.base.ln1(x1)))

        x2 = self.base.conv2(x1, edge_index, edge_attr)
        x2 = self.base.dropout(torch.relu(self.base.ln2(x2 + x1)))

        x3 = self.base.conv3(x2, edge_index, edge_attr)
        x3 = self.base.dropout(torch.relu(self.base.ln3(x3 + x2)))

        x_mean = global_mean_pool(x3, batch)
        x_max = global_max_pool(x3, batch)
        return torch.cat([x_mean, x_max], dim=-1)

    def forward(self, data, conditions):
        pooled = self._pooled_graph_features(data)
        if conditions.dim() == 1:
            conditions = conditions.unsqueeze(0)
        x = torch.cat([pooled, conditions], dim=-1)
        return self.conditioned_mlp(x)


class DielsAlderOptEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        graph_data,
        model,
        ts_mean,
        ts_std,
        device,
        max_steps=1,
    ):
        super().__init__()
        self.graph_data = graph_data.to(device)
        self.model = model
        self.ts_mean = ts_mean
        self.ts_std = ts_std
        self.device = device
        self.max_steps = max_steps
        self.steps = 0
        # Thermal: [0,1] → 20°C to solvent-dependent max (see step())
        self.temp_min_k = 293.15
        # Solvent choice: 0=DCM (39.6°C), 1=THF (66°C), 2=Toluene (110.6°C)
        self.solvent_max_temp_c = (39.6, 66.0, 110.6)
        self.temp_max_k = 273.15 + 39.6  # default DCM; updated in step() from solvent
        self.conc_min_m = 0.01
        self.conc_max_m = 5.0
        # Kinetic: [0,1] → 1 min to 24 hrs (seconds)
        self.t_min_s = 60.0
        self.t_max_s = 24.0 * 3600.0
        self.t_ref_s = 3600.0
        # Catalytic: [0,1] → 0 to 0.5 eq Lewis acid
        self.lewis_min_eq = 0.0
        self.lewis_max_eq = 0.5
        # Saturation: reduction = 5.0 * (lewis_equiv / (lewis_equiv + 0.05))
        self.lewis_saturation_kcal = 5.0
        self.lewis_saturation_half_eq = 0.05
        self.side_reaction_temp_k = 400.0
        self.side_reaction_penalty_per_k = 0.01
        self.ref_temp_k = 298.15
        self.ref_conc_m = 1.0
        self.lewis_ref_eq = 0.15
        self.reward_log_span = 5.0
        # Weighting factor for explicit ΔG‡-based reward shaping (kept modest for realism)
        self.delta_g_reward_weight = 0.25
        # Economic/efficiency: quadratic-on-excess (favor interior conditions)
        self.lambda_conc = 0.08
        self.lambda_time = 0.08
        self.lambda_lewis = 0.15
        self.penalty_toluene = 0.05
        # Retro-Diels-Alder equilibrium: ΔG° = -10 kcal/mol (exothermic)
        self.delta_G_reaction_kcal = -10.0

        self.kb_j_per_k = 1.380649e-23
        self.h_j_s = 6.62607015e-34
        self.r_kcal_per_mol_k = 1.987204258e-3
        self.rate_floor = 1e-40

        # Actions: T_norm, conc_diene_norm, conc_dienophile_norm, residence_time_norm, lewis_acid_norm, solvent_norm (binned 0→DCM, 1→THF, 2→Toluene)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)

        with torch.no_grad():
            self.model.eval()
            graph_features = self.model._pooled_graph_features(self.graph_data)
        self.observation = graph_features.detach().cpu().numpy().astype(np.float32).squeeze()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.observation.shape, dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return self.observation, {}

    def _equilibrium_fraction(self, temp_k):
        """Retro-Diels-Alder: Keq = exp(-ΔG°/(R*T)); cap yield by Keq/(1+Keq)."""
        Keq = np.exp(-self.delta_G_reaction_kcal / (self.r_kcal_per_mol_k * temp_k))
        return Keq / (1.0 + Keq)

    def _calculate_reaction_rate(self, predicted_delta_g, temp_norm, conc_diene_norm, conc_dienophile_norm):
        temp_k = self.temp_min_k + temp_norm * (self.temp_max_k - self.temp_min_k)
        conc_diene_m = self.conc_min_m + conc_diene_norm * (self.conc_max_m - self.conc_min_m)
        conc_dienophile_m = self.conc_min_m + conc_dienophile_norm * (self.conc_max_m - self.conc_min_m)

        k = (self.kb_j_per_k * temp_k / self.h_j_s) * np.exp(
            -predicted_delta_g / (self.r_kcal_per_mol_k * temp_k)
        )
        rate = k * conc_diene_m * conc_dienophile_m
        return rate, k, temp_k, conc_diene_m, conc_dienophile_m

    def _calculate_reference_rate(self, predicted_delta_g):
        k_ref = (self.kb_j_per_k * self.ref_temp_k / self.h_j_s) * np.exp(
            -predicted_delta_g / (self.r_kcal_per_mol_k * self.ref_temp_k)
        )
        rate_ref = k_ref * (self.ref_conc_m**2)
        return rate_ref, k_ref

    def _normalized_log_reward(self, rate, rate_ref):
        log_rate = np.log10(max(rate, self.rate_floor))
        log_ref = np.log10(max(rate_ref, self.rate_floor))
        return (log_rate - log_ref) / self.reward_log_span

    def _normalized_log_reward_time_weighted(self, rate, t_sec, rate_ref, t_ref_s, equilibrium_fraction=1.0):
        """Reward based on rate * time (conversion proxy), capped by equilibrium fraction."""
        product_floor = 1e-40
        product = max(rate * t_sec * equilibrium_fraction, product_floor)
        product_ref = max(rate_ref * t_ref_s, product_floor)
        return (np.log10(product) - np.log10(product_ref)) / self.reward_log_span

    def _normalized_delta_g_reward(self, delta_g_eff):
        """
        Additional shaping term that directly rewards lower effective ΔG‡.

        We convert ΔΔG‡ (relative to ts_mean) into an approximate log10(k_ratio)
        at the reference temperature via Eyring, then normalize by reward_log_span.
        This keeps the term on the same rough scale as the rate-based reward
        while modestly increasing the influence of the GNN ΔG‡ predictions.
        """
        delta_g_ref = self.ts_mean
        # Positive when the barrier is lower than the reference value
        dg_diff = delta_g_ref - delta_g_eff
        # log10(k2/k1) ≈ -ΔΔG‡ / (2.303 * R * T); sign already in dg_diff
        denom = max(2.303 * self.r_kcal_per_mol_k * self.ref_temp_k, 1e-8)
        log10_k_ratio = dg_diff / denom
        return log10_k_ratio / self.reward_log_span

    def step(self, action):
        self.steps += 1
        action = np.clip(action, 0.0, 1.0).astype(np.float32)
        temperature = float(action[0])
        conc_diene_norm = float(action[1])
        conc_dienophile_norm = float(action[2])
        residence_time_norm = float(action[3])
        lewis_acid_norm = float(action[4])
        # Bin 6th dimension to solvent: 0=DCM, 1=THF, 2=Toluene
        solvent_norm = float(action[5])
        solvent_idx = min(2, int(solvent_norm * 3))
        self.temp_max_k = 273.15 + self.solvent_max_temp_c[solvent_idx]

        t_sec = self.t_min_s + residence_time_norm * (self.t_max_s - self.t_min_s)
        lewis_equiv = self.lewis_min_eq + lewis_acid_norm * (
            self.lewis_max_eq - self.lewis_min_eq
        )

        conditions = torch.tensor(
            [
                temperature,
                conc_diene_norm,
                conc_dienophile_norm,
                residence_time_norm,
                lewis_acid_norm,
            ],
            device=self.device,
        )
        with torch.no_grad():
            pred_scaled = self.model(self.graph_data, conditions).view(-1)[0]

        predicted_delta_g = float((pred_scaled * self.ts_std) + self.ts_mean)
        # Non-linear catalyst: saturation formula (diminishing returns)
        lewis_reduction = self.lewis_saturation_kcal * (
            lewis_equiv / (lewis_equiv + self.lewis_saturation_half_eq)
        )
        delta_g_eff = predicted_delta_g - lewis_reduction

        rate, k, temp_k, conc_diene_m, conc_dienophile_m = self._calculate_reaction_rate(
            delta_g_eff, temperature, conc_diene_norm, conc_dienophile_norm
        )
        rate_ref, k_ref = self._calculate_reference_rate(predicted_delta_g)
        equilibrium_fraction = self._equilibrium_fraction(temp_k)
        base_reward = float(
            self._normalized_log_reward_time_weighted(
                rate, t_sec, rate_ref, self.t_ref_s, equilibrium_fraction
            )
        )
        # Blend the original rate/time-based reward with a mild, explicit ΔG‡ term
        delta_g_reward = float(self._normalized_delta_g_reward(delta_g_eff))
        reward = (
            (1.0 - self.delta_g_reward_weight) * base_reward
            + self.delta_g_reward_weight * delta_g_reward
        )
        if temp_k > self.side_reaction_temp_k:
            reward -= self.side_reaction_penalty_per_k * (
                temp_k - self.side_reaction_temp_k
            )
        # Quadratic-on-excess (above reference) to favor interior, economical conditions
        r_d = conc_diene_m / self.ref_conc_m
        r_ph = conc_dienophile_m / self.ref_conc_m
        reward -= self.lambda_conc * (max(0.0, r_d - 1.0) ** 2 + max(0.0, r_ph - 1.0) ** 2)
        r_t = t_sec / self.t_ref_s
        reward -= self.lambda_time * max(0.0, r_t - 1.0) ** 2
        reward -= self.lambda_lewis * (lewis_equiv**2)
        if solvent_idx == 2:
            reward -= self.penalty_toluene  # Toluene: higher energy cost of removal

        terminated = False
        truncated = self.steps >= self.max_steps
        info = {
            "predicted_delta_g": predicted_delta_g,
            "delta_g_eff": float(delta_g_eff),
            "rate": float(rate),
            "rate_ref": float(rate_ref),
            "k": float(k),
            "k_ref": float(k_ref),
            "temp_k": float(temp_k),
            "conc_diene_m": float(conc_diene_m),
            "conc_dienophile_m": float(conc_dienophile_m),
            "t_sec": float(t_sec),
            "lewis_equiv": float(lewis_equiv),
            "solvent_idx": int(solvent_idx),
            "equilibrium_fraction": float(equilibrium_fraction),
        }
        return self.observation, float(reward), terminated, truncated, info


def load_stats(data_root: Path, stats_path: str | None):
    if stats_path:
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
    else:
        stats = {}

    train_path = data_root / "data" / "train.csv"
    if "ts_mean" not in stats or "ts_std" not in stats or "feature_stats" not in stats:
        df_training = pd.read_csv(train_path)
        if "Min_TSـall" not in df_training.columns:
            raise ValueError("Min_TSـall column not found in train.csv.")

        stats.setdefault("ts_mean", float(df_training["Min_TSـall"].mean()))
        stats.setdefault("ts_std", float(df_training["Min_TSـall"].std()))
        stats.setdefault("feature_stats", build_feature_stats(df_training))

    return stats


def build_graph_from_row(row, feature_stats: dict, node_dim: int | None = None, edge_dim: int | None = None):
    """Build reaction graph. Uses impl2 format (15 node, 5 edge) when model expects it."""
    if node_dim == 15 and edge_dim == 5:
        return create_reaction_graph_from_row_impl2(row, feature_stats)
    return create_reaction_graph_from_row(row, feature_stats)


def plot_optimization_landscape(env, steps=12):
    """Plot reward over 3D slice (temp, conc_diene, conc_dienophile); t_norm=0.5, lewis_norm=0, solvent=DCM."""
    temps = np.linspace(0.0, 1.0, steps)
    conc_diene = np.linspace(0.0, 1.0, steps)
    conc_dienophile = np.linspace(0.0, 1.0, steps)
    tt, cd, cph = np.meshgrid(temps, conc_diene, conc_dienophile, indexing="ij")
    tt = tt.ravel()
    cd = cd.ravel()
    cph = cph.ravel()
    rewards = np.zeros(len(tt), dtype=float)
    t_norm_fixed = 0.5
    lewis_norm_fixed = 0.0
    env.temp_max_k = 273.15 + env.solvent_max_temp_c[0]  # DCM for plot

    env.model.eval()
    with torch.no_grad():
        for i in range(len(tt)):
            t, c_d, c_p = float(tt[i]), float(cd[i]), float(cph[i])
            conditions = torch.tensor(
                [t, c_d, c_p, t_norm_fixed, lewis_norm_fixed], device=env.device
            )
            pred_scaled = env.model(env.graph_data, conditions).view(-1)[0]
            predicted_delta_g = float((pred_scaled * env.ts_std) + env.ts_mean)
            lewis_equiv = env.lewis_min_eq + lewis_norm_fixed * (
                env.lewis_max_eq - env.lewis_min_eq
            )
            lewis_reduction = env.lewis_saturation_kcal * (
                lewis_equiv / (lewis_equiv + env.lewis_saturation_half_eq)
            )
            delta_g_eff = predicted_delta_g - lewis_reduction
            rate, _, temp_k, conc_diene_m, conc_dienophile_m = env._calculate_reaction_rate(
                delta_g_eff, t, c_d, c_p
            )
            rate_ref, _ = env._calculate_reference_rate(predicted_delta_g)
            t_sec = env.t_min_s + t_norm_fixed * (env.t_max_s - env.t_min_s)
            eq_frac = env._equilibrium_fraction(temp_k)
            r = float(
                env._normalized_log_reward_time_weighted(
                    rate, t_sec, rate_ref, env.t_ref_s, eq_frac
                )
            )
            if temp_k > env.side_reaction_temp_k:
                r -= env.side_reaction_penalty_per_k * (
                    temp_k - env.side_reaction_temp_k
                )
            r_d = conc_diene_m / env.ref_conc_m
            r_ph = conc_dienophile_m / env.ref_conc_m
            r -= env.lambda_conc * (max(0.0, r_d - 1.0) ** 2 + max(0.0, r_ph - 1.0) ** 2)
            r_t = t_sec / env.t_ref_s
            r -= env.lambda_time * max(0.0, r_t - 1.0) ** 2
            r -= env.lambda_lewis * (lewis_equiv**2)
            rewards[i] = r

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(tt, cd, cph, c=rewards, cmap="viridis", alpha=0.7, s=15)
    plt.colorbar(sc, ax=ax, label="Reward")
    ax.set_xlabel("Temperature (scaled)")
    ax.set_ylabel("Conc diene (scaled)")
    ax.set_zlabel("Conc dienophile (scaled)")
    ax.set_title(
        "Optimization Landscape (3D slice; t_norm=0.5, lewis=0; color = reward)"
    )
    plt.tight_layout()
    plt.show()


def run_rl_optimization(env, algorithm="ppo", total_timesteps=5000, **algo_kwargs):
    """
    Train an RL agent on the given env using the specified algorithm.

    Supported algorithms (require stable-baselines3): ppo, a2c, sac, td3.
    All use MlpPolicy and are suitable for continuous Box action spaces.

    Returns:
        The trained SB3 BaseAlgorithm (e.g. PPO, SAC) with .predict(obs, deterministic=...).
    """
    log_dir = f"./logs/{algorithm}/"
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)
    if not _SB3_AVAILABLE:
        raise ImportError("stable-baselines3 is required for RL optimization.")
    algo_key = algorithm.lower().strip()
    if algo_key not in RL_ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. Choose from: {list(RL_ALGORITHMS.keys())}"
        )
    AlgoClass, defaults = RL_ALGORITHMS[algo_key]
    kwargs = {**defaults, **algo_kwargs}
    policy = kwargs.pop("policy", "MlpPolicy")
    verbose = kwargs.pop("verbose", 1)
    model = AlgoClass(policy, env, verbose=verbose, **kwargs)
    model.learn(total_timesteps=total_timesteps)
    return model


def run_ppo_optimization(env, total_timesteps=5000):
    """Convenience wrapper: train using PPO (same as run_rl_optimization(env, algorithm='ppo', ...))."""
    return run_rl_optimization(env, algorithm="ppo", total_timesteps=total_timesteps)


def _infer_edge_dim_from_state(state_dict) -> int | None:
    for key in ("conv1.lin_edge.weight", "conv2.lin_edge.weight", "conv3.lin_edge.weight"):
        if key in state_dict:
            return int(state_dict[key].shape[1])
    return None


def _infer_node_dim_from_state(state_dict) -> int | None:
    for key in (
        "conv1.lin_key.weight",
        "conv1.lin_query.weight",
        "conv1.lin_value.weight",
        "conv1.lin_skip.weight",
    ):
        if key in state_dict:
            return int(state_dict[key].shape[1])
    return None


def _match_edge_attr_dim(graph_data, edge_dim: int):
    current_dim = graph_data.edge_attr.size(1)
    if current_dim == edge_dim:
        return graph_data
    if current_dim < edge_dim:
        pad = edge_dim - current_dim
        padding = torch.zeros(
            graph_data.edge_attr.size(0), pad, device=graph_data.edge_attr.device
        )
        graph_data.edge_attr = torch.cat([graph_data.edge_attr, padding], dim=1)
    else:
        graph_data.edge_attr = graph_data.edge_attr[:, :edge_dim]
    return graph_data


def _match_node_attr_dim(graph_data, node_dim: int):
    current_dim = graph_data.x.size(1)
    if current_dim == node_dim:
        return graph_data
    if current_dim < node_dim:
        pad = node_dim - current_dim
        padding = torch.zeros(
            graph_data.x.size(0), pad, device=graph_data.x.device
        )
        graph_data.x = torch.cat([graph_data.x, padding], dim=1)
    else:
        graph_data.x = graph_data.x[:, :node_dim]
    return graph_data


def _resolve_run_paths(data_root: Path, model_path: str | None, stats_path: str | None):
    if model_path and stats_path:
        return model_path, stats_path

    outputs_dir = data_root / "outputs"
    if not outputs_dir.exists():
        return model_path, stats_path

    runs = [p for p in outputs_dir.iterdir() if p.is_dir() and p.name.startswith("run-")]
    if not runs:
        return model_path, stats_path

    latest = max(runs, key=lambda p: p.name)
    inferred_model = str((latest / "model.pt").resolve())
    inferred_stats = str((latest / "stats.json").resolve())

    return model_path or inferred_model, stats_path or inferred_stats


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize Diels-Alder conditions.")
    parser.add_argument("--data-root", default="deepchem", help="Path to deepchem folder.")
    parser.add_argument("--model-path", default=None, help="Path to model.pt")
    parser.add_argument("--stats-path", default=None, help="Path to stats.json")
    parser.add_argument(
        "--data-file",
        default=DEFAULT_DATA_FILE,
        help="CSV to optimize reactions from (relative to data-root).",
    )
    parser.add_argument("--row-idx", type=int, default=DEFAULT_ROW_IDX)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        choices=["ppo", "a2c", "sac", "td3"],
        help=f"RL algorithm (default: use RL_ALGORITHM in file, currently {RL_ALGORITHM!r}).",
    )
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()

def plot_training_curve(log_dir, algo_name):
    results = load_results(log_dir)
    x, y = ts2xy(results, 'timesteps')
    window = min(1000, max(1, len(y) // 50))
    y_smooth = np.convolve(y, np.ones(window) / window, mode='valid')
    x_smooth = x[window - 1 :]
    plt.figure(figsize=(8, 4))
    # Plot raw rewards with low opacity for context
    plt.plot(x, y, color="C0", alpha=0.2, linewidth=1)
    # Plot smoothed rewards as the main curve
    plt.plot(x_smooth, y_smooth, color="C0", linewidth=2)
    # Use a fixed y-scale so different algorithms are easy to compare
    plt.ylim(-50, 5)
    plt.title(f"Learning Curve: {algo_name.upper()}")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.show()

def plot_action_radar(action, algo_name):
    labels = ['Temp', 'Conc D', 'Conc dPh', 'Time', 'Lewis Acid', 'Solvent']
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    stats = np.concatenate((action, [action[0]]))
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.fill(angles, stats, alpha=0.3)
    ax.set_xticklabels(labels)
    plt.title(f"Condition Profile: {algo_name.upper()}")
    plt.show()


def main():
    args = parse_args()
    data_root = Path(args.data_root)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model_path, stats_path = _resolve_run_paths(
        data_root, args.model_path, args.stats_path
    )

    stats = load_stats(data_root, stats_path)
    ts_mean = float(stats["ts_mean"])
    ts_std = float(stats["ts_std"])
    feature_stats = stats["feature_stats"]

    data_path = data_root / args.data_file
    data_df = pd.read_csv(data_path)
    if args.row_idx < 0 or args.row_idx >= len(data_df):
        raise IndexError("row-idx is out of range for selected data file.")
    row = data_df.iloc[args.row_idx]

    state_dict = None
    expected_edge_dim = None
    expected_node_dim = None
    if model_path and os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        expected_edge_dim = _infer_edge_dim_from_state(state_dict)
        expected_node_dim = _infer_node_dim_from_state(state_dict)

    graph_data = build_graph_from_row(row, feature_stats, expected_node_dim, expected_edge_dim)
    graph_data = graph_data.to(device)

    edge_dim = graph_data.edge_attr.size(1)
    node_dim = graph_data.x.size(1)
    graph_data = _match_node_attr_dim(graph_data, node_dim)
    graph_data = _match_edge_attr_dim(graph_data, edge_dim)

    base_model = DielsAlderTransformer(input_dim=node_dim, edge_dim=edge_dim).to(device)
    if state_dict is not None:
        base_model.load_state_dict(state_dict)

    model = ConditionAwareTransformer(base_model).to(device)

    env = DielsAlderOptEnv(
        graph_data=graph_data,
        model=model,
        ts_mean=ts_mean,
        ts_std=ts_std,
        device=device,
        max_steps=args.max_steps,
    )

    algorithm = args.algorithm if args.algorithm is not None else RL_ALGORITHM
    total_timesteps = RL_TIMESTEPS[algorithm]
    rl_model = run_rl_optimization(
        env, algorithm=algorithm, total_timesteps=total_timesteps
    )
    save_dir = data_root / "trainedmodels"
    save_dir.mkdir(parents=True, exist_ok=True)
    model_name = Path(model_path).parent.name if model_path else "default"
    save_path = save_dir / f"{model_name}_{algorithm}_row{args.row_idx}"
    rl_model.save(str(save_path))
    obs, _ = env.reset(seed=random.seed(42))
    action, _ = rl_model.predict(obs, deterministic=True)
    _, reward, _, _, info = env.step(action)
    diene_id = row["diene"]
    dienophile_id = row["dienophile"]
    diene_smi = id_to_smiles(diene_id, is_diene=True)
    dienophile_smi = id_to_smiles(dienophile_id, is_diene=False)
    print(
        f"Data file: {data_path} | row idx {args.row_idx} | "
        f"diene={diene_id} ({diene_smi}), "
        f"dienophile={dienophile_id} ({dienophile_smi})"
    )
    solvent_names = ("DCM", "THF", "Toluene")
    solvent_idx = info.get("solvent_idx", 0)
    print(
        f"Best action (T_norm, C_diene_norm, C_dienophile_norm, t_norm, lewis_norm, solvent_norm): {action}, "
        f"solvent={solvent_names[solvent_idx]}, reward: {reward:.4f}, info: {info}"
    )

    if args.plot:
        #plot_optimization_landscape(env)
        plot_training_curve(f"./logs/{algorithm}/", algorithm)
        plot_action_radar(action, algorithm)


if __name__ == "__main__":
    main()
