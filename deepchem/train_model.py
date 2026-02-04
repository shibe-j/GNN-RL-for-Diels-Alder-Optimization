import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.nn import Dropout, Linear, ReLU, Sequential
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import LayerNorm, TransformerConv, global_max_pool, global_mean_pool

TARGET_COL = "Min_TSـall"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def id_to_smiles(id_str, is_diene: bool = True) -> str:
    lookup = {
        "1": "F",
        "2": "C#N",
        "3": "OC",
        "4": "C",
        "5": "C(C)(C)C",
        "6": "",  # Hydrogen (implicit)
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
        smiles = f"C{c1}=C{c2}C{c3}=C{c4}"
    else:
        g1, g2 = groups
        c1 = f"({g1})" if g1 else ""
        c2 = f"({g2})" if g2 else ""
        smiles = f"C{c1}=C{c2}"

    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Chem.MolToSmiles(mol)
    return smiles


def _normalize_list(values, means, stds):
    values = np.array(values, dtype=float)
    stds = np.where(stds == 0, 1.0, stds)
    return ((values - means) / stds).tolist()


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
        x1 = self.dropout(F.relu(self.ln1(x1)))

        x2 = self.conv2(x1, edge_index, edge_attr)
        x2 = self.dropout(F.relu(self.ln2(x2 + x1)))

        x3 = self.conv3(x2, edge_index, edge_attr)
        x3 = self.dropout(F.relu(self.ln3(x3 + x2)))

        x_mean = global_mean_pool(x3, batch)
        x_max = global_max_pool(x3, batch)
        x = torch.cat([x_mean, x_max], dim=-1)

        return self.mlp(x)


def train_epoch(model, train_loader, optimizer, device, loss_fn):
    model.train()
    total_loss = 0.0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        prediction = model(data)
        loss = loss_fn(prediction.view(-1), data.y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_loader.dataset)


def eval_epoch(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            prediction = model(data)
            loss = loss_fn(prediction.view(-1), data.y.view(-1))
            total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


def evaluate_loader(model, loader, device, ts_mean, ts_std):
    model.eval()
    all_preds = []
    all_actuals = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            pred_kcal = (out.cpu().numpy().flatten() * ts_std) + ts_mean
            actual_kcal = (data.y.cpu().numpy().flatten() * ts_std) + ts_mean
            all_preds.extend(pred_kcal)
            all_actuals.extend(actual_kcal)

    mae = mean_absolute_error(all_actuals, all_preds)
    rmse = np.sqrt(mean_squared_error(all_actuals, all_preds))
    r2 = r2_score(all_actuals, all_preds)
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "n": len(all_actuals),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train Diels-Alder GNN model.")
    parser.add_argument("--data-root", default=".", help="Path to deepchem folder.")
    parser.add_argument("--train-file", default="data/train.csv")
    parser.add_argument("--eval-file", default="data/evaluate.csv")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    train_path = data_root / args.train_file
    eval_path = data_root / args.eval_file

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = data_root / "outputs" / f"run-{run_id}"

    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    df_training = pd.read_csv(train_path)
    df_evaluation = pd.read_csv(eval_path)

    train_df, val_df = train_test_split(
        df_training, test_size=0.2, random_state=args.seed, shuffle=True
    )

    ts_mean = train_df[TARGET_COL].mean()
    ts_std = train_df[TARGET_COL].std()
    feature_stats = build_feature_stats(train_df)

    for frame in (train_df, val_df, df_evaluation):
        frame["diene_smiles"] = frame["diene"].apply(lambda x: id_to_smiles(x, True))
        frame["dienophile_smiles"] = frame["dienophile"].apply(lambda x: id_to_smiles(x, False))
        frame["Min_TS_scaled"] = (frame[TARGET_COL] - ts_mean) / ts_std

    train_dataset = []
    for _, row in train_df.iterrows():
        graph = create_reaction_graph_from_row(row, feature_stats)
        graph.y = torch.tensor([row["Min_TS_scaled"]], dtype=torch.float)
        train_dataset.append(graph)

    val_dataset = []
    for _, row in val_df.iterrows():
        graph = create_reaction_graph_from_row(row, feature_stats)
        graph.y = torch.tensor([row["Min_TS_scaled"]], dtype=torch.float)
        val_dataset.append(graph)

    eval_dataset = []
    for _, row in df_evaluation.iterrows():
        graph = create_reaction_graph_from_row(row, feature_stats)
        graph.y = torch.tensor([row["Min_TS_scaled"]], dtype=torch.float)
        eval_dataset.append(graph)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = DielsAlderTransformer(
        input_dim=5,
        edge_dim=3,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-5, verbose=True
    )
    loss_fn = torch.nn.SmoothL1Loss()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False)
    eval_loader = DataLoader(eval_dataset, batch_size=args.val_batch_size, shuffle=False)

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = []

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, loss_fn)
        val_loss = eval_epoch(model, val_loader, device, loss_fn)
        scheduler.step(val_loss)

        improved = val_loss < best_val - 1e-4
        if improved:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )

        print(
            f"Epoch {epoch:03d}, Train: {train_loss:.6f}, Val: {val_loss:.6f}, "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if epochs_no_improve >= args.patience:
            print("Early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_metrics = evaluate_loader(model, val_loader, device, ts_mean, ts_std)
    external_metrics = evaluate_loader(model, eval_loader, device, ts_mean, ts_std)

    metrics = {
        "val": val_metrics,
        "external": external_metrics,
        "best_val_loss": float(best_val),
        "epochs_ran": len(history),
    }

    torch.save(model.state_dict(), output_dir / "model.pt")
    with open(output_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "ts_mean": float(ts_mean),
                "ts_std": float(ts_std),
                "feature_stats": {k: v.tolist() for k, v in feature_stats.items()},
            },
            f,
            indent=2,
        )
    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Validation metrics:", val_metrics)
    print("External metrics:", external_metrics)
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
