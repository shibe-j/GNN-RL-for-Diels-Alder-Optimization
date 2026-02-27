# %%
import numpy as np
import pandas as pd
from rdkit import Chem
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

# %% [markdown]
# #### Loads data from files generated in data.ipynb

# %%
df_main = pd.read_csv("data_mingap.csv")
df_training = pd.read_csv("data/train.csv")
df_evaluation = pd.read_csv("data/evaluate.csv")

# Base training dataframe
# (We'll create a validation split for early stopping / model selection.)
df = df_training
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# Target stats (train only)
ts_mean = train_df["Min_TSـall"].mean()
ts_std = train_df["Min_TSـall"].std()

# Feature stats (train only) for normalization
pz_cols = ["pz_pop_C1_D", "pz_pop_C2_D", "pz_pop_C3_D", "pz_pop_C4_D", "pz_pop_C1_dPh", "pz_pop_C2_dPh"]
nbo_cols = ["NBO_1_D", "NBO_2_D", "NBO_3_D", "NBO_4_D", "NBO_1dPh", "NBO_2_dPh"]
vol_cols = ["Volume_D_1", "Volume_D_2", "Volume_D_3", "Volume_D_4", "Volume_dPh_1", "Volume_dPh_2"]

pz_means = train_df[pz_cols].mean().values
pz_stds = train_df[pz_cols].std().replace(0, 1.0).values
nbo_means = train_df[nbo_cols].mean().values
nbo_stds = train_df[nbo_cols].std().replace(0, 1.0).values
vol_means = train_df[vol_cols].mean().values
vol_stds = train_df[vol_cols].std().replace(0, 1.0).values

feature_stats = {
    "pz_means": pz_means,
    "pz_stds": pz_stds,
    "nbo_means": nbo_means,
    "nbo_stds": nbo_stds,
    "vol_means": vol_means,
    "vol_stds": vol_stds,
}

# %% [markdown]
# #### Adds smiles strings to laoded dataframes and normalizes (again) using the Z score

# %%
def id_to_smiles(id_str, is_diene=True, mapped=False):
    # Mapping based on the lookup table provided
    lookup = {
        '1': 'F',
        '2': 'C#N',
        '3': 'OC',
        '4': 'C',
        '5': 'C(C)(C)C',
        '6': '', # Hydrogen (implicit)
        '7': 'c1ccccc1',
        '8': 'C(=O)OC',
        '9': 'C=O'
    }
    
    parts = str(id_str).split('_')
    groups = [lookup.get(p, '') for p in parts]

    if is_diene:
        # Scaffold: C1=C2-C3=C4
        g1, g2, g3, g4 = groups
        
        # Build pieces; handle empty strings (hydrogens) gracefully
        c1 = f"({g1})" if g1 else ""
        c2 = f"({g2})" if g2 else ""
        c3 = f"({g3})" if g3 else ""
        c4 = f"({g4})" if g4 else ""
        
        if mapped:
            smiles = f"[C:1]{c1}=[C:2]{c2}[C:3]{c3}=[C:4]{c4}"
        else:
            smiles = f"C{c1}=C{c2}C{c3}=C{c4}"
    else:
        # Scaffold: C1=C2
        g1, g2 = groups
        c1 = f"({g1})" if g1 else ""
        c2 = f"({g2})" if g2 else ""
        if mapped:
            smiles = f"[C:5]{c1}=[C:6]{c2}"
        else:
            smiles = f"C{c1}=C{c2}"

    # Canonicalize to ensure the SMILES is clean and valid
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Chem.MolToSmiles(mol)
    return smiles # Return raw if RDKit fails

# Add SMILES columns to train/val/eval
for frame in (train_df, val_df, df_evaluation):
    frame['diene_smiles'] = frame['diene'].apply(lambda x: id_to_smiles(x, is_diene=True))
    frame['dienophile_smiles'] = frame['dienophile'].apply(lambda x: id_to_smiles(x, is_diene=False))

# Standardize target with train stats
train_df["Min_TS_scaled"] = (train_df["Min_TSـall"] - ts_mean) / ts_std
val_df["Min_TS_scaled"] = (val_df["Min_TSـall"] - ts_mean) / ts_std
df_evaluation["Min_TS_scaled"] = (df_evaluation["Min_TSـall"] - ts_mean) / ts_std

# %% [markdown]
# #### Creates datasets for both training/evaluation of pyg objects

# %%
def _normalize_list(values, means, stds):
    values = np.array(values, dtype=float)
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


def mol_to_pyg_data(smiles, pz_map, nbo_map, vol_map):
    """
    Converts a molecule and its CSV electronic data into a PyG Data object.
    """
    mol = Chem.MolFromSmiles(smiles)
    frag_atom_idxs = Chem.GetMolFrags(mol, asMols=False)
    atom_to_frag = {}
    for frag_id, atom_ids in enumerate(frag_atom_idxs):
        for atom_id in atom_ids:
            atom_to_frag[atom_id] = frag_id

    # 1. Node Features (x)
    node_features = []
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        atom_map = atom.GetAtomMapNum()
        frag_id = atom_to_frag.get(atom_idx, 0)

        # Atomic features (scaled to similar ranges)
        atomic_num = atom.GetAtomicNum() / 20.0
        degree = atom.GetDegree() / 4.0
        formal_charge = atom.GetFormalCharge()
        aromatic = 1.0 if atom.GetIsAromatic() else 0.0
        ring = 1.0 if atom.IsInRing() else 0.0
        num_hs = atom.GetTotalNumHs() / 4.0
        mass = atom.GetMass() / 200.0
        hyb = _hybridization_onehot(atom)

        # Electronic features tied to mapped core atoms (1-4 diene, 5-6 dienophile)
        pz = pz_map.get(atom_map, 0.0)
        nbo = nbo_map.get(atom_map, 0.0)
        vol = vol_map.get(atom_map, 0.0)

        # Fragment indicator (0 diene, 1 dienophile)
        frag_flag = 0.0 if frag_id == 0 else 1.0

        node_features.append([
            atomic_num, degree, formal_charge, aromatic, ring, num_hs, mass,
            *hyb, pz, nbo, vol, frag_flag
        ])
    
    x = torch.tensor(node_features, dtype=torch.float)

    # 2. Edge Index (Adjacency)
    edge_indices = []
    edge_attrs = []
    
    bond_type_map = {
        Chem.rdchem.BondType.SINGLE: [1, 0, 0],
        Chem.rdchem.BondType.DOUBLE: [0, 1, 0],
        Chem.rdchem.BondType.AROMATIC: [0, 0, 1]
    }

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        # Get one-hot bond type + conjugation + ring flag
        b_type = bond_type_map.get(bond.GetBondType(), [0, 0, 0])
        conj = 1.0 if bond.GetIsConjugated() else 0.0
        ring = 1.0 if bond.IsInRing() else 0.0
        b_feat = b_type + [conj, ring]
        
        # Bidirectional edges for GNN
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [b_feat, b_feat]

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def create_reaction_graph_from_row(row):
    diene_smi = id_to_smiles(row['diene'], is_diene=True, mapped=True)
    dienophile_smi = id_to_smiles(row['dienophile'], is_diene=False, mapped=True)
    combined_smi = f"{diene_smi}.{dienophile_smi}"
    
    # 2. Combine Core Carbon features using your actual headers: dPh
    pz_list = [row['pz_pop_C1_D'], row['pz_pop_C2_D'], row['pz_pop_C3_D'], row['pz_pop_C4_D'], 
               row['pz_pop_C1_dPh'], row['pz_pop_C2_dPh']]
    
    # Based on headers: NBO_1_D... and NBO_1dPh / NBO_2_dPh
    nbo_list = [row['NBO_1_D'], row['NBO_2_D'], row['NBO_3_D'], row['NBO_4_D'], 
                row['NBO_1dPh'], row['NBO_2_dPh']]
    
    # Based on headers: Volume_D_1... and Volume_dPh_1...
    vol_list = [row['Volume_D_1'], row['Volume_D_2'], row['Volume_D_3'], row['Volume_D_4'], 
                row['Volume_dPh_1'], row['Volume_dPh_2']]

    # Normalize electronic features with train stats
    pz_list = _normalize_list(pz_list, feature_stats["pz_means"], feature_stats["pz_stds"])
    nbo_list = _normalize_list(nbo_list, feature_stats["nbo_means"], feature_stats["nbo_stds"])
    vol_list = _normalize_list(vol_list, feature_stats["vol_means"], feature_stats["vol_stds"])

    pz_map = {1: pz_list[0], 2: pz_list[1], 3: pz_list[2], 4: pz_list[3], 5: pz_list[4], 6: pz_list[5]}
    nbo_map = {1: nbo_list[0], 2: nbo_list[1], 3: nbo_list[2], 4: nbo_list[3], 5: nbo_list[4], 6: nbo_list[5]}
    vol_map = {1: vol_list[0], 2: vol_list[1], 3: vol_list[2], 4: vol_list[3], 5: vol_list[4], 6: vol_list[5]}
    
    graph = mol_to_pyg_data(combined_smi, pz_map, nbo_map, vol_map)
    graph.y = torch.tensor([row['Min_TS_scaled']], dtype=torch.float)
    return graph

train_dataset = [create_reaction_graph_from_row(row) for _, row in train_df.iterrows()]
val_dataset = [create_reaction_graph_from_row(row) for _, row in val_df.iterrows()]
evaluation_dataset = [create_reaction_graph_from_row(row) for _, row in df_evaluation.iterrows()]

print(f"Standardized Mean (train): {train_df['Min_TS_scaled'].mean():.2f}") # Should be 0.0
print(f"Standardized Std  (train): {train_df['Min_TS_scaled'].std():.2f}")  # Should be 1.0

# test
create_reaction_graph_from_row(train_df.iloc[0])

# %% [markdown]
# #### Transformer Class

# %%
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, Dropout
from torch_geometric.nn import TransformerConv, global_mean_pool, global_max_pool, LayerNorm

#best hidden dim = 96 so far
#best heads = 4
class DielsAlderTransformer(torch.nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim=96, heads=4, dropout=0.1):
        super(DielsAlderTransformer, self).__init__()
        
        self.conv1 = TransformerConv(input_dim, hidden_dim, heads=heads, edge_dim=edge_dim, dropout=dropout)
        self.ln1 = LayerNorm(hidden_dim * heads)
        
        self.conv2 = TransformerConv(hidden_dim * heads, hidden_dim, heads=heads, edge_dim=edge_dim, dropout=dropout)
        self.ln2 = LayerNorm(hidden_dim * heads)

        self.conv3 = TransformerConv(hidden_dim * heads, hidden_dim, heads=heads, edge_dim=edge_dim, dropout=dropout)
        self.ln3 = LayerNorm(hidden_dim * heads)

        self.dropout = Dropout(dropout)

        # Output MLP (mean + max pooled graph embedding)
        self.mlp = Sequential(
            Linear(hidden_dim * heads * 2, hidden_dim),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Layer 1
        x1 = self.conv1(x, edge_index, edge_attr)
        x1 = self.dropout(F.relu(self.ln1(x1)))
        
        # Layer 2 (residual)
        x2 = self.conv2(x1, edge_index, edge_attr)
        x2 = self.dropout(F.relu(self.ln2(x2 + x1)))
        
        # Layer 3 (residual)
        x3 = self.conv3(x2, edge_index, edge_attr)
        x3 = self.dropout(F.relu(self.ln3(x3 + x2)))

        # Pooling over both diene and dienophile atoms
        x_mean = global_mean_pool(x3, batch)
        x_max = global_max_pool(x3, batch)
        x = torch.cat([x_mean, x_max], dim=-1)
        
        return self.mlp(x)

# %% [markdown]
# #### Model Training

# %%
import random
from torch_geometric.loader import DataLoader

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


def train_epoch(model, train_loader, optimizer, device, loss_fn):
    model.train()
    total_loss = 0
    
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        prediction = model(data)
        
        # target (y) should be (batch_size, 1)
        loss = loss_fn(prediction.view(-1), data.y.view(-1))
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        
    return total_loss / len(train_loader.dataset)


def eval_epoch(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            prediction = model(data)
            loss = loss_fn(prediction.view(-1), data.y.view(-1))
            total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
model = DielsAlderTransformer(input_dim=15, edge_dim=5).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-5, verbose=True
)

loss_fn = torch.nn.SmoothL1Loss()

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

best_val = float("inf")
best_state = None
patience = 25
epochs_no_improve = 0

train_losses = []
val_losses = []
lr_history = []

for epoch in range(300):
    train_loss = train_epoch(model, train_loader, optimizer, device, loss_fn)
    val_loss = eval_epoch(model, val_loader, device, loss_fn)
    scheduler.step(val_loss)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    lr_history.append(optimizer.param_groups[0]["lr"])

    if val_loss < best_val - 1e-4:
        best_val = val_loss
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    print(f"Epoch {epoch:03d}, Train: {train_loss:.6f}, Val: {val_loss:.6f}")

    if epochs_no_improve >= patience:
        print("Early stopping triggered.")
        break

if best_state is not None:
    model.load_state_dict(best_state)

# %% [markdown]
# #### Model Evaluation

# %%
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_on_test(model, test_loader, ts_mean, ts_std, label="Test"):
    model.eval()
    all_preds = []
    all_actuals = []
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            # Unscale the predictions and targets back to kcal/mol
            pred_kcal = (out.cpu().numpy().flatten() * ts_std) + ts_mean
            actual_kcal = (data.y.cpu().numpy().flatten() * ts_std) + ts_mean
            
            all_preds.extend(pred_kcal)
            all_actuals.extend(actual_kcal)

    mae = mean_absolute_error(all_actuals, all_preds)
    rmse = np.sqrt(mean_squared_error(all_actuals, all_preds))
    r2 = r2_score(all_actuals, all_preds)

    print(f"{label} MAE:  {mae:.2f} kcal/mol")
    print(f"{label} RMSE: {rmse:.2f} kcal/mol")
    print(f"{label} R2:   {r2:.4f}")
    
    return all_actuals, all_preds

def evaluate_external_dataset(model, ts_mean, ts_std):
    # 1. Load the new 200-reaction dataset
    df_new = df_evaluation
    
    # 2. Convert to PyG Graphs (Using your existing graph creation function)
    # Important: Apply the OLD ts_mean and ts_std for scaling
    new_graphs = []
    for _, row in df_new.iterrows():
        graph = create_reaction_graph_from_row(row)
        
        # Scale the new targets using the original training stats
        scaled_y = (row['Min_TSـall'] - ts_mean) / ts_std
        graph.y = torch.tensor([scaled_y], dtype=torch.float)
        new_graphs.append(graph)
    
    # 3. Create Loader
    external_loader = DataLoader(new_graphs, batch_size=32, shuffle=False)
    
    # 4. Run Evaluation
    print("--- Evaluating External Dataset ---")
    actuals, predictions = evaluate_on_test(model, external_loader, ts_mean, ts_std, label="External")
    
    return actuals, predictions

# Execute on validation and external dataset
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
val_actuals, val_preds = evaluate_on_test(model, val_loader, ts_mean, ts_std, label="Validation")
actuals_200, preds_200 = evaluate_external_dataset(model, ts_mean, ts_std)

# %%
from datetime import datetime
from pathlib import Path
import json
import torch

run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
output_dir = Path("deepchem") / "outputs" / f"run-{run_id}"
output_dir.mkdir(parents=True, exist_ok=True)

torch.save(model.state_dict(), output_dir / "model.pt")

stats = {
    "ts_mean": float(ts_mean),
    "ts_std": float(ts_std),
}
if "feature_stats" in globals():
    stats["feature_stats"] = {k: v.tolist() for k, v in feature_stats.items()}

with open(output_dir / "stats.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)

print(f"Saved model + stats to: {output_dir}")

# %%
import numpy as np
import torch

if "device" not in globals():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.eval()

sample_count = 5
sample_idx = np.random.choice(len(train_df), size=sample_count, replace=False)
print("Sample predicted ΔG‡ (kcal/mol):")
for idx in sample_idx:
    row = train_df.iloc[idx]
    graph = create_reaction_graph_from_row(row)
    if not hasattr(graph, "batch") or graph.batch is None:
        graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    graph = graph.to(device)
    with torch.no_grad():
        pred_scaled = model(graph).view(-1)[0].item()
    pred_kcal = (pred_scaled * ts_std) + ts_mean
    actual_kcal = row["Min_TSـall"]
    print(f"idx={idx} | pred={pred_kcal:.2f} | actual={actual_kcal:.2f}")

# %% [markdown]
# #### Training Curves and Evaluation Plots

# %%
import matplotlib.pyplot as plt
import numpy as np

# --- Training loss and learning rate history ---
if 'train_losses' in globals() and 'val_losses' in globals():
    epochs = np.arange(1, len(train_losses) + 1)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color_train = 'tab:blue'
    color_val = 'tab:orange'

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color_train)
    ax1.plot(epochs, train_losses, label='Train loss', color=color_train)
    ax1.plot(epochs, val_losses, label='Val loss', color=color_val)
    ax1.tick_params(axis='y', labelcolor=color_train)
    ax1.legend(loc='upper right')

    if 'lr_history' in globals() and len(lr_history) == len(epochs):
        ax2 = ax1.twinx()
        ax2.set_ylabel('Learning rate', color='tab:green')
        ax2.plot(epochs, lr_history, label='LR', color='tab:green', linestyle='--', alpha=0.7)
        ax2.tick_params(axis='y', labelcolor='tab:green')

    plt.title('GNN Training and Validation Loss')
    plt.tight_layout()
    plt.show()
else:
    print('Run the training cell first to populate train_losses and val_losses.')

# --- Evaluation: predicted vs actual (validation and external) ---
if 'val_actuals' in globals() and 'val_preds' in globals():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Validation set
    ax = axes[0]
    ax.scatter(val_actuals, val_preds, alpha=0.6)
    diag_min = min(min(val_actuals), min(val_preds))
    diag_max = max(max(val_actuals), max(val_preds))
    ax.plot([diag_min, diag_max], [diag_min, diag_max], 'k--', linewidth=1)
    ax.set_xlabel('Actual ΔG‡ (kcal/mol)')
    ax.set_ylabel('Predicted ΔG‡ (kcal/mol)')
    ax.set_title('Validation: Predicted vs Actual')

    # External / 200-reaction dataset
    if 'actuals_200' in globals() and 'preds_200' in globals():
        ax = axes[1]
        ax.scatter(actuals_200, preds_200, alpha=0.6, color='tab:orange')
        diag_min_ext = min(min(actuals_200), min(preds_200))
        diag_max_ext = max(max(actuals_200), max(preds_200))
        ax.plot([diag_min_ext, diag_max_ext], [diag_min_ext, diag_max_ext], 'k--', linewidth=1)
        ax.set_xlabel('Actual ΔG‡ (kcal/mol)')
        ax.set_ylabel('Predicted ΔG‡ (kcal/mol)')
        ax.set_title('External: Predicted vs Actual')
    else:
        axes[1].axis('off')

    plt.tight_layout()
    plt.show()
else:
    print('Run the evaluation cell first to populate val_actuals/val_preds and actuals_200/preds_200.')


