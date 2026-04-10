"""
test.py
=======

Run inference with a trained GNN model (produced by train.py) on a folder
of PDB files.

Pipeline
--------
1. Extract features (global + graph) from test PDBs via extract_all().
2. Load the saved model bundle (state_dict + scaler_state + config).
3. Reconstruct the StandardScaler from the saved mean_ / scale_ and apply
   it to the global features — exactly as done during training.
4. Run the GNN forward pass in eval mode.
5. Write predictions to a CSV; optionally compute and report metrics if
   experimental labels are provided.

Usage — prediction only (no labels):
    python test.py \
        --pdb_dirs /path/to/test_pdbs \
        --model_path /path/to/model.pt \
        --out_dir /path/to/test_out

Usage — with labels for benchmarking:
    python test.py \
        --pdb_dirs /path/to/test_pdbs \
        --model_path /path/to/model.pt \
        --out_dir /path/to/test_out \
        --labels_csv /path/to/test_labels.csv \
        --target_col exp_dG \
        --id_col PDB
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

from PPI_GNN.extract_features import extract_all, GLOBAL_FEATURES


# ============================================================
# Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Model  (must match train.py exactly)
# ============================================================

class GNNWithGlobal(nn.Module):
    def __init__(self, node_in, global_in, hidden):
        super().__init__()
        self.conv1 = GCNConv(node_in, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden + global_in, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        edge_weight = None
        if (hasattr(batch, "edge_attr") and
                batch.edge_attr is not None and
                batch.edge_attr.numel() > 0):
            d = batch.edge_attr.view(-1)
            edge_weight = torch.exp(-(d / 5.0) ** 2)
        x = self.conv1(x, edge_index, edge_weight=edge_weight).relu()
        x = self.conv2(x, edge_index, edge_weight=edge_weight).relu()
        g = global_mean_pool(x, batch.batch)
        u = batch.u.view(batch.num_graphs, -1)
        return self.mlp(torch.cat([g, u], dim=1)).view(-1)


# ============================================================
# Scaler  (reconstructed from saved state — no sklearn at runtime)
# ============================================================

class SavedScaler:
    """Minimal StandardScaler reconstructed from a saved scaler_state dict."""

    def __init__(self, scaler_state: dict):
        self.mean_  = np.array(scaler_state["mean_"],  dtype=np.float64)
        self.scale_ = np.array(scaler_state["scale_"], dtype=np.float64)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_


# ============================================================
# Bundle loading
# ============================================================

def load_bundle(model_path: str):
    """
    Load the model bundle saved by train.py.

    Returns
    -------
    model      : GNNWithGlobal in eval mode on the correct device
    scaler     : SavedScaler reconstructed from bundle["scaler_state"]
    config     : dict with training hyperparameters and feature list
    """
    bundle = torch.load(model_path, map_location=device)

    config     = bundle["config"]
    node_in    = config["node_in"]
    global_in  = config["global_in"]
    hidden     = config.get("hidden", 128)

    model = GNNWithGlobal(node_in=node_in, global_in=global_in, hidden=hidden)
    model.load_state_dict(bundle["state_dict"])
    model.to(device)
    model.eval()

    scaler = SavedScaler(bundle["scaler_state"])

    print(f"Loaded model from: {model_path}")
    print(f"  node_in={node_in}  global_in={global_in}  hidden={hidden}")
    print(f"  NIS mode : {config.get('nis', False)}")
    print(f"  Trained  : {config.get('created', 'unknown')}")
    print(f"  OOF metrics (train CV):")
    for k, v in config.get("oof_metrics", {}).items():
        print(f"    {k} = {v:.4f}")

    return model, scaler, config


# ============================================================
# Data preparation
# ============================================================

def build_test_data_list(df, graphs, global_features, scaler,
                         target_col=None, id_col="PDB"):
    """
    Build a list of torch_geometric.data.Data objects for inference.

    Parameters
    ----------
    df              : DataFrame with at least [id_col] + global_features columns
    graphs          : dict  pdb_id (upper) → torch_geometric graph
    global_features : list of feature column names (from GLOBAL_FEATURES)
    scaler          : SavedScaler — applied to raw global features
    target_col      : str or None  — if provided, attach .y for metric computation
    id_col          : str — column name for PDB identifiers
    """
    X_raw = df[global_features].to_numpy(dtype=np.float64)
    X_scaled = scaler.transform(X_raw).astype(np.float32)

    data_list = []
    skipped   = []

    for row_idx, (_, row) in enumerate(df.iterrows()):
        pdb_id = str(row[id_col]).upper()

        if pdb_id not in graphs:
            skipped.append(pdb_id)
            continue

        g = graphs[pdb_id]
        data = Data(
            x          = g.x,
            pos        = g.pos,
            edge_index = g.edge_index,
            edge_attr  = g.edge_attr,
            pdb_id     = pdb_id,
        )
        data.u = torch.tensor(
            X_scaled[row_idx].reshape(1, -1), dtype=torch.float32
        )
        if target_col is not None and target_col in row.index:
            data.y = torch.tensor([float(row[target_col])], dtype=torch.float32)

        data_list.append(data)

    if skipped:
        print(f"  [WARN] No graph found for {len(skipped)} PDB(s): "
              f"{skipped[:10]}{'...' if len(skipped) > 10 else ''}")
    print(f"  Built {len(data_list)} test samples  (skipped={len(skipped)})")
    return data_list


# ============================================================
# Inference
# ============================================================

def run_inference(model, data_list, batch_size=32):
    """
    Forward pass over all test samples.

    Returns
    -------
    pdb_ids : list[str]
    preds   : np.ndarray  shape (n,)
    trues   : np.ndarray  shape (n,) or None if no .y attached
    """
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)

    pdb_ids, preds, trues = [], [], []
    has_labels = hasattr(data_list[0], "y") and data_list[0].y is not None

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            yhat  = model(batch).detach().cpu().numpy().astype(float)
            preds.extend(yhat.tolist())
            pdb_ids.extend(
                list(batch.pdb_id) if hasattr(batch, "pdb_id")
                else [""] * len(yhat)
            )
            if has_labels:
                trues.extend(
                    batch.y.view(-1).detach().cpu().numpy().astype(float).tolist()
                )

    return (
        pdb_ids,
        np.array(preds),
        np.array(trues) if trues else None,
    )


# ============================================================
# Metrics
# ============================================================

def compute_metrics(true, pred):
    r,   p_r   = pearsonr(true, pred)
    rho, p_rho = spearmanr(true, pred)
    mae  = float(mean_absolute_error(true, pred))
    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    return {
        "pearson_r":  float(r),
        "pearson_p":  float(p_r),
        "spearman":   float(rho),
        "spearman_p": float(p_rho),
        "MAE":        mae,
        "RMSE":       rmse,
    }


# ============================================================
# Plot
# ============================================================

def plot_predictions(true, pred, metrics, pdb_ids=None, save_dir=None):
    r, rho     = metrics["pearson_r"], metrics["spearman"]
    mae, rmse  = metrics["MAE"],       metrics["RMSE"]

    plt.figure(figsize=(6, 6))
    plt.scatter(true, pred, edgecolor="k", alpha=0.7)

    if pdb_ids is not None:
        for pid, xt, yp in zip(pdb_ids, true, pred):
            plt.annotate(str(pid), (xt, yp),
                         fontsize=6, alpha=0.8,
                         xytext=(3, 3), textcoords="offset points")

    mn = min(true.min(), pred.min())
    mx = max(true.max(), pred.max())
    plt.plot([mn, mx], [mn, mx], linestyle="--", color="gray")
    plt.xlabel("Experimental ΔG (kcal/mol)")
    plt.ylabel("Predicted ΔG (kcal/mol)")
    plt.title(f"GNN Test-Set Predictions\n"
              f"r={r:.3f}  ρ={rho:.3f}  MAE={mae:.2f}  RMSE={rmse:.2f}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_dir:
        path = os.path.join(save_dir, "test_scatter.png")
        plt.savefig(path, dpi=150)
        print(f"Plot saved: {path}")
    plt.close()


# ============================================================
# Main
# ============================================================

def main(args):
    print(f"Device: {device}\n")
    os.makedirs(args.out_dir, exist_ok=True)

    # ── 1. Load model bundle ─────────────────────────────────
    model, scaler, config = load_bundle(args.model_path)

    global_features = config.get("global_features", list(GLOBAL_FEATURES))
    use_nis         = config.get("nis", False)

    # Warn if --nis flag disagrees with the saved config
    if args.nis != use_nis:
        print(f"  [WARN] --nis={args.nis} but model was trained with nis={use_nis}. "
              f"Using nis={use_nis} from the saved config.")

    # ── 2. Extract features from test PDBs ──────────────────
    print(f"\nExtracting features from: {args.pdb_dirs}")
    feature_df, graphs = extract_all(
        pdb_dirs  = args.pdb_dirs,
        out_dir   = args.out_dir,
        use_cache = not args.no_cache,
        use_nis   = use_nis,
        verbose   = not args.quiet,
    )

    # ── 3. Merge with labels (optional) ─────────────────────
    df = pd.read_csv(os.path.join(args.out_dir, "global_features.csv"))
    df[args.id_col] = df[args.id_col].astype(str).str.upper()

    has_labels = args.labels_csv is not None
    if has_labels:
        labels = pd.read_csv(args.labels_csv)
        labels[args.id_col] = labels[args.id_col].astype(str).str.upper()
        df = df.merge(labels[[args.id_col, args.target_col]],
                      on=args.id_col, how="inner")
        print(f"\nMerged with labels: {len(df)} samples")

    # Drop rows missing any required global feature
    needed = [args.id_col] + global_features
    if has_labels:
        needed.append(args.target_col)
    df = df.dropna(subset=needed).reset_index(drop=True)
    print(f"Samples after dropping NaN in required features: {len(df)}")

    if len(df) == 0:
        raise ValueError(
            "No usable test samples after feature extraction and NaN filtering. "
            "Check that PDB files are readable and contain interface residues."
        )

    # ── 4. Validate feature compatibility with saved model ───
    saved_feats = set(global_features)
    avail_feats = set(df.columns)
    missing = saved_feats - avail_feats
    if missing:
        raise ValueError(
            f"The following global features expected by the saved model are "
            f"missing from extracted features: {sorted(missing)}\n"
            f"Ensure you are using the same extract_features version "
            f"and the same --nis setting as during training."
        )

    node_in_data = next(iter(graphs.values())).x.shape[1]
    node_in_model = config["node_in"]
    if node_in_data != node_in_model:
        raise ValueError(
            f"Node feature dimension mismatch: model expects {node_in_model} "
            f"but extracted graphs have {node_in_data}. "
            f"Check that --nis matches the training configuration (nis={use_nis})."
        )

    # ── 5. Build data list with scaled global features ───────
    print("\nBuilding test data list...")
    data_list = build_test_data_list(
        df              = df,
        graphs          = graphs,
        global_features = global_features,
        scaler          = scaler,
        target_col      = args.target_col if has_labels else None,
        id_col          = args.id_col,
    )

    if len(data_list) == 0:
        raise ValueError(
            "No test samples could be matched to extracted graphs. "
            "Verify that PDB IDs in the feature CSV match the PDB filenames."
        )

    # ── 6. Run inference ────────────────────────────────────
    print("\nRunning inference...")
    pdb_ids, preds, trues = run_inference(model, data_list, batch_size=32)

    # ── 7. Report metrics (if labels provided) ───────────────
    if has_labels and trues is not None:
        m = compute_metrics(trues, preds)
        print(f"\n{'='*50}")
        print(f"Test Results  ({len(preds)} samples)")
        print(f"  Pearson r  = {m['pearson_r']:.3f}  (p={m['pearson_p']:.4f})")
        print(f"  Spearman ρ = {m['spearman']:.3f}  (p={m['spearman_p']:.4f})")
        print(f"  MAE        = {m['MAE']:.3f} kcal/mol")
        print(f"  RMSE       = {m['RMSE']:.3f} kcal/mol")
        print(f"{'='*50}\n")
        plot_predictions(trues, preds, m, pdb_ids=pdb_ids, save_dir=args.out_dir)

    # ── 8. Save predictions ──────────────────────────────────
    out_dict = {"PDB": pdb_ids, "pred_dG": preds}
    if has_labels and trues is not None:
        out_dict["exp_dG"] = trues

    out_df = pd.DataFrame(out_dict)
    out_csv = os.path.join(args.out_dir, "test_predictions.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"Predictions saved: {out_csv}")

    # Pretty-print the first few rows
    print(f"\n{out_df.to_string(index=False, float_format=lambda x: f'{x:.3f}')}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GNN inference script — loads a bundle saved by train.py."
    )
    parser.add_argument(
        "--pdb_dirs", nargs="+", required=True,
        help="One or more directories containing test PDB files.",
    )
    parser.add_argument(
        "--model_path", required=True,
        help="Path to the .pt model bundle produced by train.py.",
    )
    parser.add_argument(
        "--out_dir", required=True,
        help="Directory for extracted features, predictions, and plots.",
    )
    parser.add_argument(
        "--labels_csv", default=None,
        help="Optional CSV with experimental labels for benchmarking.",
    )
    parser.add_argument(
        "--target_col", default="exp_dG",
        help="Column name for experimental ΔG in labels_csv (default: exp_dG).",
    )
    parser.add_argument(
        "--id_col", default="PDB",
        help="Column name for PDB identifiers (default: PDB).",
    )
    parser.add_argument(
        "--nis", action="store_true",
        help="Flag kept for CLI compatibility; actual NIS setting is read from "
             "the saved model config and takes precedence.",
    )
    parser.add_argument(
        "--no_cache", action="store_true",
        help="Force re-extraction of features even if a cache exists.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output from extract_all.",
    )
    args = parser.parse_args()
    main(args)