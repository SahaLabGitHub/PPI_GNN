"""
train.py
========

Train a GNN model from a folder of PDBs by extracting both
- global features (IC, HB, BSA, SB, IC_density, %NIS_*), and
- graph features (interface graph or NIS-aware graph)

CV strategy: sequence-similarity clustering
  1. Extract chain sequences from each PDB
  2. Smith-Waterman local alignment with BLOSUM62 scoring
  3. Normalised score: SW(a,b) / min(SW(a,a), SW(b,b))  → [0, 1]
  4. Distance matrix: distance = 1 - normalised_score
  5. Single-linkage hierarchical clustering
  6. Cut dendrogram at SEQ_ID_THRESHOLD → clusters become CV folds

Global features are StandardScaler-normalised within each CV fold
and on the full training set for the final model.
The fitted scaler is saved in the model bundle for use at inference.

Dependencies:
    pip install biopython

Usage:
    python train.py \
        --pdb_dirs /path/to/pdbs \
        --labels_csv /path/to/labels.csv \
        --out_dir /path/to/out \
        --save_path /path/to/model.pt \
        --nis \
        --cv_type random
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
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from datetime import datetime

# Smith-Waterman + BLOSUM62 via Biopython
from Bio import Align
from Bio.Align import substitution_matrices

from PPI_GNN.extract_features import extract_all, GLOBAL_FEATURES


# ============================================================
# Hyperparameters
# ============================================================

HIDDEN       = 128
LR           = 1e-3
MAX_EPOCHS   = 300
WEIGHT_DECAY = 1e-4
BATCH_TRAIN  = 16
BATCH_EVAL   = 32

PATIENCE     = 25
MIN_DELTA    = 1e-4

N_SPLITS     = 10
SEED         = 0

# Sequence clustering threshold:
# complexes with normalised SW score > this share a fold.
SEQ_ID_THRESHOLD = 0.60

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Weighted MSE loss
# ============================================================

LOSS_MEAN  = -9.9
LOSS_SCALE =  3.0


def weighted_mse(pred, true):
    deviation = (true - LOSS_MEAN).abs()
    max_dev   = deviation.max().clamp(min=1.0)
    weights   = 1.0 + LOSS_SCALE * (deviation / max_dev)
    return (weights * (pred - true) ** 2).mean()


# ============================================================
# Metrics
# ============================================================

def compute_metrics(true, pred):
    r,   p_r   = pearsonr(true, pred)
    rho, p_rho = spearmanr(true, pred)
    mae  = mean_absolute_error(true, pred)
    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    return {
        "pearson_r":  float(r),
        "pearson_p":  float(p_r),
        "spearman":   float(rho),
        "spearman_p": float(p_rho),
        "MAE":        float(mae),
        "RMSE":       float(rmse),
    }


# ============================================================
# Smith-Waterman aligner (BLOSUM62, initialised once)
# ============================================================

def _make_sw_aligner() -> Align.PairwiseAligner:
    aligner = Align.PairwiseAligner()
    aligner.mode                  = "local"
    aligner.substitution_matrix   = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score        = -11.0
    aligner.extend_gap_score      = -1.0
    return aligner

_SW_ALIGNER = _make_sw_aligner()


def sw_normalised_score(seq_a: str, seq_b: str) -> float:
    if not seq_a or not seq_b:
        return 0.0
    score_ab = _SW_ALIGNER.score(seq_a, seq_b)
    score_aa = _SW_ALIGNER.score(seq_a, seq_a)
    score_bb = _SW_ALIGNER.score(seq_b, seq_b)
    denom = min(score_aa, score_bb)
    if denom <= 0:
        return 0.0
    return float(min(score_ab / denom, 1.0))


def max_chain_sw_similarity(seqs_a: dict, seqs_b: dict) -> float:
    best = 0.0
    for sa in seqs_a.values():
        for sb in seqs_b.values():
            best = max(best, sw_normalised_score(sa, sb))
            if best >= 1.0:
                return best
    return best


# ============================================================
# Sequence extraction from PDB files
# ============================================================

AA3_TO_1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C",
    "GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
    "LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P",
    "SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
}


def extract_sequences_from_pdb(pdb_path: str) -> dict:
    seqs = defaultdict(dict)
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            chain   = line[21].strip()
            resname = line[17:20].strip()
            aa1     = AA3_TO_1.get(resname)
            if aa1 is None:
                continue
            try:
                resnum = int(line[22:26])
            except ValueError:
                continue
            seqs[chain][resnum] = aa1
    return {
        chain: "".join(aa for _, aa in sorted(resnums.items()))
        for chain, resnums in seqs.items()
    }


# ============================================================
# Build distance matrix  (SW-based)
# ============================================================

def build_distance_matrix(pdb_ids: list, pdb_dirs: list) -> np.ndarray:
    n = len(pdb_ids)
    pdb_path_map = {}
    for pdb_dir in pdb_dirs:
        for fname in os.listdir(pdb_dir):
            stem = fname.upper().replace(".PDB", "")
            if stem in {p.upper() for p in pdb_ids}:
                pdb_path_map[stem] = os.path.join(pdb_dir, fname)

    print(f"  Found PDB files for {len(pdb_path_map)}/{n} complexes")

    all_seqs = {}
    for pdb_id in pdb_ids:
        path = pdb_path_map.get(pdb_id.upper())
        if path and os.path.exists(path):
            all_seqs[pdb_id] = extract_sequences_from_pdb(path)
        else:
            all_seqs[pdb_id] = {}

    dist = np.zeros((n, n), dtype=np.float32)
    total_pairs = n * (n - 1) // 2
    done = 0
    for i in range(n):
        for j in range(i + 1, n):
            sim = max_chain_sw_similarity(all_seqs[pdb_ids[i]],
                                          all_seqs[pdb_ids[j]])
            d = 1.0 - sim
            dist[i, j] = d
            dist[j, i] = d
            done += 1
            if done % 2000 == 0:
                print(f"    SW pairs computed: {done}/{total_pairs} "
                      f"({100*done/total_pairs:.0f}%)")
    return dist


# ============================================================
# Clustering-based CV fold assignment
# ============================================================

def make_cluster_folds(pdb_ids, pdb_dirs, n_splits, threshold):
    n = len(pdb_ids)
    print(f"\nBuilding SW-BLOSUM62 distance matrix ({n}×{n})...")
    dist_matrix = build_distance_matrix(pdb_ids, pdb_dirs)

    condensed   = squareform(dist_matrix)
    Z           = linkage(condensed, method="single")
    cluster_ids = fcluster(Z, t=threshold, criterion="distance")

    counts = np.bincount(cluster_ids)[1:]
    print(f"  Threshold={threshold:.2f} → {len(np.unique(cluster_ids))} clusters "
          f"(sizes: min={counts.min()}, max={counts.max()})")

    cluster_to_idx = defaultdict(list)
    for idx, cid in enumerate(cluster_ids):
        cluster_to_idx[cid].append(idx)

    clusters_sorted = sorted(cluster_to_idx.values(), key=len, reverse=True)
    fold_sizes  = [0] * n_splits
    fold_assign = [[] for _ in range(n_splits)]

    for cluster_indices in clusters_sorted:
        target_fold = int(np.argmin(fold_sizes))
        fold_assign[target_fold].extend(cluster_indices)
        fold_sizes[target_fold] += len(cluster_indices)

    fold_assign  = [f for f in fold_assign if len(f) > 0]
    print(f"  Assigned to {len(fold_assign)} CV folds  "
          f"(sizes: {sorted([len(f) for f in fold_assign], reverse=True)})")

    all_indices = np.arange(n)
    splits = []
    for val_indices in fold_assign:
        val_set   = np.array(val_indices)
        train_set = np.array([i for i in all_indices
                               if i not in set(val_indices)])
        splits.append((train_set, val_set))
    return splits


# ============================================================
# Data loading
# ============================================================

def load_dataframe(features_csv, labels_csv, target_col, id_col):
    df = pd.read_csv(features_csv)
    df[id_col] = df[id_col].astype(str).str.upper()

    if labels_csv is not None:
        labels = pd.read_csv(labels_csv)
        labels[id_col] = labels[id_col].astype(str).str.upper()
        df = df.merge(labels[[id_col, target_col]], on=id_col, how="inner")

    needed = [id_col, target_col] + GLOBAL_FEATURES
    df = df.dropna(subset=needed).reset_index(drop=True)

    print(f"Loaded {len(df)} samples with complete features.")
    print(f"  exp_dG mean = {df[target_col].mean():.3f}  "
          f"std = {df[target_col].std():.3f}  "
          f"range = [{df[target_col].min():.2f}, {df[target_col].max():.2f}]")
    print(f"  (LOSS_MEAN is currently set to {LOSS_MEAN} — "
          f"update if your mean differs by >0.5 kcal/mol)")
    return df


def build_data_list(df, graphs, target_col, id_col):
    data_list = []
    skipped = 0
    for _, row in df.iterrows():
        pdb_id = str(row[id_col]).upper()
        if pdb_id not in graphs:
            skipped += 1
            continue
        g = graphs[pdb_id]
        data = Data(
            x=g.x, pos=g.pos,
            edge_index=g.edge_index, edge_attr=g.edge_attr,
            pdb_id=pdb_id,
        )
        data.u = torch.tensor(
            row[GLOBAL_FEATURES].to_numpy(dtype=np.float32).reshape(1, -1),
            dtype=torch.float32,
        )
        data.y = torch.tensor([float(row[target_col])], dtype=torch.float32)
        data_list.append(data)
    print(f"Built data list: {len(data_list)} samples  (skipped={skipped} missing graphs)")
    return data_list


# ============================================================
# Model
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
# Training
# ============================================================

def train_one_fold(train_loader, val_loader, node_in, global_in):
    model = GNNWithGlobal(node_in=node_in, global_in=global_in, hidden=HIDDEN).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val   = float("inf")
    best_state = None
    bad        = 0

    for ep in range(MAX_EPOCHS):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            loss = weighted_mse(model(batch), batch.y.view(-1))
            loss.backward()
            opt.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                val_losses.append(
                    weighted_mse(model(batch), batch.y.view(-1)).item()
                )
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

        if val_loss < best_val - MIN_DELTA:
            best_val   = val_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def fit_final_model(data_list, node_in, global_in):
    loader = DataLoader(data_list, batch_size=BATCH_TRAIN, shuffle=True)
    model  = GNNWithGlobal(node_in=node_in, global_in=global_in, hidden=HIDDEN).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=20, min_lr=1e-5
    )

    best_loss  = float("inf")
    best_state = None

    for ep in range(800):
        model.train()
        losses = []
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            loss = weighted_mse(model(batch), batch.y.view(-1))
            loss.backward()
            opt.step()
            losses.append(loss.item())

        epoch_loss = float(np.mean(losses))
        scheduler.step(epoch_loss)

        if epoch_loss < best_loss:
            best_loss  = epoch_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

        if (ep + 1) % 50 == 0:
            print(f"  [FINAL] epoch {ep+1:3d}  "
                  f"train_wmse={epoch_loss:.4f}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}")

    model.load_state_dict(best_state)
    model.eval()
    return model


# ============================================================
# Evaluation
# ============================================================

def eval_loader(model, loader):
    model.eval()
    preds, trues, ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            yhat  = model(batch).detach().cpu().numpy().astype(float)
            ytrue = batch.y.view(-1).detach().cpu().numpy().astype(float)
            preds.extend(yhat.tolist())
            trues.extend(ytrue.tolist())
            ids.extend(
                list(batch.pdb_id) if hasattr(batch, "pdb_id")
                else [""] * len(yhat)
            )
    return np.array(trues), np.array(preds), ids


# ============================================================
# Clustering CV  — with StandardScaler on global features
# ============================================================

def run_cv(data_list, X_global, y, node_in, global_in,
           pdb_ids, pdb_dirs, cv_type="clustering"):
    n = len(data_list)

    if cv_type == "random":
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=min(N_SPLITS, n), shuffle=True, random_state=SEED)
        splits = [(tr, va) for tr, va in kf.split(np.arange(n))]
        print(f"\nRunning random KFold CV ({len(splits)} folds)\n")
    else:
        splits = make_cluster_folds(
            pdb_ids   = pdb_ids,
            pdb_dirs  = pdb_dirs,
            n_splits  = N_SPLITS,
            threshold = 1.0 - SEQ_ID_THRESHOLD,
        )
        print(f"\nRunning SW-BLOSUM62 clustering CV "
              f"({len(splits)} folds, threshold={SEQ_ID_THRESHOLD:.2f})\n")

    oof_pred = np.zeros(n, dtype=float)
    oof_true = np.zeros(n, dtype=float)
    oof_ids  = [""] * n

    print(f"  Loss: weighted_mse  (LOSS_MEAN={LOSS_MEAN}, LOSS_SCALE={LOSS_SCALE})")
    print(f"  Global features: StandardScaler applied within each fold\n")

    for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
        if len(tr_idx) == 0 or len(va_idx) == 0:
            print(f"  [Fold {fold:2d}] SKIPPED — empty split")
            continue

        # Scale global features within fold to avoid data leakage.
        # The scaled features are stored back into data.u so the GNN
        # sees normalised inputs during this fold's training and evaluation.
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_global[tr_idx])
        X_va_scaled = scaler.transform(X_global[va_idx])

        for k, idx in enumerate(tr_idx):
            data_list[idx].u = torch.tensor(
                X_tr_scaled[k].reshape(1, -1), dtype=torch.float32
            )
        for k, idx in enumerate(va_idx):
            data_list[idx].u = torch.tensor(
                X_va_scaled[k].reshape(1, -1), dtype=torch.float32
            )

        train_loader = DataLoader(
            [data_list[i] for i in tr_idx], batch_size=BATCH_TRAIN, shuffle=True
        )
        val_loader = DataLoader(
            [data_list[i] for i in va_idx], batch_size=BATCH_EVAL, shuffle=False
        )

        model = train_one_fold(train_loader, val_loader, node_in, global_in)
        trues, preds, ids = eval_loader(model, val_loader)

        for k, idx in enumerate(va_idx):
            oof_pred[idx] = preds[k]
            oof_true[idx] = trues[k]
            oof_ids[idx]  = (ids[k] if ids[k]
                             else getattr(data_list[idx], "pdb_id", f"idx_{idx}"))

        m = compute_metrics(trues, preds)
        print(f"  [Fold {fold:2d}  n_val={len(va_idx):3d}]  "
              f"r={m['pearson_r']:.3f}  rho={m['spearman']:.3f}  "
              f"MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}")

    # Restore unscaled global features so the final model trains on a fresh scaler
    for i, d in enumerate(data_list):
        d.u = torch.tensor(X_global[i].reshape(1, -1), dtype=torch.float32)

    return oof_true, oof_pred, oof_ids


# ============================================================
# Plot
# ============================================================

def plot_oof(oof_true, oof_pred, metrics, oof_ids=None, save_dir=None):
    r, rho = metrics["pearson_r"], metrics["spearman"]
    mae, rmse = metrics["MAE"], metrics["RMSE"]

    plt.figure(figsize=(6, 6))
    plt.scatter(oof_true, oof_pred, edgecolor="k", alpha=0.7)

    if oof_ids is not None:
        for pdb_id, x, y_pt in zip(oof_ids, oof_true, oof_pred):
            plt.annotate(str(pdb_id), (x, y_pt),
                         fontsize=6, alpha=0.8,
                         xytext=(3, 3), textcoords="offset points")

    mn = min(oof_true.min(), oof_pred.min())
    mx = max(oof_true.max(), oof_pred.max())
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("Experimental ΔG (kcal/mol)")
    plt.ylabel("Predicted ΔG (OOF, kcal/mol)")
    plt.title(f"GNN Interface OOF  (SW-BLOSUM62 clustering CV)\n"
              f"r={r:.3f}  ρ={rho:.3f}  MAE={mae:.2f}  RMSE={rmse:.2f}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_dir:
        plot_path = os.path.join(save_dir, "oof_scatter.png")
        plt.savefig(plot_path, dpi=150)
        print(f"Plot saved: {plot_path}")
    plt.close()


# ============================================================
# Main
# ============================================================

def main(args):
    print(f"Device: {device}\n")
    os.makedirs(args.out_dir, exist_ok=True)

    feature_df, graphs = extract_all(
        pdb_dirs=args.pdb_dirs,
        out_dir=args.out_dir,
        use_cache=not args.no_cache,
        use_nis=args.nis,
        verbose=not args.quiet,
    )

    if args.labels_csv:
        labels_df = pd.read_csv(args.labels_csv)
        labels_df[args.id_col] = labels_df[args.id_col].astype(str).str.upper()

    df = load_dataframe(
        features_csv=os.path.join(args.out_dir, "global_features.csv"),
        labels_csv=args.labels_csv,
        target_col=args.target_col,
        id_col=args.id_col,
    )

    data_list = build_data_list(df, graphs, args.target_col, args.id_col)
    if len(data_list) == 0:
        raise ValueError(
            "No usable samples. Check PDB IDs in labels match extracted graphs."
        )

    sample_node_in = data_list[0].x.shape[1]
    node_in = sample_node_in
    print(f"  node_in inferred from data: {node_in}  "
        f"(--nis flag says: {29 if args.nis else 23})")
    if node_in != (29 if args.nis else 23):
        print(f"  [WARN] --nis flag suggests {29 if args.nis else 23} "
            f"but graphs have {node_in} node features. "
            f"Using actual graph dimension.")
    global_in = len(GLOBAL_FEATURES)

    n = len(data_list)
    pdb_ids_ordered = [
        getattr(d, "pdb_id", df.iloc[i][args.id_col])
        for i, d in enumerate(data_list)
    ]

    X_global = np.vstack([d.u.view(-1).cpu().numpy() for d in data_list])
    y        = np.array([float(d.y.item()) for d in data_list])

    # ── Clustering-based CV ──────────────────────────────────
    oof_true, oof_pred, oof_ids = run_cv(
        data_list = data_list,
        X_global  = X_global,
        y         = y,
        node_in   = node_in,
        global_in = global_in,
        pdb_ids   = pdb_ids_ordered,
        pdb_dirs  = args.pdb_dirs,
        cv_type   = args.cv_type,
    )

    m = compute_metrics(oof_true, oof_pred)

    print(f"\n{'='*50}")
    print(f"OOF Results ({n} samples, SW-BLOSUM62 clustering CV)")
    print(f"  Pearson r  = {m['pearson_r']:.3f}  (p={m['pearson_p']:.4f})")
    print(f"  Spearman ρ = {m['spearman']:.3f}  (p={m['spearman_p']:.4f})")
    print(f"  MAE        = {m['MAE']:.3f} kcal/mol")
    print(f"  RMSE       = {m['RMSE']:.3f} kcal/mol")
    print(f"{'='*50}\n")

    save_dir = (os.path.dirname(os.path.abspath(args.save_path))
                if args.save_path else args.out_dir)
    os.makedirs(save_dir, exist_ok=True)

    oof_df = pd.DataFrame({"PDB": oof_ids, "exp_dG": oof_true, "pred_dG": oof_pred})
    oof_df.to_csv(os.path.join(save_dir, "oof_predictions.csv"), index=False)
    print(f"OOF predictions saved: {os.path.join(save_dir, 'oof_predictions.csv')}")

    plot_oof(oof_true, oof_pred, m, oof_ids=oof_ids, save_dir=save_dir)

    # ── Final model on all data ──────────────────────────────
    print("\nTraining final model on all data...")

    # Fit a single scaler on all training data and apply it before final training.
    # This scaler is saved in the bundle so test.py can reproduce the same transform.
    final_scaler = StandardScaler()
    X_global_scaled = final_scaler.fit_transform(X_global)
    for i, d in enumerate(data_list):
        d.u = torch.tensor(X_global_scaled[i].reshape(1, -1), dtype=torch.float32)

    final_model = fit_final_model(data_list, node_in, global_in)

    scaler_state = {
        "mean_":          final_scaler.mean_.tolist(),
        "scale_":         final_scaler.scale_.tolist(),
        "n_features_in_": int(final_scaler.n_features_in_),
    }

    model_path = args.save_path or os.path.join(args.out_dir, "gnn_final.pt")
    bundle = {
        "state_dict":   final_model.state_dict(),
        "scaler_state": scaler_state,
        "config": {
            "node_in":         node_in,
            "hidden":          HIDDEN,
            "global_in":       global_in,
            "global_features": list(GLOBAL_FEATURES),
            "nis":             args.nis,
            "lr":              LR,
            "max_epochs":      MAX_EPOCHS,
            "weight_decay":    WEIGHT_DECAY,
            "loss":            "weighted_mse",
            "loss_mean":       LOSS_MEAN,
            "loss_scale":      LOSS_SCALE,
            "global_scaling":  "StandardScaler",
            "cv": {
                "type":                 args.cv_type,
                "method":               "SW-BLOSUM62",
                "similarity_threshold": SEQ_ID_THRESHOLD,
                "linkage":              "single",
                "gap_open":             -11.0,
                "gap_extend":           -1.0,
                "normalisation":        "score(a,b)/min(score(a,a),score(b,b))",
                "n_target_folds":       N_SPLITS,
            },
            "oof_metrics": {
                "pearson_r": m["pearson_r"],
                "spearman":  m["spearman"],
                "MAE":       m["MAE"],
                "RMSE":      m["RMSE"],
            },
            "created": datetime.now().isoformat(timespec="seconds"),
        },
    }
    torch.save(bundle, model_path)
    print(f"\n✅ Saved final model: {model_path}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train GNN with SW-BLOSUM62 clustering CV and feature scaling."
    )
    parser.add_argument("--pdb_dirs",   nargs="+", required=True)
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument("--out_dir",    required=True)
    parser.add_argument("--save_path",  default=None)
    parser.add_argument("--target_col", default="exp_dG")
    parser.add_argument("--id_col",     default="PDB")
    parser.add_argument("--nis",        action="store_true")
    parser.add_argument("--cache",      action="store_true", default=True)
    parser.add_argument("--no_cache",   action="store_true")
    parser.add_argument("--quiet",      action="store_true")
    parser.add_argument("--cv_type",    default="clustering",
                        choices=["clustering", "random"],
                        help="CV strategy: clustering (SW-BLOSUM62) or random KFold")
    args = parser.parse_args()
    main(args)