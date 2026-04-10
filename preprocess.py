"""
preprocess.py
=============
Preprocesses raw PDB files for the GNN binding affinity pipeline.

For each PDB:
  1. Parses REMARK 350 to identify the correct biological assembly chains
  2. Keeps only MODEL 1 (for NMR multi-model structures)
  3. Removes HETATM records (ligands, waters, ions)
  4. Keeps only the first alternate location (altloc A)
  5. Validates exactly 2 protein chains remain, each with >= min_residues
  6. Adds missing heavy atoms and hydrogens via PDBFixer at specified pH

Outputs:
  data/processed/*.pdb          — cleaned, protonated PDB files
  data/processed/preprocess_log.csv  — per-PDB status and atom counts

Usage:
    python preprocess.py \\
    --in_dir  /home/uwm/qingshu/MD_ML/data/train/raw_PDB \\
    --out_dir /home/uwm/qingshu/MD_ML/data/train/processed

    # Skip adding hydrogens:
    python preprocess.py \\
        --in_dir  /path/to/raw_pdbs \\
        --out_dir /path/to/processed_pdbs \\
        --no_h

    # Reprocess already-done PDBs:
    python preprocess.py \\
        --in_dir  /path/to/raw_pdbs \\
        --out_dir /path/to/processed_pdbs \\
        --overwrite
"""

import os
import re
import glob
import argparse
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import pandas as pd
import requests
from Bio.PDB import PDBParser, PDBIO, Select
from pdbfixer import PDBFixer
from openmm.app import PDBFile


# ============================================================
# REMARK 350 / NMR parsing
# ============================================================

BIOMOL_RE = re.compile(r"^REMARK 350 BIOMOLECULE:\s*(\d+)\s*$")
APPLY_RE  = re.compile(r"^REMARK 350 APPLY THE FOLLOWING TO CHAINS:\s*(.+)\s*$")
NMR_RE    = re.compile(r"^REMARK 210\s+EXPERIMENT TYPE\s*:\s*NMR\b", re.IGNORECASE)


def parse_remark350_biomol1_chains_and_is_nmr(pdb_path: str) -> Tuple[List[str], bool]:
    """
    Returns:
      chains_biomol1 : list of chain IDs for BIOMOLECULE 1 from REMARK 350
      is_nmr         : True if REMARK 210 indicates NMR experiment
    """
    chains: List[str] = []
    is_nmr = False
    current_biomol: Optional[int] = None

    with open(pdb_path, "r", errors="ignore") as f:
        for line in f:
            if NMR_RE.search(line):
                is_nmr = True

            m_b = BIOMOL_RE.match(line)
            if m_b:
                current_biomol = int(m_b.group(1))
                continue

            if current_biomol == 1:
                m_a = APPLY_RE.match(line)
                if m_a:
                    raw = m_a.group(1).strip().replace("AND", ",")
                    for part in raw.split(","):
                        c = part.strip().strip(",")
                        if c and re.match(r"^[A-Za-z0-9]$", c):
                            chains.append(c)

    # Deduplicate preserving order
    seen, chains_uniq = set(), []
    for c in chains:
        if c not in seen:
            seen.add(c)
            chains_uniq.append(c)

    return chains_uniq, is_nmr


# ============================================================
# Biopython selector
# ============================================================

class ProteinChainSelect(Select):
    """Keep only ATOM records from specified chains, MODEL 1, altloc A/' '."""

    def __init__(self, keep_chains: List[str], keep_model_index: int = 0):
        super().__init__()
        self.keep_chains = set(keep_chains)
        self.keep_model_index = keep_model_index

    def accept_model(self, model):
        return model.id == self.keep_model_index

    def accept_chain(self, chain):
        return chain.id in self.keep_chains

    def accept_residue(self, residue):
        return residue.id[0] == " "   # ATOM only, no HETATM

    def accept_atom(self, atom):
        altloc = atom.get_altloc()
        return altloc in (" ", "A")


# ============================================================
# Atom counting
# ============================================================

def count_atoms_in_pdb(pdb_path: str) -> Tuple[int, int]:
    """
    Returns (n_ATOM_lines, n_H_atoms) from ATOM records.
    Hydrogen detection by atom name starting with 'H'.
    """
    n_atom = n_h = 0
    with open(pdb_path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("ATOM"):
                n_atom += 1
                if line[12:16].strip().startswith("H"):
                    n_h += 1
    return n_atom, n_h


def count_residues_per_chain(pdb_path: str) -> Dict[str, int]:
    """Count unique residues per chain from ATOM records."""
    residues: Dict[str, set] = {}
    with open(pdb_path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("ATOM"):
                chain = line[21]
                resseq = line[22:26].strip()
                residues.setdefault(chain, set()).add(resseq)
    return {ch: len(res) for ch, res in residues.items()}


# ============================================================
# PDBFixer: add missing atoms + hydrogens
# ============================================================

def add_missing_hydrogens_pdbfixer(
    in_pdb: str,
    out_pdb: str,
    pH: float = 7.0,
) -> Dict[str, int]:
    """
    Uses PDBFixer to:
      - Find and add missing residues
      - Find and add missing heavy atoms
      - Add missing hydrogens at specified pH

    Returns atom count statistics.
    """
    n_atom_before, n_h_before = count_atoms_in_pdb(in_pdb)

    fixer = PDBFixer(filename=in_pdb)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=pH)

    with open(out_pdb, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)

    n_atom_after, n_h_after = count_atoms_in_pdb(out_pdb)

    return {
        "n_atom_before":  n_atom_before,
        "n_h_before":     n_h_before,
        "n_atom_after":   n_atom_after,
        "n_h_after":      n_h_after,
        "n_h_added_est":  max(0, n_h_after - n_h_before),
    }


# ============================================================
# Validation
# ============================================================

def validate_two_chains(pdb_path: str, min_residues: int = 10) -> Tuple[bool, str]:
    """
    Checks that the PDB has exactly 2 protein chains, each with >= min_residues.

    Returns:
        (passed, reason_string)
    """
    res_per_chain = count_residues_per_chain(pdb_path)

    if len(res_per_chain) < 2:
        return False, f"Only {len(res_per_chain)} chain(s) found after cleaning"

    if len(res_per_chain) > 2:
        chain_summary = ", ".join(f"{c}:{n}" for c, n in res_per_chain.items())
        return False, f"{len(res_per_chain)} chains found ({chain_summary}) — expected exactly 2"

    short = [c for c, n in res_per_chain.items() if n < min_residues]
    if short:
        chain_summary = ", ".join(f"{c}:{res_per_chain[c]}" for c in short)
        return False, f"Chain(s) too short (<{min_residues} residues): {chain_summary}"

    return True, ""


# ============================================================
# Single PDB preprocessing
# ============================================================

def preprocess_one_pdb(
    pdb_path:     str,
    out_path:     str,
    add_h:        bool  = True,
    pH:           float = 7.0,
    min_residues: int   = 10,
) -> Dict[str, object]:
    """
    Full preprocessing pipeline for a single PDB file.

    Steps:
      1. Parse REMARK 350 for biological assembly chains
      2. Strip HETATM, alt-locs, keep MODEL 1 via Biopython
      3. Validate exactly 2 chains with >= min_residues each
      4. Add missing atoms + hydrogens via PDBFixer (optional)

    Returns a dict of metadata/stats for logging.
    Raises ValueError on any validation failure.
    """
    chains_biomol1, is_nmr = parse_remark350_biomol1_chains_and_is_nmr(pdb_path)
    if not chains_biomol1:
        raise ValueError("No BIOMOLECULE 1 chains found in REMARK 350")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    # --- Step 1: Write protein-only PDB (no HETATM, MODEL 1, altloc A) ---
    tmp_out = out_path + ".tmp_clean.pdb"
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("s", pdb_path)

        io = PDBIO()
        io.set_structure(structure)
        io.save(tmp_out, select=ProteinChainSelect(
            keep_chains=chains_biomol1, keep_model_index=0
        ))

        # --- Step 2: Validate chain count and length ---
        valid, reason = validate_two_chains(tmp_out, min_residues=min_residues)
        if not valid:
            raise ValueError(reason)

        # --- Step 3: Add hydrogens (writes final out_path) ---
        if add_h:
            h_stats = add_missing_hydrogens_pdbfixer(tmp_out, out_path, pH=pH)
        else:
            os.replace(tmp_out, out_path)
            tmp_out = None   # already moved, nothing to clean up
            h_stats = {k: None for k in
                       ["n_atom_before","n_h_before","n_atom_after","n_h_after","n_h_added_est"]}

    finally:
        if tmp_out and os.path.exists(tmp_out):
            os.remove(tmp_out)

    # --- Final counts ---
    n_atoms_final, n_h_final = count_atoms_in_pdb(out_path)

    return {
        "chains_biomol1":   ",".join(chains_biomol1),
        "is_nmr":           bool(is_nmr),
        "model_kept":       1,
        "n_atoms_written":  int(n_atoms_final),
        "n_H_atoms_written":int(n_h_final),
        "add_hydrogens":    bool(add_h),
        "pH":               float(pH) if add_h else "",
        "n_H_added_est":    h_stats["n_h_added_est"],
        "out_path":         out_path,
    }


# ============================================================
# Batch preprocessing
# ============================================================

def preprocess_dir(
    in_dir:       str,
    out_dir:      str,
    pattern:      str   = "*.pdb",
    overwrite:    bool  = False,
    add_h:        bool  = True,
    pH:           float = 7.0,
    min_residues: int   = 10,
) -> pd.DataFrame:
    """
    Preprocess all PDB files in in_dir, write results to out_dir.
    Returns a summary DataFrame logged to preprocess_log.csv.
    """
    paths = sorted(glob.glob(os.path.join(in_dir, pattern)))
    os.makedirs(out_dir, exist_ok=True)
    rows = []

    print(f"Found {len(paths)} PDB files in {in_dir}\n")

    for pdb_path in paths:
        pdb_id   = Path(pdb_path).stem.upper()
        out_path = os.path.join(out_dir, f"{pdb_id}.pdb")

        # --- Skip if already done ---
        if not overwrite and os.path.exists(out_path):
            print(f"[SKIP] {pdb_id}: already exists")
            rows.append(_row(pdb_id, "SKIP_EXISTS", out_path=out_path, add_h=add_h, pH=pH))
            continue

        # --- Process ---
        try:
            rec = preprocess_one_pdb(
                pdb_path, out_path,
                add_h=add_h, pH=pH, min_residues=min_residues
            )
            print(
                f"[OK] {pdb_id}  chains={rec['chains_biomol1']}  "
                f"atoms={rec['n_atoms_written']}  H={rec['n_H_atoms_written']}  "
                f"(+{rec['n_H_added_est']} H added)"
            )
            rows.append(_row(pdb_id, "OK", rec=rec))

        except Exception as e:
            print(f"[FAIL] {pdb_id}: {e}")
            rows.append(_row(pdb_id, "FAIL", error=str(e),
                             out_path=out_path, add_h=add_h, pH=pH))

    df = pd.DataFrame(rows)

    # Save log
    log_path = os.path.join(out_dir, "preprocess_log.csv")
    df.to_csv(log_path, index=False)

    # Summary
    counts = df["status"].value_counts().to_dict()
    print(f"\n{'='*50}")
    print(f"PREPROCESSING COMPLETE")
    print(f"  OK          : {counts.get('OK', 0)}")
    print(f"  SKIPPED     : {counts.get('SKIP_EXISTS', 0)}")
    print(f"  FAILED      : {counts.get('FAIL', 0)}")
    print(f"  Log saved   : {log_path}")
    print(f"{'='*50}\n")

    return df


def download_pdb_from_rcsb(pdb_id: str, out_path: str, retry: int = 3, sleep_s: float = 1.0) -> None:
    """Download PDB file from RCSB to out_path."""
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    for attempt in range(1, retry + 1):
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200 and resp.content:
            with open(out_path, "wb") as f:
                f.write(resp.content)
            return
        if attempt < retry:
            time.sleep(sleep_s)
    raise RuntimeError(f"Failed to download {pdb_id} from RCSB (status={resp.status_code})")


def download_pdbs_from_csv(
    csv_path: str,
    pdb_column: str,
    out_dir: str,
    lowercase: bool = False,
    overwrite: bool = False,
    max_rows: Optional[int] = None,
) -> List[str]:
    """Read CSV, extract unique PDB IDs from column, download PDBs into out_dir.

    Returns list of downloaded/available pdb paths.
    """
    df = pd.read_csv(csv_path)
    if pdb_column not in df.columns:
        raise ValueError(f"CSV does not contain column: {pdb_column}")

    pdb_ids_raw = df[pdb_column].dropna().astype(str)
    pdb_ids = []
    for x in pdb_ids_raw:
        token = x.strip().split()[0].upper()
        if lowercase:
            token = token.lower()
        # Keep first 4 chars as PDB ID, if extra suffix present
        token = token[:4]
        if re.match(r"^[A-Za-z0-9]{4}$", token):
            pdb_ids.append(token)
    pdb_ids = sorted(set(pdb_ids))
    if max_rows is not None:
        pdb_ids = pdb_ids[:max_rows]

    os.makedirs(out_dir, exist_ok=True)
    downloaded_files = []
    for pdb_id in pdb_ids:
        out_path = os.path.join(out_dir, f"{pdb_id}.pdb")
        if not overwrite and os.path.exists(out_path):
            downloaded_files.append(out_path)
            continue
        try:
            download_pdb_from_rcsb(pdb_id, out_path)
            downloaded_files.append(out_path)
            print(f"[DOWNLOAD] {pdb_id}")
        except Exception as e:
            print(f"[DOWNLOAD FAIL] {pdb_id}: {e}")
    return downloaded_files


def preprocess_csv(
    csv_path: str,
    pdb_column: str,
    raw_pdb_dir: str,
    out_dir: str,
    overwrite: bool = False,
    add_h: bool = True,
    pH: float = 7.0,
    min_residues: int = 10,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Read CSV IDs, download PDBs to raw_pdb_dir, then process with preprocess_dir."""
    download_pdbs_from_csv(
        csv_path=csv_path,
        pdb_column=pdb_column,
        out_dir=raw_pdb_dir,
        overwrite=overwrite,
        max_rows=max_rows,
    )
    return preprocess_dir(
        in_dir=raw_pdb_dir,
        out_dir=out_dir,
        overwrite=overwrite,
        add_h=add_h,
        pH=pH,
        min_residues=min_residues,
    )


def _row(pdb_id, status, rec=None, error="", out_path="", add_h=True, pH=7.0):
    """Helper to build a consistent log row."""
    if rec:
        return {
            "pdb":              pdb_id,
            "status":           status,
            "error":            "",
            "chains_biomol1":   rec["chains_biomol1"],
            "is_nmr":           rec["is_nmr"],
            "model_kept":       rec["model_kept"],
            "n_atoms_written":  rec["n_atoms_written"],
            "n_H_atoms_written":rec["n_H_atoms_written"],
            "add_hydrogens":    rec["add_hydrogens"],
            "pH":               rec["pH"],
            "n_H_added_est":    rec["n_H_added_est"],
            "out_path":         rec["out_path"],
        }
    return {
        "pdb":              pdb_id,
        "status":           status,
        "error":            error,
        "chains_biomol1":   "",
        "is_nmr":           "",
        "model_kept":       "",
        "n_atoms_written":  "",
        "n_H_atoms_written":"",
        "add_hydrogens":    add_h,
        "pH":               pH if add_h else "",
        "n_H_added_est":    "",
        "out_path":         out_path,
    }


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess raw PDB files: clean, validate, and protonate."
    )
    parser.add_argument("--in_dir",       required=False,
                        help="Directory of raw PDB files (required unless --csv is used)")
    parser.add_argument("--out_dir",      required=True,
                        help="Output directory for processed PDB files")
    parser.add_argument("--pattern",      default="*.pdb",
                        help="Filename glob pattern (default: *.pdb)")
    parser.add_argument("--overwrite",    action="store_true",
                        help="Reprocess PDBs that already exist in out_dir")
    parser.add_argument("--no_h",         action="store_true",
                        help="Skip adding hydrogens")
    parser.add_argument("--pH",           type=float, default=7.0,
                        help="pH for hydrogen addition (default: 7.0)")
    parser.add_argument("--min_residues", type=int, default=10,
                        help="Minimum residues per chain (default: 10)")
    parser.add_argument("--csv", dest="csv_path", default=None,
                        help="Optional CSV file containing PDB IDs to download")
    parser.add_argument("--pdb_column", default="pdb",
                        help="Column name in CSV for PDB IDs (default: pdb)")
    parser.add_argument("--download_dir", default=None,
                        help="Directory to download PDBs before preprocessing (default: in_dir)")
    parser.add_argument("--max_rows", type=int, default=None,
                        help="Optional max number of PDB IDs to process from CSV")

    args = parser.parse_args()

    if args.csv_path:
        raw_dir = args.download_dir or args.in_dir
        if raw_dir is None:
            parser.error("--download_dir or --in_dir must be provided when --csv is used")
        os.makedirs(raw_dir, exist_ok=True)
        preprocess_csv(
            csv_path=args.csv_path,
            pdb_column=args.pdb_column,
            raw_pdb_dir=raw_dir,
            out_dir=args.out_dir,
            overwrite=args.overwrite,
            add_h=not args.no_h,
            pH=args.pH,
            min_residues=args.min_residues,
            max_rows=args.max_rows,
        )
    else:
        if args.in_dir is None:
            parser.error("--in_dir is required when not using --csv")
        preprocess_dir(
            in_dir=args.in_dir,
            out_dir=args.out_dir,
            pattern=args.pattern,
            overwrite=args.overwrite,
            add_h=not args.no_h,
            pH=args.pH,
            min_residues=args.min_residues,
        )