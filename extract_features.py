"""
extract_features.py
===========================

Standalone feature extraction for PPI binding affinity prediction.

This script combines:
  - Global feature extraction (IC, HB, BSA, SB, IC_density, %NIS_*).
  - Graph construction (interface + optional NIS nodes) for GNN input.

Usage (similar to extract_features_nis.py):
    python features/extract_features.py \
        --pdb_dirs /path/to/pdbs \
        --out_dir /path/to/outputs \
        [--nis] [--cache] [--no_cache]

The script writes:
  - {out_dir}/global_features.csv
  - {out_dir}/graphs.pkl

It also exposes functions for use from training/testing scripts.
"""

import os
import glob
import tempfile
import pickle
import argparse
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Ensure repo root is on path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import MDAnalysis as mda
import freesasa
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis
from MDAnalysis.lib.distances import distance_array
from Bio.PDB import PDBParser, NeighborSearch

try:
    from graph_utils import Data
except ImportError:
    from pipeline_scripts.graph_utils import Data

# Helper to parse residue numbers that may include insertion codes (e.g. '83C')
# Some PDBs embed insertion codes in the residue ID field (e.g. '184A').
_resid_re = re.compile(r"^(-?\d+)")

def _resid_to_int(resid):
    s = str(resid).strip()
    m = _resid_re.match(s)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot parse residue id: {resid}")


# For interface-only graphs (23-dim) when --nis is not requested
try:
    from pipeline_scripts.graph_features_old_keep import build_graph as build_graph_interface
except ImportError:
    try:
        from graph_features_old_keep import build_graph as build_graph_interface
    except ImportError as e:
        raise ImportError(
            "Could not import build_graph_interface from pipeline_scripts.graph_features_old_keep or graph_features_old_keep. "
            "Ensure this script is run from the repo root or that Python path includes the repo root."
        ) from e


# ============================================================
# Global Feature Extraction (from global_features.py)
# ============================================================

IC_CUTOFF    = 5.5   # Å, heavy-atom interface contact cutoff
RSA_THRESHOLD = 0.05  # minimum RSA for NIS residues
SB_CUTOFF    = 4.0   # Å, salt bridge cutoff

DONOR_ATOMS    = "name N NE NH1 NH2 ND2 NE2 NZ OG OG1 OH SG"
ACCEPTOR_ATOMS = "name O OD1 OD2 OE1 OE2 OG OG1 OH SD SG NE2"
HYDROGEN_ATOMS = "name H*"

POLAR   = {"SER","THR","ASN","GLN","TYR","CYS","TRP","HIS"}
CHARGED = {"ASP","GLU","LYS","ARG"}
APOLAR  = {"ALA","VAL","ILE","LEU","MET","PHE","PRO","GLY"}

REL_ASA_TOTAL = {
    "ALA":107.95,"CYS":134.28,"ASP":140.39,"GLU":172.25,"PHE":199.48,"GLY":80.10,
    "HIS":182.88,"ILE":175.12,"LYS":200.81,"LEU":178.63,"MET":194.15,"ASN":143.94,
    "PRO":136.13,"GLN":178.50,"ARG":238.76,"SER":116.50,"THR":139.27,"VAL":151.44,
    "TRP":249.36,"TYR":212.76,
}


@dataclass
class HBondParams:
    d_a_cutoff: float = 3.89
    d_h_cutoff: float = 1.2
    angle_min:  float = 90
    angle_max:  float = 270
    a_h_cutoff: float = 4.0


# ------------------------------------------------------------
# Shared helpers (chain detection)
# ------------------------------------------------------------

def autodetect_two_protein_chains_mda(u: mda.Universe) -> Tuple[str, str, str]:
    """Returns (chainA, chainB, field) using MDAnalysis Universe."""
    chainIDs = [c for c in set(u.atoms.chainIDs) if str(c).strip()]
    if len(chainIDs) >= 2:
        field, labels = "chainID", sorted(chainIDs)
    else:
        segids = [s for s in set(u.atoms.segids) if str(s).strip()]
        if len(segids) >= 2:
            field, labels = "segid", sorted(segids)
        else:
            raise ValueError("Cannot find >=2 chains via chainID or segid.")

    protein_chains = [
        lab for lab in labels
        if len(u.select_atoms(f"protein and {field} {lab}")) > 0
    ]
    if len(protein_chains) < 2:
        raise ValueError(f"Found <2 protein chains: {protein_chains}")
    return str(protein_chains[0]), str(protein_chains[1]), field


def get_two_chains_biopython(model) -> Tuple[str, str]:
    """Returns (chainA, chainB) using BioPython model."""
    chains = [c.id for c in model.get_chains()]
    if len(chains) < 2:
        raise ValueError("Found <2 chains in BioPython model.")
    return chains[0], chains[1]


# ------------------------------------------------------------
# Global features
# ------------------------------------------------------------

def compute_ic(pdb_path: str, d_cutoff: float = IC_CUTOFF) -> int:
    """Count interface residue-residue contacts (heavy-atom, cross-chain)."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", pdb_path)
    model = next(structure.get_models())

    atom_list = [a for a in model.get_atoms() if a.element != "H"]
    ns = NeighborSearch(atom_list)
    all_contacts = ns.search_all(radius=d_cutoff, level="R")

    ic_list = [c for c in all_contacts if c[0].parent.id != c[1].parent.id]
    return len(ic_list)


def _has_hydrogens(u: mda.Universe, field: str, chainA: str, chainB: str) -> bool:
    ch = f"({field} {chainA} or {field} {chainB})"
    return len(u.select_atoms(f"protein and {ch} and {HYDROGEN_ATOMS}")) > 0


def _count_hbonds_with_H(u, chainA, chainB, field, p: HBondParams) -> int:
    ch = f"({field} {chainA} or {field} {chainB})"
    hb = HydrogenBondAnalysis(
        universe=u,
        donors_sel=f"protein and {ch} and ({DONOR_ATOMS})",
        hydrogens_sel=f"protein and {ch} and ({HYDROGEN_ATOMS})",
        acceptors_sel=f"protein and {ch} and ({ACCEPTOR_ATOMS})",
        d_h_cutoff=p.d_h_cutoff,
        d_a_cutoff=p.d_a_cutoff,
        d_h_a_angle_cutoff=p.angle_min,
    )
    hb.run()

    n = 0
    for fr, d_idx, h_idx, a_idx, d_a_dist, ang in hb.results.hbonds:
        D = u.atoms[int(d_idx)]
        A = u.atoms[int(a_idx)]
        H = u.atoms[int(h_idx)]
        d_chain = str(getattr(D, field, "")).strip()
        a_chain = str(getattr(A, field, "")).strip()
        if {d_chain, a_chain} != {str(chainA), str(chainB)}:
            continue
        if not (p.angle_min <= float(ang) <= p.angle_max):
            continue
        if float(np.linalg.norm(H.position - A.position)) > p.a_h_cutoff:
            continue
        n += 1
    return n


def _count_hbonds_heavy_only(u, chainA, chainB, field, p: HBondParams) -> int:
    donors_A  = u.select_atoms(f"protein and {field} {chainA} and ({DONOR_ATOMS})")
    donors_B  = u.select_atoms(f"protein and {field} {chainB} and ({DONOR_ATOMS})")
    acc_A     = u.select_atoms(f"protein and {field} {chainA} and ({ACCEPTOR_ATOMS})")
    acc_B     = u.select_atoms(f"protein and {field} {chainB} and ({ACCEPTOR_ATOMS})")
    if min(len(donors_A), len(donors_B), len(acc_A), len(acc_B)) == 0:
        return 0
    n  = int(np.sum(distance_array(donors_A.positions, acc_B.positions, box=u.dimensions) <= p.d_a_cutoff))
    n += int(np.sum(distance_array(donors_B.positions, acc_A.positions, box=u.dimensions) <= p.d_a_cutoff))
    return n


def compute_hbonds(pdb_path: str, params: Optional[HBondParams] = None) -> int:
    """Count interface hydrogen bonds across two chains."""
    p = params or HBondParams()
    u = mda.Universe(pdb_path)
    chainA, chainB, field = autodetect_two_protein_chains_mda(u)
    if _has_hydrogens(u, field, chainA, chainB):
        return _count_hbonds_with_H(u, chainA, chainB, field, p)
    else:
        return _count_hbonds_heavy_only(u, chainA, chainB, field, p)


def compute_bsa(pdb_path: str) -> float:
    """Compute buried surface area (BSA) via freesasa."""
    u = mda.Universe(pdb_path)
    chainA, chainB, field = autodetect_two_protein_chains_mda(u)

    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as fA, \
         tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as fB:
        tmp_A, tmp_B = fA.name, fB.name

    try:
        u.select_atoms(f"protein and {field} {chainA}").write(tmp_A)
        u.select_atoms(f"protein and {field} {chainB}").write(tmp_B)

        sasa_complex = freesasa.calc(freesasa.Structure(pdb_path)).totalArea()
        sasa_A       = freesasa.calc(freesasa.Structure(tmp_A)).totalArea()
        sasa_B       = freesasa.calc(freesasa.Structure(tmp_B)).totalArea()
    finally:
        os.remove(tmp_A)
        os.remove(tmp_B)

    return (sasa_A + sasa_B - sasa_complex) / 2.0


def compute_salt_bridges(pdb_path: str, cutoff: float = SB_CUTOFF) -> int:
    """Count atom-atom salt bridge contacts across chains."""
    u = mda.Universe(pdb_path)
    _, _, field = autodetect_two_protein_chains_mda(u)

    pos_sel = "protein and (resname LYS ARG) and (name NZ NE NH1 NH2)"
    neg_sel = "protein and (resname ASP GLU) and (name OD1 OD2 OE1 OE2)"

    pos = u.select_atoms(pos_sel)
    neg = u.select_atoms(neg_sel)

    if len(pos) == 0 or len(neg) == 0:
        return 0

    D = distance_array(pos.positions, neg.positions, box=u.dimensions)
    i_pos, i_neg = np.where(D <= cutoff)

    n = 0
    for ip, ineg in zip(i_pos, i_neg):
        p_chain = str(getattr(pos[ip],  field, "")).strip()
        n_chain = str(getattr(neg[ineg], field, "")).strip()
        if p_chain and n_chain and p_chain != n_chain:
            n += 1
    return n


def _parse_resnum(resnum_str: str) -> Tuple[int, str]:
    s = str(resnum_str).strip()
    m = re.match(r"^(-?\d+)\s*([A-Za-z]?)$", s)
    if m:
        return int(m.group(1)), (m.group(2) or "")
    m2 = re.match(r"^(-?\d+)", s)
    if m2:
        tail = s[len(m2.group(1)):].strip()
        return int(m2.group(1)), (tail[:1] if tail and tail[0].isalpha() else "")
    raise ValueError(f"Cannot parse residueNumber='{resnum_str}'")


def _interface_residue_keys(model, cutoff: float = IC_CUTOFF):
    chA, chB = get_two_chains_biopython(model)
    atom_list = list(model.get_atoms())
    ns = NeighborSearch(atom_list)

    iface = set()
    for a in model[chA].get_atoms():
        if a.element == "H":
            continue
        for b in ns.search(a.coord, cutoff, level="A"):
            if b.element == "H":
                continue
            if b.get_parent().get_parent().id != chB:
                continue
            for res in (a.get_parent(), b.get_parent()):
                iface.add((
                    res.get_parent().id,
                    int(res.id[1]),
                    (res.id[2] or "").strip()
                ))
    return iface


def _residue_asa(pdb_path: str) -> Tuple[Dict, Dict]:
    fs = freesasa.Structure(pdb_path)
    result = freesasa.calc(fs)
    res_asa, res_name = {}, {}
    for i in range(fs.nAtoms()):
        chain = (str(fs.chainLabel(i)) or "").strip()
        resseq, icode = _parse_resnum(fs.residueNumber(i))
        resn = (fs.residueName(i) or "").strip().upper()
        key = (chain, resseq, icode)
        res_asa[key]  = res_asa.get(key, 0.0) + float(result.atomArea(i))
        res_name[key] = resn
    return res_asa, res_name


def compute_nis(pdb_path: str) -> Tuple[float, float, float]:
    """Returns (%NIS_polar, %NIS_charged, %NIS_apolar)."""
    parser = PDBParser(QUIET=True)
    model = parser.get_structure("x", pdb_path)[0]
    iface = _interface_residue_keys(model, cutoff=IC_CUTOFF)
    res_asa, res_name = _residue_asa(pdb_path)

    counts = {"polar": 0, "charged": 0, "apolar": 0}
    total = 0

    for key, asa in res_asa.items():
        resn = res_name.get(key)
        if resn not in REL_ASA_TOTAL:
            continue
        rsa = float(asa) / float(REL_ASA_TOTAL[resn])
        if rsa < RSA_THRESHOLD or key in iface:
            continue
        if resn in POLAR:
            counts["polar"] += 1
        elif resn in CHARGED:
            counts["charged"] += 1
        elif resn in APOLAR:
            counts["apolar"] += 1
        else:
            continue
        total += 1

    if total == 0:
        return 0.0, 0.0, 0.0
    return (
        100.0 * counts["polar"]   / total,
        100.0 * counts["charged"] / total,
        100.0 * counts["apolar"]  / total,
    )


def compute_all_features(pdb_path: str) -> Dict:
    """Compute all global features for a single PDB file."""
    pdb_id = os.path.splitext(os.path.basename(pdb_path))[0].upper()

    ic  = compute_ic(pdb_path)
    hb  = compute_hbonds(pdb_path)
    bsa = compute_bsa(pdb_path)
    sb  = compute_salt_bridges(pdb_path)
    ic_density = float(ic) / float(bsa) if bsa > 0 else 0.0
    nis_polar, nis_charged, nis_apolar = compute_nis(pdb_path)

    return {
        "PDB":          pdb_id,
        "IC":           ic,
        "HB":           hb,
        "BSA":          round(bsa, 3),
        "SB":           sb,
        "IC_density":   round(ic_density, 6),
        "%NIS_polar":   round(nis_polar, 3),
        "%NIS_charged": round(nis_charged, 3),
        "%NIS_apolar":  round(nis_apolar, 3),
    }


# ============================================================
# Graph Feature Extraction (from graph_features_nis.py)
# ============================================================

AA_LIST = [
    "ALA","CYS","ASP","GLU","PHE","GLY","HIS","ILE","LYS","LEU",
    "MET","ASN","PRO","GLN","ARG","SER","THR","VAL","TRP","TYR"
]
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

CHARGED = {"ASP","GLU","LYS","ARG","HIS"}
POLAR   = {"SER","THR","ASN","GLN","TYR","CYS"}
APOLAR  = {"ALA","VAL","ILE","LEU","MET","PHE","TRP","PRO","GLY"}

# Zone indices
ZONE_INTERFACE  = 0
ZONE_NIS_SHELL  = 1
ZONE_NIS_DISTAL = 2

NODE_DIM = 29  # 20 AA + 3 class + 3 zone + 3 shell_weight


def one_hot(idx, K: int) -> np.ndarray:
    v = np.zeros(K, dtype=np.float32)
    if idx is not None:
        v[idx] = 1.0
    return v


def residue_class_onehot(resname: str) -> np.ndarray:
    """3-dim: [charged, polar, apolar]"""
    r = str(resname).upper()
    v = np.zeros(3, dtype=np.float32)
    if r in CHARGED:  v[0] = 1.0
    elif r in POLAR:  v[1] = 1.0
    elif r in APOLAR: v[2] = 1.0
    return v


def zone_onehot(zone: int) -> np.ndarray:
    """3-dim: [interface, NIS_shell, NIS_distal]"""
    v = np.zeros(3, dtype=np.float32)
    v[zone] = 1.0
    return v


def shell_weight_vector(zone: int, d_to_interface: float, decay: float = 20.0) -> np.ndarray:
    """Compute shell weight vector for a residue."""
    v = np.zeros(3, dtype=np.float32)
    if zone == ZONE_INTERFACE:
        v[0] = 1.0
    else:
        v[1] = float(np.exp(-d_to_interface / decay))
    return v


def autodetect_two_protein_chains(u: mda.Universe):
    """Returns (chain_field, chainA, chainB). Prefers chainID; falls back to segid."""
    chainIDs = [c for c in set(u.atoms.chainIDs) if str(c).strip()]
    if len(chainIDs) >= 2:
        chain_field, labels = "chainID", sorted(chainIDs)
    else:
        segids = [s for s in set(u.atoms.segids) if str(s).strip()]
        if len(segids) >= 2:
            chain_field, labels = "segid", sorted(segids)
        else:
            raise ValueError("Could not find >=2 chains via chainID or segid.")

    protein_chains = [
        ch for ch in labels
        if len(u.select_atoms(f"protein and {chain_field} {ch}")) > 0
    ]
    if len(protein_chains) < 2:
        raise ValueError(f"Found <2 protein chains: {protein_chains}")

    return chain_field, protein_chains[0], protein_chains[1]


def interface_resids_heavy_atom(
    u: mda.Universe,
    chain_field: str,
    chainA: str,
    chainB: str,
    cutoff: float = 5.5,
):
    """Find interface residues by ANY heavy-atom cross-chain contact <= cutoff Å."""
    heavyA = u.select_atoms(f"protein and {chain_field} {chainA} and not name H*")
    heavyB = u.select_atoms(f"protein and {chain_field} {chainB} and not name H*")

    if len(heavyA) == 0 or len(heavyB) == 0:
        return [], []

    pairs = mda.lib.distances.capped_distance(
        heavyA.positions,
        heavyB.positions,
        max_cutoff=float(cutoff),
        return_distances=False,
    )

    if pairs is None or len(pairs) == 0:
        return [], []

    residsA, residsB = set(), set()
    for iA, iB in pairs:
        residsA.add(_resid_to_int(heavyA[int(iA)].resid))
        residsB.add(_resid_to_int(heavyB[int(iB)].resid))

    return sorted(residsA), sorted(residsB)


MAX_ASA = {
    "ALA": 113.0, "ARG": 241.0, "ASN": 158.0, "ASP": 151.0, "CYS": 140.0,
    "GLN": 189.0, "GLU": 183.0, "GLY":  85.0, "HIS": 194.0, "ILE": 182.0,
    "LEU": 180.0, "LYS": 211.0, "MET": 204.0, "PHE": 218.0, "PRO": 143.0,
    "SER": 122.0, "THR": 146.0, "TRP": 259.0, "TYR": 229.0, "VAL": 160.0,
}
DEFAULT_MAX_ASA = 180.0


def compute_rsa_per_residue(pdb_path: str, chain_field: str,
                             chainA: str, chainB: str) -> dict:
    """Compute RSA per residue using freesasa."""
    u = mda.Universe(pdb_path)

    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        u.select_atoms(
            f"protein and ({chain_field} {chainA} or {chain_field} {chainB})"
        ).write(tmp_path)

        structure = freesasa.Structure(tmp_path)
        result    = freesasa.calc(structure)

        rsa_map = {}
        for i in range(structure.nAtoms()):
            chain  = structure.chainLabel(i)
            resnum = _resid_to_int(structure.residueNumber(i))
            resname= structure.residueName(i).strip().upper()
            asa    = result.atomArea(i)

            key = (chain, resnum)
            rsa_map[key] = rsa_map.get(key, 0.0) + asa

        rsa_result = {}
        u2 = mda.Universe(tmp_path)
        for res in u2.residues:
            chain  = str(getattr(res.atoms[0], "chainID", "")).strip()
            if not chain:
                chain = str(getattr(res.atoms[0], "segid", "")).strip()
            resnum  = _resid_to_int(res.resid)
            resname = str(res.resname).upper()
            max_asa = MAX_ASA.get(resname, DEFAULT_MAX_ASA)
            total_asa = rsa_map.get((chain, resnum), 0.0)
            rsa_result[(chain, resnum)] = min(1.0, total_asa / max_asa)

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return rsa_result


def interface_center(u: mda.Universe, chain_field: str,
                     chainA: str, chainB: str,
                     iface_residsA: list, iface_residsB: list) -> np.ndarray:
    """Compute centroid of interface CA atoms."""
    selA = (f"protein and name CA and {chain_field} {chainA} "
            f"and resid {' '.join(map(str, iface_residsA))}")
    selB = (f"protein and name CA and {chain_field} {chainB} "
            f"and resid {' '.join(map(str, iface_residsB))}")
    ca_iface = u.select_atoms(selA) + u.select_atoms(selB)
    return ca_iface.positions.mean(axis=0).astype(np.float32)


def build_graph_nis(
    pdb_path:        str,
    edge_cutoff:     float = 8.0,
    iface_cutoff:    float = 5.5,
    nis_edge_cutoff: float = 12.0,
    nis_shell_cutoff:float = 15.0,
    rsa_cutoff:      float = 0.05,
    shell_decay:     float = 20.0,
) -> Data:
    """Build a NIS-aware PyG Data object for one PDB file."""
    pdb_id = os.path.splitext(os.path.basename(pdb_path))[0].upper()

    u = mda.Universe(pdb_path)
    chain_field, chainA, chainB = autodetect_two_protein_chains(u)

    residsA_iface, residsB_iface = interface_resids_heavy_atom(
        u, chain_field, chainA, chainB, cutoff=iface_cutoff
    )
    iface_set = set()
    for r in residsA_iface:
        iface_set.add((chainA, r))
    for r in residsB_iface:
        iface_set.add((chainB, r))

    if not iface_set:
        raise ValueError(f"{pdb_id}: No interface residues found.")

    rsa_map = compute_rsa_per_residue(pdb_path, chain_field, chainA, chainB)

    iface_center_pos = interface_center(
        u, chain_field, chainA, chainB, residsA_iface, residsB_iface
    )

    all_ca = u.select_atoms(
        f"protein and name CA and "
        f"({chain_field} {chainA} or {chain_field} {chainB})"
    )

    node_atoms  = []
    node_zones  = []
    node_dists  = []

    for atom in all_ca:
        chain = str(getattr(atom, "chainID", "")).strip()
        if not chain:
            chain = str(getattr(atom, "segid", "")).strip()
        resid = _resid_to_int(atom.resid)

        rsa = rsa_map.get((chain, resid), 0.0)

        is_interface = (chain, resid) in iface_set
        is_surface   = (rsa >= rsa_cutoff) or is_interface

        if not is_surface:
            continue

        d = float(np.linalg.norm(atom.position - iface_center_pos))

        if is_interface:
            zone = ZONE_INTERFACE
        elif d <= nis_shell_cutoff:
            zone = ZONE_NIS_SHELL
        else:
            zone = ZONE_NIS_DISTAL

        node_atoms.append(atom)
        node_zones.append(zone)
        node_dists.append(d)

    if len(node_atoms) == 0:
        raise ValueError(f"{pdb_id}: No surface nodes found.")

    pos = np.array([a.position for a in node_atoms], dtype=np.float32)
    zones = np.array(node_zones, dtype=np.int32)

    src_list, dst_list = [], []
    n = len(node_atoms)

    d2_matrix = np.sum((pos[:, None, :] - pos[None, :, :]) ** 2, axis=-1)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            zi, zj = int(zones[i]), int(zones[j])
            d = float(np.sqrt(d2_matrix[i, j]))

            if zi == ZONE_INTERFACE and zj == ZONE_INTERFACE:
                if d < edge_cutoff:
                    src_list.append(i)
                    dst_list.append(j)

            elif (zi == ZONE_INTERFACE and zj == ZONE_NIS_SHELL) or \
                 (zi == ZONE_NIS_SHELL  and zj == ZONE_INTERFACE):
                if d < nis_edge_cutoff:
                    src_list.append(i)
                    dst_list.append(j)

    if len(src_list) == 0:
        for i in range(n):
            for j in range(n):
                if i != j and zones[i] == ZONE_INTERFACE and zones[j] == ZONE_INTERFACE:
                    src_list.append(i)
                    dst_list.append(j)

    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    edge_dist  = np.sqrt(d2_matrix[src_list, dst_list]).reshape(-1, 1).astype(np.float32)

    x_list = []
    for idx, atom in enumerate(node_atoms):
        resname = str(atom.resname).upper()
        zone    = int(node_zones[idx])
        d_iface = float(node_dists[idx])

        aa_oh      = one_hot(AA_TO_IDX.get(resname, None), len(AA_LIST))
        cls_oh     = residue_class_onehot(resname)
        zone_oh    = zone_onehot(zone)
        shell_w    = shell_weight_vector(zone, d_iface, decay=20.0)

        x_list.append(np.concatenate([aa_oh, cls_oh, zone_oh, shell_w]))

    x = np.stack(x_list, axis=0).astype(np.float32)

    n_interface  = int(np.sum(zones == ZONE_INTERFACE))
    n_nis_shell  = int(np.sum(zones == ZONE_NIS_SHELL))
    n_nis_distal = int(np.sum(zones == ZONE_NIS_DISTAL))

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        pos=torch.tensor(pos, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_dist, dtype=torch.float32),
        pdb_id=pdb_id,
    )
    data.n_interface  = n_interface
    data.n_nis_shell  = n_nis_shell
    data.n_nis_distal = n_nis_distal

    return data


# ============================================================
# Extraction wrapper
# ============================================================

GLOBAL_FEATURES = [
    "IC",
    "HB",
    "BSA",
    "SB",
    "IC_density",
    "%NIS_polar",
    "%NIS_charged",
    "%NIS_apolar",
]


def collect_pdb_paths(pdb_dirs) -> list:
    pdb_dirs = pdb_dirs if isinstance(pdb_dirs, (list, tuple)) else [pdb_dirs]
    paths = []
    for d in pdb_dirs:
        if not os.path.isdir(d):
            print(f"[WARN] Not a directory, skipping: {d}")
            continue
        paths.extend(glob.glob(os.path.join(d, "*.pdb")))
    return sorted(set(paths))


def extract_all(
    pdb_dirs,
    out_dir:      str  = ".",
    use_cache:    bool = True,
    use_nis:      bool = False,
    verbose:      bool = True,
):
    """Extract global features and graphs for all PDBs."""
    os.makedirs(out_dir, exist_ok=True)

    global_csv  = os.path.join(out_dir, "global_features.csv")
    graphs_pkl  = os.path.join(out_dir, "graphs.pkl")

    cached_global_ids = set()
    global_rows = []
    if use_cache and os.path.exists(global_csv):
        existing_df = pd.read_csv(global_csv)
        cached_global_ids = set(existing_df["PDB"].astype(str).str.upper())
        global_rows = existing_df.to_dict("records")
        if verbose:
            print(f"[CACHE] Loaded {len(cached_global_ids)} global feature entries.")

    cached_graph_ids = set()
    graphs = {}
    if use_cache and os.path.exists(graphs_pkl):
        with open(graphs_pkl, "rb") as f:
            graphs = pickle.load(f)
        cached_graph_ids = set(graphs.keys())
        if verbose:
            print(f"[CACHE] Loaded {len(cached_graph_ids)} cached graphs.")

    pdb_paths = collect_pdb_paths(pdb_dirs)
    if verbose:
        print(f"\nFound {len(pdb_paths)} PDB files.")
        print("-" * 50)

    n_global_ok = n_global_fail = n_global_cached = 0
    n_graph_ok  = n_graph_fail  = n_graph_cached  = 0

    for pdb_path in pdb_paths:
        pdb_id = os.path.splitext(os.path.basename(pdb_path))[0].upper()

        if pdb_id in cached_global_ids:
            n_global_cached += 1
            if verbose:
                print(f"[CACHED] {pdb_id} global features")
        else:
            try:
                row = compute_all_features(pdb_path)
                global_rows.append(row)
                n_global_ok += 1
                if verbose:
                    print(
                        f"[OK] {pdb_id}  "
                        f"IC={row['IC']}  HB={row['HB']}  "
                        f"BSA={row['BSA']:.1f}  SB={row['SB']}"
                    )
            except Exception as e:
                global_rows.append({"PDB": pdb_id, "error": str(e)})
                n_global_fail += 1
                if verbose:
                    print(f"[FAIL] {pdb_id} global: {e}")

        if pdb_id in cached_graph_ids:
            n_graph_cached += 1
            if verbose:
                print(f"[CACHED] {pdb_id} graph")
        else:
            try:
                if use_nis:
                    data = build_graph_nis(pdb_path)
                else:
                    data = build_graph_interface(pdb_path)

                graphs[pdb_id] = data
                n_graph_ok += 1
                if verbose:
                    if use_nis and hasattr(data, "n_interface"):
                        print(
                            f"[OK] {pdb_id}  "
                            f"nodes={data.x.shape[0]}  "
                            f"edges={data.edge_index.shape[1]}  "
                            f"(iface={data.n_interface}  "
                            f"shell={data.n_nis_shell}  "
                            f"distal={data.n_nis_distal})"
                        )
                    else:
                        print(
                            f"[OK] {pdb_id}  "
                            f"nodes={data.x.shape[0]}  "
                            f"edges={data.edge_index.shape[1]}"
                        )
            except Exception as e:
                n_graph_fail += 1
                if verbose:
                    print(f"[FAIL] {pdb_id} graph: {e}")

    global_df = pd.DataFrame(global_rows)
    global_df.to_csv(global_csv, index=False)

    with open(graphs_pkl, "wb") as f:
        pickle.dump(graphs, f)

    if verbose:
        graph_mode = "NIS-aware (29-dim)" if use_nis else "interface-only (23-dim)"
        print("\n" + "=" * 50)
        print(f"EXTRACTION COMPLETE  [{graph_mode}]")
        print("=" * 50)
        print(f"Global features:")
        print(f"  OK={n_global_ok}  CACHED={n_global_cached}  FAIL={n_global_fail}")
        print(f"Graphs:")
        print(f"  OK={n_graph_ok}  CACHED={n_graph_cached}  FAIL={n_graph_fail}")
        print(f"\nOutputs written to: {out_dir}")
        print(f"  {global_csv}")
        print(f"  {graphs_pkl}")

    return global_df, graphs


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract global and graph features from a folder of PDB files."
    )
    parser.add_argument(
        "--pdb_dirs", nargs="+", required=True,
        help="One or more directories containing .pdb files"
    )
    parser.add_argument(
        "--out_dir", required=True,
        help="Output directory for global_features.csv and graphs.pkl"
    )
    parser.add_argument(
        "--nis", action="store_true",
        help="Use NIS-aware graph builder (29-dim nodes: interface + surface NIS)."
    )
    parser.add_argument(
        "--cache", action="store_true", default=True,
        help="Skip PDBs already present in existing outputs (default: True)"
    )
    parser.add_argument(
        "--no_cache", action="store_true",
        help="Reprocess all PDBs, ignoring any existing cache"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-PDB output"
    )
    args = parser.parse_args()

    use_cache = not args.no_cache

    extract_all(
        pdb_dirs=args.pdb_dirs,
        out_dir=args.out_dir,
        use_cache=use_cache,
        use_nis=args.nis,
        verbose=not args.quiet,
    )
