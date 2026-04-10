## Learning Protein–Protein Binding Free Energies from Interface Graphs and Physicochemical Descriptors

<img width="910" height="333" alt="cover_art_QZ" src="https://github.com/user-attachments/assets/0d10b41d-f045-439f-9bd2-5c6ce0cba508" />

Predicting protein–protein binding free energy (ΔG) from structure remains a central challenge in computational biophysics. Here, we present a graph neural network (GNN) that jointly learns from a residue-level graph representation of the binding interface with global physicochemical descriptors. We systematically investigate how training data dis-tribution affects model performance by comparing a full training set with a balanced subset enriched for extreme-affinity complexes. The proposed model is computationally efficient and provides interpretable insights into residue-level and physi-cochemical contributions to binding. On external validation, the model achieves a mean absolute er-ror (MAE) of 2.31 kcal/mol and shows moderate agreement with experimental ΔG values (Pearson r = 0.54, Spearman ρ = 0.58). 

For inference, please follow the steps below:

## Create and activate the environment
conda env create -f environment.yml
conda activate gnn-ddg

## Install PyG dependencies pinned to your torch build
pip install torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

## Download and preprocess PPI complex 
preprocess.py \
        --in_dir  /path/to/raw_pdbs \
        --out_dir /path/to/processed_pdbs

## Run inference with a trained GNN model on a folder of PDB files.
test.py \
        --pdb_dirs /path/to/test_pdbs \
        --model_path /path/to/model.pt \
        --out_dir /path/to/test_out \
        --labels_csv /path/to/test_labels.csv \
        --target_col exp_dG \
        --id_col PDB
