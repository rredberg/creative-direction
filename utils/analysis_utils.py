import os
import torch
from collections import defaultdict
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Activations Handling
# -------------------------------

MAX_TOKEN = 20
N_LAYERS = 32

def load_activations(folder, max_token=MAX_TOKEN):
    """
    Loads activations from .pt files and organizes them by layer and token position from instruction end.
    Args:
        folder (str): Path to folder containing activation .pt files.
        max_token (int): Number of token positions after instruction to consider (e.g., 20 → +1 to +20).
        instruction_end_pos (int): Position where instruction ends. If None, assumes it's consistent across samples.
    Returns:
        activations: dict[layer_idx][token_pos_from_instruction] = list of token vectors (torch.Tensor)
    """
    activations = defaultdict(lambda: defaultdict(list))
    print(folder)
    files = sorted([f for f in os.listdir(folder) if f.endswith(".pt")])
    print(f"Found {len(files)} activation files in {folder}")
    
    for fname in files:
        path = os.path.join(folder, fname)
        data = torch.load(path)  # Dict[layer_idx] -> Tensor[num_tokens, hidden_dim]
        for layer_idx, tensor in data.items():
            seq_len = tensor.shape[0]
            
            for token_idx in range(1, max_token + 1):  # positions 1, 2, ..., max_token after instruction
                if token_idx < seq_len:  # Make sure we don't go beyond sequence length
                    vec = tensor[token_idx]  # token at position instruction_end + pos
                    activations[layer_idx][token_idx].append(vec)  # result: activations[layer][pos][n_sample]
    
    return activations



# -------------------------------
# Compute MDV
# -------------------------------

def save_mdvs():

    train_formal_activations_folder = "activations/train/formal"
    train_creative_activations_folder = "activations/train/creative"

    mdvs_output_file = "representations/mdvs.pt"
    thresholds_output_file = "representations/thresholds.pt"
    
    os.makedirs("representations", exist_ok=True)
    
    # Load activations from training dataset (both encouraging and formal)
    a_train_formal = load_activations(train_formal_activations_folder)
    a_train_creative = load_activations(train_creative_activations_folder)

    # hidden_size = a_train_formal[0][-1][0].shape[0]
    hidden_size = 32

    mdvs = {}  # Will store MDVs as mdvs[(layer, token)] = tensor of shape (4096,)
    thresholds= {} # store thresholds between classes as thresholds[(layer, token)]
    
    for layer in range(N_LAYERS):
        for token in range(1, MAX_TOKEN + 1):
            formal_vecs = torch.stack(a_train_formal[layer][token])  # shape: (n_samples, 4096)
            creative_vecs = torch.stack(a_train_creative[layer][token])          # shape: (n_samples, 4096)
            
            # Compute mean difference vector
            mdv = creative_vecs.mean(dim=0) - formal_vecs.mean(dim=0)  # shape: (4096,)
            mdvs[(layer, token)] = mdv

            # Compute classification threshold
            proj_formal = project_onto(formal_vecs, mdv)  # (n_formal,)
            proj_creative = project_onto(creative_vecs, mdv)          # (n_enc,)
            threshold = (proj_creative.mean() + proj_formal.mean()) / 2
            thresholds[(layer, token)] = threshold

            
    torch.save(mdvs, mdvs_output_file)
    torch.save(thresholds, thresholds_output_file)
    return mdvs, thresholds

# -------------------------------
# Projections
# -------------------------------

def project_onto(vector, direction):
    """Project vector(s) onto direction (not necessarily normalized)."""
    direction = direction / direction.norm()
    return (vector @ direction)