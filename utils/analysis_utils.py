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

    train_neutral_activations_folder = "activations/train/formal"
    train_creative_activations_folder = "activations/train/creative"

    mdvs_output_file = "representations/mdvs.pt"
    thresholds_output_file = "representations/thresholds.pt"
    
    os.makedirs("representations", exist_ok=True)
    
    # Load activations from training dataset (both encouraging and neutral)
    a_train_neutral = load_activations(train_neutral_activations_folder)
    a_train_creative = load_activations(train_creative_activations_folder)

    # hidden_size = a_train_neutral[0][-1][0].shape[0]
    hidden_size = 32

    mdvs = {}  # Will store MDVs as mdvs[(layer, token)] = tensor of shape (4096,)
    thresholds= {} # store thresholds between classes as thresholds[(layer, token)]
    
    for layer in range(N_LAYERS):
        for token in range(1, MAX_TOKEN + 1):
            neutral_vecs = torch.stack(a_train_neutral[layer][token])  # shape: (n_samples, 4096)
            creative_vecs = torch.stack(a_train_creative[layer][token])          # shape: (n_samples, 4096)
            
            # Compute mean difference vector
            mdv = creative_vecs.mean(dim=0) - neutral_vecs.mean(dim=0)  # shape: (4096,)
            mdvs[(layer, token)] = mdv

            # Compute classification threshold
            proj_neutral = project_onto(neutral_vecs, mdv)  # (n_neutral,)
            proj_creative = project_onto(creative_vecs, mdv)          # (n_enc,)
            threshold = (proj_creative.mean() + proj_neutral.mean()) / 2
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

# -------------------------------
# PCA
# -------------------------------

def save_pca_lastlayer_lasttoken():
    # Input folders
    train_neutral_activations_folder = "activations/train/neutral"
    train_enc_activations_folder     = "activations/train/creative"
    val_neutral_activations_folder   = "activations/val/neutral"
    val_enc_activations_folder       = "activations/val/creative"
    test_neutral_activations_folder  = "activations/test/neutral"
    test_enc_activations_folder      = "activations/test/creative"

    # Output file
    pca_output_file = "representations/pca_lastlayer_lasttoken.pt"
    os.makedirs("representations", exist_ok=True)
    
    # Load activations (dicts: layer → token → list[tensors])
    a_train_neutral = load_activations(train_neutral_activations_folder)
    a_train_enc     = load_activations(train_enc_activations_folder)
    a_val_neutral   = load_activations(val_neutral_activations_folder)
    a_val_enc       = load_activations(val_enc_activations_folder)
    a_test_neutral  = load_activations(test_neutral_activations_folder)
    a_test_enc      = load_activations(test_enc_activations_folder)

    # Last layer + last token only
    last_layer = N_LAYERS - 1
    last_token = -1

    # Stack train vectors
    train_neutral_vecs = torch.stack(a_train_neutral[last_layer][last_token])
    train_enc_vecs     = torch.stack(a_train_enc[last_layer][last_token])
    X_train = torch.cat([train_neutral_vecs, train_enc_vecs], dim=0).cpu().numpy()
    y_train = torch.tensor([0]*len(train_neutral_vecs) + [1]*len(train_enc_vecs))

    # Stack val vectors
    val_neutral_vecs = torch.stack(a_val_neutral[last_layer][last_token])
    val_enc_vecs     = torch.stack(a_val_enc[last_layer][last_token])
    X_val = torch.cat([val_neutral_vecs, val_enc_vecs], dim=0).cpu().numpy()
    y_val = torch.tensor([0]*len(val_neutral_vecs) + [1]*len(val_enc_vecs))

    # Stack test vectors
    test_neutral_vecs = torch.stack(a_test_neutral[last_layer][last_token])
    test_enc_vecs     = torch.stack(a_test_enc[last_layer][last_token])
    X_test = torch.cat([test_neutral_vecs, test_enc_vecs], dim=0).cpu().numpy()
    y_test = torch.tensor([0]*len(test_neutral_vecs) + [1]*len(test_enc_vecs))

    # Fit PCA on train only
    pca = PCA(n_components=50)  # adjust as needed
    pca.fit(X_train)

    # Project all sets
    X_train_proj = pca.transform(X_train)
    X_val_proj   = pca.transform(X_val)
    X_test_proj  = pca.transform(X_test)

