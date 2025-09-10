import torch
import os
from utils.analysis_utils import load_activations, MAX_TOKEN, N_LAYERS, save_mdvs, project_onto
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

import numpy as np

def evaluate_on_split(split_name, formal_folder, creative_folder, mdvs, thresholds):
    print(f"\nEvaluating on {split_name} set...")
    a_formal = load_activations(formal_folder)
    a_creative = load_activations(creative_folder)
    
    best_auc = -float('inf')
    best_coords = None
    metrics = {}
    
    for layer in range(N_LAYERS):
        for token in range(1, MAX_TOKEN + 1):
            mdv = mdvs[(layer, token)]
            threshold = thresholds[(layer, token)]

            # Get activations
            test_formal = torch.stack(a_formal[layer][token])  # (n_samples, 4096)
            test_creative = torch.stack(a_creative[layer][token])          # (n_samples, 4096)

            # Project onto MDV
            proj_formal = project_onto(test_formal, mdv)  # (n_formal,)
            proj_creative = project_onto(test_creative, mdv)          # (n_enc,)

            # Combine and label
            X = torch.cat([proj_formal, proj_creative]).numpy()
            y = torch.cat([
                torch.zeros(len(proj_formal)),
                torch.ones(len(proj_creative))
            ]).numpy()

            # Threshold = midpoint between means
            y_pred = (X > threshold.item()).astype(int)

            acc = accuracy_score(y, y_pred)
            auc = roc_auc_score(y, X)

            metrics[(layer, token)] = {'accuracy': acc, 'auc': auc}

            print(f"{split_name} classification AUC at layer {layer}, token {token}: {auc:.4f}  Acc: {acc:.4f}")

            if auc > best_auc:
                best_auc = auc
                best_coords = (layer, token)

    print(f"\nBest {split_name} classification AUC at layer {best_coords[0]}, token {best_coords[1]}: {best_auc:.4f}")

    return metrics, best_coords, best_auc

def find_best_coords_on_val(val_formal_folder, val_creative_folder, mdvs, thresholds):
    print("\nFinding best coords on validation set...")
    val_metrics, best_coords, best_auc = evaluate_on_split("Validation", val_formal_folder, val_creative_folder, mdvs, thresholds)
    return best_coords

def evaluate_test_given_coords(test_formal_folder, test_creative_folder, mdvs, best_coords, thresholds):
    print("\nEvaluating on test set with best coords from validation...")
    
    layer, token = best_coords
    a_formal = load_activations(test_formal_folder)
    a_creative = load_activations(test_creative_folder)

    mdv = mdvs[(layer, token)]
    thresholds = threshold[(layer, token)]

    test_formal = torch.stack(a_formal[layer][token])  # (n_samples, hidden_dim)
    test_creative = torch.stack(a_creative[layer][token])

    proj_formal = project_onto(test_formal, mdv)
    proj_creative = project_onto(test_creative, mdv)

    X = torch.cat([proj_formal, proj_creative]).numpy()
    y = torch.cat([
        torch.zeros(len(proj_formal)),
        torch.ones(len(proj_creative))
    ]).numpy()

    y_pred = (X > threshold.item()).astype(int)

    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, X)

    print(f"Test classification AUC at layer {layer}, token {token}: {auc:.4f}  Acc: {acc:.4f}")

    return {'accuracy': acc, 'auc': auc}, (layer, token)



def main():
    # Paths for each split
    train_formal_folder = "activations/train/formal"
    train_creative_folder = "activations/train/creative"
    val_formal_folder = "activations/val/formal"
    val_creative_folder = "activations/val/creative"
    test_formal_folder = "activations/test/formal"
    test_creative_folder = "activations/test/creative"

    # Compute or load MDVs and thresholds from training data
    mdvs, thresholds = save_mdvs()  # assumes it uses train activations internally

    # 1) Evaluate on validation set to find best coords
    val_metrics, val_best_coords, val_best_auc = evaluate_on_split("Validation", val_formal_folder, val_creative_folder, mdvs, thresholds)

    print(f"\nBest Validation coords: {val_best_coords} with AUC: {val_best_auc:.4f}")

    # 2) Evaluate on test set **only** at best coords from validation
    layer, token = val_best_coords

    # Load test activations
    a_formal = load_activations(test_formal_folder)
    a_creative = load_activations(test_creative_folder)

    mdv = mdvs[(layer, token)]
    threshold = thresholds[(layer, token)]

    a_val_formal = load_activations(val_formal_folder)
    a_val_creative = load_activations(val_creative_folder)


    # Extract test activations for chosen layer/token
    test_formal = torch.stack(a_formal[layer][token])
    test_creative = torch.stack(a_creative[layer][token])

    # Project onto MDV
    proj_formal = project_onto(test_formal, mdv)
    proj_creative = project_onto(test_creative, mdv)

    # Combine and label
    X = torch.cat([proj_formal, proj_creative]).numpy()
    y = torch.cat([torch.zeros(len(proj_formal)), torch.ones(len(proj_creative))]).numpy()

    y_pred = (X > threshold.item()).astype(int)

    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, X)

    print(f"\nTest classification AUC at layer {layer}, token {token}: {auc:.4f}  Acc: {acc:.4f}")

    # Plot histograms with different colors
    plt.figure(figsize=(10, 6))
    
    # Convert to numpy if they're torch tensors
    proj_formal_np = proj_formal.cpu().numpy() if hasattr(proj_formal, 'cpu') else proj_formal
    proj_creative_np = proj_creative.cpu().numpy() if hasattr(proj_creative, 'cpu') else proj_creative

    os.makedirs("plots", exist_ok=True)
    
    # Create histograms with some transparency so overlapping regions are visible

    plt.hist(proj_formal_np, bins=20, alpha=0.7, color='blue', label='Formal', density=True)
    plt.hist(proj_creative_np, bins=20, alpha=0.7, color='red', label='Creative', density=True)
    
    plt.xlabel('Projection onto Mean Difference Vector')
    plt.ylabel('Density')
    plt.title('Distribution of Projections: Formal vs Creative')
    plt.legend()
    plt.grid(True, alpha=0.3)

    threshold = (proj_creative.mean() + proj_formal.mean()) / 2

    
    # Add vertical lines at means for reference
    plt.axvline(np.mean(proj_formal_np), color='blue', linestyle='--', alpha=0.8, label='Formal Mean')
    plt.axvline(np.mean(proj_creative_np), color='red', linestyle='--', alpha=0.8, label='Creative Mean')
    plt.axvline(threshold, color='purple', linestyle='--', alpha=0.8, label='Threshold')
    plt.legend()

    
    plt.tight_layout()
    plt.show()
    plt.savefig(f"plots/mdv_projection_formal_v_creative.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"plots/mdv_projection_formal_v_creative.pdf", bbox_inches='tight')


if __name__ == "__main__":
    main()
