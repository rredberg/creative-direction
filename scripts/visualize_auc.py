import torch
from utils.analysis_utils import load_activations, MAX_TOKEN, N_LAYERS, save_mdvs, project_onto
from sklearn.metrics import accuracy_score, roc_auc_score

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import random
import os

RANDOM_SEED = 32

def get_best_auc_per_token(split_name, formal_folder, creative_folder, mdvs, thresholds):
    """Get the best AUC for each token position across all layers"""
    print(f"\nFinding best AUC per token on {split_name} set...")
    a_formal = load_activations(formal_folder)
    a_creative = load_activations(creative_folder)
    
    token_results = {}  # token -> {'auc': best_auc, 'layer': best_layer, 'acc': best_acc}
    
    for token in range(1, MAX_TOKEN + 1):
        best_auc_for_token = -float('inf')
        best_layer_for_token = None
        best_acc_for_token = None
        
        for layer in range(N_LAYERS):
            mdv = mdvs[(layer, token)]
            threshold = thresholds[(layer, token)]
            # Get activations
            test_formal = torch.stack(a_formal[layer][token])
            test_creative = torch.stack(a_creative[layer][token])
            # Project onto MDV
            proj_formal = project_onto(test_formal, mdv)
            proj_creative = project_onto(test_creative, mdv)
            # Combine and label
            X = torch.cat([proj_formal, proj_creative]).numpy()
            y = torch.cat([
                torch.zeros(len(proj_formal)),
                torch.ones(len(proj_creative))
            ]).numpy()
                
            y_pred = (X > threshold.item()).astype(int)
            acc = accuracy_score(y, y_pred)
            auc = roc_auc_score(y, X)
            
            if auc > best_auc_for_token:
                best_auc_for_token = auc
                best_layer_for_token = layer
                best_acc_for_token = acc
        
        token_results[token] = {
            'auc': best_auc_for_token, 
            'layer': best_layer_for_token, 
            'acc': best_acc_for_token
        }
        print(f"{split_name} best AUC for token {token}: {best_auc_for_token:.4f} (layer {best_layer_for_token})  Acc: {best_acc_for_token:.4f}")
    
    return token_results

def get_best_auc_per_layer(split_name, formal_folder, creative_folder, mdvs, thresholds):
    """Get the best AUC for each layer across all token positions"""
    print(f"\nFinding best AUC per layer on {split_name} set...")
    a_formal = load_activations(formal_folder)
    a_creative = load_activations(creative_folder)
    
    layer_results = {}  # layer -> {'auc': best_auc, 'token': best_token, 'acc': best_acc}
    
    for layer in range(N_LAYERS):
        best_auc_for_layer = -float('inf')
        best_token_for_layer = None
        best_acc_for_layer = None
        
        # for token in range(-MAX_TOKEN, 0):
        for token in range(1, MAX_TOKEN + 1):
            mdv = mdvs[(layer, token)]
            threshold = thresholds[(layer, token)]
            # Get activations
            test_formal = torch.stack(a_formal[layer][token])
            test_creative = torch.stack(a_creative[layer][token])
            # Project onto MDV
            proj_formal = project_onto(test_formal, mdv)
            proj_creative = project_onto(test_creative, mdv)
            # Combine and label
            X = torch.cat([proj_formal, proj_creative]).numpy()
            y = torch.cat([
                torch.zeros(len(proj_formal)),
                torch.ones(len(proj_creative))
            ]).numpy()
            y_pred = (X > threshold.item()).astype(int)
            acc = accuracy_score(y, y_pred)
            auc = roc_auc_score(y, X)
            
            if auc > best_auc_for_layer:
                best_auc_for_layer = auc
                best_token_for_layer = token
                best_acc_for_layer = acc
        
        layer_results[layer] = {
            'auc': best_auc_for_layer, 
            'token': best_token_for_layer, 
            'acc': best_acc_for_layer
        }
        print(f"{split_name} best AUC for layer {layer}: {best_auc_for_layer:.4f} (token {best_token_for_layer})  Acc: {best_acc_for_layer:.4f}")
    
    return layer_results

def plot_auc_results(token_results, layer_results, save_dir="plots"):
    """Generate and save AUC plots"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    plt.ylim(0.75, 1.0)
    # plt.legend(['AUC = 0.9'], loc='lower left')


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: AUC across token positions
    tokens = sorted(token_results.keys())  # Should be [-20, -19, ..., -1]
    token_aucs = [token_results[t]['auc'] for t in tokens]

    
    ax1.plot(tokens, token_aucs, 'o-', linewidth=2, markersize=6, color='#2E86C1')
    ax1.set_xlabel('Post-Instruction Token Position', fontsize=12)
    ax1.set_ylabel('Best AUC across layers', fontsize=12)
    ax1.set_title('AUC vs Token Position\n(Max over all layers for each token)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.75, 1.0)  # Focus on the interesting range
    
    # Add horizontal line at AUC = 0.9 for reference
    ax1.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='AUC = 0.8')
    ax1.legend(loc='lower left', bbox_to_anchor=(0.02, 0.02))
    
    # Plot 2: AUC across layers  
    layers = sorted(layer_results.keys())
    layer_aucs = [layer_results[l]['auc'] for l in layers]
    
    ax2.plot(layers, layer_aucs, 'o-', linewidth=2, markersize=6, color='#2E86C1')
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('Best AUC across tokens', fontsize=12)
    ax2.set_title('AUC vs Layer\n(Max over all tokens for each layer)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.75, 1.0)
    
    # Add horizontal line at AUC = 0.9 for reference
    ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='AUC = 0.8')
    ax2.legend(loc="lower left")
    
    # plt.tight_layout()

    plt.savefig(f"{save_dir}/auc_analysis_formal_v_creative.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_dir}/auc_analysis_formal_v_creative.pdf", bbox_inches='tight')  # Also save as PDF
    plt.show()

    plt.clf()

    plt.style.use('default')
    plt.ylim(0.7, 1.0)
    plt.legend(['Accuracy = 0.9'], loc='lower left')


    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))

    token_accs = [token_results[t]['acc'] for t in tokens]

    ax3.plot(tokens, token_accs, 'o-', linewidth=2, markersize=6, color='#2E86C1')
    ax3.set_xlabel('Post-Instruction Token Position', fontsize=12)
    ax3.set_ylabel('Best accuracy across layers', fontsize=12)
    ax3.set_title('Accuracy vs Token Position\n(Max over all layers for each token)', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0.7, 1.0)  # Focus on the interesting range
    
    # Add horizontal line at accuracy = 0.7 for reference
    ax3.axhline(y=0.75, color='red', linestyle='--', alpha=0.7, label='accuracy = 0.75')
    ax3.legend(loc='lower left', bbox_to_anchor=(0.02, 0.02))
    
    # Plot 2: AUC across layers  
    layers = sorted(layer_results.keys())
    layer_accs = [layer_results[l]['acc'] for l in layers]
    
    ax4.plot(layers, layer_accs, 'o-', linewidth=2, markersize=6, color='#2E86C1')
    ax4.set_xlabel('Layer', fontsize=12)
    ax4.set_ylabel('Best accuracy across tokens', fontsize=12)
    ax4.set_title('Accuracy vs Layer\n(Max over all tokens for each layer)', fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0.7, 1.0)
    
    # Add horizontal line at acc = 0.7 for reference
    ax4.axhline(y=0.75, color='red', linestyle='--', alpha=0.7, label='accuracy = 0.75')
    ax4.legend(loc="lower left")
    
    # plt.tight_layout()

    plt.savefig(f"{save_dir}/acc_analysis_formal_v_creative.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_dir}/acc_analysis_formal_v_creative.pdf", bbox_inches='tight')  # Also save as PDF
    
    # Print summary stats
    print(f"\nToken AUC stats:")
    print(f"  Mean: {np.mean(token_aucs):.4f}")
    print(f"  Std:  {np.std(token_aucs):.4f}")
    print(f"  Min:  {np.min(token_aucs):.4f} (token {tokens[np.argmin(token_aucs)]})")
    print(f"  Max:  {np.max(token_aucs):.4f} (token {tokens[np.argmax(token_aucs)]})")
    
    print(f"\nLayer AUC stats:")
    print(f"  Mean: {np.mean(layer_aucs):.4f}")
    print(f"  Std:  {np.std(layer_aucs):.4f}")
    print(f"  Min:  {np.min(layer_aucs):.4f} (layer {layers[np.argmin(layer_aucs)]})")
    print(f"  Max:  {np.max(layer_aucs):.4f} (layer {layers[np.argmax(layer_aucs)]})")


def main():

    val_formal_folder = "activations/val/formal"
    val_creative_folder = "activations/val/creative"
    mdvs = torch.load("representations/mdvs.pt")
    thresholds = torch.load("representations/thresholds.pt")
        

    # Get best AUC for each token position
    token_results = get_best_auc_per_token("validation", val_formal_folder, val_creative_folder, mdvs, thresholds)
    # Get best AUC for each token position
    layer_results = get_best_auc_per_layer("validation", val_formal_folder, val_creative_folder, mdvs, thresholds)

    # For plotting
    tokens = list(token_results.keys())
    token_aucs = [token_results[t]['auc'] for t in tokens]
    
    layers = list(layer_results.keys())  
    layer_aucs = [layer_results[l]['auc'] for l in layers]
    
    plot_auc_results(token_results, layer_results)


if __name__ == "__main__":
    main()