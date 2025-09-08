import os
import random
from tqdm import tqdm
import torch
from utils.analysis_utils import load_activations, MAX_TOKEN, N_LAYERS, project_onto
from sklearn.metrics import accuracy_score, roc_auc_score


def get_best_auc_per_token(split_name, neutral_folder, creative_folder, mdvs):
    """Get the best AUC for each token position across all layers"""
    print(f"\nFinding best AUC per token on {split_name} set...")
    a_neutral = load_activations(neutral_folder)
    a_creative = load_activations(creative_folder)
    
    
    token_results = {}  # token -> {'auc': best_auc, 'layer': best_layer, 'acc': best_acc}
    
    # for token in range(-MAX_TOKEN, 0):
    for token in range(1, MAX_TOKEN + 1):
        best_auc_for_token = -float('inf')
        best_layer_for_token = None
        best_acc_for_token = None
        
        for layer in range(N_LAYERS):
            mdv = mdvs[(layer, token)]
            # Get activations
            test_neutral = torch.stack(a_neutral[layer][token])
            test_creative = torch.stack(a_creative[layer][token])
            # Project onto MDV
            proj_neutral = project_onto(test_neutral, mdv)
            proj_creative = project_onto(test_creative, mdv)
            # Combine and label
            X = torch.cat([proj_neutral, proj_creative]).numpy()
            y = torch.cat([
                torch.zeros(len(proj_neutral)),
                torch.ones(len(proj_creative))
            ]).numpy()
                
            # Threshold = midpoint between means
            threshold = (proj_creative.mean() + proj_neutral.mean()) / 2
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

def get_best_auc_per_layer(split_name, neutral_folder, creative_folder, mdvs):
    """Get the best AUC for each layer across all token positions"""
    print(f"\nFinding best AUC per layer on {split_name} set...")
    a_neutral = load_activations(neutral_folder)
    a_creative = load_activations(creative_folder)
    
    layer_results = {}  # layer -> {'auc': best_auc, 'token': best_token, 'acc': best_acc}
    
    for layer in range(N_LAYERS):
        best_auc_for_layer = -float('inf')
        best_token_for_layer = None
        best_acc_for_layer = None
        
        # for token in range(-MAX_TOKEN, 0):
        for token in range(1, MAX_TOKEN + 1):
            mdv = mdvs[(layer, token)]
            # Get activations
            test_neutral = torch.stack(a_neutral[layer][token])
            test_creative = torch.stack(a_creative[layer][token])
            # Project onto MDV
            proj_neutral = project_onto(test_neutral, mdv)
            proj_creative = project_onto(test_creative, mdv)
            # Combine and label
            X = torch.cat([proj_neutral, proj_creative]).numpy()
            y = torch.cat([
                torch.zeros(len(proj_neutral)),
                torch.ones(len(proj_creative))
            ]).numpy()
            # Threshold = midpoint between means
            threshold = (proj_creative.mean() + proj_neutral.mean()) / 2
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

def create_shuffled_folders_symlinks(neutral_folder, creative_folder, output_base_dir, random_seed=42):
    """
    Create shuffled folders using symbolic links instead of copying files.
    Uses zero additional disk space.
    """
    print(f"Creating shuffled folders with symlinks in {output_base_dir}...")
    
    # Create output directories
    shuffled_neutral_dir = os.path.join(output_base_dir, "shuffled_neutral")
    shuffled_creative_dir = os.path.join(output_base_dir, "shuffled_creative")
    
    # Remove existing directories if they exist
    import shutil
    if os.path.exists(shuffled_neutral_dir):
        shutil.rmtree(shuffled_neutral_dir)
    if os.path.exists(shuffled_creative_dir):
        shutil.rmtree(shuffled_creative_dir)
    
    os.makedirs(shuffled_neutral_dir, exist_ok=True)
    os.makedirs(shuffled_creative_dir, exist_ok=True)
    
    # Get all .pt files from both folders
    neutral_files = [f for f in os.listdir(neutral_folder) if f.endswith('.pt')]
    creative_files = [f for f in os.listdir(creative_folder) if f.endswith('.pt')]
    
    print(f"Found {len(neutral_files)} neutral files and {len(creative_files)} creative files")
    
    # Create list of (file_path, original_label) tuples
    all_files = []
    for f in neutral_files:
        all_files.append((os.path.join(os.path.abspath(neutral_folder), f), 'neutral'))
    for f in creative_files:
        all_files.append((os.path.join(os.path.abspath(creative_folder), f), 'creative'))
    
    # Shuffle all files
    random.seed(random_seed)
    random.shuffle(all_files)
    
    # Split into two balanced groups
    mid = len(all_files) // 2
    new_neutral_files = all_files[:mid]
    new_creative_files = all_files[mid:]
    
    print(f"Creating {len(new_neutral_files)} symlinks in shuffled_neutral")
    print(f"Creating {len(new_creative_files)} symlinks in shuffled_creative")
    
    # Create symbolic links for neutral folder
    print("Creating symlinks for shuffled_neutral...")
    for i, (original_path, original_label) in tqdm(enumerate(new_neutral_files), total=len(new_neutral_files)):
        new_path = os.path.join(shuffled_neutral_dir, f"prompt_{i}.pt")
        try:
            os.symlink(original_path, new_path)
        except OSError as e:
            print(f"Error creating symlink {new_path}: {e}")
    
    # Create symbolic links for creative folder  
    print("Creating symlinks for shuffled_creative...")
    for i, (original_path, original_label) in tqdm(enumerate(new_creative_files), total=len(new_creative_files)):
        new_path = os.path.join(shuffled_creative_dir, f"prompt_{i}.pt")
        try:
            os.symlink(original_path, new_path)
        except OSError as e:
            print(f"Error creating symlink {new_path}: {e}")
    
    print(f"Shuffled symlink folders created successfully!")
    print(f"  - {shuffled_neutral_dir}: {len(new_neutral_files)} symlinks")
    print(f"  - {shuffled_creative_dir}: {len(new_creative_files)} symlinks")
    
    # Print shuffle statistics
    neutral_to_neutral = sum(1 for _, orig_label in new_neutral_files if orig_label == 'neutral')
    creative_to_neutral = len(new_neutral_files) - neutral_to_neutral
    neutral_to_creative = sum(1 for _, orig_label in new_creative_files if orig_label == 'neutral') 
    creative_to_creative = len(new_creative_files) - neutral_to_creative
    
    print(f"\nShuffle statistics:")
    print(f"  shuffled_neutral folder: {neutral_to_neutral} orig neutral + {creative_to_neutral} orig creative")
    print(f"  shuffled_creative folder: {neutral_to_creative} orig neutral + {creative_to_creative} orig creative")
    
    # Verify symlinks work
    print(f"\nVerifying symlinks...")
    sample_file = os.path.join(shuffled_neutral_dir, "prompt_0.pt")
    if os.path.exists(sample_file):
        print(f"✓ Sample symlink works: {sample_file} -> {os.readlink(sample_file)}")
    else:
        print(f"✗ Sample symlink failed")

def create_all_shuffled_splits_symlinks(base_dir="activations", random_seed=42):
    """Create shuffled versions for all splits using symlinks"""
    
    splits = ['train', 'test', 'val']
    
    for split in splits:
        print(f"\n" + "="*50)
        print(f"Processing {split} split")
        print("="*50)
        
        neutral_folder = os.path.join(base_dir, split, "neutral")
        creative_folder = os.path.join(base_dir, split, "creative")
        output_base = os.path.join(base_dir, split)
        
        if os.path.exists(neutral_folder) and os.path.exists(creative_folder):
            create_shuffled_folders_symlinks(neutral_folder, creative_folder, output_base, 
                                           random_seed + hash(split) % 1000)
        else:
            print(f"Skipping {split} - folders don't exist")

def plot_auc_results(token_results, layer_results, shuffled_token_results=None, shuffled_layer_results=None, save_dir="plots"):
    """Generate and save AUC plots with optional shuffled baseline"""
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: AUC across token positions
    tokens = sorted(token_results.keys())
    token_aucs = [token_results[t]['auc'] for t in tokens]
    
    # Main results
    ax1.plot(tokens, token_aucs, 'o-', linewidth=2, markersize=6, color='#2E86C1', label='Real Labels')
    
    # Add shuffled baseline if provided
    if shuffled_token_results is not None:
        shuffled_tokens = sorted(shuffled_token_results.keys())
        shuffled_token_aucs = [shuffled_token_results[t]['auc'] for t in shuffled_tokens]
        ax1.plot(shuffled_tokens, shuffled_token_aucs, 's--', linewidth=2, markersize=4, 
                color='gray', alpha=0.7, label='Shuffled Labels')
    
    ax1.set_xlabel('Post-Instruction Token Position', fontsize=12)
    ax1.set_ylabel('Best AUC across layers', fontsize=12)
    ax1.set_title('AUC vs Token Position\n(Max over all layers for each token)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Adjust y-limits based on data
    if shuffled_token_results is not None:
        y_min = min(min(token_aucs), min(shuffled_token_aucs)) - 0.05
        y_max = max(max(token_aucs), max(shuffled_token_aucs)) + 0.05
        ax1.set_ylim(max(0.4, y_min), min(1.0, y_max))
    else:
        ax1.set_ylim(0.7, 1.0)  # Original range for real data only
    
    # Add horizontal reference lines
    ax1.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axhline(y=0.5, color='black', linestyle=':', alpha=0.5, linewidth=1)
    
    # Add reference line labels
    ax1.text(max(tokens) - 0.5, 0.91, 'AUC = 0.9', fontsize=10, ha='right', va='bottom', color='red')
    ax1.text(max(tokens) - 0.5, 0.51, 'Random (0.5)', fontsize=10, ha='right', va='bottom', color='black')
    
    ax1.legend(loc='lower left', bbox_to_anchor=(0.02, 0.02))
    
    # Plot 2: AUC across layers  
    layers = sorted(layer_results.keys())
    layer_aucs = [layer_results[l]['auc'] for l in layers]
    
    # Main results
    ax2.plot(layers, layer_aucs, 'o-', linewidth=2, markersize=6, color='#E74C3C', label='Real Labels')
    
    # Add shuffled baseline if provided
    if shuffled_layer_results is not None:
        shuffled_layers = sorted(shuffled_layer_results.keys())
        shuffled_layer_aucs = [shuffled_layer_results[l]['auc'] for l in shuffled_layers]
        ax2.plot(shuffled_layers, shuffled_layer_aucs, 's--', linewidth=2, markersize=4, 
                color='gray', alpha=0.7, label='Shuffled Labels')
    
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('Best AUC across tokens', fontsize=12)
    ax2.set_title('AUC vs Layer\n(Max over all tokens for each layer)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Adjust y-limits based on data
    if shuffled_layer_results is not None:
        y_min = min(min(layer_aucs), min(shuffled_layer_aucs)) - 0.05
        y_max = max(max(layer_aucs), max(shuffled_layer_aucs)) + 0.05
        ax2.set_ylim(max(0.4, y_min), min(1.0, y_max))
    else:
        ax2.set_ylim(0.7, 1.0)  # Original range for real data only
    
    # Add horizontal reference lines
    ax2.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axhline(y=0.5, color='black', linestyle=':', alpha=0.5, linewidth=1)
    
    # Add reference line labels
    ax2.text(max(layers) - 1, 0.91, 'AUC = 0.9', fontsize=10, ha='right', va='bottom', color='red')
    ax2.text(max(layers) - 1, 0.51, 'Random (0.5)', fontsize=10, ha='right', va='bottom', color='black')
    
    ax2.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/auc_analysis_with_baseline.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_dir}/auc_analysis_with_baseline.pdf", bbox_inches='tight')
    plt.show()
    
    # Print summary stats
    print(f"\n{'='*60}")
    print(f"REAL LABELS - Token AUC stats:")
    print(f"  Mean: {np.mean(token_aucs):.4f}")
    print(f"  Std:  {np.std(token_aucs):.4f}")
    print(f"  Min:  {np.min(token_aucs):.4f} (token {tokens[np.argmin(token_aucs)]})")
    print(f"  Max:  {np.max(token_aucs):.4f} (token {tokens[np.argmax(token_aucs)]})")
    
    print(f"\nREAL LABELS - Layer AUC stats:")
    print(f"  Mean: {np.mean(layer_aucs):.4f}")
    print(f"  Std:  {np.std(layer_aucs):.4f}")
    print(f"  Min:  {np.min(layer_aucs):.4f} (layer {layers[np.argmin(layer_aucs)]})")
    print(f"  Max:  {np.max(layer_aucs):.4f} (layer {layers[np.argmax(layer_aucs)]})")
    
    # Print shuffled stats if provided
    if shuffled_token_results is not None:
        shuffled_token_aucs = [shuffled_token_results[t]['auc'] for t in shuffled_tokens]
        print(f"\n{'='*60}")
        print(f"SHUFFLED LABELS - Token AUC stats:")
        print(f"  Mean: {np.mean(shuffled_token_aucs):.4f}")
        print(f"  Std:  {np.std(shuffled_token_aucs):.4f}")
        print(f"  Min:  {np.min(shuffled_token_aucs):.4f} (token {shuffled_tokens[np.argmin(shuffled_token_aucs)]})")
        print(f"  Max:  {np.max(shuffled_token_aucs):.4f} (token {shuffled_tokens[np.argmax(shuffled_token_aucs)]})")
        
        # Calculate difference
        print(f"\nCOMPARISON - Token AUC:")
        print(f"  Real mean - Shuffled mean: {np.mean(token_aucs) - np.mean(shuffled_token_aucs):.4f}")
    
    if shuffled_layer_results is not None:
        shuffled_layer_aucs = [shuffled_layer_results[l]['auc'] for l in shuffled_layers]
        print(f"\nSHUFFLED LABELS - Layer AUC stats:")
        print(f"  Mean: {np.mean(shuffled_layer_aucs):.4f}")
        print(f"  Std:  {np.std(shuffled_layer_aucs):.4f}")
        print(f"  Min:  {np.min(shuffled_layer_aucs):.4f} (layer {shuffled_layers[np.argmin(shuffled_layer_aucs)]})")
        print(f"  Max:  {np.max(shuffled_layer_aucs):.4f} (layer {shuffled_layers[np.argmax(shuffled_layer_aucs)]})")
        
        # Calculate difference
        print(f"\nCOMPARISON - Layer AUC:")
        print(f"  Real mean - Shuffled mean: {np.mean(layer_aucs) - np.mean(shuffled_layer_aucs):.4f}")
    
    print(f"{'='*60}")


def main():
    mdvs = torch.load("representations/mdvs.pt")

    create_shuffled_folders_symlinks("activations/test/neutral", "activations/test/creative", "activations/test")

    shuffled_results = get_best_auc_per_token("test_shuffled", 
                                         "activations/test/shuffled_neutral", 
                                         "activations/test/shuffled_creative", 
                                         mdvs)

    # Get your regular results
    token_results = get_best_auc_per_token("test", "activations/test/neutral", "activations/test/creative", mdvs)
    layer_results = get_best_auc_per_layer("test", "activations/test/neutral", "activations/test/creative", mdvs)
    
    # Get shuffled results
    shuffled_token_results = get_best_auc_per_token("test_shuffled", "activations/test/shuffled_neutral", "activations/test/shuffled_creative", mdvs)
    shuffled_layer_results = get_best_auc_per_layer("test_shuffled", "activations/test/shuffled_neutral", "activations/test/shuffled_creative", mdvs)
    
    # Plot with baseline
    plot_auc_results(token_results, layer_results, shuffled_token_results, shuffled_layer_results)


if __name__ == "__main__":
    main()