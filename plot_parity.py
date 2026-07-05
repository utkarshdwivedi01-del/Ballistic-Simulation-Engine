import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pinn_dataloader import BallisticPINNDataset
from train_simple_nn import BallisticFCNN

def main():
    print("="*60)
    print("  Phase 2: Generating Parity Plots (Pred vs True)")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "models/fcnn_baseline.pt"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Run 'python train_simple_nn.py' first.")
        return
        
    val_dir = "pinn_data/val"
    if not os.path.exists(val_dir):
        print("Validation data not found.")
        return
        
    ds = BallisticPINNDataset(val_dir, normalize=True)
    loader = DataLoader(ds, batch_size=2048, shuffle=False)
    
    model = BallisticFCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    all_preds = []
    all_trues = []
    
    print("Running inference on validation set...")
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x)
            
            all_preds.append(preds.cpu().numpy())
            all_trues.append(batch_y.numpy())
            
    all_preds = np.vstack(all_preds)
    all_trues = np.vstack(all_trues)
    
    # Denormalize
    preds_phys = ds.denormalize_targets(all_preds)
    trues_phys = ds.denormalize_targets(all_trues)
    
    # Plotting
    target_names = ["Velocity (m/s)", "Stress (MPa)", "Temperature (K)", "Strain", "Damage"]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(5):
        ax = axes[i]
        true_val = trues_phys[:, i]
        pred_val = preds_phys[:, i]
        
        # Subsample for plotting speed if necessary
        idx = np.random.choice(len(true_val), min(10000, len(true_val)), replace=False)
        
        ax.scatter(true_val[idx], pred_val[idx], alpha=0.1, s=2)
        
        # Parity line
        min_val = min(true_val.min(), pred_val.min())
        max_val = max(true_val.max(), pred_val.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        ax.set_title(target_names[i])
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    axes[-1].axis('off') # Hide the empty 6th subplot
        
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/parity_plot.png", dpi=300, bbox_inches='tight')
    print("Saved parity plots to plots/parity_plot.png")

if __name__ == "__main__":
    main()
