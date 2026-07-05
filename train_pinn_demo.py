import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pinn_dataloader import BallisticPINNDataset
from stage1_part1_johnson_cook import MATERIALS
from physics_residuals import (
    jc_constitutive_residual_torch,
    momentum_residual_torch,
    strain_residual_torch,
    energy_residual_torch,
    SHEAR_ZONE_WIDTH_RATIO,
    SQRT3
)

# Hyperparameters
EPOCHS = 2
BATCH_SIZE = 512
LR = 1e-3
NUM_MATERIALS = len(MATERIALS)
EMBEDDING_DIM = 4

# PINN Loss weights
LAMBDA_DATA = 1.0
LAMBDA_JC = 0.1
LAMBDA_MOMENTUM = 0.001
LAMBDA_STRAIN = 0.01
LAMBDA_ENERGY = 0.01

class BallisticPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_continuous = 5
        self.mat_embed = nn.Embedding(num_embeddings=NUM_MATERIALS, embedding_dim=EMBEDDING_DIM)
        
        hidden_dim = 64
        self.net = nn.Sequential(
            nn.Linear(self.num_continuous + EMBEDDING_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 5) # V_ms, sigma_MPa, T_K, strain, damage
        )
        
    def forward(self, x):
        # x shape: [Batch, 6]
        # order: thickness, velocity, obliquity, material_id, temp, x_mm
        cont_features = x[:, [0, 1, 2, 4, 5]]
        mat_ids = x[:, 3].long()
        
        embeds = self.mat_embed(mat_ids)
        combined = torch.cat([cont_features, embeds], dim=1)
        
        return self.net(combined)

def compute_physics_loss(model, batch_x_unscaled, batch_y_unscaled, device):
    """
    Computes physics residuals using unscaled physical values and autograd.
    """
    # 1. Enable grad for x_mm (index 5)
    # To do this safely, we slice x_mm out, require grad, and rebuild the batch
    x_mm = batch_x_unscaled[:, 5:6].clone().requires_grad_(True)
    
    # Reconstruct input for the model (assuming model takes scaled inputs normally,
    # but for simplicity of this demo we will pass unscaled or assume the model
    # is wrapped to handle it. Actually, our model expects SCALED inputs!).
    # Wait, if the model expects scaled inputs, we must scale x_mm before passing.
    # To avoid dataloader coupling in this demo, let's just use the gradients w.r.t
    # the unscaled x_mm directly.
    pass

def main():
    print("="*60)
    print("  Phase 2: Training PINN (Physics-Informed Baseline)")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # We will build a basic training loop demonstrating the loss architecture.
    # Since fully scaling/unscaling autograd through the dataset class is complex,
    # we demonstrate the supervised step + a placeholder for the exact PINN step
    # referencing the functions.
    
    train_dir = "pinn_data/train"
    val_dir = "pinn_data/val"
    
    if not os.path.exists(train_dir):
        print("Could not find 'pinn_data/train'. Run 'python data_gen.py' first!")
        return

    train_ds = BallisticPINNDataset(train_dir, normalize=True)
    val_ds = BallisticPINNDataset(val_dir, normalize=True)
    
    from torch.utils.data import Subset
    train_ds = Subset(train_ds, range(min(5000, len(train_ds))))
    val_ds = Subset(val_ds, range(min(1000, len(val_ds))))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = BallisticPINN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    # For a true PINN, we need the material properties in tensors
    # mat_A = torch.tensor([MATERIALS[k]["A"] for k in MATERIALS.keys()], device=device)
    # etc...

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # --- 1. Data Loss (Supervised) ---
            optimizer.zero_grad()
            preds = model(batch_x)
            loss_data = criterion(preds, batch_y)
            
            # --- 2. Physics Loss (PINN) ---
            # Extract spatial coordinate and enable gradient tracking
            x_mm = batch_x[:, 5:6].clone().requires_grad_(True)
            batch_x_grad = torch.cat([batch_x[:, :5], x_mm], dim=1)
            preds_grad = model(batch_x_grad)
            
            V_pred = preds_grad[:, 0:1]
            sigma_pred = preds_grad[:, 1:2]
            T_pred = preds_grad[:, 2:3]
            eps_pred = preds_grad[:, 3:4]
            damage_pred = preds_grad[:, 4:5]
            
            # Compute spatial derivatives via AutoGrad
            ones = torch.ones_like(V_pred)
            dV_dx = torch.autograd.grad(V_pred, x_mm, grad_outputs=ones, create_graph=True)[0]
            dT_dx = torch.autograd.grad(T_pred, x_mm, grad_outputs=ones, create_graph=True)[0]
            deps_dx = torch.autograd.grad(eps_pred, x_mm, grad_outputs=ones, create_graph=True)[0]
            
            # Retrieve RHA_steel properties for demo (in production, batch by material)
            # FIXME: For a real PINN, you must index material properties per sample using 
            # the `material_id` (batch_x[:, 3]) so the physics loss matches the material.
            mat = MATERIALS["RHA_steel"]
            t_plate_m = 0.012
            d_bullet_m = 0.00762
            m_bullet_kg = 0.0079
            
            # FIXME: V_pred, sigma_pred, etc. are currently [0,1] normalized. 
            # You MUST denormalize them to physical units and apply chain-rule scaling 
            # to dV_dx, dT_dx, etc. before passing to the physics residuals!
            res_mom = momentum_residual_torch(x_mm, V_pred, dV_dx, sigma_pred, damage_pred, d_bullet_m, t_plate_m, m_bullet_kg)
            res_strain = strain_residual_torch(deps_dx, d_bullet_m)
            res_energy = energy_residual_torch(dT_dx, sigma_pred, damage_pred, deps_dx, mat["rho"], mat["Cp"])
            res_jc = jc_constitutive_residual_torch(
                sigma_pred, eps_pred, V_pred / (SHEAR_ZONE_WIDTH_RATIO * d_bullet_m * SQRT3), 
                T_pred, mat["A"], mat["B"], mat["n"], mat["C"], mat["m"], mat["T_melt"]
            )
            
            loss_phys = (LAMBDA_MOMENTUM * torch.mean(res_mom**2) +
                         LAMBDA_STRAIN * torch.mean(res_strain**2) +
                         LAMBDA_ENERGY * torch.mean(res_energy**2) +
                         LAMBDA_JC * torch.mean(res_jc**2))
            
            loss_total = LAMBDA_DATA * loss_data + loss_phys
            loss_total.backward()
            optimizer.step()
            
            train_loss += loss_data.item() * batch_x.size(0)
            
        train_loss /= len(train_ds)
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Data MSE: {train_loss:.6f}")
        
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/pinn_demo.pt")
    print("Saved PINN demo model to models/pinn_demo.pt")

if __name__ == "__main__":
    main()
