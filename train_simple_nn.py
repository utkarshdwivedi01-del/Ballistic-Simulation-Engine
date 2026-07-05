import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pinn_dataloader import BallisticPINNDataset
from stage1_part1_johnson_cook import MATERIALS

# Hyperparameters
EPOCHS = 20
BATCH_SIZE = 1024
LR = 1e-3
NUM_MATERIALS = len(MATERIALS)
EMBEDDING_DIM = 4

class BallisticFCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Continuous inputs: thickness, velocity, obliquity, temperature, x
        self.num_continuous = 5
        
        # Categorical embedding for material_id
        self.mat_embed = nn.Embedding(num_embeddings=NUM_MATERIALS, embedding_dim=EMBEDDING_DIM)
        
        # Input to first hidden layer: 5 continuous + 4 embedded = 9
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
            nn.Linear(hidden_dim, 5) # 5 targets: V_ms, sigma_MPa, T_K, strain, damage
        )
        
    def forward(self, x):
        # x shape: [Batch, 6]
        # features: thickness, velocity, obliquity, material_id, temp, x_mm
        cont_features = x[:, [0, 1, 2, 4, 5]]
        mat_ids = x[:, 3].long()
        
        embeds = self.mat_embed(mat_ids)
        combined = torch.cat([cont_features, embeds], dim=1)
        
        return self.net(combined)

def main():
    print("="*60)
    print("  Phase 2: Training Simple FCNN (Supervised Baseline)")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Data
    train_dir = "pinn_data/train"
    val_dir = "pinn_data/val"
    
    if not os.path.exists(train_dir):
        print("Could not find 'pinn_data/train'. Run 'python data_gen.py' first!")
        return

    train_ds = BallisticPINNDataset(train_dir, normalize=True)
    val_ds = BallisticPINNDataset(val_dir, normalize=True)
    
    # Subset to demonstrate architecture quickly
    from torch.utils.data import Subset
    train_ds = Subset(train_ds, range(min(50000, len(train_ds))))
    val_ds = Subset(val_ds, range(min(10000, len(val_ds))))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = BallisticFCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    print(f"Training on {len(train_ds)} samples. Validating on {len(val_ds)} samples.")
    
    history = {"train_loss": [], "val_loss": []}
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(train_ds)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                
        val_loss /= len(val_ds)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")
        
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/fcnn_baseline.pt")
    print("Saved baseline model to models/fcnn_baseline.pt")
    
if __name__ == "__main__":
    main()
