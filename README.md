# Ballistic Simulation Engine

A high-speed, physics-based numerical simulation engine designed for rapid ballistic impact analysis and Physics-Informed Neural Network (PINN) dataset generation. 

This engine bridges the gap between purely analytical formulations (which lack transient fidelity) and full 3D Finite Element Analysis (which is computationally expensive). By coupling a non-linear ODE solver with dynamic material constitutive models (Johnson-Cook and Cowper-Symonds) and adiabatic shear plug failure mechanisms (Recht-Ipson), the engine predicts the spatiotemporal evolution of temperature, strain, flow stress, and velocity during impact.

## Project Vision & Goals

**1. Solving the "Data Starvation" Problem**
Interns and researchers at defense organizations often lack access to proprietary experimental datasets or classified internal data. This engine is intentionally designed as a **synthetic data generator** to solve this problem. It produces highly realistic, diverse, physics-grounded data spanning the entire operational envelope, enabling researchers to begin ML experimentation immediately.

**2. Long-Term Scalability & PINN-Readiness**
The architecture is deliberately modular, extensible, and research-oriented. The current codebase serves as a structurally sound foundation for future advanced ML applications. The extraction of exact physical ODE residuals makes the engine completely **PINN-ready**—allowing features like surrogate models, inverse modeling, parameter identification, and uncertainty quantification to be integrated later with minimal restructuring.
## Features

- **Extreme Performance**: Written in Python and compiled to C-level machine code via Numba `@njit`. Capable of simulating millions of impacts in seconds. (~1000x faster than traditional FEM).
- **Physical Fidelity**: Implements Johnson-Cook (strain hardening, thermal softening) and Cowper-Symonds (dynamic strain rate) material models.
- **Robust Integration**: Custom explicit Euler integration stabilized by strict energy bounding to prevent non-physical NaNs at low velocities.
- **PINN Data Pipeline**: Natively generates and exports massive datasets into chunked Parquet shards.
- **PyTorch Integration**: Includes a ready-to-use `BallisticPINNDataset` for seamlessly loading normalized tensors into PyTorch models.

---

## Project Structure

- `stage1_part1_johnson_cook.py`: Canonical implementation of the Johnson-Cook constitutive model and the central `MATERIALS` dictionary.
- `stage1_part2_cowper_symonds.py`: Canonical implementation of the Cowper-Symonds dynamic strain-rate increase factor (DIF).
- `stage1_part3_recht_ipson.py`: Analytical Recht-Ipson benchmark model for calculating ballistic limits ($V_{50}$) and residual velocities.
- `simulation.py`: The core physics engine. Combines the material models into a time-stepping ODE solver to generate spatiotemporal trajectories.
- `data_gen.py` / `eda.py`: Scripts for generating massive LHS (Latin Hypercube Sampling) PINN datasets and performing exploratory data analysis.
- `pinn_dataloader.py`: PyTorch `Dataset` wrapper that handles batch loading, normalization, and inverse-transforms for the generated Parquet data.

---

## Requirements

Ensure you have the following installed:
```bash
pip install numpy pandas pyarrow matplotlib seaborn numba torch scipy
```
*(Note: `torch` is technically optional for running the core physics simulation, but required to use `pinn_dataloader.py` as a PyTorch dataset).*

---

## Example Usage

### 1. Running a Single Impact Simulation
You can simulate a single impact trajectory to inspect the transient physics (velocity, stress, temperature, strain) at each depth step.

```python
from simulation import simulate_trajectory
import matplotlib.pyplot as plt

# Simulate a 7.62mm, 7.9g bullet hitting a 10mm RHA Steel plate at 850 m/s
traj = simulate_trajectory(
    plate_thickness_mm=10.0,
    material_name="RHA_steel",
    bullet_velocity_ms=850.0,
    bullet_mass_g=7.9,
    bullet_diameter_mm=7.62
)

depths = traj["x_mm"]
velocities = traj["V_ms"]
temperatures = traj["T_K"]

print(f"Simulation completed with {traj['n_points']} active steps.")
print(f"Exit Velocity: {velocities[-1]:.2f} m/s")

# Plot velocity decay vs depth
plt.plot(depths, velocities)
plt.xlabel("Depth (mm)")
plt.ylabel("Velocity (m/s)")
plt.title("Projectile Velocity vs. Penetration Depth")
plt.show()
```

### 2. Finding the Ballistic Limit (V50)
Calculate the exact terminal ballistic parameters for a given armor/threat combination using the coupled numerical integration.

```python
from simulation import simulate_ballistic_impact

res = simulate_ballistic_impact(
    plate_thickness_mm=12.0,
    material_name="aluminium_7075_T6",
    plate_temperature_K=298.0,
    bullet_velocity_ms=900.0,
    bullet_mass_g=7.9,
    bullet_diameter_mm=7.62
)

print(f"Penetrated: {res.penetrated}")
print(f"Residual Velocity: {res.residual_velocity_ms:.2f} m/s")
print(f"Ballistic Limit (V50): {res.ballistic_limit_ms:.2f} m/s")
print(f"Energy Absorbed: {res.energy_absorbed_J:.2f} J")
```

### 3. Loading PINN Data into PyTorch
Once data has been generated (via `python data_gen.py`), use `BallisticPINNDataset` to feed normalized data into your Neural Network.

```python
import torch
from torch.utils.data import DataLoader
from pinn_dataloader import BallisticPINNDataset

# Initialize the dataset (automatically handles min-max scaling & log-transforms)
dataset = BallisticPINNDataset("pinn_data/train", normalize=True)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# Fetch a batch
x_batch, y_batch = next(iter(dataloader))
print("Input shape:", x_batch.shape)   # [256, 6]
print("Target shape:", y_batch.shape)  # [256, 4]

# IMPORTANT: Handling the Ordinal Material ID
# x_batch[:, 3] is the `material_id`. It is explicitly NOT normalized.
# Route it through an embedding layer in your PyTorch model:
# 
# mat_id = x_batch[:, 3].long()
# cont_x = torch.cat([x_batch[:, :3], x_batch[:, 4:]], dim=1)
# mat_embed = embedding_layer(mat_id)
# x_final = torch.cat([cont_x, mat_embed], dim=1)

# Denormalizing network predictions back to physical units:
predictions = model(x_final).detach().numpy()
physical_predictions = dataset.denormalize_targets(predictions)
```

## Adding Custom Materials
Materials are dynamically loaded from `materials_config.json`. To add a new armor material, simply add an entry to the JSON file with the required Johnson-Cook and Cowper-Symonds parameters:

```json
{
    "titanium_Ti6Al4V": {
        "label": "Titanium Alloy (Ti-6Al-4V)",
        "A": 862.0,
        "B": 331.0,
        "n": 0.34,
        "C": 0.012,
        "m": 0.8,
        "T_melt": 1900.0,
        "rho": 4430.0,
        "E": 114000.0,
        "Cp": 526.0,
        "cs_D": 100000.0,
        "cs_q": 4.0,
        "hardness_HRC": 36.0
    }
}
```
No code recompilation is required; the engine will automatically parse and JIT-compile the new parameters on the next run.

## Known Limitations & Assumptions

> **⚠️ Important Physical Constraints**
> - **Material-Dependent Failure Strain:** Metals and ceramics fail at vastly different strains. The integration now properly respects the `failure_strain` parameter defined in `materials_config.json`. Once $\epsilon > \epsilon_f$, the flow stress drops to zero.
> - **Geometric Strain Rate:** The strain rate is currently simplified to a purely geometric formulation: $\dot{\epsilon} = V / (w \sqrt{3})$. This is a massive macroscopic simplification. Advanced users building PINNs should note that the network is forced to learn this rigid relationship rather than discovering a true independent rate sensitivity.
> - **Single-Point Calibration:** The adiabatic shear zone width ($w_{ratio}$) was formally calibrated to $V_{50} = 619.5$ m/s for 12mm RHA Steel. *Note: After introducing a realistic failure strain of 1.2 for RHA, the $w_{ratio}$ was re-calibrated from 0.130 to 0.3965 to maintain strict agreement with the experimental $V_{50}$ limit.* Sweeping target thickness heavily or evaluating non-RHA materials may require re-calibration or defining $w_{ratio}(t, V)$ as a function.
