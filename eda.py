import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_data_subset(split="train", n_shards=5):
    """Load a subset of parquet shards for EDA."""
    path = os.path.join("pinn_data", split, "*.parquet")
    files = sorted(glob.glob(path))
    if not files:
        raise ValueError(f"No parquet files found in {path}")

    # Load first n_shards
    dfs = [pd.read_parquet(f) for f in files[:n_shards]]
    return pd.concat(dfs, ignore_index=True)


def plot_lhs_coverage(df, output_dir):
    """Plot input distributions to verify LHS sampling across all 5 dimensions."""

    # Extract unique trajectories by dropping duplicate input combinations
    input_cols = ['thickness_mm', 'velocity_ms', 'obliquity_deg', 'material_id']

    # Check if plate_temperature_K exists in the data
    has_temp = 'plate_temperature_K' in df.columns
    if has_temp:
        input_cols.append('plate_temperature_K')

    df_unique = df.drop_duplicates(subset=input_cols)

    n_plots = 5 if has_temp else 4
    rows = 3 if has_temp else 2
    plt.figure(figsize=(15, 5 * rows))

    # 1. Thickness
    plt.subplot(rows, 2, 1)
    sns.histplot(df_unique['thickness_mm'], bins=30, kde=True)
    plt.title("Thickness Distribution (Uniform LHS)")

    # 2. Velocity (Log scale)
    plt.subplot(rows, 2, 2)
    sns.histplot(df_unique['velocity_ms'], bins=30, kde=True, log_scale=True)
    plt.title("Velocity Distribution (Log-Scale LHS)")

    # 3. Obliquity
    plt.subplot(rows, 2, 3)
    sns.histplot(df_unique['obliquity_deg'], bins=30, kde=True)
    plt.title("Obliquity Distribution (Uniform LHS)")

    # 4. Material ID
    plt.subplot(rows, 2, 4)
    sns.countplot(data=df_unique, x='material_id')
    plt.title("Material Class Balance")

    # 5. Temperature (if available)
    if has_temp:
        plt.subplot(rows, 2, 5)
        sns.histplot(df_unique['plate_temperature_K'], bins=30, kde=True)
        plt.title("Plate Temperature Distribution (Uniform LHS)")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lhs_coverage.png"), dpi=150)
    plt.close()


def plot_trajectories(df, output_dir, n_traj=8):
    """Plot V vs x, T vs x, and sigma vs x for a few trajectories."""
    # Group by inputs to get unique trajectories
    cols = ['thickness_mm', 'velocity_ms', 'obliquity_deg', 'material_id']

    # Pre-filter: only keep rows for the first n_traj unique trajectories
    unique_keys = df[cols].drop_duplicates().head(n_traj)
    df_filtered = df.merge(unique_keys, on=cols)
    grouped = df_filtered.groupby(cols)

    plt.figure(figsize=(18, 5))

    for name, group in grouped:
        group = group.sort_values('x_mm')
        label = f"Mat {name[3]}, V0={name[1]:.0f}"

        plt.subplot(1, 3, 1)
        plt.plot(group['x_mm'], group['V_ms'], marker='.', label=label)

        plt.subplot(1, 3, 2)
        plt.plot(group['x_mm'], group['T_K'], marker='.', label=label)

        plt.subplot(1, 3, 3)
        plt.plot(group['x_mm'], group['sigma_MPa'], marker='.', label=label)

    plt.subplot(1, 3, 1)
    plt.title("Velocity Decay vs Depth")
    plt.xlabel("Depth x (mm)"); plt.ylabel("Velocity (m/s)")
    plt.legend(fontsize=7)

    plt.subplot(1, 3, 2)
    plt.title("Adiabatic Heating vs Depth")
    plt.xlabel("Depth x (mm)"); plt.ylabel("Temperature (K)")

    plt.subplot(1, 3, 3)
    plt.title("Flow Stress vs Depth")
    plt.xlabel("Depth x (mm)"); plt.ylabel("Flow Stress (MPa)")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trajectory_profiles.png"), dpi=150)
    plt.close()


def plot_phase_portraits(df, output_dir):
    """Plot phase portraits of the physical manifold."""
    # Subsample for clearer scatter plot (e.g. 10k points)
    if len(df) > 10000:
        df_sub = df.sample(10000, random_state=42)
    else:
        df_sub = df

    plt.figure(figsize=(12, 5))

    # 1. Temperature vs Velocity Phase
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df_sub, x='V_ms', y='T_K', hue='material_id', palette='tab10', s=10, alpha=0.5)
    plt.title("Phase Portrait: Temp vs Velocity")
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Temperature (K)")

    # 2. Stress vs Strain Phase (Work Hardening)
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=df_sub, x='strain', y='sigma_MPa', hue='material_id', palette='tab10', s=10, alpha=0.5)
    plt.title("Phase Portrait: Stress vs Strain")
    plt.xlabel("Plastic Strain")
    plt.ylabel("Flow Stress (MPa)")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "phase_portraits.png"), dpi=150)
    plt.close()


def main():
    print("Loading data subset for EDA...")
    df = load_data_subset("train", n_shards=10)
    print(f"Loaded {len(df)} collocation points.")
    print(f"Columns: {list(df.columns)}")

    out_dir = "data_pipeline/eda_plots"
    os.makedirs(out_dir, exist_ok=True)

    print("Generating LHS Coverage plots...")
    plot_lhs_coverage(df, out_dir)

    print("Generating Trajectory Profiles...")
    plot_trajectories(df, out_dir, n_traj=8)

    print("Generating Phase Portraits...")
    plot_phase_portraits(df, out_dir)

    print(f"EDA plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
