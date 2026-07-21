"""
Ballistic Simulation — Comprehensive Report Graph Generator
===================================================================
Generates 20+ publication-quality graphs for mentor presentations.

Usage:  python generate_all_graphs.py

Output: plots/all_graphs/  (created automatically)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from simulation import simulate_trajectory, simulate_ballistic_impact
from stage1_part1_johnson_cook import MATERIALS, T_ROOM, johnson_cook_stress
from stage1_part3_recht_ipson import (
    recht_ipson, _plug_mass_ri, _perforation_energy,
    _estimate_tau_f, validate_against_simulation
)

# ── Global plot style ──────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

OUT = "plots/all_graphs"
os.makedirs(OUT, exist_ok=True)

# ── Default scenario ───────────────────────────────────────────────
T_PLATE  = 10.0    # mm
MAT      = "RHA_steel"
M_BULLET = 7.9     # g
D_BULLET = 7.62    # mm
V0       = 850.0   # m/s

# Material display names (short)
MAT_LABELS = {k: v.get("label", k)[:25] for k, v in MATERIALS.items()
              if not k.startswith("_")}
MAT_NAMES  = list(MAT_LABELS.keys())

# Colors
COLORS = ['#e63946', '#457b9d', '#2a9d8f', '#e9c46a', '#f4a261',
          '#264653', '#9b5de5', '#f15bb5', '#00bbf9', '#00f5d4']


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  HELPER                                                          ║
# ╚═══════════════════════════════════════════════════════════════════╝

def compute_time(x_mm, V_ms):
    """Compute cumulative time array from depth and velocity."""
    dx_m = np.diff(x_mm) / 1000.0
    V_avg = np.maximum((V_ms[:-1] + V_ms[1:]) / 2.0, 1e-6)
    dt_s = dx_m / V_avg
    return np.insert(np.cumsum(dt_s), 0, 0.0) * 1e6  # µs


def save(name):
    path = f"{OUT}/{name}.png"
    plt.savefig(path, dpi=250)
    plt.close()
    print(f"  [OK] {name}.png")


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  SECTION 1 — SINGLE TRAJECTORY DEEP DIVE  (6 graphs)            ║
# ╚═══════════════════════════════════════════════════════════════════╝

def section_1_single_trajectory():
    print("\n[Section 1] Single Trajectory Analysis ...")
    traj = simulate_trajectory(
        plate_thickness_mm=T_PLATE, material_name=MAT,
        bullet_velocity_ms=V0, bullet_mass_g=M_BULLET,
        bullet_diameter_mm=D_BULLET)

    x   = traj["x_mm"]
    V   = traj["V_ms"]
    sig = traj["sigma_MPa"]
    T   = traj["T_K"]
    eps = traj["strain"]
    edr = traj["strain_rate"]
    dmg = traj["damage"]
    t_us = compute_time(x, V)

    m_kg = M_BULLET * 1e-3
    KE   = 0.5 * m_kg * V**2
    E_abs = KE[0] - KE

    # ── 1a. Velocity vs Depth ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(x, V, alpha=0.15, color='#1f77b4')
    ax.plot(x, V, lw=2.2, color='#1f77b4')
    ax.axhline(0, color='grey', ls='--', lw=0.8)
    ax.set_xlabel("Penetration Depth (mm)")
    ax.set_ylabel("Projectile Velocity (m/s)")
    ax.set_title(f"Velocity Decay vs Depth — {V0:.0f} m/s into {T_PLATE}mm RHA")
    save("01_velocity_vs_depth")

    # ── 1b. All 5 State Variables vs Depth  (Subplot grid) ─────────
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f"Complete State Evolution — {D_BULLET}mm, {M_BULLET}g bullet "
                 f"into {T_PLATE}mm RHA @ {V0:.0f} m/s", fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    ax.plot(x, V, lw=2, color='#1f77b4')
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("① Velocity")

    ax = axes[0, 1]
    ax.plot(x, T, lw=2, color='#d62728')
    ax.axhline(MATERIALS[MAT]["T_melt"], color='r', ls=':', label=f'T_melt = {MATERIALS[MAT]["T_melt"]:.0f} K')
    ax.set_ylabel("Temperature (K)")
    ax.set_title("② Temperature")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(x, eps, lw=2, color='#9467bd')
    ax.set_ylabel("Plastic Strain (m/m)")
    ax.set_title("③ Strain")

    ax = axes[1, 1]
    ax.plot(x, sig, lw=2, color='#2ca02c')
    ax.set_ylabel("Flow Stress (MPa)")
    ax.set_title("④ Flow Stress (Effective)")

    ax = axes[2, 0]
    ax.plot(x, dmg, lw=2, color='#ff7f0e')
    ax.axhline(1.0, color='r', ls=':', label='Full Failure (D=1)')
    ax.set_ylabel("Damage Parameter D")
    ax.set_xlabel("Depth (mm)")
    ax.set_title("⑤ Damage Evolution")
    ax.set_ylim(-0.05, 1.1)
    ax.legend()

    ax = axes[2, 1]
    ax.plot(x, edr, lw=2, color='#8c564b')
    ax.set_ylabel("Strain Rate (s⁻¹)")
    ax.set_xlabel("Depth (mm)")
    ax.set_title("⑥ Strain Rate")
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save("02_all_state_variables")

    # ── 1c. Energy Bookkeeping (KE + Absorbed) ─────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(t_us, KE, alpha=0.15, color='#2ca02c')
    ax.plot(t_us, KE, lw=2, color='#2ca02c', label='Projectile KE')
    ax.fill_between(t_us, E_abs, alpha=0.15, color='#d62728')
    ax.plot(t_us, E_abs, lw=2, color='#d62728', label='Energy Absorbed by Plate')
    ax.plot(t_us, KE + E_abs - KE[0]*np.ones_like(t_us) + KE, lw=1,
            ls='--', color='grey', alpha=0.0)  # phantom for layout
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Energy (J)")
    ax.set_title("Energy Conservation — KE Transfer During Impact")
    ax.legend()
    save("03_energy_vs_time")

    # ── 1d. Stress–Strain Curve (Dynamic Hardening + Thermal Softening)
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(eps, sig, c=T, cmap='hot', s=12, zorder=3)
    ax.plot(eps, sig, lw=1.5, color='grey', alpha=0.4, zorder=2)
    cbar = plt.colorbar(sc, ax=ax, label='Temperature (K)')
    ax.set_xlabel("Plastic Strain (m/m)")
    ax.set_ylabel("Effective Flow Stress (MPa)")
    ax.set_title("Dynamic Stress–Strain Curve (colored by Temperature)")
    save("04_stress_strain_colored")

    # ── 1e. Damage + Stress combined ───────────────────────────────
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(x, sig, lw=2, color='#2ca02c', label='Flow Stress')
    ax1.set_xlabel("Depth (mm)")
    ax1.set_ylabel("Flow Stress (MPa)", color='#2ca02c')
    ax1.tick_params(axis='y', labelcolor='#2ca02c')

    ax2 = ax1.twinx()
    ax2.plot(x, dmg, lw=2, color='#e63946', label='Damage (D)')
    ax2.axhline(1.0, ls=':', color='grey', alpha=0.6)
    ax2.set_ylabel("Damage Parameter D", color='#e63946')
    ax2.tick_params(axis='y', labelcolor='#e63946')
    ax2.set_ylim(-0.05, 1.15)

    ax1.set_title("Stress Collapse & Damage Accumulation vs Depth")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    save("05_damage_vs_stress")

    # ── 1f. Force vs Time ──────────────────────────────────────────
    dt_us = np.diff(t_us)
    dV    = np.diff(V)
    dt_s  = np.maximum(dt_us * 1e-6, 1e-12)
    decel = -dV / dt_s
    force_kN = (m_kg * decel) / 1000.0
    force_kN = np.insert(force_kN, 0, force_kN[0])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(t_us, force_kN, alpha=0.2, color='#8c564b')
    ax.plot(t_us, force_kN, lw=2, color='#8c564b')
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Contact Force (kN)")
    ax.set_title("Impact Contact Force vs Time")
    save("06_force_vs_time")


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  SECTION 2 — MULTI-MATERIAL COMPARISON  (4 graphs)              ║
# ╚═══════════════════════════════════════════════════════════════════╝

def section_2_multi_material():
    print("\n[Section 2] Multi-Material Comparison ...")

    # ── 2a. V50 Ballistic Limit Bar Chart ──────────────────────────
    v50_data = {}
    for mn in MAT_NAMES:
        try:
            res = simulate_ballistic_impact(T_PLATE, mn, 298.0, 1000.0,
                                            M_BULLET, D_BULLET, 0.0)
            v50_data[MAT_LABELS[mn]] = res.ballistic_limit_ms
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(list(v50_data.keys()), list(v50_data.values()),
                   color=COLORS[:len(v50_data)], edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, v50_data.values()):
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
                f'{val:.0f} m/s', va='center', fontweight='bold', fontsize=10)
    ax.set_xlabel("Ballistic Limit V₅₀ (m/s)")
    ax.set_title(f"Ballistic Limit Comparison — {D_BULLET}mm bullet vs {T_PLATE}mm plates")
    save("07_v50_comparison_bar")

    # ── 2b. Velocity Decay overlay (all materials) ─────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, mn in enumerate(MAT_NAMES):
        try:
            traj = simulate_trajectory(T_PLATE, mn, 298.0, V0, M_BULLET, D_BULLET)
            ax.plot(traj["x_mm"], traj["V_ms"], lw=2,
                    color=COLORS[i], label=MAT_LABELS[mn])
        except Exception:
            pass
    ax.set_xlabel("Depth (mm)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title(f"Velocity Decay Comparison — {V0:.0f} m/s into {T_PLATE}mm plates")
    ax.legend()
    save("08_velocity_decay_all_materials")

    # ── 2c. Temperature Rise overlay ───────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, mn in enumerate(MAT_NAMES):
        try:
            traj = simulate_trajectory(T_PLATE, mn, 298.0, V0, M_BULLET, D_BULLET)
            ax.plot(traj["x_mm"], traj["T_K"], lw=2,
                    color=COLORS[i], label=MAT_LABELS[mn])
            ax.axhline(MATERIALS[mn]["T_melt"], color=COLORS[i], ls=':', alpha=0.4)
        except Exception:
            pass
    ax.set_xlabel("Depth (mm)")
    ax.set_ylabel("Temperature (K)")
    ax.set_title(f"Adiabatic Temperature Rise — {V0:.0f} m/s into {T_PLATE}mm plates")
    ax.legend()
    save("09_temperature_all_materials")

    # ── 2d. Stress-Strain overlay ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, mn in enumerate(MAT_NAMES):
        try:
            traj = simulate_trajectory(T_PLATE, mn, 298.0, V0, M_BULLET, D_BULLET)
            ax.plot(traj["strain"], traj["sigma_MPa"], lw=2,
                    color=COLORS[i], label=MAT_LABELS[mn])
        except Exception:
            pass
    ax.set_xlabel("Plastic Strain")
    ax.set_ylabel("Flow Stress (MPa)")
    ax.set_title("Dynamic Stress–Strain Curves — All Materials")
    ax.legend()
    save("10_stress_strain_all_materials")


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  SECTION 3 — THICKNESS SWEEP  (2 graphs)                        ║
# ╚═══════════════════════════════════════════════════════════════════╝

def section_3_thickness_sweep():
    print("\n[Section 3] Thickness Sweep ...")
    thicknesses = [5, 8, 10, 12, 15, 20, 25, 30]

    # ── 3a. V50 vs Thickness ───────────────────────────────────────
    v50_num = []
    v50_ri  = []
    for t in thicknesses:
        res = simulate_ballistic_impact(float(t), MAT, 298.0, 1000.0,
                                        M_BULLET, D_BULLET)
        v50_num.append(res.ballistic_limit_ms)
        ri = recht_ipson(1000.0, float(t), MAT, M_BULLET, D_BULLET)
        v50_ri.append(ri.V50_ms)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thicknesses, v50_num, 'o-', lw=2, color='#e63946',
            markersize=7, label='Numerical Engine')
    ax.plot(thicknesses, v50_ri, 's--', lw=2, color='#457b9d',
            markersize=7, label='Recht-Ipson (Analytical)')
    ax.set_xlabel("Plate Thickness (mm)")
    ax.set_ylabel("Ballistic Limit V₅₀ (m/s)")
    ax.set_title("V₅₀ vs Plate Thickness — Numerical vs Analytical")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save("11_v50_vs_thickness")

    # ── 3b. Velocity decay at different thicknesses ────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, t in enumerate([5, 10, 15, 20]):
        traj = simulate_trajectory(float(t), MAT, 298.0, V0, M_BULLET, D_BULLET)
        ax.plot(traj["x_mm"], traj["V_ms"], lw=2,
                color=COLORS[i], label=f't = {t} mm')
        ax.axvline(t, color=COLORS[i], ls=':', alpha=0.4)
    ax.set_xlabel("Depth (mm)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title(f"Effect of Plate Thickness on Velocity Decay — {V0:.0f} m/s RHA")
    ax.legend()
    save("12_velocity_thickness_sweep")


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  SECTION 4 — VELOCITY SENSITIVITY  (2 graphs)                   ║
# ╚═══════════════════════════════════════════════════════════════════╝

def section_4_velocity_sensitivity():
    print("\n[Section 4] Velocity Sensitivity ...")

    # ── 4a. Residual Velocity vs Impact Velocity (with V50 marker) ─
    V_sweep = np.linspace(200, 1500, 40)
    Vr_num = []
    res_base = simulate_ballistic_impact(T_PLATE, MAT, 298.0, 1000.0,
                                         M_BULLET, D_BULLET)
    V50_num = res_base.ballistic_limit_ms

    ri_base = recht_ipson(1000.0, T_PLATE, MAT, M_BULLET, D_BULLET)
    V50_ri = ri_base.V50_ms

    Vr_ri = []
    for v in V_sweep:
        res = simulate_ballistic_impact(T_PLATE, MAT, 298.0, float(v),
                                         M_BULLET, D_BULLET)
        Vr_num.append(res.residual_velocity_ms)
        ri = recht_ipson(float(v), T_PLATE, MAT, M_BULLET, D_BULLET)
        Vr_ri.append(ri.residual_velocity_ms)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(V_sweep, Vr_num, 'o-', color='#e63946', markersize=4, lw=2,
            label='Numerical Engine')
    ax.plot(V_sweep, Vr_ri, '--', color='#457b9d', lw=2,
            label='Recht-Ipson (Analytical)')
    ax.axvline(V50_num, color='black', ls=':', lw=1.2,
               label=f'Num V₅₀ = {V50_num:.0f} m/s')
    ax.axvline(V50_ri, color='grey', ls=':', lw=1.2,
               label=f'RI V₅₀ = {V50_ri:.0f} m/s')
    ax.fill_between(V_sweep, 0, np.maximum(Vr_num, 0), alpha=0.05, color='red')
    ax.set_xlabel("Impact Velocity (m/s)")
    ax.set_ylabel("Residual Velocity (m/s)")
    ax.set_title(f"Residual vs Impact Velocity — {T_PLATE}mm RHA")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save("13_residual_vs_impact")

    # ── 4b. Velocity decay curves at different V0 ──────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, v0 in enumerate([400, 600, 800, 1000, 1200]):
        traj = simulate_trajectory(T_PLATE, MAT, 298.0, float(v0),
                                   M_BULLET, D_BULLET)
        ax.plot(traj["x_mm"], traj["V_ms"], lw=2,
                color=COLORS[i], label=f'V₀ = {v0} m/s')
    ax.axvline(T_PLATE, color='black', ls='--', alpha=0.5, label='Plate exit')
    ax.set_xlabel("Depth (mm)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Velocity Profiles at Different Impact Speeds")
    ax.legend()
    save("14_velocity_profiles_v0_sweep")


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  SECTION 5 — JOHNSON-COOK MODEL DECOMPOSITION  (2 graphs)       ║
# ╚═══════════════════════════════════════════════════════════════════╝

def section_5_jc_decomposition():
    print("\n[Section 5] Johnson-Cook Decomposition ...")
    mat = MATERIALS[MAT]

    # ── 5a. Effect of Temperature on JC flow stress ────────────────
    eps_range = np.linspace(0.001, 1.0, 200)
    temps = [298, 600, 900, 1200, 1500]

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, T in enumerate(temps):
        sig, _ = johnson_cook_stress(
            eps_range, 1e4, float(T),
            mat["A"], mat["B"], mat["n"], mat["C"], mat["m"],
            T_melt=mat["T_melt"])
        ax.plot(eps_range, sig, lw=2, color=COLORS[i], label=f'T = {T} K')

    ax.set_xlabel("Plastic Strain")
    ax.set_ylabel("Flow Stress σ (MPa)")
    ax.set_title(f"JC Model — Thermal Softening Effect (RHA Steel, ε̇ = 10⁴ s⁻¹)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save("15_jc_temperature_effect")

    # ── 5b. Effect of Strain Rate on JC flow stress ────────────────
    rates = [1, 1e2, 1e3, 1e4, 1e5]
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, rate in enumerate(rates):
        sig, _ = johnson_cook_stress(
            eps_range, rate, 298.0,
            mat["A"], mat["B"], mat["n"], mat["C"], mat["m"],
            T_melt=mat["T_melt"])
        ax.plot(eps_range, sig, lw=2, color=COLORS[i],
                label=f'ε̇ = {rate:.0e} s⁻¹')

    ax.set_xlabel("Plastic Strain")
    ax.set_ylabel("Flow Stress σ (MPa)")
    ax.set_title(f"JC Model — Strain Rate Hardening Effect (RHA Steel, T = 298 K)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save("16_jc_strain_rate_effect")


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  SECTION 6 — ENERGY ANALYSIS  (2 graphs)                        ║
# ╚═══════════════════════════════════════════════════════════════════╝

def section_6_energy_analysis():
    print("\n[Section 6] Energy Analysis ...")

    # ── 6a. Energy Absorption % vs Impact Velocity ─────────────────
    V_sweep = np.linspace(300, 1500, 30)
    e_pct = []
    for v in V_sweep:
        res = simulate_ballistic_impact(T_PLATE, MAT, 298.0, float(v),
                                         M_BULLET, D_BULLET)
        e_pct.append(res.energy_absorption_pct)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(V_sweep, e_pct, 'o-', lw=2, color='#d62728', markersize=5)
    ax.axhline(100, ls='--', color='grey', alpha=0.5, label='100% (bullet stopped)')
    ax.set_xlabel("Impact Velocity (m/s)")
    ax.set_ylabel("Energy Absorbed by Plate (%)")
    ax.set_title(f"Energy Absorption Efficiency — {T_PLATE}mm RHA")
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(True, alpha=0.3)
    save("17_energy_absorption_pct")

    # ── 6b. Energy Absorption for all materials at fixed V0 ────────
    mat_names_clean = []
    energies = []
    for mn in MAT_NAMES:
        try:
            res = simulate_ballistic_impact(T_PLATE, mn, 298.0, V0,
                                            M_BULLET, D_BULLET)
            mat_names_clean.append(MAT_LABELS[mn])
            energies.append(res.energy_absorbed_J)
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(mat_names_clean, energies,
                   color=COLORS[:len(mat_names_clean)], edgecolor='black', lw=0.5)
    for bar, val in zip(bars, energies):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f} J', va='center', fontweight='bold')
    ax.set_xlabel("Energy Absorbed (J)")
    ax.set_title(f"Energy Absorbed by Different Materials — {V0:.0f} m/s, {T_PLATE}mm plates")
    save("18_energy_absorbed_materials")


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  SECTION 7 — VALIDATION  (3 graphs)                             ║
# ╚═══════════════════════════════════════════════════════════════════╝

def section_7_validation():
    print("\n[Section 7] Validation Plots ...")

    # ── 7a. Parity Plot (Numerical vs Analytical V_r) ──────────────
    V_list = np.linspace(300, 1500, 25)
    vr_num = []
    vr_ri  = []
    for v in V_list:
        res = simulate_ballistic_impact(T_PLATE, MAT, 298.0, float(v),
                                         M_BULLET, D_BULLET)
        vr_num.append(res.residual_velocity_ms)
        ri = recht_ipson(float(v), T_PLATE, MAT, M_BULLET, D_BULLET)
        vr_ri.append(ri.residual_velocity_ms)

    vr_num = np.array(vr_num)
    vr_ri  = np.array(vr_ri)

    fig, ax = plt.subplots(figsize=(7, 7))
    lim = max(max(vr_num), max(vr_ri)) * 1.1
    ax.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.5, label='Perfect Agreement')
    ax.scatter(vr_ri, vr_num, c=V_list, cmap='plasma', s=50, edgecolors='black',
               linewidth=0.5, zorder=3)
    cbar = plt.colorbar(ax.collections[0], ax=ax, label='Impact Velocity (m/s)')
    ax.set_xlabel("Recht-Ipson V_r (Analytical) [m/s]")
    ax.set_ylabel("Numerical V_r (Simulation) [m/s]")
    ax.set_title("Parity Plot — Numerical vs Analytical Residual Velocity")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    save("19_parity_plot")

    # ── 7b. V_r Error (Absolute) vs Impact Velocity ────────────────
    delta = np.abs(vr_num - vr_ri)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(V_list, delta, width=40, color='#e9c46a', edgecolor='black', lw=0.5)
    ax.set_xlabel("Impact Velocity (m/s)")
    ax.set_ylabel("|ΔV_r| (m/s)")
    ax.set_title("Prediction Error — |Numerical − Analytical| Residual Velocity")
    ax.grid(True, alpha=0.3)
    save("20_prediction_error")

    # ── 7c. Failure Mode Classification ────────────────────────────
    V_sweep = np.linspace(200, 1500, 50)
    modes = []
    for v in V_sweep:
        res = simulate_ballistic_impact(T_PLATE, MAT, 298.0, float(v),
                                         M_BULLET, D_BULLET)
        modes.append(res.failure_mode)

    unique_modes = list(set(modes))
    mode_colors = {m: COLORS[i] for i, m in enumerate(unique_modes)}

    fig, ax = plt.subplots(figsize=(10, 4))
    for v, mode in zip(V_sweep, modes):
        ax.barh(0, 1, left=v, color=mode_colors[mode], edgecolor='none', height=0.5)

    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=mode_colors[m], label=m.replace('_', ' ').title())
               for m in unique_modes]
    ax.legend(handles=handles, loc='upper left')
    ax.set_xlabel("Impact Velocity (m/s)")
    ax.set_yticks([])
    ax.set_title(f"Failure Mode Classification — {T_PLATE}mm RHA")
    save("21_failure_modes")


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  SECTION 8 — PEAK VALUES HEATMAP  (1 graph)                     ║
# ╚═══════════════════════════════════════════════════════════════════╝

def section_8_heatmap():
    print("\n[Section 8] Peak Values Heatmap ...")
    thicknesses = [5, 8, 10, 12, 15, 20]
    velocities  = [400, 600, 800, 1000, 1200]

    peak_T_grid = np.zeros((len(thicknesses), len(velocities)))

    for i, t in enumerate(thicknesses):
        for j, v in enumerate(velocities):
            res = simulate_ballistic_impact(float(t), MAT, 298.0, float(v),
                                            M_BULLET, D_BULLET)
            peak_T_grid[i, j] = res.peak_temperature_K

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(peak_T_grid, cmap='inferno', aspect='auto', origin='lower')
    ax.set_xticks(range(len(velocities)))
    ax.set_xticklabels([f'{v}' for v in velocities])
    ax.set_yticks(range(len(thicknesses)))
    ax.set_yticklabels([f'{t} mm' for t in thicknesses])
    ax.set_xlabel("Impact Velocity (m/s)")
    ax.set_ylabel("Plate Thickness")
    ax.set_title("Peak Temperature Heatmap (K) — RHA Steel")

    # Annotate each cell
    for i in range(len(thicknesses)):
        for j in range(len(velocities)):
            val = peak_T_grid[i, j]
            color = 'white' if val > (peak_T_grid.max() + peak_T_grid.min())/2 else 'black'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                    color=color, fontweight='bold', fontsize=9)

    plt.colorbar(im, ax=ax, label='Peak Temperature (K)')
    save("22_peak_temperature_heatmap")


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                            ║
# ╚═══════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    print("=" * 60)
    print("  DMSRDE — Comprehensive Report Graph Generator")
    print("=" * 60)

    section_1_single_trajectory()
    section_2_multi_material()
    section_3_thickness_sweep()
    section_4_velocity_sensitivity()
    section_5_jc_decomposition()
    section_6_energy_analysis()
    section_7_validation()
    section_8_heatmap()

    total = len([f for f in os.listdir(OUT) if f.endswith('.png')])
    print(f"\n{'=' * 60}")
    print(f"  DONE! {total} graphs saved to:  {OUT}/")
    print(f"{'=' * 60}")
