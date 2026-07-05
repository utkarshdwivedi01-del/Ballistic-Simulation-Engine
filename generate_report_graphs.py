import os
import numpy as np
import matplotlib.pyplot as plt
from simulation import simulate_trajectory, simulate_ballistic_impact
from stage1_part3_recht_ipson import _plug_mass_ri, _perforation_energy, _estimate_tau_f, RechtIpsonResult, MATERIALS

def compute_time(x_mm, V_ms):
    """Compute time array from x and V arrays."""
    dx_m = np.diff(x_mm) / 1000.0
    V_avg = (V_ms[:-1] + V_ms[1:]) / 2.0
    # Avoid division by zero
    V_avg = np.maximum(V_avg, 1e-6)
    dt_s = dx_m / V_avg
    time_s = np.insert(np.cumsum(dt_s), 0, 0.0)
    return time_s

def ri_analytical(V0, V50, plug_mass, bullet_mass):
    if V0 <= V50:
        return 0.0
    a = bullet_mass / (bullet_mass + plug_mass)
    return a * np.sqrt(V0**2 - V50**2)

def generate_graphs(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Scenario ---
    t_plate = 10.0 # mm
    mat_name = "RHA_steel"
    m_bullet = 7.9 # g
    d_bullet = 7.62 # mm
    V0 = 850.0 # m/s
    
    print(f"Generating graphs for {d_bullet}mm, {m_bullet}g bullet vs {t_plate}mm {mat_name} at {V0}m/s")
    
    # 1. Get Trajectory Data
    traj = simulate_trajectory(
        plate_thickness_mm=t_plate,
        material_name=mat_name,
        bullet_velocity_ms=V0,
        bullet_mass_g=m_bullet,
        bullet_diameter_mm=d_bullet
    )
    
    x_mm = traj["x_mm"]
    V_ms = traj["V_ms"]
    sigma_MPa = traj["sigma_MPa"]
    T_K = traj["T_K"]
    strain = traj["strain"]
    
    time_s = compute_time(x_mm, V_ms)
    time_us = time_s * 1e6
    
    m_kg = m_bullet * 1e-3
    ke_J = 0.5 * m_kg * (V_ms ** 2)
    energy_absorbed_J = ke_J[0] - ke_J
    
    # Deceleration & Force
    dt_s = np.diff(time_s)
    dV_ms = np.diff(V_ms)
    # Avoid division by zero
    dt_s_safe = np.maximum(dt_s, 1e-9)
    decel_ms2 = -dV_ms / dt_s_safe
    decel_ms2 = np.insert(decel_ms2, 0, decel_ms2[0]) # pad
    force_kN = (m_kg * decel_ms2) / 1000.0
    
    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Projectile Velocity vs Time
    plt.figure(figsize=(8, 5))
    plt.plot(time_us, V_ms, lw=2, color='#1f77b4')
    plt.title('Projectile Velocity vs Time')
    plt.xlabel('Time (µs)')
    plt.ylabel('Velocity (m/s)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/velocity_vs_time.png", dpi=300)
    plt.close()
    
    # 2. Penetration Depth vs Time
    plt.figure(figsize=(8, 5))
    plt.plot(time_us, x_mm, lw=2, color='#ff7f0e')
    plt.title('Penetration Depth vs Time')
    plt.xlabel('Time (µs)')
    plt.ylabel('Depth (mm)')
    plt.axhline(y=t_plate, color='r', linestyle='--', label='Plate Thickness')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/depth_vs_time.png", dpi=300)
    plt.close()
    
    # 3. Kinetic Energy and Absorbed Energy vs Time
    plt.figure(figsize=(8, 5))
    plt.plot(time_us, ke_J, lw=2, label='Projectile Kinetic Energy', color='#2ca02c')
    plt.plot(time_us, energy_absorbed_J, lw=2, label='Energy Absorbed by Plate', color='#d62728')
    plt.title('Energy Evolution vs Time')
    plt.xlabel('Time (µs)')
    plt.ylabel('Energy (Joules)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/energy_vs_time.png", dpi=300)
    plt.close()
    
    # 4. Stress vs Strain (Phase Portrait for Material Model)
    plt.figure(figsize=(8, 5))
    plt.plot(strain, sigma_MPa, lw=2, color='#9467bd')
    plt.title('Flow Stress vs Plastic Strain (Dynamic Hardening)')
    plt.xlabel('Plastic Strain (m/m)')
    plt.ylabel('Flow Stress (MPa)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/stress_vs_strain.png", dpi=300)
    plt.close()
    
    # 5. Contact Force vs Time
    plt.figure(figsize=(8, 5))
    plt.plot(time_us, force_kN, lw=2, color='#8c564b')
    plt.title('Impact Contact Force vs Time')
    plt.xlabel('Time (µs)')
    plt.ylabel('Force (kN)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/force_vs_time.png", dpi=300)
    plt.close()
    
    # 6. Temperature and Strain Evolution vs Depth (Layer-wise equivalent)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = 'tab:red'
    ax1.set_xlabel('Depth (mm)')
    ax1.set_ylabel('Temperature (K)', color=color)
    ax1.plot(x_mm, T_K, color=color, lw=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(y=MATERIALS[mat_name]['T_melt'], color='r', linestyle=':', label='Melting Temp')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    color = 'tab:purple'
    ax2.set_ylabel('Plastic Strain', color=color)
    ax2.plot(x_mm, strain, color=color, lw=2)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Thermo-Mechanical State vs Depth (Plate Interior)')
    fig.tight_layout()
    plt.savefig(f"{output_dir}/state_vs_depth.png", dpi=300)
    plt.close()
    
    # 7. Residual Velocity vs Impact Velocity (V_r vs V_i)
    V_sweep = np.linspace(200, 1500, 30)
    Vr_num = []
    
    res_base = simulate_ballistic_impact(t_plate, mat_name, 298.0, 1000.0, m_bullet, d_bullet, 0.0)
    V50_num = res_base.ballistic_limit_ms
    
    for v in V_sweep:
        res = simulate_ballistic_impact(t_plate, mat_name, 298.0, float(v), m_bullet, d_bullet, 0.0)
        Vr_num.append(res.residual_velocity_ms)
        
    Vr_num = np.array(Vr_num)
    
    mat = MATERIALS[mat_name]
    t_m = t_plate * 1e-3
    r_m = (d_bullet / 2.0) * 1e-3
    rho = mat["rho"]
    tau_f = _estimate_tau_f(mat) * 1e6
    
    plug_m = _plug_mass_ri(r_m, t_m, rho)
    E_perf = _perforation_energy(t_m, r_m, tau_f)
    V50_ri = np.sqrt(2.0 * E_perf / m_kg)
    
    Vr_ri = [ri_analytical(v, V50_ri, plug_m, m_kg) for v in V_sweep]
    
    plt.figure(figsize=(8, 5))
    plt.plot(V_sweep, Vr_num, 'o-', color='#e377c2', label='Numerical Model')
    plt.plot(V_sweep, Vr_ri, '--', color='#7f7f7f', label='Recht-Ipson (Analytical)')
    plt.axvline(x=V50_num, color='k', linestyle=':', label=f'Numerical V50 ({V50_num:.1f} m/s)')
    plt.title('Residual Velocity vs Impact Velocity')
    plt.xlabel('Impact Velocity (m/s)')
    plt.ylabel('Residual Velocity (m/s)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/residual_vs_impact.png", dpi=300)
    plt.close()
    
    print("All graphs generated successfully.")

if __name__ == "__main__":
    output_dir = "plots/report_graphs"
    generate_graphs(output_dir)
