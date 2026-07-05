"""
DMSRDE Ballistic Simulation — Part 3: Recht-Ipson Model

ANALYTICAL residual velocity and ballistic limit (V50).
This is the VALIDATION BENCHMARK for the numerical integration
in simulation.py. If the numerical V_r diverges significantly
from the analytical Recht-Ipson V_r, something is wrong.

Physics:
  Recht-Ipson assumes adiabatic shear plugging — the projectile
  punches a cylindrical plug out of the plate. Momentum conservation
  between projectile + plug gives the residual velocity.

  V_r = a · √(V₀² - V₅₀²)   for V₀ > V₅₀
  V_r = 0                     for V₀ ≤ V₅₀

  where a = m_p / (m_p + m_plug)    (momentum sharing factor)
        V₅₀ = √(2 · E_perforation / m_p)

References:
  - Recht & Ipson (1963), J. Applied Mechanics, 30(3), 384-390
  - Ipson & Recht (1977), Experimental Mechanics, 17, 249-253

Run: python stage1_part3_recht_ipson.py
"""
from __future__ import annotations
import math
from typing import Dict, Optional
from dataclasses import dataclass

from stage1_part1_johnson_cook import MATERIALS, T_ROOM

# ===================================================================
#  PLUG GEOMETRY (shared with simulation.py — same formulas)
# ===================================================================

def _plug_mass_ri(
    r_bullet_m: float,
    t_plate_m: float,
    rho_plate: float,
    shear_angle_deg: float = 6.0,
) -> float:
    """
    Truncated-cone plug mass.
    Matches simulation._plug_mass exactly.
    """
    alpha = math.radians(shear_angle_deg)
    r1 = r_bullet_m
    r2 = r_bullet_m + t_plate_m * math.tan(alpha)
    return rho_plate * math.pi * t_plate_m / 3.0 * (r1*r1 + r1*r2 + r2*r2)


def _perforation_energy(
    t_plate_m: float,
    r_bullet_m: float,
    tau_f_Pa: float,
    shear_angle_deg: float = 6.0,
) -> float:
    """
    Energy to shear the plug free.
    E = τ_f · π · d · t²   (simple adiabatic shear model)

    For the truncated cone:
    E = τ_f · π · (r1 + r2) · L_shear · t
    where L_shear = t / cos(α)  is the shear band length.
    """
    alpha = math.radians(shear_angle_deg)
    r2 = r_bullet_m + t_plate_m * math.tan(alpha)
    L_shear = t_plate_m / math.cos(alpha)
    perimeter_avg = math.pi * (r_bullet_m + r2)
    return 0.5 * tau_f_Pa * perimeter_avg * L_shear * t_plate_m


# ===================================================================
#  RECHT-IPSON MODEL
# ===================================================================

@dataclass
class RechtIpsonResult:
    """Output of the Recht-Ipson analytical model."""
    V50_ms: float               # Ballistic limit (m/s)
    residual_velocity_ms: float # Residual velocity (m/s) — 0 if stopped
    penetrated: bool            # True if V₀ > V₅₀
    momentum_factor: float      # a = m_p / (m_p + m_plug)
    plug_mass_kg: float         # Mass of the sheared plug
    perforation_energy_J: float # Energy to perforate the plate
    tau_f_MPa: float            # Shear failure stress used


def _estimate_tau_f(mat: dict) -> float:
    """
    Estimate dynamic shear failure stress from JC parameters.

    τ_f ≈ σ_y / √3   (von Mises)

    For a more accurate estimate, we evaluate JC at representative
    ballistic conditions: ε=0.3, ε̇=10⁴ s⁻¹, T = 0.5·T_melt.
    This accounts for strain hardening + rate effects + thermal softening
    that the plug actually experiences during perforation.
    """
    from stage1_part1_johnson_cook import _jc_sigma_scalar

    # Representative ballistic conditions for plug shearing
    eps = 0.3           # typical shear strain at plug boundary
    eps_dot = 1e4       # typical strain rate during perforation
    T_avg = 0.5 * (T_ROOM + mat["T_melt"])  # average plug temperature

    eps_dot_ref = mat.get("eps_dot_ref", 1.0)
    T_room = mat.get("T_room", T_ROOM)

    sigma_jc, _, _, _ = _jc_sigma_scalar(
        eps, eps_dot, T_avg,
        mat["A"], mat["B"], mat["n"], mat["C"], mat["m"],
        mat["T_melt"], eps_dot_ref, T_room)

    # Convert from normal stress to shear stress (von Mises)
    tau_f_MPa = sigma_jc / math.sqrt(3.0)
    return tau_f_MPa


def recht_ipson(
    V0_ms: float,
    plate_thickness_mm: float,
    material_name: str = "RHA_steel",
    bullet_mass_g: float = 7.9,
    bullet_diameter_mm: float = 7.62,
    obliquity_angle_deg: float = 0.0,
    *,
    shear_angle_deg: float = 6.0,
    tau_f_override_MPa: Optional[float] = None,
) -> RechtIpsonResult:
    """
    Recht-Ipson analytical residual velocity model.

    Parameters
    ----------
    V0_ms : float
        Impact velocity (m/s).
    plate_thickness_mm : float
        Plate thickness (mm).
    material_name : str
        Key into MATERIALS dict.
    bullet_mass_g : float
        Projectile mass (grams).
    bullet_diameter_mm : float
        Projectile diameter (mm).
    obliquity_angle_deg : float
        Angle of obliquity (degrees).
    shear_angle_deg : float
        Plug shear cone half-angle (degrees).
    tau_f_override_MPa : float, optional
        Override the computed shear failure stress.

    Returns
    -------
    RechtIpsonResult
        Analytical predictions.
    """
    if material_name not in MATERIALS:
        raise ValueError(f"Unknown material: {material_name}")
    if V0_ms <= 0:
        raise ValueError("V0_ms must be > 0")
    if plate_thickness_mm <= 0:
        raise ValueError("plate_thickness_mm must be > 0")
    if bullet_mass_g <= 0:
        raise ValueError("bullet_mass_g must be > 0")
    if not (0.0 <= obliquity_angle_deg < 90.0):
        raise ValueError("obliquity_angle_deg must be in [0, 90)")

    mat = MATERIALS[material_name]

    # Unit conversions
    t_m = plate_thickness_mm * 1e-3
    m_p = bullet_mass_g * 1e-3
    r_m = (bullet_diameter_mm / 2.0) * 1e-3
    t_eff = t_m / math.cos(math.radians(obliquity_angle_deg))

    # Shear failure stress
    if tau_f_override_MPa is not None:
        tau_f_MPa = tau_f_override_MPa
    else:
        tau_f_MPa = _estimate_tau_f(mat)
    tau_f_Pa = tau_f_MPa * 1e6

    # Plug mass (truncated cone)
    m_plug = _plug_mass_ri(r_m, t_eff, mat["rho"], shear_angle_deg)

    # Momentum sharing factor
    a = m_p / (m_p + m_plug)

    # Perforation energy → V₅₀
    E_perf = _perforation_energy(t_eff, r_m, tau_f_Pa, shear_angle_deg)
    V50 = math.sqrt(2.0 * E_perf / m_p) if m_p > 0 else float("inf")

    # Residual velocity
    if V0_ms > V50:
        V_r = a * math.sqrt(V0_ms**2 - V50**2)
        penetrated = True
    else:
        V_r = 0.0
        penetrated = False

    return RechtIpsonResult(
        V50_ms=round(V50, 2),
        residual_velocity_ms=round(V_r, 2),
        penetrated=penetrated,
        momentum_factor=round(a, 4),
        plug_mass_kg=m_plug,
        perforation_energy_J=round(E_perf, 2),
        tau_f_MPa=round(tau_f_MPa, 1),
    )


# ===================================================================
#  GENERALIZED LAMBERT-JONAS FORM
# ===================================================================

def lambert_jonas(
    V0_ms: float,
    V50_ms: float,
    m_p_kg: float,
    m_plug_kg: float,
    p: float = 2.0,
) -> float:
    """
    Generalized Lambert-Jonas residual velocity model.

    V_r = a · (V₀ᵖ - V₅₀ᵖ)^(1/p)

    p = 2.0 → standard Recht-Ipson (plugging)
    p = 2.0 is appropriate for most metallic armour.
    p ≠ 2 can be fitted to experimental data.

    Reference:
        Lambert & Jonas (1976), ARBRL-TR-02021
    """
    if V0_ms <= V50_ms:
        return 0.0
    a = m_p_kg / (m_p_kg + m_plug_kg)
    return a * (V0_ms**p - V50_ms**p) ** (1.0 / p)


# ===================================================================
#  VALIDATION: Compare analytical vs numerical
# ===================================================================

def validate_against_simulation(
    plate_thickness_mm: float = 10.0,
    material_name: str = "RHA_steel",
    velocities_ms: list = None,
    **kwargs,
) -> list:
    """
    Run both Recht-Ipson (analytical) and simulation.py (numerical)
    side by side. Returns list of comparison dicts.

    This is the key validation: if they diverge wildly, either
    the analytical model assumptions are violated or the numerical
    integration has a bug.
    """
    from simulation import simulate_ballistic_impact

    if velocities_ms is None:
        velocities_ms = [200, 400, 600, 800, 1000, 1200]

    results = []
    for V in velocities_ms:
        ri = recht_ipson(V, plate_thickness_mm, material_name, **kwargs)
        sim = simulate_ballistic_impact(
            plate_thickness_mm, material_name,
            bullet_velocity_ms=V,
            bullet_mass_g=kwargs.get("bullet_mass_g", 7.9),
            bullet_diameter_mm=kwargs.get("bullet_diameter_mm", 7.62),
            obliquity_angle_deg=kwargs.get("obliquity_angle_deg", 0.0),
        )
        results.append({
            "V0_ms": V,
            "RI_V_r": ri.residual_velocity_ms,
            "Sim_V_r": sim.residual_velocity_ms,
            "RI_V50": ri.V50_ms,
            "Sim_V_BL": sim.ballistic_limit_ms,
            "RI_penetrated": ri.penetrated,
            "Sim_penetrated": sim.penetrated,
            "delta_V_r": abs(ri.residual_velocity_ms - sim.residual_velocity_ms),
        })
    return results


# ===================================================================
#  TESTS
# ===================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  Recht-Ipson Model -- DMSRDE")
    print("=" * 70)

    # --- Basic predictions ---
    print("\n>>> Single shot: 7.62mm @ 715 m/s into 10mm RHA\n")
    r = recht_ipson(715.0, 10.0, "RHA_steel")
    print(f"  V50 (ballistic limit): {r.V50_ms:.1f} m/s")
    print(f"  Residual velocity:     {r.residual_velocity_ms:.1f} m/s")
    print(f"  Penetrated:            {r.penetrated}")
    print(f"  Momentum factor (a):   {r.momentum_factor:.4f}")
    print(f"  Plug mass:             {r.plug_mass_kg*1000:.2f} g")
    print(f"  tau_f (shear failure): {r.tau_f_MPa:.1f} MPa")

    # --- Velocity sweep ---
    print("\n>>> Velocity sweep: 10mm RHA\n")
    print(f"  {'V0 (m/s)':>10} | {'V_r (m/s)':>10} | {'V50':>8} | {'Status':>6}")
    print("  " + "-" * 48)
    for V in [200, 400, 600, 800, 1000, 1200]:
        r = recht_ipson(V, 10.0, "RHA_steel")
        s = "PEN" if r.penetrated else "STOP"
        print(f"  {V:>10.0f} | {r.residual_velocity_ms:>10.1f} | "
              f"{r.V50_ms:>8.1f} | {s:>6}")

    # --- Thickness sweep ---
    print("\n>>> Thickness sweep: 7.62mm @ 715 m/s into RHA\n")
    print(f"  {'t (mm)':>8} | {'V50 (m/s)':>10} | {'V_r (m/s)':>10} | {'a':>6}")
    print("  " + "-" * 44)
    for t in [5, 10, 15, 20, 30]:
        r = recht_ipson(715.0, t, "RHA_steel")
        print(f"  {t:>8} | {r.V50_ms:>10.1f} | "
              f"{r.residual_velocity_ms:>10.1f} | {r.momentum_factor:>6.4f}")

    # --- Multi-material ---
    print("\n>>> Multi-material: 10mm plates @ 715 m/s\n")
    for mat_name in MATERIALS:
        r = recht_ipson(715.0, 10.0, mat_name)
        label = MATERIALS[mat_name]['label'].encode('ascii', 'replace').decode()
        print(f"  {label:>40} | "
              f"V50={r.V50_ms:>7.1f} | V_r={r.residual_velocity_ms:>6.1f} | "
              f"tau_f={r.tau_f_MPa:>6.0f} MPa")

    # --- Analytical vs Numerical validation ---
    print("\n>>> Validation: Recht-Ipson vs Numerical (10mm RHA)\n")
    print(f"  {'V0':>6} | {'RI V_r':>8} | {'Sim V_r':>8} | {'dV_r':>8} | "
          f"{'RI V50':>8} | {'Sim VBL':>8}")
    print("  " + "-" * 62)
    comparisons = validate_against_simulation()
    for c in comparisons:
        print(f"  {c['V0_ms']:>6.0f} | {c['RI_V_r']:>8.1f} | {c['Sim_V_r']:>8.1f} | "
              f"{c['delta_V_r']:>8.1f} | {c['RI_V50']:>8.1f} | {c['Sim_V_BL']:>8.1f}")

    # --- Lambert-Jonas generalized ---
    print("\n>>> Lambert-Jonas (p=2.0 vs p=2.5):\n")
    r = recht_ipson(715.0, 10.0, "RHA_steel")
    for p in [2.0, 2.5, 3.0]:
        V_r = lambert_jonas(715.0, r.V50_ms, 7.9e-3,
                            r.plug_mass_kg, p=p)
        print(f"  p={p:.1f}: V_r = {V_r:.1f} m/s")

    print("\n" + "=" * 70)

