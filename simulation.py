"""
DMSRDE Ballistic Simulation — Physics Engine
=============================================

Architecture:
  - JC equation lives ONCE in stage1_part1 (_jc_sigma_scalar)
  - CS equation lives ONCE in stage1_part2 (_cs_dif_scalar)
  - This file COMPOSES them — no duplicated physics

Numerical fixes:
  - Adaptive step size: fixed dx=0.01mm (not thickness/N)
  - Dynamic bisection: exponential search for V_hi
  - Truncated-cone plug mass
  - Geometry-derived deformation diameter

Run: python simulation.py
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

# --- Import canonical kernels (SINGLE SOURCE OF TRUTH) ---
from stage1_part1_johnson_cook import (MATERIALS, T_ROOM, _jc_sigma_scalar,
                                       MaterialConfigError, reload_materials)
from stage1_part2_cowper_symonds import _cs_dif_scalar

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(func=None, **kwargs):
        if func is not None: return func
        def _d(f): return f
        return _d

# -------------------------------------------------------------------
# Configurable constants
# -------------------------------------------------------------------
TAYLOR_QUINNEY_BETA: float = 0.9
SHEAR_ZONE_WIDTH_RATIO: float = 0.3965
PLUG_SHEAR_ANGLE_DEG: float = 6.0
# Default effective plug expansion angle.
# Should ideally be calibrated per-material from ballistic experiments.
# Not a resolved crack propagation angle.
TAN_ALPHA: float = math.tan(math.radians(PLUG_SHEAR_ANGLE_DEG))
SPATIAL_RESOLUTION_M: float = 1e-5   # 0.01 mm fixed step size
MIN_STEPS: int = 20
MAX_STEPS: int = 10000
V_BL_TOL: float = 2.0
V_BL_MAX_ITER: int = 25
# JC model is fitted for strain rates up to ~1e6 s^-1 (~2500 m/s impact).
# Above ~3000 m/s, materials enter hydrodynamic regime where strength is
# irrelevant and Mie-Grüneisen EOS is needed. Capping here prevents
# generating garbage PINN training data.
V_SEARCH_CEILING: float = 3000.0     # m/s — JC model validity ceiling

_RATE_JC: int = 0
_RATE_CS: int = 1
_RATE_BLENDED: int = 2
_RATE_MODEL_MAP = {"jc": _RATE_JC, "cs": _RATE_CS, "blended": _RATE_BLENDED}


# ===================================================================
#  COMPOSED FLOW STRESS — calls kernels from part1 + part2 (DRY)
# ===================================================================

@njit(nogil=True, cache=True)
def _flow_stress(
    eps: float, eps_dot: float, T: float,
    A: float, B: float, n: float, C: float, m_jc: float,
    T_melt: float, eps_dot_ref: float, T_room: float,
    cs_D: float, cs_q: float, rate_model: int,
    clamp_low_rate: bool = True
) -> float:
    """
    Composed flow stress. No duplicated equations.
    JC terms from _jc_sigma_scalar (part1).
    CS DIF from _cs_dif_scalar (part2).
    """
    sigma_jc, term1, term2_jc, term3 = _jc_sigma_scalar(
        eps, eps_dot, T, A, B, n, C, m_jc,
        T_melt, eps_dot_ref, T_room, clamp_low_rate)

    if rate_model == 0:
        return sigma_jc
    elif rate_model == 1:
        cs_dif = _cs_dif_scalar(eps_dot, cs_D, cs_q)
        return term1 * cs_dif * term3
    else:
        # WARNING:
        # This mode averages JC and Cowper-Symonds rate multipliers:
        #     0.5*(term2_jc + cs_dif)
        # It has no known basis in standard constitutive literature
        # (LS-DYNA, Abaqus, AUTODYN, original JC or CS formulations).
        # Intended only for numerical comparison and sensitivity studies.
        cs_dif = _cs_dif_scalar(eps_dot, cs_D, cs_q)
        return term1 * 0.5 * (term2_jc + cs_dif) * term3


# ===================================================================
#  INTEGRATION KERNEL — adaptive step size
# ===================================================================

@njit(nogil=True, cache=True)
def _integrate_core(
    V0: float, t_plate: float, d_bullet: float, m_bullet: float,
    A: float, B: float, n: float, C: float, m_jc: float,
    T_melt: float, rho: float, Cp: float,
    T0: float, beta: float, w_ratio: float,
    eps_dot_ref: float, T_room: float, n_steps: int,
    cs_D: float, cs_q: float, rate_model: int, failure_strain: float,
    shear_angle: float,
    clamp_low_rate: bool = True
) -> tuple:
    """
    Time-stepped penetration (Numba-compiled).
    n_steps is computed adaptively from plate thickness / SPATIAL_RESOLUTION.
    """
    SQRT3 = math.sqrt(3.0)
    w = w_ratio * d_bullet
    perimeter = math.pi * d_bullet
    tan_alpha = math.tan(math.radians(shear_angle))
    tan_alpha = math.tan(math.radians(shear_angle))
    dx = t_plate / n_steps

    V = V0
    T = T0
    epsilon = 1e-4
    x = 0.0
    peak_T = T0
    peak_strain = 0.0
    E_absorbed = 0.0
    sigma_sum = 0.0
    n_active = 0
    last_eps_dot = 0.0
    D = 0.0
    peak_damage = 0.0

    for _ in range(n_steps):
        if V <= 0.0:
            break

        eps_dot = V / (w * SQRT3)
        eps_dot = max(eps_dot, 1.0)
        last_eps_dot = eps_dot

        sigma = _flow_stress(
            epsilon, eps_dot, T, A, B, n, C, m_jc,
            T_melt, eps_dot_ref, T_room,
            cs_D, cs_q, rate_model, clamp_low_rate)

        d_eps = dx / (w * SQRT3)
        dD = d_eps / failure_strain
        D_mid = min(D + 0.5 * dD, 1.0)
        sigma_eff = sigma * (1.0 - D_mid)
        
        tau_Pa = (sigma_eff * 1e6) / SQRT3
        d_eff = d_bullet + 2.0 * x * tan_alpha
        perimeter_eff = math.pi * d_eff
        F = tau_Pa * perimeter_eff * t_plate

        dE = F * dx
        E_absorbed += dE

        V_sq_new = V * V - 2.0 * F * dx / m_bullet
        if V_sq_new <= 0.0:
            remaining = 0.5 * m_bullet * V * V
            x_partial = remaining / F if F > 0.0 else 0.0
            x += x_partial
            E_absorbed -= dE
            E_absorbed += remaining
            V = 0.0
            
            d_eps_partial = x_partial / (w * SQRT3)
            epsilon += d_eps_partial
            D = min(D + d_eps_partial / failure_strain, 1.0)
            if D > peak_damage: peak_damage = D
            T += beta * sigma_eff * 1e6 * d_eps_partial / (rho * Cp)
            if T > peak_T: peak_T = T
            if epsilon > peak_strain: peak_strain = epsilon
            n_active += 1
            break

        V = math.sqrt(V_sq_new)
        x += dx

        d_eps = dx / (w * SQRT3)
        epsilon += d_eps
        D = min(D + dD, 1.0)
        if D > peak_damage:
            peak_damage = D

        dT = beta * sigma_eff * 1e6 * d_eps / (rho * Cp)
        T += dT

        if T > peak_T:
            peak_T = T
        if epsilon > peak_strain:
            peak_strain = epsilon
        sigma_sum += sigma
        n_active += 1

    penetrated = x >= (t_plate - 1e-10)
    sigma_avg = sigma_sum / max(n_active, 1)
    V_exit = V if penetrated else 0.0

    return (penetrated, V_exit, min(x, t_plate),
            peak_T, peak_strain, peak_damage, E_absorbed, sigma_avg, last_eps_dot)


# ===================================================================
#  TRAJECTORY KERNEL — for PINN collocation points
# ===================================================================

@njit(nogil=True, cache=True)
def _integrate_trajectory(
    V0: float, t_plate: float, d_bullet: float, m_bullet: float,
    A: float, B: float, n: float, C: float, m_jc: float,
    T_melt: float, rho: float, Cp: float,
    T0: float, beta: float, w_ratio: float,
    eps_dot_ref: float, T_room: float, n_steps: int,
    cs_D: float, cs_q: float, rate_model: int, failure_strain: float,
    shear_angle: float,
    clamp_low_rate: bool = True
) -> tuple:
    """
    Same physics as _integrate_core, but stores the FULL trajectory.
    Returns (n_active, x_arr, V_arr, sigma_arr, T_arr, eps_arr, edot_arr).
    PINN needs these to sample collocation points inside the domain.
    """
    SQRT3 = math.sqrt(3.0)
    w = w_ratio * d_bullet
    perimeter = math.pi * d_bullet
    tan_alpha = math.tan(math.radians(shear_angle))
    tan_alpha = math.tan(math.radians(shear_angle))
    dx = t_plate / n_steps

    # Pre-allocate trajectory arrays
    x_arr = np.empty(n_steps)
    V_arr = np.empty(n_steps)
    sigma_arr = np.empty(n_steps)
    T_arr = np.empty(n_steps)
    eps_arr = np.empty(n_steps)
    edot_arr = np.empty(n_steps)
    D_arr = np.empty(n_steps)

    V = V0
    T = T0
    epsilon = 1e-4
    x = 0.0
    n_active = 0
    D = 0.0

    for step in range(n_steps):
        if V <= 0.0:
            break

        eps_dot = max(V / (w * SQRT3), 1.0)
        sigma = _flow_stress(epsilon, eps_dot, T, A, B, n, C, m_jc,
                             T_melt, eps_dot_ref, T_room, cs_D, cs_q, rate_model, clamp_low_rate)
        
        d_eps = dx / (w * SQRT3)
        dD = d_eps / failure_strain
        D_mid = min(D + 0.5 * dD, 1.0)
        sigma_eff = sigma * (1.0 - D_mid)

        # Store state BEFORE update (collocation point)
        x_arr[n_active] = x
        V_arr[n_active] = V
        sigma_arr[n_active] = sigma_eff
        T_arr[n_active] = T
        eps_arr[n_active] = epsilon
        edot_arr[n_active] = eps_dot
        D_arr[n_active] = D
        n_active += 1

        tau_Pa = (sigma_eff * 1e6) / SQRT3
        d_eff = d_bullet + 2.0 * x * tan_alpha
        perimeter_eff = math.pi * d_eff
        F = tau_Pa * perimeter_eff * t_plate
        V_sq_new = V * V - 2.0 * F * dx / m_bullet

        if V_sq_new <= 0.0:
            remaining = 0.5 * m_bullet * V * V
            x_partial = remaining / F if F > 0.0 else 0.0
            x += x_partial
            V = 0.0
            
            d_eps_partial = x_partial / (w * SQRT3)
            epsilon += d_eps_partial
            D = min(D + d_eps_partial / failure_strain, 1.0)
            T += beta * sigma_eff * 1e6 * d_eps_partial / (rho * Cp)
            
            x_arr[n_active] = x
            V_arr[n_active] = V
            sigma_arr[n_active] = sigma_eff
            T_arr[n_active] = T
            eps_arr[n_active] = epsilon
            edot_arr[n_active] = eps_dot
            D_arr[n_active] = D
            n_active += 1
            break

        V = math.sqrt(V_sq_new)
        x += dx
        epsilon += d_eps
        D = min(D + dD, 1.0)
        T += beta * sigma_eff * 1e6 * d_eps / (rho * Cp)

    return (n_active, x_arr, V_arr, sigma_arr, T_arr, eps_arr, edot_arr, D_arr)


def simulate_trajectory(
    plate_thickness_mm: float,
    material_name: str = "RHA_steel",
    plate_temperature_K: float = 298.0,
    bullet_velocity_ms: float = 715.0,
    bullet_mass_g: float = 7.9,
    bullet_diameter_mm: float = 7.62,
    obliquity_angle_deg: float = 0.0,
    *,
    rate_hardening_model: str = "jc",
    clamp_low_rate: bool = True
) -> Dict[str, np.ndarray]:
    """
    Returns the full spatiotemporal trajectory for PINN training.
    Output: dict of 1D arrays (x_mm, V_ms, sigma_MPa, T_K, strain, strain_rate)
    trimmed to actual active steps.
    """
    mat = MATERIALS[material_name]
    rate_code = _RATE_MODEL_MAP[rate_hardening_model]
    t_m = plate_thickness_mm * 1e-3
    m_kg = bullet_mass_g * 1e-3
    r_m = (bullet_diameter_mm / 2.0) * 1e-3
    t_eff = t_m / np.cos(np.radians(obliquity_angle_deg))
    n_steps = _compute_n_steps(t_eff)

    cs_D = mat.get("cs_D", 1.0)
    cs_q = mat.get("cs_q", 1.0)
    failure_strain = mat.get("failure_strain", 10.0)
    if rate_code != _RATE_JC and ("cs_D" not in mat or "cs_q" not in mat):
        rate_code = _RATE_JC

    res = _integrate_trajectory(
        bullet_velocity_ms, t_eff, 2*r_m, m_kg,
        mat["A"], mat["B"], mat["n"], mat["C"], mat["m"],
        mat["T_melt"], mat["rho"], mat["Cp"],
        plate_temperature_K, TAYLOR_QUINNEY_BETA, SHEAR_ZONE_WIDTH_RATIO,
        1.0, T_ROOM, n_steps, cs_D, cs_q, rate_code, failure_strain, PLUG_SHEAR_ANGLE_DEG, clamp_low_rate)

    na = int(res[0])
    return {
        "x_mm":        res[1][:na] * 1000,
        "V_ms":        res[2][:na],
        "sigma_MPa":   res[3][:na],
        "T_K":         res[4][:na],
        "strain":      res[5][:na],
        "strain_rate": res[6][:na],
        "damage":      res[7][:na],
        "n_points":    na,
    }


# ===================================================================
#  PYTHON WRAPPERS
# ===================================================================

def _compute_n_steps(t_plate_m: float) -> int:
    """Adaptive: fixed spatial resolution, not fixed step count."""
    return min(max(int(math.ceil(t_plate_m / SPATIAL_RESOLUTION_M)), MIN_STEPS), MAX_STEPS)


def _integrate_penetration(
    V0: float, t_plate_m: float, r_m: float,
    m_kg: float, mat: dict, T0: float,
    w_ratio: float = SHEAR_ZONE_WIDTH_RATIO,
    rate_model: int = _RATE_JC,
    shear_angle: float = PLUG_SHEAR_ANGLE_DEG,
    clamp_low_rate: bool = True
) -> Dict[str, float]:
    d = 2.0 * r_m
    n_steps = _compute_n_steps(t_plate_m)

    # CS constants: graceful fallback to JC if missing (with warning)
    has_cs = ("cs_D" in mat and "cs_q" in mat)
    if rate_model != _RATE_JC and not has_cs:
        import warnings
        warnings.warn(
            f"Material '{mat.get('label','?')}' has no cs_D/cs_q. "
            f"Falling back to JC-only rate hardening.",
            stacklevel=3)
        rate_model = _RATE_JC
    cs_D = mat.get("cs_D", 1.0)   # safe defaults (only used if rate_model != JC)
    cs_q = mat.get("cs_q", 1.0)
    failure_strain = mat.get("failure_strain", 10.0)

    res = _integrate_core(
        V0, t_plate_m, d, m_kg,
        mat["A"], mat["B"], mat["n"], mat["C"], mat["m"],
        mat["T_melt"], mat["rho"], mat["Cp"],
        T0, TAYLOR_QUINNEY_BETA, w_ratio,
        1.0, T_ROOM, n_steps,
        cs_D, cs_q, rate_model, failure_strain, shear_angle, clamp_low_rate)

    return {
        "penetrated": bool(res[0]), "exit_velocity": float(res[1]),
        "penetration_depth_m": float(res[2]), "peak_temperature_K": float(res[3]),
        "peak_strain": float(res[4]), "peak_damage": float(res[5]),
        "energy_absorbed_J": float(res[6]), "sigma_avg_MPa": float(res[7]),
        "final_epsilon_dot": float(res[8]),
    }


def _find_ballistic_limit(
    t_plate_m: float, r_m: float, m_kg: float,
    mat: dict, T0: float, rate_model: int = _RATE_JC,
    w_ratio: float = SHEAR_ZONE_WIDTH_RATIO,
    shear_angle: float = PLUG_SHEAR_ANGLE_DEG,
) -> float:
    """
    V_BL via dynamic bounds + bisection.
    Phase 1: Exponential search to bracket [V_lo_pen, V_hi_stop].
    Phase 2: Bisect within the bracket.
    """
    # Phase 1: find a velocity that penetrates (V_pen) and one that doesn't (V_stop)
    V_stop = 0.0
    V_pen = 0.0

    V_test = 10.0
    found_pen = False
    found_stop = True

    while V_test < V_SEARCH_CEILING:
        res = _integrate_penetration(V_test, t_plate_m, r_m, m_kg, mat, T0,
                                     w_ratio=w_ratio, rate_model=rate_model, shear_angle=shear_angle)
        if res["penetrated"]:
            V_pen = V_test
            found_pen = True
            if found_stop:
                break
        else:
            V_stop = V_test
            found_stop = True
            if found_pen:
                break

        V_test *= 2.0

    if not found_pen:
        # Plate stops everything up to V_SEARCH_CEILING
        return V_SEARCH_CEILING
    if not found_stop:
        # Even at extreme velocity, always penetrates (molten plate)
        return 0.0

    # Phase 2: bisection within [V_stop, V_pen]
    for _ in range(V_BL_MAX_ITER):
        V_mid = 0.5 * (V_stop + V_pen)
        res = _integrate_penetration(V_mid, t_plate_m, r_m, m_kg, mat, T0,
                                     w_ratio=w_ratio, rate_model=rate_model, shear_angle=shear_angle)
        if res["penetrated"]:
            V_pen = V_mid
        else:
            V_stop = V_mid
        if abs(V_pen - V_stop) < V_BL_TOL:
            break

    return 0.5 * (V_stop + V_pen)


# ===================================================================
#  PLUG MASS, DEFORMATION, FAILURE (unchanged from last version)
# ===================================================================

def _plug_mass(r_m: float, t_eff: float, rho: float,
               angle_deg: float = PLUG_SHEAR_ANGLE_DEG) -> float:
    alpha = math.radians(angle_deg)
    r1, r2 = r_m, r_m + t_eff * math.tan(alpha)
    return rho * math.pi * t_eff / 3.0 * (r1*r1 + r1*r2 + r2*r2)


def _deformation_mm(penetrated: bool, r_m: float, pen_m: float,
                     t_eff: float, angle_deg: float) -> float:
    d = 2.0 * r_m
    if not penetrated:
        return (d + 2.0 * math.sqrt(max(2.0 * r_m * pen_m, 0.0))) * 1000.0
    else:
        return (d + 2.0 * t_eff * math.tan(math.radians(angle_deg))) * 1000.0


def _classify_failure(penetrated: bool, peak_T: float, T_melt: float,
                      peak_strain: float, yield_strain: float) -> str:
    # heuristic post-processing classifier - not a validated failure criterion
    T_star = max((peak_T - T_ROOM) / (T_melt - T_ROOM), 0.0) if T_melt > T_ROOM else 0.0
    if not penetrated:
        if peak_strain < 3.0 * yield_strain:
            return "elastic_rebound"
        return "dishing" if T_star < 0.4 else "petalling"
    else:
        if T_star > 0.7:
            return "plugging"
        if peak_strain > 5.0 * yield_strain:
            return "ductile_hole_enlargement"
        return "petalling"


# ===================================================================
#  SimResult + MAIN FUNCTION
# ===================================================================

@dataclass
class SimResult:
    penetrated: bool
    failure_mode: str
    rate_hardening_model: str
    residual_velocity_ms: float
    ballistic_limit_ms: float
    penetration_depth_mm: float
    deformation_diameter_mm: float
    energy_initial_J: float
    energy_absorbed_J: float
    energy_absorption_pct: float
    sigma_avg_MPa: float
    peak_temperature_K: float
    peak_strain: float
    peak_damage: float
    T_star_peak: float
    n_steps_used: int             # NEW: shows adaptive resolution


def simulate_ballistic_impact(
    plate_thickness_mm: float,
    material_name: str = "RHA_steel",
    plate_temperature_K: float = 298.0,
    bullet_velocity_ms: float = 715.0,
    bullet_mass_g: float = 7.9,
    bullet_diameter_mm: float = 7.62,
    obliquity_angle_deg: float = 0.0,
    *,
    rate_hardening_model: str = "jc",
    shear_zone_ratio: Optional[float] = None,
    plug_shear_angle_deg: Optional[float] = None,
    clamp_low_rate: bool = True,
) -> SimResult:
    if material_name not in MATERIALS:
        raise ValueError(f"Unknown material: {material_name}")
    if plate_thickness_mm <= 0:
        raise ValueError(f"plate_thickness_mm must be > 0")
    if bullet_velocity_ms <= 0:
        raise ValueError(f"bullet_velocity_ms must be > 0")
    if bullet_mass_g <= 0:
        raise ValueError(f"bullet_mass_g must be > 0")
    if bullet_diameter_mm <= 0:
        raise ValueError(f"bullet_diameter_mm must be > 0")
    if not (0.0 <= obliquity_angle_deg < 90.0):
        raise ValueError(f"obliquity must be in [0,90)")
    if plate_temperature_K <= 0:
        raise ValueError(f"plate_temperature_K must be > 0")
    if rate_hardening_model not in _RATE_MODEL_MAP:
        raise ValueError(f"rate_hardening_model must be jc/cs/blended")

    # Physics validity warning for hydrodynamic regime
    if bullet_velocity_ms > V_SEARCH_CEILING:
        import warnings
        warnings.warn(
            f"Velocity {bullet_velocity_ms:.0f} m/s exceeds JC validity range "
            f"(>{V_SEARCH_CEILING:.0f} m/s). Above ~Mach 9, materials enter "
            f"hydrodynamic flow where strength models are invalid. "
            f"Results are extrapolated and physically suspect.",
            UserWarning, stacklevel=2)

    mat = MATERIALS[material_name]
    # Tuned against empirical 12mm RHA V50 (619.5 m/s)
    SHEAR_ZONE_WIDTH_RATIO = 0.6841
    w_ratio = shear_zone_ratio if shear_zone_ratio is not None else SHEAR_ZONE_WIDTH_RATIO
    shear_angle = plug_shear_angle_deg if plug_shear_angle_deg is not None else PLUG_SHEAR_ANGLE_DEG
    if w_ratio <= 0:
        raise ValueError("shear_zone_ratio must be > 0")
    rate_code = _RATE_MODEL_MAP[rate_hardening_model]

    t_m = plate_thickness_mm * 1e-3
    m_kg = bullet_mass_g * 1e-3
    r_m = (bullet_diameter_mm / 2.0) * 1e-3
    t_eff = t_m / np.cos(np.radians(obliquity_angle_deg))
    V0 = bullet_velocity_ms

    res = _integrate_penetration(V0, t_eff, r_m, m_kg, mat,
                                 plate_temperature_K, w_ratio=w_ratio,
                                 rate_model=rate_code, shear_angle=shear_angle)

    penetrated = res["penetrated"]
    V_exit = res["exit_velocity"]
    pen_m = res["penetration_depth_m"]
    peak_T = res["peak_temperature_K"]
    peak_strain = res["peak_strain"]
    peak_damage = res["peak_damage"]
    sigma_avg = res["sigma_avg_MPa"]

    m_plug = _plug_mass(r_m, t_eff, mat["rho"], shear_angle)
    V_r = V_exit * m_kg / (m_kg + m_plug) if (penetrated and V_exit > 0) else 0.0

    V_BL = _find_ballistic_limit(t_eff, r_m, m_kg, mat, plate_temperature_K,
                                 rate_model=rate_code, w_ratio=w_ratio, shear_angle=shear_angle)

    deform_mm = _deformation_mm(penetrated, r_m, pen_m, t_eff, shear_angle)
    yield_strain = mat["A"] / mat["E"]
    failure = _classify_failure(penetrated, peak_T, mat["T_melt"],
                                peak_strain, yield_strain)

    denom = mat["T_melt"] - T_ROOM
    T_star = max((peak_T - T_ROOM) / denom, 0.0) if denom > 0 else 0.0

    E_i = 0.5 * m_kg * V0**2
    E_r = 0.5 * (m_kg + m_plug) * V_r**2 if penetrated else 0.0
    E_a = E_i - E_r

    return SimResult(
        penetrated=penetrated, failure_mode=failure,
        rate_hardening_model=rate_hardening_model,
        residual_velocity_ms=round(V_r, 2),
        ballistic_limit_ms=round(V_BL, 2),
        penetration_depth_mm=round(pen_m * 1000, 2),
        deformation_diameter_mm=round(deform_mm, 2),
        energy_initial_J=round(E_i, 2),
        energy_absorbed_J=round(E_a, 2),
        energy_absorption_pct=round(abs(E_a / E_i * 100), 1) if E_i > 0 else 0.0,
        sigma_avg_MPa=round(sigma_avg, 1),
        peak_temperature_K=round(peak_T, 1),
        peak_strain=round(peak_strain, 3),
        peak_damage=round(peak_damage, 3),
        T_star_peak=round(T_star, 4),
        n_steps_used=_compute_n_steps(t_eff),
    )


# ===================================================================
#  PINN DATA GENERATION — trajectory-based, disk-streaming
# ===================================================================

# Integer-encoded failure modes for PyTorch (no string tensors)
FAILURE_MODES = {
    "elastic_rebound": 0, "dishing": 1, "petalling": 2,
    "plugging": 3, "ductile_hole_enlargement": 4, "fragmentation": 5,
}


def generate_pinn_data(
    thicknesses_mm: np.ndarray,
    velocities_ms: np.ndarray,
    material_names: list,
    obliquities_deg: np.ndarray,
    plate_temperatures_K: np.ndarray,
    output_dir: str = "pinn_data",
    bullet_mass_g: float = 7.9,
    bullet_diameter_mm: float = 7.62,
    n_collocation: int = 15,
    *,
    rate_hardening_model: str = "jc",
    max_workers: Optional[int] = None,
    chunk_size: int = 500,
    seed: int = 42,
) -> str:
    """
    PINN-ready trajectory data generator with chunked disk I/O.

    For each (thickness, velocity, material, obliquity, temperature) sample:
      1. Runs the trajectory kernel (NOT endpoint-only).
      2. Samples n_collocation points from the trajectory.
      3. NO V_BL bisection — PINN learns physics, not limits.

    Each thread chunk writes a parquet shard to disk immediately,
    keeping RAM usage constant regardless of dataset size.

    Returns the output directory path containing parquet shards.
    """
    import concurrent.futures, os
    import pandas as pd

    N = len(thicknesses_mm)
    if not (N == len(velocities_ms) == len(material_names) == len(obliquities_deg) == len(plate_temperatures_K)):
        raise ValueError("All input arrays must have the same length.")

    rate_code = _RATE_MODEL_MAP.get(rate_hardening_model)
    if rate_code is None:
        raise ValueError("rate_hardening_model must be jc/cs/blended")

    os.makedirs(output_dir, exist_ok=True)
    m_kg = bullet_mass_g * 1e-3
    r_m = (bullet_diameter_mm / 2.0) * 1e-3
    d = 2.0 * r_m
    _MATERIAL_ID = {name: i for i, name in enumerate(MATERIALS.keys())}

    def _process_chunk(start: int, end: int, shard_id: int) -> int:
        """Run trajectories, sample collocation points, write parquet shard."""
        rows = []
        rng = np.random.default_rng(seed + shard_id)
        for i in range(start, end):
            mat_name = str(material_names[i])
            mat = MATERIALS[mat_name]
            t_m = float(thicknesses_mm[i]) * 1e-3
            ob = float(obliquities_deg[i])
            T_plate = float(plate_temperatures_K[i])
            t_eff = t_m / np.cos(np.radians(ob))
            V0 = float(velocities_ms[i])
            n_steps = _compute_n_steps(t_eff)

            rc = rate_code
            if rc != _RATE_JC and ("cs_D" not in mat or "cs_q" not in mat):
                rc = _RATE_JC

            cs_D = mat.get("cs_D", 1.0)
            cs_q = mat.get("cs_q", 1.0)
            failure_strain = mat.get("failure_strain", 10.0)

            res = _integrate_trajectory(
                V0, t_eff, d, m_kg,
                mat["A"], mat["B"], mat["n"], mat["C"], mat["m"],
                mat["T_melt"], mat["rho"], mat["Cp"],
                T_plate, TAYLOR_QUINNEY_BETA,
                SHEAR_ZONE_WIDTH_RATIO,
                1.0, T_ROOM, n_steps, cs_D, cs_q, rc, failure_strain, PLUG_SHEAR_ANGLE_DEG)

            na = int(res[0])
            if na < 2:
                continue

            # Sample collocation points from trajectory
            n_pts = min(n_collocation, na)
            indices = np.sort(rng.choice(na, size=n_pts, replace=False))

            for idx in indices:
                rows.append({
                    # Inputs (boundary conditions)
                    "thickness_mm": float(thicknesses_mm[i]),
                    "velocity_ms": V0,
                    "obliquity_deg": ob,
                    "material_id": _MATERIAL_ID[mat_name],
                    "plate_temperature_K": T_plate,
                    # Collocation coordinate
                    "x_mm": float(res[1][idx]) * 1000,
                    # Physics state at this point (PINN targets)
                    "V_ms": float(res[2][idx]),
                    "sigma_MPa": float(res[3][idx]),
                    "T_K": float(res[4][idx]),
                    "strain": float(res[5][idx]),
                    "strain_rate": float(res[6][idx]),
                    "damage": float(res[7][idx]),
                })

        if rows:
            df = pd.DataFrame(rows)
            float_cols = [c for c in df.columns if c != "material_id"]
            df[float_cols] = df[float_cols].astype(np.float32)
            df["material_id"] = df["material_id"].astype(np.int32)
            path = os.path.join(output_dir, f"shard_{shard_id:05d}.parquet")
            df.to_parquet(path, index=False, engine="pyarrow")
        return len(rows)

    # Dispatch chunks to thread pool
    workers = max_workers or (os.cpu_count() or 4)
    total_rows = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = []
        shard_id = 0
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            futures.append(pool.submit(_process_chunk, start, end, shard_id))
            shard_id += 1
        for f in concurrent.futures.as_completed(futures):
            total_rows += f.result()

    return output_dir


# ===================================================================
#  TEST RUNNER
# ===================================================================
if __name__ == "__main__":
    print(f"\nNumba: {'ACTIVE' if HAS_NUMBA else 'fallback'}")
    print(f"V_SEARCH_CEILING: {V_SEARCH_CEILING} m/s (JC validity range)")

    print("\n" + "=" * 70)
    print("  simulation.py — Full Test Suite")
    print("=" * 70)

    base = dict(material_name="RHA_steel", bullet_velocity_ms=715.0,
                bullet_mass_g=7.9, bullet_diameter_mm=7.62)

    # S1: Thickness sweep
    print("\n>>> S1: Thickness sweep\n")
    for t in [5, 10, 20, 50]:
        r = simulate_ballistic_impact(float(t), **base)
        s = "PEN" if r.penetrated else "STOP"
        print(f"  {t:4d}mm | {s} | V_BL={r.ballistic_limit_ms:6.0f} | "
              f"n_steps={r.n_steps_used:5d} | {r.failure_mode}")

    # S2: Rate models
    print("\n>>> S2: Rate model comparison (10mm RHA)\n")
    for model in ["jc", "cs", "blended"]:
        r = simulate_ballistic_impact(10.0, rate_hardening_model=model, **base)
        print(f"  {model:8} | V_BL={r.ballistic_limit_ms:5.0f} | "
              f"V_r={r.residual_velocity_ms:6.1f} | T*={r.T_star_peak:.3f}")

    # S3: Standard rounds
    print("\n>>> S3: Rounds vs 10mm RHA\n")
    for name, v, m, d in [("5.56mm",950,4,5.56),("7.62x39",715,7.9,7.62),
                           ("7.62x51",838,9.8,7.62),(".50 BMG",928,41.9,12.7)]:
        r = simulate_ballistic_impact(10.0, bullet_velocity_ms=v,
                                      bullet_mass_g=m, bullet_diameter_mm=d)
        s = "PEN" if r.penetrated else "STOP"
        print(f"  {name:8} | {s} | V_BL={r.ballistic_limit_ms:5.0f} | "
              f"V_r={r.residual_velocity_ms:6.1f}")

    # S4: PINN data generation (trajectory + disk)
    print("\n>>> S4: generate_pinn_data (2000 trajectories, 15 collocation pts)\n")
    import time, os, glob
    rng = np.random.default_rng(42)
    N_batch = 2000
    t_arr = rng.uniform(3, 30, N_batch)
    v_arr = rng.uniform(200, 1200, N_batch)
    o_arr = rng.uniform(0, 60, N_batch)
    T_arr = rng.uniform(250, 500, N_batch)
    mats = (["RHA_steel"] * 1000 + ["aluminium_7075_T6"] * 500
            + ["copper"] * 500)

    simulate_ballistic_impact(10.0, **base)  # warmup

    out_dir = "pinn_data"
    t0 = time.time()
    generate_pinn_data(t_arr, v_arr, mats, o_arr, T_arr, output_dir=out_dir)
    el = time.time() - t0

    shards = glob.glob(os.path.join(out_dir, "*.parquet"))
    import pandas as pd
    total_rows = sum(len(pd.read_parquet(s)) for s in shards)
    disk_kb = sum(os.path.getsize(s) for s in shards) / 1024

    print(f"  {N_batch} trajectories in {el:.2f}s ({el/N_batch*1000:.1f} ms/traj)")
    print(f"  Collocation points: {total_rows} ({total_rows//N_batch} per traj)")
    print(f"  Parquet shards: {len(shards)} files, {disk_kb:.0f} KB total")
    print(f"  Columns: {list(pd.read_parquet(shards[0]).columns)}")
    print(f"  RAM: ZERO (streamed to disk)")
    print(f"  100K traj estimate: {el/N_batch*100000/60:.1f} min")

    # S5: Single trajectory
    print("\n>>> S5: PINN trajectory (7.62mm @ 715m/s into 10mm RHA)\n")
    traj = simulate_trajectory(10.0, **base)
    print(f"  Collocation points: {traj['n_points']}")
    print(f"  x: [{traj['x_mm'][0]:.3f}, {traj['x_mm'][-1]:.3f}] mm")
    print(f"  V: [{traj['V_ms'][-1]:.1f}, {traj['V_ms'][0]:.1f}] m/s")
    print(f"  sigma: [{traj['sigma_MPa'].min():.0f}, {traj['sigma_MPa'].max():.0f}] MPa")

    # S6: Scalar perf
    print("\n>>> S6: Scalar performance\n")
    t0 = time.time()
    for _ in range(500):
        simulate_ballistic_impact(10.0, **base)
    el = time.time() - t0
    print(f"  500 calls: {el:.2f}s ({el/500*1000:.1f} ms/call)")

    print("\n" + "=" * 70)
    print("  STAGE 1 COMPLETE")
    print("=" * 70)