"""
DMSRDE Ballistic Simulation — Part 1: Johnson-Cook Model

CANONICAL SOURCE for the JC equation. The Numba scalar kernel
`_jc_sigma_scalar` is the SINGLE implementation. Array processing
uses a compiled Numba batch loop (NOT np.vectorize).

Run: python stage1_part1_johnson_cook.py
"""
from __future__ import annotations
import json, os, math
from typing import Dict, Tuple, Union
import numpy as np
from numpy.typing import NDArray

try:
    from numba import njit
except ImportError:
    def njit(func=None, **kwargs):
        if func is not None: return func
        def _d(f): return f
        return _d

Numeric = Union[float, NDArray[np.floating]]
T_ROOM: float = 298.0


# ===========================================================
# CUSTOM EXCEPTION
# ===========================================================

class MaterialConfigError(Exception):
    """Raised when material configuration is invalid or incomplete."""
    pass


# ===========================================================
# MATERIALS — from JSON, hardcoded fallback, hot-reloadable
# ===========================================================

_CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "materials_config.json")

_FALLBACK: Dict[str, dict] = {
    "RHA_steel": {
        "label": "Rolled Homogeneous Armour Steel",
        "A": 1175.0, "B": 519.0, "n": 0.26, "C": 0.007, "m": 1.0,
        "T_melt": 1783.0, "rho": 7850.0, "E": 207e3, "Cp": 450.0,
        "cs_D": 40.4, "cs_q": 5.0, "hardness_HRC": 50.0,
    },
    "aluminium_7075_T6": {
        "label": "Aluminium Alloy 7075-T6",
        "A": 520.0, "B": 477.0, "n": 0.52, "C": 0.001, "m": 1.61,
        "T_melt": 893.0, "rho": 2810.0, "E": 71e3, "Cp": 960.0,
        "cs_D": 6500.0, "cs_q": 4.0, "hardness_HRC": 18.0,
    },
    "copper": {
        "label": "Copper (bullet jacket)",
        "A": 90.0, "B": 292.0, "n": 0.31, "C": 0.025, "m": 1.09,
        "T_melt": 1356.0, "rho": 8960.0, "E": 120e3, "Cp": 385.0,
        "cs_D": 1e6, "cs_q": 4.0, "hardness_HRC": 8.0,
    },
    "alumina_Al2O3": {
        "label": "Alumina Ceramic (Al₂O₃ — 99.5%)",
        "A": 2100.0, "B": 0.0, "n": 0.01, "C": 0.003, "m": 1.0,
        "T_melt": 2345.0, "rho": 3900.0, "E": 370e3, "Cp": 880.0,
        "cs_D": 10000.0, "cs_q": 4.0, "hardness_HRC": 70.0,
    },
    "UHMWPE": {
        "label": "Ultra-High-Molecular-Weight Polyethylene (Dyneema/Spectra)",
        "A": 20.0, "B": 150.0, "n": 0.65, "C": 0.04, "m": 1.2,
        "T_melt": 420.0, "rho": 970.0, "E": 1.2e3, "Cp": 1900.0,
        "cs_D": 100.0, "cs_q": 3.0, "hardness_HRC": 3.0,
    },
}

_REQUIRED_KEYS = {"A", "B", "n", "C", "m", "T_melt", "rho", "E", "Cp"}


def _load_materials() -> Dict[str, dict]:
    if os.path.exists(_CFG):
        with open(_CFG, "r") as f:
            data = json.load(f)
        # Validate required keys
        valid_data = {}
        for name, props in data.items():
            if name.startswith("_"):
                continue
            missing = _REQUIRED_KEYS - set(props.keys())
            if missing:
                raise MaterialConfigError(
                    f"Material '{name}' missing required keys: {missing}")
            
            if "cs_D" in props and props["cs_D"] <= 0:
                raise ValueError(f"Material '{name}': Cowper-Symonds D must be > 0.")
            if "cs_q" in props and props["cs_q"] <= 0:
                raise ValueError(f"Material '{name}': Cowper-Symonds q must be > 0.")
            if "failure_strain" in props and props["failure_strain"] <= 0:
                raise ValueError(f"Material '{name}': failure_strain must be > 0.")
            if "rho" in props and props["rho"] <= 0:
                raise ValueError(f"Material '{name}': rho must be > 0.")
            if "Cp" in props and props["Cp"] <= 0:
                raise ValueError(f"Material '{name}': Cp must be > 0.")
            if "failure_strain" not in props:
                props["failure_strain"] = 10.0
            if "cs_D" not in props:
                props["cs_D"] = 1.0
            if "cs_q" not in props:
                props["cs_q"] = 1.0
                
            valid_data[name] = props
        return valid_data
    return _FALLBACK.copy()


MATERIALS: Dict[str, dict] = _load_materials()


def reload_materials() -> None:
    """Hot-reload materials from JSON without restarting the runtime."""
    global MATERIALS
    MATERIALS.clear()
    MATERIALS.update(_load_materials())


# ===========================================================
# CANONICAL JC KERNEL — Numba scalar. SINGLE SOURCE OF TRUTH.
# ===========================================================

@njit(nogil=True, cache=True)
def _jc_sigma_scalar(
    eps: float, eps_dot: float, T: float,
    A: float, B: float, n: float, C: float, m: float,
    T_melt: float, eps_dot_ref: float, T_room: float,
    clamp_low_rate: bool = True
) -> tuple:
    """
    Johnson-Cook flow stress (scalar, MPa).
    Returns (sigma, term1, term2, term3).
    """
    eps = max(eps, 0.0)
    term1 = A + B * (eps ** n)

    eps_star = max(eps_dot / eps_dot_ref, 1e-10)
    rate_term = 1.0 + C * math.log(eps_star)
    # Ensure residual strength stays > 0 even for very large C and small eps_star
    term2 = max(rate_term, 1.0) if clamp_low_rate else max(rate_term, 0.01)

    denom = T_melt - T_room
    T_star = min(max((T - T_room) / denom, 0.0), 1.0) if denom > 0.0 else 0.0
    term3 = 1.0 - (T_star ** m)

    return term1 * term2 * term3, term1, term2, term3


# ===========================================================
# NUMBA BATCH LOOP — replaces np.vectorize (C-speed arrays)
# ===========================================================

@njit(nogil=True, cache=True)
def _jc_sigma_batch(
    eps: np.ndarray, edot: np.ndarray, T: np.ndarray,
    A: float, B: float, n: float, C: float, m: float,
    T_melt: float, eps_dot_ref: float, T_room: float,
    clamp_low_rate: bool = True
) -> tuple:
    """Compiled loop over arrays. Each element calls the scalar kernel."""
    N = eps.shape[0]
    sigma = np.empty(N)
    t1 = np.empty(N)
    t2 = np.empty(N)
    t3 = np.empty(N)
    for i in range(N):
        s, a, b, c = _jc_sigma_scalar(
            eps[i], edot[i], T[i],
            A, B, n, C, m, T_melt, eps_dot_ref, T_room, clamp_low_rate)
        sigma[i] = s
        t1[i] = a
        t2[i] = b
        t3[i] = c
    return sigma, t1, t2, t3


# ===========================================================
# PUBLIC API — scalar fast-path, array via compiled batch
# ===========================================================

def johnson_cook_stress(
    epsilon: Numeric, epsilon_dot: Numeric, T: Numeric,
    A: float, B: float, n: float, C: float, m: float,
    T_melt: float,
    epsilon_dot_ref: float = 1.0,
    T_room: float = T_ROOM,
    clamp_low_rate: bool = True
) -> Tuple[Numeric, Dict[str, Numeric]]:
    """
    JC model. T_melt is REQUIRED.
    Scalar inputs → direct kernel call.
    Array inputs → compiled Numba batch loop (NOT np.vectorize).
    """
    if epsilon_dot_ref <= 0.0:
        raise ValueError("epsilon_dot_ref must be strictly positive.")

    # Fast scalar path (most common in simulation hot loop)
    if isinstance(epsilon, (int, float)):
        s, t1, t2, t3 = _jc_sigma_scalar(
            float(epsilon), float(epsilon_dot), float(T),
            A, B, n, C, m, T_melt, epsilon_dot_ref, T_room, clamp_low_rate)
        return s, {"strain_hardening": t1, "rate_hardening": t2,
                   "thermal_softening": t3}

    # Array path: broadcast + compiled batch
    e, ed, Tv = np.broadcast_arrays(
        np.asarray(epsilon, dtype=np.float64),
        np.asarray(epsilon_dot, dtype=np.float64),
        np.asarray(T, dtype=np.float64))
    shape = e.shape
    sigma, t1, t2, t3 = _jc_sigma_batch(
        e.ravel(), ed.ravel(), Tv.ravel(),
        A, B, n, C, m, T_melt, epsilon_dot_ref, T_room, clamp_low_rate)
    return sigma.reshape(shape), {
        "strain_hardening": t1.reshape(shape),
        "rate_hardening": t2.reshape(shape),
        "thermal_softening": t3.reshape(shape),
    }


# ===========================================================
# TESTS
# ===========================================================
if __name__ == "__main__":
    print("=" * 55)
    print("  Johnson-Cook Model — DMSRDE")
    print("=" * 55)

    mat = MATERIALS["RHA_steel"]
    print(f"\nMaterial: {mat['label']}")
    print(f"Source: {'JSON' if os.path.exists(_CFG) else 'fallback'}")

    print(f"\n{'Strain':>8} | {'Stress (MPa)':>14}")
    print("-" * 28)
    for eps in [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]:
        sigma, _ = johnson_cook_stress(
            eps, 1e5, 298.0,
            mat["A"], mat["B"], mat["n"], mat["C"], mat["m"],
            T_melt=mat["T_melt"])
        print(f"{eps:>8.2f} | {sigma:>14.1f}")

    # Array test — compiled batch, NOT np.vectorize
    eps_arr = np.array([0.0, 0.1, 0.5])
    sig_arr, terms = johnson_cook_stress(
        eps_arr, 1e5, 298.0,
        mat["A"], mat["B"], mat["n"], mat["C"], mat["m"],
        T_melt=mat["T_melt"])
    print(f"\nArray test (Numba batch): {sig_arr}")

    # Hot-reload test
    reload_materials()
    print(f"Hot-reload: {len(MATERIALS)} materials loaded")

    print("=" * 55)