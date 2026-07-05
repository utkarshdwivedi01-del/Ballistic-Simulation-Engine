"""
DMSRDE Ballistic Simulation — Part 2: Cowper-Symonds Model

CANONICAL SOURCE for CS DIF. The Numba scalar kernel `_cs_dif_scalar`
is the SINGLE implementation. simulation.py imports it directly.

Run: python stage1_part2_cowper_symonds.py
"""
from __future__ import annotations
import math
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

from stage1_part1_johnson_cook import johnson_cook_stress, MATERIALS, T_ROOM

Numeric = Union[float, NDArray[np.floating]]



# ===========================================================
# CANONICAL CS KERNEL — Numba scalar. SINGLE SOURCE OF TRUTH.
# ===========================================================

@njit(nogil=True, cache=True)
def _cs_dif_scalar(eps_dot: float, D: float, q: float) -> float:
    """CS Dynamic Increase Factor (scalar). Guards eps_dot <= 0."""
    safe_dot = max(eps_dot, 0.0)
    return 1.0 + (safe_dot / D) ** (1.0 / q)

# ===========================================================
# NUMBA BATCH LOOP — replaces np.vectorize (C-speed)
# ===========================================================

@njit(nogil=True, cache=True)
def _cs_dif_batch(edot: np.ndarray, D: float, q: float) -> np.ndarray:
    """Compiled loop over arrays. Each element calls the scalar kernel."""
    N = edot.shape[0]
    out = np.empty(N)
    for i in range(N):
        out[i] = _cs_dif_scalar(edot[i], D, q)
    return out


def cowper_symonds_dif(epsilon_dot: Numeric, D: float, q: float) -> Numeric:
    """Scalar fast-path; array via compiled batch (NOT np.vectorize)."""
    if D <= 0.0 or q <= 0.0:
        raise ValueError("Cowper-Symonds constants D and q must be strictly positive.")
    if isinstance(epsilon_dot, (int, float)):
        return _cs_dif_scalar(float(epsilon_dot), D, q)
    arr = np.asarray(epsilon_dot, dtype=np.float64)
    return _cs_dif_batch(arr.ravel(), D, q).reshape(arr.shape)


def cowper_symonds_stress(
    sigma_static: Numeric, epsilon_dot: Numeric, D: float, q: float,
) -> Tuple[Numeric, Numeric]:
    dif = cowper_symonds_dif(epsilon_dot, D, q)
    return sigma_static * dif, dif

# ===========================================================
# TESTS
# ===========================================================
if __name__ == "__main__":
    print("=" * 65)
    print("  Cowper-Symonds Model — DMSRDE")
    print("=" * 65)

    test_rates = [1e0, 1e2, 1e3, 1e4, 1e5, 1e6]

    print("\n--- DIF vs Strain Rate ---\n")
    for mat_name, mat in MATERIALS.items():
        if "cs_D" not in mat or "cs_q" not in mat:
            continue
        print(f"  {mat['label']}  (D={mat['cs_D']}, q={mat['cs_q']})")
        sigma_s, _ = johnson_cook_stress(
            0.3, 1.0, T_ROOM, mat["A"], mat["B"], mat["n"],
            mat["C"], mat["m"], T_melt=mat["T_melt"])
        for rate in test_rates:
            _, dif = cowper_symonds_stress(sigma_s, rate, mat["cs_D"], mat["cs_q"])
            print(f"    {rate:>10.0e} | DIF={dif:.3f}")
        print()

    # Edge case
    print("--- Edge: eps_dot=0 ---")
    print(f"  DIF(0) = {cowper_symonds_dif(0.0, 40.4, 5.0):.4f}")
    print(f"  DIF([-1,0,1e5]) = {cowper_symonds_dif(np.array([-1,0,1e5]), 40.4, 5.0)}")

    print("\n" + "=" * 65)