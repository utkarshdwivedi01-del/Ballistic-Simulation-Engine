import pytest
import numpy as np

from stage1_part1_johnson_cook import johnson_cook_stress, MATERIALS
from stage1_part2_cowper_symonds import cowper_symonds_dif

def test_materials_loaded():
    assert "RHA_steel" in MATERIALS
    assert "aluminium_7075_T6" in MATERIALS
    
    mat = MATERIALS["RHA_steel"]
    assert mat["T_melt"] == 1783.0
    assert mat["hardness_HRC"] == 50.0

def test_johnson_cook_basic():
    mat = MATERIALS["RHA_steel"]
    # At room temperature and 0 strain, stress should be exactly A
    stress, _ = johnson_cook_stress(
        epsilon=0.0,
        epsilon_dot=1.0,
        T=298.0,
        A=mat["A"], B=mat["B"], n=mat["n"], C=mat["C"], m=mat["m"],
        T_melt=mat["T_melt"], T_room=298.0
    )
    assert np.isclose(stress, mat["A"])

def test_johnson_cook_hardening():
    mat = MATERIALS["RHA_steel"]
    stress_0, _ = johnson_cook_stress(
        epsilon=0.0, epsilon_dot=1.0, T=298.0,
        A=mat["A"], B=mat["B"], n=mat["n"], C=mat["C"], m=mat["m"],
        T_melt=mat["T_melt"]
    )
    stress_1, _ = johnson_cook_stress(
        epsilon=0.5, epsilon_dot=1.0, T=298.0,
        A=mat["A"], B=mat["B"], n=mat["n"], C=mat["C"], m=mat["m"],
        T_melt=mat["T_melt"]
    )
    # Strain hardening should increase stress
    assert stress_1 > stress_0

def test_johnson_cook_thermal_softening():
    mat = MATERIALS["RHA_steel"]
    stress_cold, _ = johnson_cook_stress(
        epsilon=0.2, epsilon_dot=1.0, T=298.0,
        A=mat["A"], B=mat["B"], n=mat["n"], C=mat["C"], m=mat["m"],
        T_melt=mat["T_melt"]
    )
    stress_hot, _ = johnson_cook_stress(
        epsilon=0.2, epsilon_dot=1.0, T=1000.0,
        A=mat["A"], B=mat["B"], n=mat["n"], C=mat["C"], m=mat["m"],
        T_melt=mat["T_melt"]
    )
    # Thermal softening should decrease stress
    assert stress_hot < stress_cold

def test_cowper_symonds():
    # DIF should be 1.0 at quasi-static rates
    dif_static = cowper_symonds_dif(epsilon_dot=0.0, D=40.4, q=5.0)
    assert np.isclose(dif_static, 1.0, atol=1e-3)
    
    # DIF should increase at high dynamic rates
    dif_dynamic = cowper_symonds_dif(epsilon_dot=1e5, D=40.4, q=5.0)
    assert dif_dynamic > 1.5
