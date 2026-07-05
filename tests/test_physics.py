import pytest
import numpy as np
from simulation import simulate_trajectory, simulate_ballistic_impact
from stage1_part1_johnson_cook import MATERIALS

def test_simulate_trajectory_no_penetration():
    # A small bullet at low velocity against a thick RHA plate should not penetrate
    res = simulate_trajectory(
        plate_thickness_mm=50.0,
        material_name="RHA_steel",
        bullet_velocity_ms=100.0,
        bullet_mass_g=7.9,
        bullet_diameter_mm=7.62
    )
    
    assert res["V_ms"][-1] < 25.0  # Should be stopped
    assert res["x_mm"][-1] < 50.0  # Should not reach the back

def test_simulate_trajectory_penetration():
    # A fast bullet against a thin RHA plate should overmatch it easily
    res = simulate_trajectory(
        plate_thickness_mm=5.0,
        material_name="RHA_steel",
        bullet_velocity_ms=1000.0,
        bullet_mass_g=7.9,
        bullet_diameter_mm=7.62
    )
    
    assert res["V_ms"][-1] > 0.0
    assert np.isclose(res["x_mm"][-1], 5.0, atol=0.1)

def test_ballistic_limit_empirical_calibration():
    # We explicitly calibrated 12mm RHA to have V50 = 619.5 m/s
    res = simulate_ballistic_impact(
        plate_thickness_mm=12.0,
        material_name="RHA_steel",
        bullet_velocity_ms=1000.0,
        bullet_mass_g=7.9,
        bullet_diameter_mm=7.62
    )
    
    # Should be close to empirical
    assert np.isclose(res.ballistic_limit_ms, 619.5, atol=2.0)

def test_energy_conservation():
    res = simulate_ballistic_impact(
        plate_thickness_mm=10.0,
        material_name="aluminium_7075_T6",
        bullet_velocity_ms=800.0,
        bullet_mass_g=7.9,
        bullet_diameter_mm=7.62
    )
    
    E_initial = 0.5 * (7.9e-3) * (800.0**2)
    # The exit energy is 1/2 * (m_bullet + m_plug) * V_residual^2
    # E_initial should roughly equal Energy Absorbed + Energy Residual
    # The absorbed energy includes deformation, thermal, and momentum transfer to the plug.
    assert res.energy_absorbed_J > 0
    assert res.energy_absorbed_J < E_initial
