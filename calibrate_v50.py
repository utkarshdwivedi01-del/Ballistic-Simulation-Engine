import numpy as np
from scipy.optimize import minimize_scalar
from simulation import simulate_ballistic_impact, SHEAR_ZONE_WIDTH_RATIO

def empirical_v50(t_mm):
    """Empirical V50 for 7.62mm AP against RHA steel."""
    return 49.75 * t_mm + 22.5

def objective(w_ratio, t_mm, target_v50):
    res = simulate_ballistic_impact(
        plate_thickness_mm=t_mm,
        material_name="RHA_steel",
        plate_temperature_K=298.0,
        bullet_velocity_ms=1000.0,
        bullet_mass_g=7.9,
        bullet_diameter_mm=7.62,
        shear_zone_ratio=w_ratio
    )
    v50 = res.ballistic_limit_ms
    print(f"w_ratio: {w_ratio:.4f} -> V50: {v50:.1f} m/s (Target: {target_v50:.1f})")
    return (v50 - target_v50)**2

if __name__ == "__main__":
    t_test = 12.0
    target = empirical_v50(t_test)
    print(f"Target V50 for {t_test}mm RHA: {target:.1f} m/s")
    
    res = minimize_scalar(objective, bounds=(0.01, 1.0), args=(t_test, target), method='bounded', options={'xatol': 1e-4})
    print("\nCalibration Results:")
    print(f"Optimal shear_zone_ratio: {res.x:.4f}")
    
    current_res = simulate_ballistic_impact(
        plate_thickness_mm=t_test, material_name="RHA_steel", plate_temperature_K=298.0, 
        bullet_velocity_ms=1000.0, bullet_mass_g=7.9, bullet_diameter_mm=7.62, shear_zone_ratio=SHEAR_ZONE_WIDTH_RATIO
    )
    print(f"Current default ({SHEAR_ZONE_WIDTH_RATIO:.4f}) V50: {current_res.ballistic_limit_ms:.1f} m/s")
