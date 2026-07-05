import pytest
import numpy as np
import os
from pinn_dataloader import _log_safe, _normalize, _denormalize, BallisticPINNDataset

def test_log_safe():
    # log10(max(x, 1.0))
    assert np.isclose(_log_safe(np.array([100.0]))[0], 2.0)
    assert np.isclose(_log_safe(np.array([10.0]))[0], 1.0)
    assert np.isclose(_log_safe(np.array([0.5]))[0], 0.0) # Clipped to 1.0

def test_normalize():
    # Min-max scaling
    arr = np.array([0.0, 50.0, 100.0])
    norm = _normalize(arr, 0.0, 100.0)
    assert np.isclose(norm[0], 0.0)
    assert np.isclose(norm[1], 0.5)
    # The max is slightly less than 1.0 due to 1e-12 epsilon
    assert np.isclose(norm[2], 1.0, atol=1e-5)

def test_denormalize():
    arr = np.array([0.0, 0.5, 1.0])
    denorm = _denormalize(arr, 0.0, 100.0)
    assert np.isclose(denorm[0], 0.0)
    assert np.isclose(denorm[1], 50.0)
    assert np.isclose(denorm[2], 100.0)

def test_dataset_initialization():
    # If the debug sample data exists, we can test the dataset directly
    data_dir = "pinn_data/train"
    if not os.path.exists(data_dir):
        data_dir = "pinn_data/_debug_sample"
        
    if not os.path.exists(data_dir):
        pytest.skip(f"No parquet data found at {data_dir} to test dataloader.")
        
    ds = BallisticPINNDataset(data_dir, normalize=True)
    assert len(ds) > 0
    
    # Check shape
    x, y = ds[0]
    # In PyTorch context it returns tensors, but standard len checks work
    assert len(x) == 6
    assert len(y) == 5
    
    stats = ds.get_stats()
    in_min, in_max = stats["input_range"]
    # Because of material_id (which isn't normalized), the max should be around num_materials
    assert in_max > 1.0 and in_max <= 10.0
    
    tgt_min, tgt_max = stats["target_range"]
    # Targets should be strictly bounded to [0, 1]
    assert tgt_min >= 0.0
    assert tgt_max <= 1.0
