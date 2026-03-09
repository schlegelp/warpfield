# Run: pytest tests/test_all.py
# All functions to be tested should start with test_prefix

import os
import warnings
import subprocess

import pytest
import numpy as np
import h5py
import nibabel as nib
import tifffile

import warpfield
from warpfield.utils import load_data

try:
    import cupy as cp

    try:
        _ = cp.cuda.runtime.getDeviceCount()  # Check if any GPU devices are available
        cupy_available = True
    except cp.cuda.runtime.CUDARuntimeError:
        cupy_available = False
        warnings.warn("No CUDA devices available. Skipping cupy backend tests.")
except (ImportError, ModuleNotFoundError):
    cupy_available = False
    warnings.warn("Cupy not installed or not importable. Skipping cupy backend tests.")

try:
    import mlx.core as mx

    mlx_available = True
except (ImportError, ModuleNotFoundError):
    mlx_available = False
    warnings.warn("No MLX detected. Skipping MLX backend tests.")


def test_trivial():
    assert True == True


def test_trivial2():
    assert False == False


def test_warpmap_invert_fast_zero_field():
    """invert_fast should return (near-)zero inverse for a zero displacement field."""
    ref_shape = (5, 6, 7)
    mov_shape = (5, 6, 7)
    block_size = np.array([1, 1, 1], dtype=np.float32)
    block_stride = np.array([1, 1, 1], dtype=np.float32)
    warp_field = np.zeros((3, *ref_shape), dtype=np.float32)

    wm = warpfield.register.WarpMap(
        warp_field=warp_field,
        block_size=block_size,
        block_stride=block_stride,
        ref_shape=ref_shape,
        mov_shape=mov_shape,
        backend="auto",
    )

    inv = wm.invert_fast(sigma=0.5, truncate=4)
    inv_np = np.array(inv.warp_field)

    expected_shape = tuple(np.ceil(np.array(mov_shape) / block_stride + 1).astype(int))
    assert inv_np.shape == (3, *expected_shape)
    np.testing.assert_allclose(inv_np, 0.0, atol=1e-6)


def test_warpmap_push_coordinates_constant_shift():
    """push_coordinates should add sampled displacement vectors to input voxel coordinates."""
    shape = (6, 6, 6)
    shift = np.array([1.5, -2.0, 0.25], dtype=np.float32)
    warp_field = np.zeros((3, *shape), dtype=np.float32)
    warp_field[0] = shift[0]
    warp_field[1] = shift[1]
    warp_field[2] = shift[2]

    wm = warpfield.register.WarpMap(
        warp_field=warp_field,
        block_size=np.array([1, 1, 1], dtype=np.float32),
        block_stride=np.array([1, 1, 1], dtype=np.float32),
        ref_shape=shape,
        mov_shape=shape,
        backend="auto",
    )

    coords = np.array([[1.0, 2.0, 3.5], [1.0, 4.0, 2.0], [2.0, 1.5, 4.0]], dtype=np.float32)

    pushed = wm.push_coordinates(coords)
    pushed_negative = wm.push_coordinates(coords, negative_shifts=True)

    np.testing.assert_allclose(pushed, coords + shift[:, None], atol=1e-5)
    np.testing.assert_allclose(pushed_negative, coords - shift[:, None], atol=1e-5)


def test_warpmap_jacobian_det_linear_field():
    """jacobian_det should match the analytic determinant for a linear displacement field."""
    shape = (7, 7, 7)
    z, y, x = np.indices(shape, dtype=np.float32)
    alpha, beta, gamma = 0.2, -0.1, 0.05
    warp_field = np.stack([alpha * z, beta * y, gamma * x]).astype(np.float32)

    wm = warpfield.register.WarpMap(
        warp_field=warp_field,
        block_size=np.array([1, 1, 1], dtype=np.float32),
        block_stride=np.array([1, 1, 1], dtype=np.float32),
        ref_shape=shape,
        mov_shape=shape,
        backend="auto",
    )

    det_j = np.array(wm.jacobian_det(units_per_voxel=[1, 1, 1], edge_order=1))
    expected = (1.0 + alpha) * (1.0 + beta) * (1.0 + gamma)

    np.testing.assert_allclose(det_j[1:-1, 1:-1, 1:-1], expected, atol=2e-3, rtol=2e-3)


def test_import_npy(tmp_path):
    """Test importing a 3D .npy file."""
    file_path = tmp_path / "test.npy"
    data = np.random.rand(10, 10, 10).astype("float32")
    np.save(file_path, data)

    loaded_data, meta = load_data(str(file_path))
    assert np.allclose(loaded_data, data), "Loaded .npy data does not match expected data."


def test_import_h5(tmp_path):
    """Test importing a 3D .h5 file."""
    file_path = tmp_path / "test.h5"
    data = np.random.rand(10, 10, 10).astype("float32")
    with h5py.File(file_path, "w") as f:
        f.create_dataset("dataset", data=data)

    loaded_data, meta = load_data(f"{file_path}:dataset")
    assert np.allclose(loaded_data, data), "Loaded .h5 data does not match expected data."


def test_import_nii(tmp_path):
    """Test importing a 3D .nii file."""
    file_path = tmp_path / "test.nii"
    data = np.random.rand(10, 10, 10).astype("float32")
    nii = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(nii, file_path)

    loaded_data, meta = load_data(str(file_path))
    assert np.allclose(loaded_data, data), "Loaded .nii data does not match expected data."


def test_import_tiff(tmp_path):
    """Test importing a 3D .tiff file."""
    file_path = tmp_path / "test.tiff"
    data = np.random.rand(10, 10, 10).astype("float32")
    tifffile.imwrite(file_path, data)

    loaded_data, meta = load_data(str(file_path))
    assert np.allclose(loaded_data, data), "Loaded .tiff data does not match expected data."


def test_register_volumes():
    """Test the register_volumes function."""
    import scipy.ndimage

    fixed = np.random.rand(256, 256, 256).astype("float32")
    fixed = scipy.ndimage.gaussian_filter(fixed, sigma=1)
    moving = np.roll(fixed, shift=5, axis=0).copy()  # Simulate a simple shift
    recipe = warpfield.Recipe.from_yaml("default.yml")

    registered, warp_map, _ = warpfield.register_volumes(fixed, moving, recipe, backend="auto", verbose=False)

    assert registered.shape == fixed.shape, "Registered volume shape mismatch."
    assert (
        np.abs(registered[10:-10, 10:-10, 10:-10] - fixed[10:-10, 10:-10, 10:-10]) < 0.2
    ).mean() > 0.9, "Registered volume does not match the fixed volume."
    assert warp_map is not None, "WarpMap object was not returned."


def test_register_volumes_noncubic():
    """Test the register_volumes function with non-cubic volumes (tests shape mismatch bug)."""
    import scipy.ndimage

    # Non-cubic volume that would expose the stacking bug
    fixed = np.random.rand(256, 256, 180).astype("float32")
    fixed = scipy.ndimage.gaussian_filter(fixed, sigma=1)
    moving = np.roll(fixed, shift=5, axis=0).copy()  # Simulate a simple shift
    recipe = warpfield.Recipe.from_yaml("default.yml")

    registered, warp_map, _ = warpfield.register_volumes(fixed, moving, recipe, backend="auto", verbose=False)

    assert registered.shape == fixed.shape, "Registered volume shape mismatch."
    assert (
        np.abs(registered[10:-10, 10:-10, 10:-10] - fixed[10:-10, 10:-10, 10:-10]) < 0.2
    ).mean() > 0.9, "Registered volume does not match the fixed volume."
    assert warp_map is not None, "WarpMap object was not returned."


def test_cli(tmp_path):
    """Test the CLI for registering volumes."""
    fixed = np.random.rand(256, 256, 256).astype("float32")
    moving = np.roll(fixed, shift=5, axis=0).copy()  # Simulate a simple shift

    # Save mock volumes to temporary files
    fixed_path = tmp_path / "fixed.npy"
    moving_path = tmp_path / "moving.npy"
    np.save(fixed_path, fixed)
    np.save(moving_path, moving)

    # Path to the default recipe
    recipe_path = "default.yml"

    # Output path
    output_path = tmp_path / "output.h5"

    # Run the CLI command
    result = subprocess.run(
        [
            "python",
            "-m",
            "warpfield",
            "--fixed",
            str(fixed_path),
            "--moving",
            str(moving_path),
            "--recipe",
            recipe_path,
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    # Check the CLI execution
    assert result.returncode == 0, f"CLI failed with error: {result.stderr}"

    # Verify the output file
    assert os.path.exists(output_path), "Output file was not created."
    import h5py

    with h5py.File(output_path, "r") as f:
        assert "moving_reg" in f, "Registered volume not found in output file."
        assert "warp_map" in f, "Warp map not found in output file."
