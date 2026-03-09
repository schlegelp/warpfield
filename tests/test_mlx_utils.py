"""Test suite for warpfield MLX utils module.

This module tests the utility functions in warpfield.backends._mlx.utils
by comparing against NumPy/SciPy reference implementations where possible.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy import signal
from scipy.ndimage import map_coordinates as scipy_map_coordinates, median_filter as scipy_median_filter

try:
    import mlx.core as mx
    mlx_available = True
except (ImportError, ModuleNotFoundError):
    mlx_available = False
    warnings.warn("MLX not installed or not importable. Skipping MLX utils tests.")

pytestmark = pytest.mark.skipif(not mlx_available, reason="MLX not available")

if mlx_available:
    from warpfield.backends._mlx.utils import (
        tukey_window,
        soften_edges,
        convolve_mlx,
        unravel_index,
        map_coordinates,
        median_filter,
    )


def _to_numpy(arr):
    """Convert array to numpy, handling MLX arrays."""
    return np.asarray(arr)


class TestTukeyWindow:
    """Test the Tukey window generation."""

    def test_tukey_alpha_zero(self):
        """When alpha=0, should return a rectangular (all ones) window."""
        n = 100
        result = tukey_window(n, alpha=0)
        expected = np.ones(n, dtype=np.float32)
        np.testing.assert_array_almost_equal(_to_numpy(result), expected)

    def test_tukey_alpha_one(self):
        """When alpha=1, should return a Hann window."""
        n = 100
        result_tukey = tukey_window(n, alpha=1)
        # Hann window: 0.5 - 0.5*cos(2*pi*i/(n-1))
        x = np.arange(n, dtype=np.float32)
        expected = 0.5 - 0.5 * np.cos(2 * np.pi * x / (n - 1))
        np.testing.assert_array_almost_equal(_to_numpy(result_tukey), expected, decimal=5)

    def test_tukey_alpha_half(self):
        """Test with alpha=0.5 (symmetric window)."""
        n = 100
        result = tukey_window(n, alpha=0.5)
        result_np = _to_numpy(result)

        # Check symmetry
        np.testing.assert_array_almost_equal(result_np, result_np[::-1], decimal=5)

        # Check bounds
        assert np.all(result_np >= 0) and np.all(result_np <= 1)

        # Check that center values are close to 1
        center_idx = n // 2
        assert result_np[center_idx] > 0.99

    def test_tukey_shape(self):
        """Test output shape is correct."""
        for n in [10, 50, 100, 256]:
            result = tukey_window(n, alpha=0.5)
            assert _to_numpy(result).shape == (n,)

    def test_tukey_bounds(self):
        """Test that window values are always in [0, 1]."""
        for alpha in [0, 0.25, 0.5, 0.75, 1.0]:
            for n in [10, 100]:
                result = tukey_window(n, alpha=alpha)
                result_np = _to_numpy(result)
                assert np.all(result_np >= 0) and np.all(result_np <= 1)

    def test_tukey_against_scipy(self):
        """Compare against scipy.signal.windows.tukey."""
        n = 256
        for alpha in [0, 0.25, 0.5, 0.75, 1.0]:
            result_mlx = _to_numpy(tukey_window(n, alpha))
            result_scipy = signal.windows.tukey(n, alpha=alpha)
            np.testing.assert_array_almost_equal(result_mlx, result_scipy, decimal=5)


class TestSoftenEdgesMLX:
    """Test the soften_edges function."""

    def test_soften_edges_uniform_soft_edge(self):
        """Test with uniform soft edge specification."""
        arr = np.ones((10, 10, 10), dtype=np.float32)
        soft_edge = 2

        result = soften_edges(arr, soft_edge)
        result_np = _to_numpy(result)

        # Result should be < 1 at edges and == 1 in center
        assert result_np.shape == arr.shape
        # Check that edges are reduced
        assert result_np[0, 5, 5] < 1.0
        # Check that center is preserved (approximately)
        assert result_np[5, 5, 5] > 0.99

    def test_soften_edges_per_axis(self):
        """Test with per-axis soft edge specification."""
        arr = np.ones((10, 20, 30), dtype=np.float32)
        soft_edge = (1, 2, 3)

        result = soften_edges(arr, soft_edge)
        result_np = _to_numpy(result)

        assert result_np.shape == arr.shape
        # Test that center is largely preserved
        assert result_np[5, 10, 15] > 0.95

    def test_soften_edges_zero_soft_edge(self):
        """When soft_edge=0, should return original array unchanged."""
        arr = np.ones((10, 10, 10), dtype=np.float32)

        result = soften_edges(arr, soft_edge=0)
        result_np = _to_numpy(result)

        np.testing.assert_array_almost_equal(result_np, arr)

    def test_soften_edges_large_soft_edge(self):
        """When soft_edge is larger than half the dimension, should clip."""
        arr = np.ones((10, 10, 10), dtype=np.float32)
        soft_edge = 100  # Much larger than array dimensions

        result = soften_edges(arr, soft_edge)
        result_np = _to_numpy(result)

        # Should not raise, result should be valid
        assert result_np.shape == arr.shape
        # Values should be valid (not NaN or Inf)
        assert np.all(np.isfinite(result_np))

    def test_soften_edges_tuple_conversion(self):
        """Test that integer soft_edge is converted to tuple."""
        arr = np.ones((10, 10, 10), dtype=np.float32)

        result_int = soften_edges(arr, 2)
        result_tuple = soften_edges(arr, (2, 2, 2))

        np.testing.assert_array_almost_equal(
            _to_numpy(result_int), _to_numpy(result_tuple)
        )

    def test_soften_edges_2d(self):
        """Test with 2D array."""
        arr = np.ones((20, 30), dtype=np.float32)
        soft_edge = 3

        result = soften_edges(arr, soft_edge)
        result_np = _to_numpy(result)

        assert result_np.shape == arr.shape
        # Edges should be lower than center
        assert result_np[0, 15] < result_np[10, 15]

    def test_soften_edges_bounds_preservation(self):
        """Test that softened values stay within [0, 1] when input is [0, 1]."""
        arr = np.random.rand(15, 15, 15).astype(np.float32)

        result = soften_edges(arr, soft_edge=2)
        result_np = _to_numpy(result)

        # Since we're multiplying by a Tukey window [0,1], result should be <= input
        assert np.all(result_np <= arr + 1e-5)  # small tolerance for rounding
        assert np.all(result_np >= 0)


class TestConvolveMLX:
    """Test the convolve_mlx function."""

    def test_convolve_identity_kernel(self):
        """Test with an identity-like kernel."""
        arr = np.random.rand(10, 10, 10).astype(np.float32)
        kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)

        result = convolve_mlx(arr, kernel)
        result_np = _to_numpy(result)

        # Should preserve the center region (approximately)
        assert result_np.shape == arr.shape

    def test_convolve_gaussian_kernel(self):
        """Test with a small Gaussian kernel."""
        arr = np.ones((10, 10, 10), dtype=np.float32)
        kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0

        result = convolve_mlx(arr, kernel)
        result_np = _to_numpy(result)

        assert result_np.shape == arr.shape
        # Convolving ones with a normalized kernel should give values close to 1
        assert np.all(np.isfinite(result_np))

    def test_convolve_output_shape(self):
        """Test that output shape matches input shape."""
        for input_shape in [(20, 20, 20), (15, 25, 18), (10, 10, 10)]:
            arr = np.random.rand(*input_shape).astype(np.float32)
            kernel = np.random.rand(3, 3).astype(np.float32)

            result = convolve_mlx(arr, kernel)
            assert _to_numpy(result).shape == arr.shape

    def test_convolve_with_mlx_arrays(self):
        """Test that MLX arrays are handled correctly."""
        arr_mx = mx.array(np.random.rand(10, 10, 10).astype(np.float32))
        kernel_mx = mx.array(np.eye(3, dtype=np.float32))

        result = convolve_mlx(arr_mx, kernel_mx)

        assert isinstance(result, mx.array)

    def test_convolve_2d_on_5d_array(self):
        """Test that 2D kernel works on 5D array."""
        arr = np.random.rand(2, 3, 10, 10, 5).astype(np.float32)
        kernel = np.random.rand(3, 3).astype(np.float32)

        result = convolve_mlx(arr, kernel)
        result_np = _to_numpy(result)

        assert result_np.shape == arr.shape

    def test_convolve_uniform_input(self):
        """Test with uniform input - verify smooth behavior."""
        arr = np.ones((15, 15, 15), dtype=np.float32)
        kernel = np.ones((3, 3), dtype=np.float32) / 9.0

        result = convolve_mlx(arr, kernel)
        result_np = _to_numpy(result)

        # Convolving uniform array should produce mostly uniform results
        # Center should be close to 1, edges have boundary effects due to padding
        assert np.mean(result_np[5:-5, 5:-5, 5:-5]) > 0.9


class TestUnravelIndex:
    """Test the unravel_index function."""

    def test_unravel_index_single(self):
        """Test with a single index."""
        indices = mx.array([5])
        shape = (3, 4)  # 3 rows, 4 cols

        rows, cols = unravel_index(indices, shape)
        rows_np = _to_numpy(rows)
        cols_np = _to_numpy(cols)

        # 5 in a 3x4 grid is row=1, col=1 (0-indexed)
        assert rows_np[0] == 1
        assert cols_np[0] == 1

    def test_unravel_index_multiple(self):
        """Test with multiple indices."""
        indices = mx.array([0, 4, 11])
        shape = (3, 4)  # 3 rows, 4 cols, total 12 elements

        rows, cols = unravel_index(indices, shape)
        rows_np = _to_numpy(rows)
        cols_np = _to_numpy(cols)

        # Index 0: (0, 0)
        # Index 4: (1, 0)
        # Index 11: (2, 3)
        expected_rows = np.array([0, 1, 2])
        expected_cols = np.array([0, 0, 3])

        np.testing.assert_array_equal(rows_np, expected_rows)
        np.testing.assert_array_equal(cols_np, expected_cols)

    def test_unravel_index_boundaries(self):
        """Test boundary indices."""
        shape = (5, 7)
        indices = mx.array([0, 34])  # First and last indices

        rows, cols = unravel_index(indices, shape)
        rows_np = _to_numpy(rows)
        cols_np = _to_numpy(cols)

        # Index 0: (0, 0)
        assert rows_np[0] == 0 and cols_np[0] == 0
        # Index 34: (4, 6) - last element in 5x7 grid
        assert rows_np[1] == 4 and cols_np[1] == 6

    def test_unravel_index_against_numpy(self):
        """Compare against numpy.unravel_index."""
        shape = (10, 15)
        test_indices = np.array([0, 25, 73, 149])
        indices_mx = mx.array(test_indices)

        rows_mlx, cols_mlx = unravel_index(indices_mx, shape)
        rows_mlx_np = _to_numpy(rows_mlx)
        cols_mlx_np = _to_numpy(cols_mlx)

        rows_np, cols_np = np.unravel_index(test_indices, shape)

        np.testing.assert_array_equal(rows_mlx_np, rows_np)
        np.testing.assert_array_equal(cols_mlx_np, cols_np)


class TestMapCoordinates:
    """Test the map_coordinates function."""

    def test_map_coordinates_single_point(self):
        """Test sampling a single point."""
        arr = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
        coords = mx.array([[1], [1], [1]], dtype=mx.float32)

        result = map_coordinates(arr, coords)
        result_np = _to_numpy(result)

        # Value at (1,1,1) should be 1*9 + 1*3 + 1 = 13
        assert result_np[0] == pytest.approx(13.0, abs=0.1)

    def test_map_coordinates_multiple_points(self):
        """Test sampling multiple points."""
        arr = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
        coords = mx.array([
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2]
        ], dtype=mx.float32)

        result = map_coordinates(arr, coords)
        result_np = _to_numpy(result)

        assert result_np.shape == (3,)
        assert np.all(np.isfinite(result_np))

    def test_map_coordinates_output_shape(self):
        """Test output shape matches coordinate shape."""
        arr = np.random.rand(10, 10, 10).astype(np.float32)

        for n_points in [1, 5, 10, 50]:
            coords = mx.array(
                np.random.rand(3, n_points).astype(np.float32) * 9
            )
            result = map_coordinates(arr, coords)
            assert _to_numpy(result).shape == (n_points,)

    def test_map_coordinates_bounds(self):
        """Test that clipping works correctly."""
        arr = np.ones((5, 5, 5), dtype=np.float32)
        # Coordinates outside bounds
        coords = mx.array([
            [-1.0, 10.0, 2.0],
            [-1.0, 10.0, 2.0],
            [-1.0, 10.0, 2.0]
        ], dtype=mx.float32)

        result = map_coordinates(arr, coords)
        result_np = _to_numpy(result)

        # Should all be 1.0 (since input is all ones and clipped to bounds)
        np.testing.assert_array_almost_equal(result_np, np.ones(3))

    def test_map_coordinates_interpolation(self):
        """Test trilinear interpolation at non-integer coordinates."""
        # Create simple test array where interpolation can be easily verified
        arr = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.float32)

        # Sample at (0.5, 0.5, 0.5) - should interpolate between all 8 corner values
        # The 8 values are: 0, 1, 2, 3, 4, 5, 6, 7
        coords = mx.array([[0.5], [0.5], [0.5]], dtype=mx.float32)
        result = map_coordinates(arr, coords)
        result_np = _to_numpy(result)

        # Trilinear interpolation at center: average of all 8 corners
        expected = 3.5
        assert result_np[0] == pytest.approx(expected, abs=0.2)

    def test_map_coordinates_against_scipy(self):
        """Compare against scipy.ndimage.map_coordinates for a simple case."""
        arr = np.arange(125, dtype=np.float32).reshape(5, 5, 5)
        coords = np.array([
            [1.5, 2.5, 3.0],
            [1.5, 2.5, 3.0],
            [1.5, 2.5, 3.0]
        ], dtype=np.float32)

        # MLX version
        result_mlx = map_coordinates(arr, mx.array(coords))
        result_mlx_np = _to_numpy(result_mlx)

        # SciPy version (also uses order=1 for linear interpolation)
        result_scipy = scipy_map_coordinates(arr, coords, order=1, mode='nearest')

        # They should be very close
        np.testing.assert_array_almost_equal(result_mlx_np, result_scipy, decimal=1)

    def test_map_coordinates_integer_vs_fractional(self):
        """Test that integer coordinates give exact values."""
        arr = np.arange(27, dtype=np.float32).reshape(3, 3, 3)

        # Integer coordinates should give exact values
        int_coords = mx.array([[1], [1], [1]], dtype=mx.float32)
        result_int = map_coordinates(arr, int_coords)

        # Fractional coords slightly offset
        frac_coords = mx.array([[1.01], [1.01], [1.01]], dtype=mx.float32)
        result_frac = map_coordinates(arr, frac_coords)

        # They should be very close but not identical
        diff = np.abs(_to_numpy(result_int) - _to_numpy(result_frac))
        assert 0 < diff[0] < 0.5

    def test_map_coordinates_3d_array(self):
        """Test with various 3D array shapes."""
        for shape in [(5, 5, 5), (10, 15, 8), (7, 7, 7)]:
            arr = np.random.rand(*shape).astype(np.float32)
            # Random coordinates within bounds
            coords = mx.array(
                np.random.rand(3, 10).astype(np.float32) *
                np.array(shape).reshape(3, 1) / 2  # Scale to keep in bounds
            )

            result = map_coordinates(arr, coords)
            assert _to_numpy(result).shape == (10,)
            assert np.all(np.isfinite(_to_numpy(result)))

    def test_map_coordinates_constant_array(self):
        """Test sampling from a constant array."""
        arr = np.full((10, 10, 10), 5.0, dtype=np.float32)
        coords = mx.array(
            np.random.rand(3, 5).astype(np.float32) * 9
        )

        result = map_coordinates(arr, coords)
        result_np = _to_numpy(result)

        # Should be approximately 5.0 everywhere
        np.testing.assert_array_almost_equal(result_np, np.full(5, 5.0), decimal=1)


class TestConvolveMLXComparison:
    """Integration tests for convolve_mlx comparing with scipy/numpy."""

    def test_convolve_separable_gaussian(self):
        """Test convolution with a separable Gaussian approximation."""
        # Create a test signal
        arr = np.random.rand(10, 10, 10).astype(np.float32)

        # Create a simple smoothing kernel
        kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0

        result = convolve_mlx(arr, kernel)
        result_np = _to_numpy(result)

        # Result should be smoother than input (lower variance)
        assert np.var(result_np) < np.var(arr)
        # But values should still be in reasonable range
        assert np.all(np.isfinite(result_np))

    def test_convolve_zero_kernel(self):
        """Test with zero kernel."""
        arr = np.random.rand(10, 10, 10).astype(np.float32)
        kernel = np.zeros((3, 3), dtype=np.float32)

        result = convolve_mlx(arr, kernel)
        result_np = _to_numpy(result)

        # Convolving with all zeros should give all zeros
        np.testing.assert_array_almost_equal(result_np, np.zeros_like(arr))


class TestMapCoordinatesWithComplexShapes:
    """Additional integration tests for map_coordinates."""

    def test_map_coordinates_random_sampling(self):
        """Test random sampling from a known 3D array."""
        # Create a linear gradient array
        arr = np.zeros((10, 10, 10), dtype=np.float32)
        for i in range(10):
            arr[i, :, :] = i

        # Sample at multiple z-coordinates
        coords = mx.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5.0, 5.0, 5.0, 5.0, 5.0],
            [5.0, 5.0, 5.0, 5.0, 5.0]
        ], dtype=mx.float32)

        result = map_coordinates(arr, coords)
        result_np = _to_numpy(result)

        # Expected: [1, 2, 3, 4, 5]
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_almost_equal(result_np, expected, decimal=1)


class TestMedianFilter:
    """Test the median_filter_mlx function."""

    def test_median_filter_shape_preservation(self):
        """Test that output shape matches input shape."""
        arr = np.random.rand(2, 5, 10, 10).astype(np.float32)
        size = [1, 3, 3, 3]

        result = median_filter(arr, size)
        result_np = _to_numpy(result)

        assert result_np.shape == arr.shape

    def test_median_filter_mlx_input(self):
        """Test with MLX array input."""
        arr_mx = mx.array(np.random.rand(2, 5, 10, 10).astype(np.float32))
        size = [1, 3, 3, 3]

        result = median_filter(arr_mx, size)

        assert isinstance(result, mx.array)
        assert result.shape == arr_mx.shape

    def test_median_filter_removes_noise(self):
        """Test that median filter reduces salt-and-pepper noise."""
        # Create a clean array
        arr_clean = np.ones((2, 10, 10, 10), dtype=np.float32)

        # Add salt-and-pepper noise
        arr_noisy = arr_clean.copy()
        np.random.seed(42)
        noise_mask = np.random.rand(2, 10, 10, 10) < 0.1
        arr_noisy[noise_mask] = np.random.choice([0, 2], size=noise_mask.sum())

        # Apply median filter
        result = median_filter(arr_noisy, size=[1, 3, 3, 3])
        result_np = _to_numpy(result)

        # Check that filtered result is closer to clean version than noisy input
        error_noisy = np.mean(np.abs(arr_noisy - arr_clean))
        error_filtered = np.mean(np.abs(result_np - arr_clean))
        assert error_filtered < error_noisy

    def test_median_filter_preserves_edges(self):
        """Test that median filter partially preserves edges."""
        # Create array with distinct regions
        arr = np.zeros((2, 10, 10, 10), dtype=np.float32)
        arr[:, :5, :, :] = 1.0  # Upper half = 1
        arr[:, 5:, :, :] = 2.0  # Lower half = 2

        result = median_filter(arr, size=[1, 3, 3, 3])
        result_np = _to_numpy(result)

        # Interior regions should be preserved
        assert result_np[:, 2, 5, 5] == pytest.approx(1.0, abs=0.1)
        assert result_np[:, 7, 5, 5] == pytest.approx(2.0, abs=0.1)

    def test_median_filter_constant_array(self):
        """Test with constant input - should be unchanged."""
        arr = np.full((2, 8, 8, 8), 5.0, dtype=np.float32)

        result = median_filter(arr, size=[1, 3, 3, 3])
        result_np = _to_numpy(result)

        # Should remain constant
        np.testing.assert_array_almost_equal(result_np, arr)

    def test_median_filter_against_scipy(self):
        """Compare against scipy.ndimage.median_filter."""
        arr = np.random.rand(2, 6, 8, 8).astype(np.float32)
        size = [1, 3, 3, 3]

        # MLX version
        result_mlx = median_filter(arr, size)
        result_mlx_np = _to_numpy(result_mlx)

        # SciPy version
        result_scipy = scipy_median_filter(arr, size=size, mode="nearest")

        # Should be approximately equal (may differ slightly due to implementation details)
        np.testing.assert_array_almost_equal(result_mlx_np, result_scipy, decimal=1)

    def test_median_filter_fallback_path(self):
        """Test fallback path for unsupported size parameters."""
        arr = np.random.rand(2, 6, 8, 8).astype(np.float32)
        # Use a size that requires fallback
        size = [1, 5, 5, 5]

        result = median_filter(arr, size)
        result_np = _to_numpy(result)

        # Should still work via scipy fallback
        assert result_np.shape == arr.shape
        assert np.all(np.isfinite(result_np))

    def test_median_filter_small_array(self):
        """Test with small array."""
        arr = np.random.rand(1, 3, 3, 3).astype(np.float32)
        size = [1, 3, 3, 3]

        result = median_filter(arr, size)
        result_np = _to_numpy(result)

        assert result_np.shape == arr.shape

    def test_median_filter_single_channel(self):
        """Test with single channel (C=1)."""
        arr = np.random.rand(1, 10, 10, 10).astype(np.float32)
        size = [1, 3, 3, 3]

        result = median_filter(arr, size)
        result_np = _to_numpy(result)

        assert result_np.shape == arr.shape

    def test_median_filter_multiple_channels(self):
        """Test with multiple channels."""
        arr = np.random.rand(4, 10, 10, 10).astype(np.float32)
        size = [1, 3, 3, 3]

        result = median_filter(arr, size)
        result_np = _to_numpy(result)

        assert result_np.shape == arr.shape
        # Each channel should be filtered independently
        assert np.all(np.isfinite(result_np))

    def test_median_filter_with_nan_handling(self):
        """Test that filter handles edge cases gracefully."""
        arr = np.random.rand(2, 8, 8, 8).astype(np.float32)
        # Ensure no NaN or Inf in input
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        size = [1, 3, 3, 3]

        result = median_filter(arr, size)
        result_np = _to_numpy(result)

        # Output should not have NaN or Inf
        assert np.all(np.isfinite(result_np))

    def test_median_filter_boundary_behavior(self):
        """Test that boundary is handled correctly with nearest mode."""
        # Create array with gradient
        arr = np.zeros((1, 5, 5, 5), dtype=np.float32)
        for i in range(5):
            arr[0, i, :, :] = i

        result = median_filter(arr, size=[1, 3, 3, 3])
        result_np = _to_numpy(result)

        # Interior values should match gradient values
        assert result_np[0, 1, 2, 2] == pytest.approx(1.0, abs=0.1)
        assert result_np[0, 2, 2, 2] == pytest.approx(2.0, abs=0.1)
        # Boundaries with edge replication will have the boundary value or lower
        assert result_np[0, 0, 2, 2] <= 1.0

    def test_median_filter_idempotent(self):
        """Test that applying filter twice is mostly idempotent for valid inputs."""
        arr = np.ones((2, 8, 8, 8), dtype=np.float32)
        size = [1, 3, 3, 3]

        # Apply filter once
        result1 = median_filter(arr, size)
        result1_np = _to_numpy(result1)

        # Apply filter again
        result2 = median_filter(result1_np, size)
        result2_np = _to_numpy(result2)

        # Should be very similar (idempotent for constant arrays)
        np.testing.assert_array_almost_equal(result1_np, result2_np, decimal=5)

    def test_median_filter_dtype_preservation(self):
        """Test that output dtype is preserved as float32."""
        arr = np.random.rand(2, 8, 8, 8).astype(np.float32)
        size = [1, 3, 3, 3]

        result = median_filter(arr, size)
        result_np = _to_numpy(result)

        assert result_np.dtype == np.float32

