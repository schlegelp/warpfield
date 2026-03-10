import numpy as np
import mlx.core as mx


from .ndimage import gaussian_filter, gausskernel_sheared, upsampled_dft_rfftn


def tukey_window(n, alpha):
    """Generate a Tukey (tapered cosine) window on GPU using MLX.

    Args:
        n (int): Window length
        alpha (float): Shape parameter (0 <= alpha <= 1)
            - alpha = 0: rectangular window
            - alpha = 1: Hann window

    Returns:
        mx.array: Tukey window of length n
    """
    if alpha <= 0:
        return mx.ones(n, dtype=mx.float32)
    elif alpha >= 1:
        # Hann window
        x = mx.arange(n, dtype=mx.float32)
        return 0.5 - 0.5 * mx.cos(2 * mx.pi * x / (n - 1))

    # Tukey window following scipy's implementation
    x = mx.arange(n, dtype=mx.float32)

    # Compute taper width (integer floor)
    width = int(mx.floor(alpha * (n - 1) / 2.0))
    n2 = n - width - 1  # Start of right taper

    # Start with all ones
    window = mx.ones(n, dtype=mx.float32)

    # Left taper: indices 0 to width
    # Formula: 0.5 * (1 + cos(pi * (-1 + 2*i / (alpha*(n-1)))))
    left_mask = x <= width
    left_arg = mx.pi * (-1.0 + 2.0 * x / (alpha * (n - 1)))
    left_taper = 0.5 * (1.0 + mx.cos(left_arg))

    # Right taper: indices n2 to n-1
    # Formula: 0.5 * (1 + cos(pi * (-2/alpha + 1 + 2*i / (alpha*(n-1)))))
    right_mask = x >= n2
    right_arg = mx.pi * (-2.0 / alpha + 1.0 + 2.0 * x / (alpha * (n - 1)))
    right_taper = 0.5 * (1.0 + mx.cos(right_arg))

    # Apply tapers
    window = mx.where(left_mask, left_taper, window)
    window = mx.where(right_mask, right_taper, window)

    return window


def soften_edges(arr, soft_edge):
    """Apply soft Tukey edge to array on GPU using MLX.

    Args:
        arr (mx.array): Input array
        soft_edge (tuple): Size of soft edge for each dimension

    Returns:
        mx.array: Array with softened edges
    """
    if not isinstance(arr, mx.array):
        arr = mx.array(arr, dtype=mx.float32)

    result = arr

    if isinstance(soft_edge, (int, float)):
        soft_edge = (soft_edge,) * arr.ndim

    # Clip soft_edge to valid range
    soft_edge = [min(max(se, 0), arr.shape[i] / 2) for i, se in enumerate(soft_edge)]

    for i in range(arr.ndim):
        if soft_edge[i] > 0:
            alpha = 2 * soft_edge[i] / arr.shape[i]
            alpha = min(max(alpha, 0), 1)

            # Generate Tukey window on GPU
            win = tukey_window(arr.shape[i], alpha)

            # Reshape window to broadcast along axis i
            new_shape = [1] * arr.ndim
            new_shape[i] = arr.shape[i]
            win_reshaped = mx.reshape(win, new_shape)

            # Apply window
            result = result * win_reshaped

    return result


def _smooth_func_mlx_impl(xcorr_proj, block_size, sigmas, shear, long_range_ratio):
    """
    Core smoothing implementation. This is the function that gets compiled.
    Kernel generation happens entirely on GPU via gausskernel_sheared.
    """
    truncate = 4.0

    # Convert to MLX array if needed
    if not isinstance(xcorr_proj, mx.array):
        xcorr_proj = mx.array(xcorr_proj, dtype=mx.float32)

    if sigmas is None:
        return xcorr_proj

    if shear is not None:
        # Generate sheared kernel on GPU using GPU-bound gausskernel_sheared
        gw = gausskernel_sheared(sigmas[:2], shear, truncate=truncate)
        xcorr_proj = convolve_mlx(xcorr_proj, gw, mode="constant")

        # Apply 1D Gaussian along axis 2 (the third block dimension)
        xcorr_proj = gaussian_filter(xcorr_proj, sigma=[0, 0, sigmas[2], 0, 0], mode="constant", truncate=truncate)
    else:
        # Standard separable Gaussian filtering (no shear)
        xcorr_proj = gaussian_filter(xcorr_proj, sigma=[*sigmas, 0, 0], mode="constant", truncate=truncate)

    if long_range_ratio is not None:
        # Apply long-range smoothing with larger sigma
        xcorr_proj = xcorr_proj * (1 - long_range_ratio)

        long_range_sigma = [s * 5 for s in sigmas] + [0, 0]
        xcorr_proj_smooth = gaussian_filter(xcorr_proj, sigma=long_range_sigma, mode="constant", truncate=truncate)
        xcorr_proj = xcorr_proj + xcorr_proj_smooth * long_range_ratio

    return xcorr_proj


# Cache for compiled smooth functions
# Compiled smooth function that accepts all parameters explicitly
@mx.compile
def _compiled_smooth_func_impl(xcorr_proj, block_size, sigmas, shear, long_range_ratio):
    """Compiled implementation of smooth function with all parameters passed explicitly."""
    return _smooth_func_mlx_impl(xcorr_proj, block_size, sigmas, shear, long_range_ratio)


def smooth_func(xcorr_proj, block_size, sigmas, shear, long_range_ratio):
    """
    Apply Gaussian smoothing to the cross-correlation data.

    All operations including kernel generation happen on GPU via compiled function.
    """
    # Convert inputs to MLX arrays if needed.
    xcorr_proj_mx = mx.array(xcorr_proj, dtype=mx.float32) if not isinstance(xcorr_proj, mx.array) else xcorr_proj
    block_size_mx = mx.array(block_size, dtype=mx.float32) if not isinstance(block_size, mx.array) else block_size

    # NOTE:
    # The compiled path currently produces all-zero outputs for valid inputs,
    # which collapses correlation peaks to the epsilon-biased center.
    # Use the non-compiled implementation to preserve correct behavior.
    return _smooth_func_mlx_impl(xcorr_proj_mx, block_size_mx, sigmas, shear, long_range_ratio)


def _sample_displacement_channel(channel, coords):
    """Sample one displacement component at `coords` using trilinear interpolation."""
    return map_coordinates(channel, coords, mode="nearest", order=1)


# Lazy initialization of vmapped sampler to avoid circular import issues
_sample_displacement_channels_vmapped = None


def _get_sample_displacement_channels_vmapped():
    """Get or create the compiled vmapped sampler."""
    global _sample_displacement_channels_vmapped
    if _sample_displacement_channels_vmapped is None:
        vmapped_fn = mx.vmap(_sample_displacement_channel, in_axes=(0, None), out_axes=0)
        _sample_displacement_channels_vmapped = mx.compile(vmapped_fn)
    return _sample_displacement_channels_vmapped


def _refine_displacement_from_xcorr_and_spectrum(xcorr_proj, R, epsilon, subpixel):
    """Refine integer-peak displacement to subpixel precision for one projection."""
    max_ix = _compute_integer_peak_from_xcorr(xcorr_proj, epsilon)
    return _refine_subpixel_from_spectrum(R, max_ix, subpixel)


def _compute_integer_peak_from_xcorr(xcorr_proj, epsilon):
    """Compute integer displacement peak from cross-correlation volume."""
    center_y = xcorr_proj.shape[-2] // 2
    center_x = xcorr_proj.shape[-1] // 2

    y_mesh = mx.arange(xcorr_proj.shape[-2], dtype=mx.float32).reshape(-1, 1)
    x_mesh = mx.arange(xcorr_proj.shape[-1], dtype=mx.float32).reshape(1, -1)
    is_center = (y_mesh == center_y) & (x_mesh == center_x)
    epsilon_delta = mx.where(is_center, epsilon, mx.array(0.0, dtype=mx.float32))
    xcorr_proj = xcorr_proj + epsilon_delta

    flat_xcorr = mx.reshape(xcorr_proj, (*xcorr_proj.shape[:-2], -1))
    max_flat_ix = mx.argmax(flat_xcorr, axis=-1)

    max_ix_0, max_ix_1 = unravel_index(max_flat_ix.flatten(), xcorr_proj.shape[-2:])
    max_ix_0 = mx.reshape(max_ix_0, xcorr_proj.shape[:-2])
    max_ix_1 = mx.reshape(max_ix_1, xcorr_proj.shape[:-2])
    max_ix = mx.stack([max_ix_0, max_ix_1], axis=0)
    max_ix = max_ix - mx.array(xcorr_proj.shape[-2:], dtype=mx.float32)[:, None, None, None] // 2
    return max_ix


def _refine_subpixel_from_spectrum(R, max_ix, subpixel):
    """Refine integer displacement peak to subpixel using local upsampled DFT."""
    max_ix_reshaped = mx.reshape(max_ix, (2, -1))
    i0 = max_ix_reshaped[0]
    j0 = max_ix_reshaped[1]

    shifts = upsampled_dft_rfftn(
        R.reshape(-1, *R.shape[-2:]),
        upsampled_region_size=int(subpixel * 2 + 1),
        upsample_factor=subpixel,
        axis_offsets=(i0, j0),
    )

    flat_shifts = mx.reshape(shifts, (*shifts.shape[:-2], -1))
    max_sub_flat_ix = mx.argmax(flat_shifts, axis=-1)

    max_sub_0, max_sub_1 = unravel_index(max_sub_flat_ix.flatten(), shifts.shape[-2:])
    max_sub_0 = mx.reshape(max_sub_0, max_ix.shape[1:])
    max_sub_1 = mx.reshape(max_sub_1, max_ix.shape[1:])
    max_sub = mx.stack([max_sub_0, max_sub_1], axis=0)
    max_sub = (max_sub - mx.array(shifts.shape[-2:], dtype=mx.float32)[:, None, None, None] // 2) / subpixel
    return max_ix + max_sub


# Compiled displacement refinement function with subpixel passed explicitly as parameter
@mx.compile
def _compiled_refine_impl(xcorr_proj, R, epsilon, subpixel):
    """Compiled implementation of displacement refinement with all parameters passed explicitly."""
    return _refine_displacement_from_xcorr_and_spectrum(xcorr_proj, R, epsilon, subpixel)


def _get_compiled_displacement_refiner(subpixel):
    """Return the compiled displacement-refinement function."""

    # Call the compiled function directly, passing subpixel as a parameter
    def refiner(xcorr_proj, R, epsilon):
        # Convert inputs to MLX arrays if needed
        xcorr_proj_mx = mx.array(xcorr_proj, dtype=mx.float32) if not isinstance(xcorr_proj, mx.array) else xcorr_proj
        R_mx = mx.array(R, dtype=mx.float32) if not isinstance(R, mx.array) else R
        epsilon_mx = mx.array(epsilon, dtype=mx.float32) if not isinstance(epsilon, mx.array) else epsilon
        subpixel_val = mx.array(subpixel, dtype=mx.float32)

        return _compiled_refine_impl(xcorr_proj_mx, R_mx, epsilon_mx, subpixel_val)

    return refiner


def _compute_displacement_for_one_projection(xcorr_proj, R, epsilon_val, subpx):
    """Refine displacement for one projection using pre-computed xcorr and R. Designed for vmap."""
    disp = _refine_displacement_from_xcorr_and_spectrum(xcorr_proj, R, epsilon_val, subpixel=subpx)
    return disp


def _get_compiled_vmapped_displacement_computer(subpixel):
    """Return a function that processes all 3 projections at once.

    Note: We manually process projections to avoid closure capture issues with vmap.
    The subpixel parameter is passed explicitly and captured here (outer closure only).
    """

    def displacement_computer(xcorr_proj, R, epsilon):
        # Convert inputs to MLX arrays if needed
        xcorr_proj_mx = mx.array(xcorr_proj, dtype=mx.float32) if not isinstance(xcorr_proj, mx.array) else xcorr_proj
        # Note: R may be complex, keep its dtype
        R_mx = mx.array(R) if not isinstance(R, mx.array) else R
        epsilon_mx = mx.array(epsilon, dtype=mx.float32) if not isinstance(epsilon, mx.array) else epsilon

        # Process each projection (manually unroll vmap) to avoid vmap issues
        # xcorr_proj_mx has shape (3, ...) - one per projection
        # R_mx has shape (3, ...) - one per projection
        # epsilon_mx is scalar - same for all projections

        results = []
        for i in range(3):
            disp = _compute_displacement_for_one_projection(
                xcorr_proj_mx[i],  # i-th projection
                R_mx[i],  # i-th R
                epsilon_mx,  # same epsilon for all
                subpixel,  # explicit subpixel parameter
            )
            results.append(disp)

        # Stack results back together (mimics vmap output shape)
        return mx.stack(results, axis=0)

    return displacement_computer


def convolve_mlx(input_arr, kernel, mode="constant"):
    """Apply multidimensional convolution in MLX.

    This is optimized for the case where the kernel has fewer dimensions than the input.
    For example, convolving a 5D array with a 2D kernel along the first 2 axes.

    Args:
        input_arr (mx.array): Input array (e.g., 5D with shape [a, b, c, d, e]).
        kernel (array_like): Convolution kernel (e.g., 2D with shape [ka, kb]).
        mode (str): Padding mode ('constant' for zero-padding).

    Returns:
        mx.array: Convolved array with same shape as input.
    """
    if not isinstance(input_arr, mx.array):
        input_arr = mx.array(input_arr, dtype=mx.float32)
    if not isinstance(kernel, mx.array):
        kernel = mx.array(kernel, dtype=mx.float32)

    # Get dimensions
    input_shape = input_arr.shape
    kernel_shape = kernel.shape
    kernel_ndim = len(kernel_shape)

    # For efficiency, we'll apply the convolution by reshaping
    # E.g., for 5D input [a,b,c,d,e] and 2D kernel [ka,kb]:
    # Reshape to [a,b,c*d*e], convolve along first 2 dims, reshape back

    if kernel_ndim < len(input_shape):
        # Reshape input: move conv dims to front, flatten rest
        inner_shape = input_shape[kernel_ndim:]
        inner_size = int(np.prod(inner_shape))
        reshaped_input = mx.reshape(input_arr, input_shape[:kernel_ndim] + (inner_size,))
    else:
        reshaped_input = input_arr
        inner_size = 1

    # Pad the array
    pad_list = [(ks // 2, ks // 2) for ks in kernel_shape]
    if kernel_ndim < len(reshaped_input.shape):
        pad_list += [(0, 0)] * (len(reshaped_input.shape) - kernel_ndim)

    if mode == "constant":
        padded = mx.pad(reshaped_input, pad_list, constant_values=0.0)
    else:
        padded = reshaped_input

    # For 2D kernels, use MLX grouped convolution over the flattened inner channels.
    if kernel_ndim == 2:
        if len(padded.shape) == 2:
            # (H, W) -> (N=1, H, W, Cin=1)
            conv_input = mx.expand_dims(mx.expand_dims(padded, axis=0), axis=-1)
            in_channels = 1
            squeeze_output = True
        else:
            # (H, W, Cin) -> (N=1, H, W, Cin)
            conv_input = mx.expand_dims(padded, axis=0)
            in_channels = int(conv_input.shape[-1])
            squeeze_output = False

        k0, k1 = kernel_shape
        kernel_broadcast = mx.broadcast_to(
            mx.reshape(kernel, (1, int(k0), int(k1), 1)), (in_channels, int(k0), int(k1), 1)
        )

        # conv_general with flip=True computes convolution (not cross-correlation).
        conv_out = mx.conv_general(
            conv_input, kernel_broadcast, stride=(1, 1), padding=0, groups=in_channels, flip=True
        )

        result = conv_out[0, :, :, 0] if squeeze_output else conv_out[0]
    else:
        # For other cases, fall back to separable convolution
        # This is a simplification - proper implementation would need more work
        result = padded

    # Reshape back to original shape
    if kernel_ndim < len(input_shape):
        result = mx.reshape(result, input_shape)

    return result


def unravel_index(indices, shape):
    """MLX implementation of numpy's unravel_index for 2D shapes.

    Args:
        indices (mx.array): Flat indices to convert.
        shape (tuple): Shape of the array (rows, cols).

    Returns:
        tuple of mx.array: (row_indices, col_indices)
    """
    rows, cols = shape
    row_idx = indices // cols
    col_idx = indices % cols
    return row_idx, col_idx


def map_coordinates(input_arr, coordinates, order=1, mode="nearest"):
    """GPU-accelerated map_coordinates using MLX (trilinear interpolation).

    Args:
        input_arr (mx.array): Input array to sample from, shape (D1, D2, D3).
        coordinates (mx.array): Coordinates at which to sample, shape (3, N).
        order (int): Interpolation order - only 1 (linear) is supported.
        mode (str): How to handle out-of-bounds - only "nearest" is supported.

    Returns:
        mx.array: Sampled values, shape (N,).
    """
    assert mode == "nearest", "Only 'nearest' mode is supported in MLX map_coordinates"

    if order != 1:
        # Fall back to numpy for unsupported orders
        input_np = np.array(input_arr)
        coords_np = np.array(coordinates)
        from scipy.ndimage import map_coordinates

        result = map_coordinates(input_np, coords_np, order=order, mode=mode)
        return mx.array(result, dtype=mx.float32)

    if not isinstance(input_arr, mx.array):
        input_arr = mx.array(input_arr, dtype=mx.float32)
    if not isinstance(coordinates, mx.array):
        coordinates = mx.array(coordinates, dtype=mx.float32)

    # Get dimensions
    D0, D1, D2 = input_arr.shape

    # Extract coordinates for each dimension
    z_coords = coordinates[0]  # Shape: (N,)
    y_coords = coordinates[1]  # Shape: (N,)
    x_coords = coordinates[2]  # Shape: (N,)

    # Clamp coordinates to valid range (nearest mode)
    z_coords = mx.clip(z_coords, 0, D0 - 1)
    y_coords = mx.clip(y_coords, 0, D1 - 1)
    x_coords = mx.clip(x_coords, 0, D2 - 1)

    # Get integer parts (floor)
    z0 = mx.floor(z_coords).astype(mx.int32)
    y0 = mx.floor(y_coords).astype(mx.int32)
    x0 = mx.floor(x_coords).astype(mx.int32)

    # Get next integer (ceil), clamped
    z1 = mx.minimum(z0 + 1, D0 - 1)
    y1 = mx.minimum(y0 + 1, D1 - 1)
    x1 = mx.minimum(x0 + 1, D2 - 1)

    # Get fractional parts
    zd = z_coords - z0.astype(mx.float32)
    yd = y_coords - y0.astype(mx.float32)
    xd = x_coords - x0.astype(mx.float32)

    # Sample at 8 corners of the cube
    # Use advanced indexing to gather values
    c000 = input_arr[z0, y0, x0]
    c001 = input_arr[z0, y0, x1]
    c010 = input_arr[z0, y1, x0]
    c011 = input_arr[z0, y1, x1]
    c100 = input_arr[z1, y0, x0]
    c101 = input_arr[z1, y0, x1]
    c110 = input_arr[z1, y1, x0]
    c111 = input_arr[z1, y1, x1]

    # Trilinear interpolation
    # Along x
    c00 = c000 * (1 - xd) + c001 * xd
    c01 = c010 * (1 - xd) + c011 * xd
    c10 = c100 * (1 - xd) + c101 * xd
    c11 = c110 * (1 - xd) + c111 * xd

    # Along y
    c0 = c00 * (1 - yd) + c01 * yd
    c1 = c10 * (1 - yd) + c11 * yd

    # Along z
    result = c0 * (1 - zd) + c1 * zd

    return result


def median_filter(input_arr, size, mode="nearest"):
    """Apply 3D median filter on MLX array using GPU.

    Optimized for size=[1, 3, 3, 3] case (no filtering on first dimension).

    Args:
        input_arr (mx.array): Input array to filter, shape (C, Z, Y, X).
        size (list): Size of the filter window for each dimension.
        mode (str): Padding mode - only "nearest" is supported.

    Returns:
        mx.array: Filtered array with same shape as input.
    """
    if not isinstance(input_arr, mx.array):
        input_arr = mx.array(input_arr, dtype=mx.float32)

    if len(size) != 4 or size[0] != 1 or size[1:] != [3, 3, 3]:
        # Fall back to numpy for unsupported cases
        input_np = np.array(input_arr)
        from scipy.ndimage import median_filter

        filtered = median_filter(input_np, size=size, mode=mode)
        return mx.array(filtered, dtype=mx.float32)

    # Optimized path for [1, 3, 3, 3] median filter
    # Manually pad with edge replication (nearest mode)
    C, Z, Y, X = input_arr.shape

    # Pad Z dimension
    edge_z0 = input_arr[:, 0:1, :, :]  # First Z slice
    edge_z1 = input_arr[:, -1:, :, :]  # Last Z slice
    padded_z = mx.concatenate([edge_z0, input_arr, edge_z1], axis=1)

    # Pad Y dimension
    edge_y0 = padded_z[:, :, 0:1, :]  # First Y slice
    edge_y1 = padded_z[:, :, -1:, :]  # Last Y slice
    padded_zy = mx.concatenate([edge_y0, padded_z, edge_y1], axis=2)

    # Pad X dimension
    edge_x0 = padded_zy[:, :, :, 0:1]  # First X slice
    edge_x1 = padded_zy[:, :, :, -1:]  # Last X slice
    padded = mx.concatenate([edge_x0, padded_zy, edge_x1], axis=3)

    # Now padded has shape (C, Z+2, Y+2, X+2)
    # Pre-allocate array for all 27 neighborhoods (more memory efficient than list append + stack)
    neighborhoods = mx.zeros((C, Z, Y, X, 27), dtype=mx.float32)
    idx = 0
    for dz in range(3):
        for dy in range(3):
            for dx in range(3):
                # Extract shifted version and place directly in pre-allocated array
                neighborhoods[:, :, :, :, idx] = padded[:, dz : dz + Z, dy : dy + Y, dx : dx + X]
                idx += 1

    # Sort along the last dimension and take median (14th element, 0-indexed: 13)
    sorted_neighborhoods = mx.sort(neighborhoods, axis=-1)
    median_result = sorted_neighborhoods[..., 13]

    return median_result