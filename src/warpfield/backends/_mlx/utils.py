import mlx.core as mx


from .ndimage import (
    convolve_mlx,
    gaussian_filter_mlx,
    gausskernel_sheared,
    upsampled_dft_rfftn,
    unravel_index_mlx,
    map_coordinates_mlx,
)


def tukey_window_mlx(n, alpha):
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


def soften_edges_mlx(arr, soft_edge):
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
            win = tukey_window_mlx(arr.shape[i], alpha)

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
        xcorr_proj = gaussian_filter_mlx(xcorr_proj, sigma=[0, 0, sigmas[2], 0, 0], mode="constant", truncate=truncate)
    else:
        # Standard separable Gaussian filtering (no shear)
        xcorr_proj = gaussian_filter_mlx(xcorr_proj, sigma=[*sigmas, 0, 0], mode="constant", truncate=truncate)

    if long_range_ratio is not None:
        # Apply long-range smoothing with larger sigma
        xcorr_proj = xcorr_proj * (1 - long_range_ratio)

        long_range_sigma = [s * 5 for s in sigmas] + [0, 0]
        xcorr_proj_smooth = gaussian_filter_mlx(xcorr_proj, sigma=long_range_sigma, mode="constant", truncate=truncate)
        xcorr_proj = xcorr_proj + xcorr_proj_smooth * long_range_ratio

    return xcorr_proj


# Cache for compiled smooth functions
# Compiled smooth function that accepts all parameters explicitly
@mx.compile
def _compiled_smooth_func_impl(xcorr_proj, block_size, sigmas, shear, long_range_ratio):
    """Compiled implementation of smooth function with all parameters passed explicitly."""
    return _smooth_func_mlx_impl(xcorr_proj, block_size, sigmas, shear, long_range_ratio)


def smooth_func_mlx(xcorr_proj, block_size, sigmas, shear, long_range_ratio):
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
    return map_coordinates_mlx(channel, coords, mode="nearest", order=1)


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
    center_y = xcorr_proj.shape[-2] // 2
    center_x = xcorr_proj.shape[-1] // 2

    y_mesh = mx.arange(xcorr_proj.shape[-2], dtype=mx.float32).reshape(-1, 1)
    x_mesh = mx.arange(xcorr_proj.shape[-1], dtype=mx.float32).reshape(1, -1)
    is_center = (y_mesh == center_y) & (x_mesh == center_x)
    epsilon_delta = mx.where(is_center, epsilon, mx.array(0.0, dtype=mx.float32))
    xcorr_proj = xcorr_proj + epsilon_delta

    flat_xcorr = mx.reshape(xcorr_proj, (*xcorr_proj.shape[:-2], -1))
    max_flat_ix = mx.argmax(flat_xcorr, axis=-1)

    max_ix_0, max_ix_1 = unravel_index_mlx(max_flat_ix.flatten(), xcorr_proj.shape[-2:])
    max_ix_0 = mx.reshape(max_ix_0, xcorr_proj.shape[:-2])
    max_ix_1 = mx.reshape(max_ix_1, xcorr_proj.shape[:-2])
    max_ix = mx.stack([max_ix_0, max_ix_1], axis=0)
    max_ix = max_ix - mx.array(xcorr_proj.shape[-2:], dtype=mx.float32)[:, None, None, None] // 2

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

    max_sub_0, max_sub_1 = unravel_index_mlx(max_sub_flat_ix.flatten(), shifts.shape[-2:])
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
                R_mx[i],           # i-th R
                epsilon_mx,        # same epsilon for all
                subpixel           # explicit subpixel parameter
            )
            results.append(disp)

        # Stack results back together (mimics vmap output shape)
        return mx.stack(results, axis=0)

    return displacement_computer
