import numpy as np
import mlx.core as mx
import scipy.ndimage
import scipy.signal


def unravel_index_mlx(indices, shape):
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


def median_filter_mlx(input_arr, size, mode="nearest"):
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


def map_coordinates_mlx(input_arr, coordinates, order=1, mode="nearest"):
    """GPU-accelerated map_coordinates using MLX (trilinear interpolation).

    Args:
        input_arr (mx.array): Input array to sample from, shape (D1, D2, D3).
        coordinates (mx.array): Coordinates at which to sample, shape (3, N).
        order (int): Interpolation order - only 1 (linear) is supported.
        mode (str): How to handle out-of-bounds - only "nearest" is supported.

    Returns:
        mx.array: Sampled values, shape (N,).
    """
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


def gaussian_kernel_1d(sigma, truncate=4.0):
    """Generate a 1D Gaussian kernel for MLX.

    Args:
        sigma (float): Standard deviation for Gaussian kernel.
        truncate (float): Truncate the filter at this many standard deviations.

    Returns:
        mx.array: 1D Gaussian kernel normalized to sum to 1.
    """
    radius = int(truncate * sigma + 0.5)
    x = mx.arange(-radius, radius + 1, dtype=mx.float32)
    kernel = mx.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / mx.sum(kernel)
    return kernel


def convolve_separable_mlx(input_arr, kernels, axes, mode="constant"):
    """Apply separable convolution along specified axes using MLX.

    Args:
        input_arr (mx.array): Input array.
        kernels (list of mx.array): List of 1D kernels, one per axis.
        axes (list of int): Axes along which to apply convolution.
        mode (str): Padding mode ('constant' for zero-padding).

    Returns:
        mx.array: Convolved array.
    """
    result = input_arr

    for kernel, axis in zip(kernels, axes):
        if len(kernel) <= 1:  # Skip if kernel is too small
            continue

        # Get array shape
        ndim = len(result.shape)
        axis = axis % ndim  # Handle negative axes

        # Pad the array along the axis
        pad_width = len(kernel) // 2
        pad_list = [(0, 0)] * ndim
        pad_list[axis] = (pad_width, pad_width)

        # Apply zero padding
        if mode == "constant":
            result = mx.pad(result, pad_list, constant_values=0.0)

        # For 1D convolution along an axis, use sliding windows
        # Move axis to position 0 for easier processing
        result = mx.moveaxis(result, axis, 0)
        # mx.as_strided assumes contiguous layout for the provided shape order.
        # Materialize this transposed view to keep strided windows correct.
        result = mx.array(result, dtype=mx.float32)
        original_shape = result.shape

        # Apply convolution using a strided window view + weighted reduction.
        # This avoids building a Python list of slices and a large mx.stack.
        kernel_flipped = kernel[::-1]
        kernel_size = len(kernel)

        # Create output array shape
        out_len = original_shape[0] - kernel_size + 1
        rest_shape = original_shape[1:]

        # Build a contiguous-layout element-stride model for the moved axis view.
        elem_strides = np.empty(len(original_shape), dtype=np.int64)
        running = 1
        for d in range(len(original_shape) - 1, -1, -1):
            elem_strides[d] = running
            running *= int(original_shape[d])

        # windows shape: (out_len, kernel_size, *rest_shape)
        win_shape = [int(out_len), int(kernel_size), *[int(s) for s in rest_shape]]
        win_strides = [int(elem_strides[0]), int(elem_strides[0]), *[int(s) for s in elem_strides[1:]]]
        windows = mx.as_strided(result, shape=win_shape, strides=win_strides)

        # Broadcast kernel over all non-window dimensions and reduce over window axis.
        kernel_shape = [1, int(kernel_size)] + [1] * len(rest_shape)
        weighted = windows * mx.reshape(kernel_flipped, kernel_shape)
        result = mx.sum(weighted, axis=1)

        # Move axis back to original position
        result = mx.moveaxis(result, 0, axis)

    return result


def gaussian_filter_mlx(input_arr, sigma, axes=None, mode="constant", truncate=4.0):
    """Apply Gaussian filter using separable convolution in MLX.

    Args:
        input_arr (mx.array): Input array.
        sigma (float or list): Standard deviation(s) for Gaussian kernel.
            If a single float, use the same sigma for all axes.
            If a list, must match the length of axes (or ndim if axes is None).
        axes (list of int, optional): Axes along which to apply filter.
            If None, apply to all axes with non-zero sigma.
        mode (str): Padding mode ('constant' for zero-padding).
        truncate (float): Truncate the filter at this many standard deviations.

    Returns:
        mx.array: Filtered array.
    """
    if not isinstance(input_arr, mx.array):
        input_arr = mx.array(input_arr, dtype=mx.float32)

    # Handle sigma as scalar or sequence
    if np.isscalar(sigma):
        sigma = [sigma] * input_arr.ndim
    else:
        sigma = list(sigma)

    # Pad sigma list if needed
    if len(sigma) < input_arr.ndim:
        sigma = sigma + [0.0] * (input_arr.ndim - len(sigma))

    # Determine which axes to filter
    if axes is None:
        axes = [i for i, s in enumerate(sigma) if s > 0]
        sigma = [sigma[i] for i in axes]
    else:
        sigma = [sigma[i] for i in axes]

    # Generate kernels for each axis
    kernels = [gaussian_kernel_1d(s, truncate=truncate) if s > 0 else mx.array([1.0]) for s in sigma]

    # Apply separable convolution
    result = convolve_separable_mlx(input_arr, kernels, axes, mode=mode)

    return result


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


def pad_reflect_mlx(arr, pad_width):
    """
    Reflect-pad an MLX array using MLX operations.

    In reflect mode, the input is reflected at the edges (excluding the edge pixel).
    Example for 1D: [a, b, c, d] with pad=(2,2) becomes [c, b | a, b, c, d | c, b]

    Args:
        arr (mx.array): Input array.
        pad_width (list of tuples): Padding specification, e.g., [(2,2)] for 1D, [(1,1), (2,2)] for 2D.

    Returns:
        mx.array: Padded array.
    """
    result = arr

    for axis_idx, (pad_before, pad_after) in enumerate(pad_width):
        if pad_before == 0 and pad_after == 0:
            continue

        ndim = len(result.shape)
        axis = axis_idx % ndim
        axis_size = result.shape[axis]

        # For reflect mode, we can only reflect if axis_size >= 2
        # If axis_size < 2, fall back to edge mode (repeat edge values)
        if axis_size < 2:
            # Edge mode: repeat the boundary value
            if pad_before > 0:
                slices_before = [slice(None)] * ndim
                slices_before[axis] = slice(0, 1)
                edge_before = result[tuple(slices_before)]
                # Repeat along axis
                repeats = [1] * ndim
                repeats[axis] = pad_before
                edge_before = mx.tile(edge_before, repeats)
                result = mx.concatenate([edge_before, result], axis=axis)
                axis_size += pad_before  # Update axis_size after concatenation
            if pad_after > 0:
                slices_after = [slice(None)] * ndim
                slices_after[axis] = slice(-1, None)
                edge_after = result[tuple(slices_after)]
                # Repeat along axis
                repeats = [1] * ndim
                repeats[axis] = pad_after
                edge_after = mx.tile(edge_after, repeats)
                result = mx.concatenate([result, edge_after], axis=axis)
        else:
            # Reflect mode
            # Build lists of parts to concatenate
            parts = []

            if pad_before > 0:
                # Reflect from the start (excluding index 0)
                # scipy 'reflect' mode reflects about the edge: [a, b, c, d] with pad=2 gives [c, b, a, b, c, d]
                num_reflect = min(pad_before, axis_size)
                # Create indices: [num_reflect-1, num_reflect-2, ..., 0]
                indices = mx.arange(num_reflect - 1, -1, -1, dtype=mx.int32)
                parts.append(mx.take(result, indices, axis=axis))
                # If we need more padding than available, repeat edge
                if pad_before > num_reflect:
                    extra = pad_before - num_reflect
                    slices_edge = [slice(None)] * ndim
                    slices_edge[axis] = slice(0, 1)
                    edge = result[tuple(slices_edge)]
                    repeats = [1] * ndim
                    repeats[axis] = extra
                    parts.append(mx.tile(edge, repeats))

            parts.append(result)

            if pad_after > 0:
                # Reflect from the end (excluding the last index)
                # For array [..., a, b, c, d] (indices ..., 0,1,2,3), reflecting with pad=2 gives [..., a,b,c,d, c, b]
                num_reflect = min(pad_after, axis_size)
                # Create indices: [axis_size-2, axis_size-3, ..., axis_size-num_reflect]
                # For axis_size=4, num_reflect=2: [2, 1] (elements c, b)
                indices = mx.arange(axis_size - 2, axis_size - num_reflect - 2, -1, dtype=mx.int32)
                parts.append(mx.take(result, indices, axis=axis))
                # If we need more padding than available, repeat edge
                if pad_after > num_reflect:
                    extra = pad_after - num_reflect
                    slices_edge = [slice(None)] * ndim
                    slices_edge[axis] = slice(-1, None)
                    edge = result[tuple(slices_edge)]
                    repeats = [1] * ndim
                    repeats[axis] = extra
                    parts.append(mx.tile(edge, repeats))

            result = mx.concatenate(parts, axis=axis)

    return result


def convolve_separable_mlx_reflect(input_arr, kernels, axes, truncate=4.0):
    """
    Apply separable convolution with reflect padding along specified axes using MLX.

    Args:
        input_arr (mx.array): Input array.
        kernels (list of mx.array): List of 1D kernels, one per axis.
        axes (list of int): Axes along which to apply convolution.
        truncate (float): Truncate parameter (used to compute padding size).

    Returns:
        mx.array: Convolved array with same shape as input.
    """
    result = input_arr

    for kernel, axis in zip(kernels, axes):
        if len(kernel) <= 1:  # Skip if kernel is too small
            continue

        # Get array shape
        ndim = len(result.shape)
        axis = axis % ndim  # Handle negative axes

        kernel_size = len(kernel)
        pad_width = kernel_size // 2

        # Create padding specification
        pad_list = [(0, 0)] * ndim
        pad_list[axis] = (pad_width, pad_width)

        # Apply reflect padding
        padded = pad_reflect_mlx(result, pad_list)

        # For 1D convolution along an axis, use sliding windows
        # Move axis to position 0 for easier processing
        padded = mx.moveaxis(padded, axis, 0)
        # mx.as_strided assumes contiguous layout for the provided shape order.
        # Materialize this transposed view to keep strided windows correct.
        padded = mx.array(padded, dtype=mx.float32)
        original_shape = padded.shape

        # Apply convolution using a strided window view + weighted reduction.
        # This avoids building a Python list of slices and a large mx.stack.
        kernel_flipped = kernel[::-1]

        out_len = original_shape[0] - kernel_size + 1
        rest_shape = original_shape[1:]

        # Build a contiguous-layout element-stride model for the moved axis view.
        elem_strides = np.empty(len(original_shape), dtype=np.int64)
        running = 1
        for d in range(len(original_shape) - 1, -1, -1):
            elem_strides[d] = running
            running *= int(original_shape[d])

        # windows shape: (out_len, kernel_size, *rest_shape)
        win_shape = [int(out_len), int(kernel_size), *[int(s) for s in rest_shape]]
        win_strides = [int(elem_strides[0]), int(elem_strides[0]), *[int(s) for s in elem_strides[1:]]]
        windows = mx.as_strided(padded, shape=win_shape, strides=win_strides)

        # Broadcast kernel over all non-window dimensions and reduce over window axis.
        kernel_shape = [1, int(kernel_size)] + [1] * len(rest_shape)
        weighted = windows * mx.reshape(kernel_flipped, kernel_shape)
        result = mx.sum(weighted, axis=1)

        # Move axis back to original position
        result = mx.moveaxis(result, 0, axis)

    return result


def gaussian_filter_mlx_reflect(input_arr, sigma, axes=None, truncate=4.0):
    """
    Apply Gaussian filter with reflect padding using separable convolution in MLX.

    Args:
        input_arr (mx.array): Input array.
        sigma (float or list): Standard deviation(s) for Gaussian kernel.
        axes (list of int, optional): Axes along which to apply filter.
            If None, apply to all axes with non-zero sigma.
        truncate (float): Truncate the filter at this many standard deviations.

    Returns:
        mx.array: Filtered array with same shape as input.
    """
    if not isinstance(input_arr, mx.array):
        input_arr = mx.array(input_arr, dtype=mx.float32)

    # Handle sigma as scalar or sequence
    if np.isscalar(sigma):
        sigma = [sigma] * input_arr.ndim
    else:
        sigma = list(sigma)

    # Pad sigma list if needed
    if len(sigma) < input_arr.ndim:
        sigma = sigma + [0.0] * (input_arr.ndim - len(sigma))

    # Determine which axes to filter
    if axes is None:
        axes = [i for i, s in enumerate(sigma) if s > 0]
        sigma = [sigma[i] for i in axes]
    else:
        sigma = [sigma[i] for i in axes]

    # Generate kernels for each axis
    kernels = [gaussian_kernel_1d(s, truncate=truncate) if s > 0 else mx.array([1.0]) for s in sigma]

    # Apply separable convolution with reflect padding
    result = convolve_separable_mlx_reflect(input_arr, kernels, axes, truncate=truncate)

    return result


def dogfilter_mlx(vol, sigma_low=1, sigma_high=4):
    """
    GPU-accelerated Difference of Gaussians filter with reflect padding.

    Args:
        vol (mx.array or np.ndarray): Input volume.
        sigma_low (scalar or sequence): Standard deviation(s) for low-pass Gaussian.
        sigma_high (scalar or sequence): Standard deviation(s) for high-pass Gaussian.

    Returns:
        mx.array: DoG filtered volume.
    """
    if not isinstance(vol, mx.array):
        vol = mx.array(vol, dtype=mx.float32)

    # Apply Gaussian filters and compute difference
    # Note: Compilation was causing incorrect results, so we use direct evaluation
    out_low = gaussian_filter_mlx_reflect(vol, sigma_low, truncate=5.0)
    out_high = gaussian_filter_mlx_reflect(vol, sigma_high, truncate=5.0)

    return out_low - out_high


def periodic_smooth_decomposition_nd_rfft(img):
    """
    Decompose ND arrays of 2D images into periodic plus smooth components. This can help with edge artifacts in
    Fourier transforms.

    Args:
        img (mlx.array): input image or volume. The last two axes are treated as the image dimensions.

    Returns:
        mlx.array: periodic component
    """
    if isinstance(img, mx.array):
        img_mx = img.astype(mx.float32)

        # Border-difference on last two dimensions using broadcast masks.
        M, N = img_mx.shape[-2:]
        top_row = img_mx[..., -1, :] - img_mx[..., 0, :]  # (..., N)
        side_delta = img_mx[..., :, -1] - img_mx[..., :, 0]  # (..., M)

        y = mx.arange(M, dtype=mx.float32).reshape(M, 1)
        x = mx.arange(N, dtype=mx.float32).reshape(1, N)
        y_edge = (y == 0).astype(mx.float32) - (y == (M - 1)).astype(mx.float32)  # (M, 1)
        x_edge = (x == 0).astype(mx.float32) - (x == (N - 1)).astype(mx.float32)  # (1, N)

        B = y_edge * top_row[..., None, :] + x_edge * side_delta[..., :, None]

        # Real FFT of border difference.
        B_rfft = mx.fft.rfftn(B, axes=(-2, -1))

        # Denominator for periodic-smooth decomposition.
        q = mx.arange(M, dtype=mx.float32).reshape(M, 1)
        r = mx.arange(N, dtype=mx.float32).reshape(1, N)
        denom_full = 2 * mx.cos(2 * mx.pi * q / M) + 2 * mx.cos(2 * mx.pi * r / N) - 4
        Nh = N // 2 + 1
        denom_half = denom_full[:, :Nh]

        # Avoid divide-by-zero at DC term and enforce zero smooth DC.
        qh = mx.arange(M, dtype=mx.float32).reshape(M, 1)
        rh = mx.arange(Nh, dtype=mx.float32).reshape(1, Nh)
        dc_mask = (qh == 0) & (rh == 0)
        denom_half = mx.where(dc_mask, mx.array(1.0, dtype=mx.float32), denom_half)

        B_rfft = B_rfft / denom_half
        keep_mask = mx.where(dc_mask, mx.array(0.0, dtype=mx.float32), mx.array(1.0, dtype=mx.float32))
        B_rfft = B_rfft * keep_mask

        # Inverse FFT back to spatial; periodic component = img - smooth.
        smooth = mx.fft.irfftn(B_rfft, s=(M, N), axes=(-2, -1))
        return img_mx - smooth

    # CPU fallback for NumPy inputs.
    img_np = np.array(img, dtype="float32")

    B = np.zeros_like(img_np)
    B[..., 0, :] = img_np[..., -1, :] - img_np[..., 0, :]
    B[..., -1, :] = -B[..., 0, :]
    B[..., :, 0] += img_np[..., :, -1] - img_np[..., :, 0]
    B[..., :, -1] -= img_np[..., :, -1] - img_np[..., :, 0]

    B_rfft = np.fft.rfftn(B, axes=(-2, -1))
    M, N = img_np.shape[-2:]
    q = np.arange(M, dtype="float32").reshape(M, 1)
    r = np.arange(N, dtype="float32").reshape(1, N)
    denom_full = 2 * np.cos(2 * np.pi * q / M) + 2 * np.cos(2 * np.pi * r / N) - 4
    denom_half = denom_full[:, : (N // 2 + 1)]
    denom_half[0, 0] = 1
    B_rfft /= denom_half
    B_rfft[..., 0, 0] = 0

    smooth = np.fft.irfftn(B_rfft, s=(M, N), axes=(-2, -1))
    return img_np - smooth


def gausskernel_sheared(sigma, shear=0, truncate=3):
    """Create Gaussian window of a given shape and sigma. The window is sheared along the first two axes.
    GPU-accelerated using MLX for array operations.

    Args:
        sigma (float or tuple of float): Standard deviation for Gaussian kernel.
        shear (float): Shear factor in d_axis0 / d_axis1
        truncate (float): Truncate the filter at this many standard deviations.

    Returns:
        window (mx.array): 2D Gaussian kernel with shear applied on GPU
    """
    # Convert inputs to MLX for GPU computation
    sigma_y = mx.array(sigma[0], dtype=mx.float32)
    sigma_x = mx.array(sigma[1], dtype=mx.float32)
    shear_val = mx.array(shear, dtype=mx.float32)
    truncate_val = mx.array(truncate, dtype=mx.float32)

    # Calculate shape on GPU
    shape_y = mx.ceil(sigma_y * truncate_val * 2)
    shape_y = mx.maximum(shape_y, mx.ceil(mx.abs(shear_val * sigma_x * truncate_val * 2)))
    shape_x = mx.ceil(sigma_x * truncate_val * 2)

    # Make shapes odd and convert to int
    shape_y_int = int(mx.array((shape_y.astype(mx.int32) // 2) * 2 + 1))
    shape_x_int = int(mx.array((shape_x.astype(mx.int32) // 2) * 2 + 1))

    # Create coordinate grids on GPU
    yy = mx.arange(shape_y_int, dtype=mx.float32) - shape_y_int // 2
    xx = mx.arange(shape_x_int, dtype=mx.float32) - shape_x_int // 2

    # Normalize by sigma
    yy = yy / sigma_y
    xx = xx / sigma_x

    # Create 2D grids
    yy_grid = mx.reshape(yy, (-1, 1)) * mx.ones((1, shape_x_int), dtype=mx.float32)
    xx_grid = mx.ones((shape_y_int, 1), dtype=mx.float32) * mx.reshape(xx, (1, -1))

    # Apply shear to y: y' = y + shear * x * (sigma_x / sigma_y)
    yy_sheared = yy_grid + shear_val * xx_grid * (sigma_x / sigma_y)

    # Compute Gaussian
    kernel = mx.exp(-(yy_sheared**2 + xx_grid**2) / 2)

    # Normalize
    kernel = kernel / mx.sum(kernel)

    return kernel


def ndwindow(shape, window_func):
    """Create a n-dimensional window function (GPU-accelerated using MLX)

    Args:
        shape (tuple): shape of the window
        window_func (function): window function to be applied to each dimension

    Returns:
        window (mx.array): n-dimensional window computed on GPU
    """
    # Start with scalar 1 on GPU
    out = mx.array(1.0, dtype=mx.float32)

    for i in range(len(shape)):
        # Call window function (returns numpy array for 1D window)
        w_np = window_func(shape[i])

        # Convert to MLX and ensure float32
        w = mx.array(w_np, dtype=mx.float32)

        # Create new shape with all 1s except at dimension i
        newshape = [1] * len(shape)
        newshape[i] = shape[i]

        # Reshape and multiply on GPU
        w_reshaped = mx.reshape(w, newshape)
        out = out * w_reshaped

    return out


def accumarray(coords, shape, weights=None, clip=False):
    """Accumulate values into an array using given coordinates and weights

    Args:
        coords (array_like): 3-by-n array of coordinates
        shape (tuple): shape of the output array
        weights (array_like): weights to be accumulated. If None, all weights are set to 1
        clip (bool): if True, clip coordinates to the shape of the output array, else ignore out-of-bounds coordinates. Default is False.

    Returns:
        accum (array_like): accumulated array of the given shape
    """
    assert coords.shape[0] == 3
    coords = np.round(coords.reshape(3, -1)).astype("int")
    if clip:
        for d in range(len(shape)):
            coords[d] = np.minimum(np.maximum(coords[d], 0), shape[d] - 1)
    else:
        valid_ix = np.all((coords >= 0) & (coords < np.array(shape)[:, None]), axis=0)
        coords = coords[:, valid_ix]
        if weights is not None:
            weights = weights.ravel()[valid_ix]
    coords_as_ix = np.ravel_multi_index((*coords,), shape).ravel()
    accum = np.bincount(coords_as_ix, minlength=np.prod(shape), weights=weights)
    accum = accum.reshape(shape)
    return accum


def infill_nans(arr, sigma=0.5, truncate=50, ignore_warning=True):
    """Infill NaNs in an array using Gaussian basis interpolation

    Args:
        arr (array_like): input array
        sigma (float): standard deviation of the Gaussian basis function
        truncate (float): truncate the filter at this many standard deviations. Note: values outside the truncation may still contain NaNs.
        ignore_warning (bool): if True, ignore warnings about invalid values during division
    """
    nans = np.isnan(arr)
    arr_zeros = arr.copy()
    arr_zeros[nans] = 0
    a = scipy.ndimage.gaussian_filter(np.array(arr_zeros, dtype="float64"), sigma=sigma, truncate=truncate)
    b = scipy.ndimage.gaussian_filter(np.array(~nans, dtype="float64"), sigma=sigma, truncate=truncate)
    if ignore_warning:
        with np.errstate(invalid="ignore"):
            out = (a / b).astype(arr.dtype)
    else:
        out = (a / b).astype(arr.dtype)
    return out


def sliding_block(data, block_size=100, block_stride=1):
    """Create a sliding window/block view into the array with the given block shape and stride. The block slides across all dimensions of the array and extracts subsets of the array at all positions.

    Args:
        data (array_like): Array to create the sliding window view from
        block_size (int or tuple of int): Size of window over each axis that takes part in the sliding block
        block_stride (int or tuple of int): Stride of teh window along each axis

    Returns:
        view (ndarray): Sliding block view of the array.

    See Also:
        numpy.lib.stride_tricks.sliding_window_view
        numpy.lib.stride_tricks.as_strided

    """
    # MLX backend path: always operate on MLX arrays.
    if not isinstance(data, mx.array):
        data_mx = mx.array(data, dtype=mx.float32)
    else:
        data_mx = data

    # Use mlx.as_strided view (strides are in element units).
    ndim = data_mx.ndim
    data_shape = np.array(data_mx.shape, dtype=np.int32)

    block_stride = np.array(block_stride, dtype=np.int32) * np.ones(ndim, dtype=np.int32)
    block_size = np.array(block_size, dtype=np.int32) * np.ones(ndim, dtype=np.int32)

    if np.any(block_size <= 0):
        raise ValueError("block_size must be positive for all dimensions")
    if np.any(block_stride <= 0):
        raise ValueError("block_stride must be positive for all dimensions")
    if np.any(block_size > data_shape):
        raise ValueError("block_size must not exceed data shape")

    blocks_shape = 1 + (data_shape - block_size) // block_stride
    out_shape = np.r_[blocks_shape, block_size].astype(np.int32)

    # mlx.as_strided expects strides in element units, assuming row-contiguous layout.
    elem_strides = np.empty(ndim, dtype=np.int64)
    running = 1
    for d in range(ndim - 1, -1, -1):
        elem_strides[d] = running
        running *= int(data_shape[d])
    out_strides = np.r_[block_stride * elem_strides, elem_strides].astype(np.int64)

    return mx.as_strided(data_mx, shape=out_shape, strides=out_strides)


def fftfreq_mlx(n, d=1.0):
    """
    GPU-accelerated FFT frequency computation using MLX.

    Computes frequency bins in the same way as numpy.fft.fftfreq:
    returns f = [0, 1, ..., ceil(n/2)-1, -floor(n/2), ..., -1] / (d*n)

    Args:
        n (int): Window length.
        d (float): Sample spacing (inverse of sampling frequency).

    Returns:
        mx.array: Array of length n containing the frequency bins.
    """
    # Create indices [0, 1, 2, ..., n-1]
    indices = mx.arange(n, dtype=mx.float32)

    # Shift indices > n//2 by subtracting n
    # This gives [0, 1, ..., n//2-1, -n//2, -n//2+1, ..., -1]
    mask = indices >= (n + 1) // 2
    indices = mx.where(mask, indices - n, indices)

    # Normalize by sample spacing and length
    return indices / (d * n)


def upsampled_dft_rfftn(data, upsampled_region_size, upsample_factor: int = 1, axis_offsets=None):
    """
    Performs an upsampled inverse DFT on a small region around given offsets,
    taking as input the output of numpy.fft.rfftn (real-to-complex FFT).

    This implements the Guizar‑Sicairos local DFT upsampling: no full zero‑padding,
    just a small m×n patch at subpixel resolution.

    GPU-accelerated version using MLX for all computations.

    Args:
        data: A real-to-complex FFT array of shape (..., M, Nf),
            where Nf = N//2 + 1 corresponds to an original real image width N.
            Can be numpy or MLX array.
        upsampled_region_size: Size of the output patch (m, n). If an int is
            provided, the same size is used for both dimensions.
        upsample_factor: The integer upsampling factor in each axis.
        axis_offsets: The center of the patch in original-pixel coordinates
            (off_y, off_x). If None, defaults to (0, 0).

    Returns:
        A real-valued array of shape (..., m, n) containing the
        upsampled inverse DFT patch (magnitude of complex result).
    """
    was_mlx = isinstance(data, mx.array)

    # Convert to MLX array if needed
    if not was_mlx:
        data_mx = mx.array(data, dtype=mx.complex64)
    else:
        data_mx = data

    if data_mx.ndim < 2:
        raise ValueError("Input must have at least 2 dimensions")

    *batch_shape, M, Nf = data_mx.shape
    batch_size = int(np.prod(batch_shape)) if batch_shape else 1

    # determine patch size
    if isinstance(upsampled_region_size, int):
        m, n = upsampled_region_size, upsampled_region_size
    else:
        m, n = upsampled_region_size

    # full width of original image
    N = (Nf - 1) * 2

    # default offset: origin
    off_y, off_x = (0.0, 0.0) if axis_offsets is None else axis_offsets

    # Reshape data to [batch_size, M, Nf]
    data_mx = mx.reshape(data_mx, (batch_size, M, Nf))

    # Reconstruct full complex FFT via Hermitian symmetry on GPU
    # Create full FFT: [batch_size, M, N] (complex)
    full_real = mx.zeros((batch_size, M, N), dtype=mx.float32)
    full_imag = mx.zeros((batch_size, M, N), dtype=mx.float32)

    # Copy the rfft part (first Nf columns)
    full_real = mx.concatenate([mx.real(data_mx), mx.zeros((batch_size, M, N - Nf), dtype=mx.float32)], axis=2)
    full_imag = mx.concatenate([mx.imag(data_mx), mx.zeros((batch_size, M, N - Nf), dtype=mx.float32)], axis=2)

    # Fill in the conjugate symmetric part [Nf:N]
    # For 2D Hermitian symmetry, we need to flip both dimensions
    # data[..., M, Nf] corresponds to full[..., M, Nf]
    # data[..., i, j] (for 1 <= i < M-1, 1 <= j < Nf-1)
    # corresponds to full[..., M-i, N-j] conjugated
    if Nf > 1:
        # Extract the interior of the rfft output (excluding DC and Nyquist)
        # Shape: [batch, M, Nf-2]
        tail = data_mx[:, :, 1:-1]
        # Flip along both row and column axes
        # tail_flipped[b, i, j] = tail[b, M-1-i, Nf-2-j]
        tail_flipped = tail[:, ::-1, ::-1]
        # Conjugate: negate imaginary part
        tail_flipped_conj_real = mx.real(tail_flipped)
        tail_flipped_conj_imag = -mx.imag(tail_flipped)

        # Place in full FFT at positions [Nf:N]
        # The flipped and conjugated tail goes from index Nf to N
        full_real = mx.concatenate([full_real[:, :, :Nf], tail_flipped_conj_real], axis=2)
        full_imag = mx.concatenate([full_imag[:, :, :Nf], tail_flipped_conj_imag], axis=2)

    # Frequency coordinates (on GPU)
    fy = fftfreq_mlx(M)  # shape (M,)
    fx = fftfreq_mlx(N)  # shape (N,)

    # Sample coordinates around offsets
    y_idx = mx.arange(m, dtype=mx.float32) - (m // 2)
    x_idx = mx.arange(n, dtype=mx.float32) - (n // 2)

    # Compute phase angles (avoid explicit complex exponentials)
    # For batch processing: off_y, off_x should be arrays of shape (batch_size,)
    if isinstance(off_y, mx.array):
        # Already an MLX array, ensure 1D
        off_y = mx.reshape(off_y, (-1,))
    elif isinstance(off_y, (list, np.ndarray)):
        off_y = mx.array(off_y, dtype=mx.float32)
    else:
        off_y = mx.array([off_y] * batch_size, dtype=mx.float32)

    if isinstance(off_x, mx.array):
        # Already an MLX array, ensure 1D
        off_x = mx.reshape(off_x, (-1,))
    elif isinstance(off_x, (list, np.ndarray)):
        off_x = mx.array(off_x, dtype=mx.float32)
    else:
        off_x = mx.array([off_x] * batch_size, dtype=mx.float32)

    # Phase angles: ky[b, i, j] = 2π * (off_y[b] + y_idx[i] / upsample_factor) * fy[j]
    # Shape: [batch_size, m, M]
    y_coords = off_y[:, None] + y_idx[None, :] / upsample_factor  # [batch, m]
    phase_y = 2 * mx.pi * y_coords[:, :, None] * fy[None, None, :]  # [batch, m, M]

    # Shape: [batch_size, n, N]
    x_coords = off_x[:, None] + x_idx[None, :] / upsample_factor  # [batch, n]
    phase_x = 2 * mx.pi * x_coords[:, :, None] * fx[None, None, :]  # [batch, n, N]

    # Compute cos and sin for phase kernels (more efficient than complex exponentials)
    # Real parts of kernels
    ky_real = mx.cos(phase_y)  # [batch, m, M]
    ky_imag = mx.sin(phase_y)  # [batch, m, M]

    kx_real = mx.cos(phase_x)  # [batch, n, N]
    kx_imag = mx.sin(phase_x)  # [batch, n, N]

    # First matrix multiplication: out1 = ky @ full (complex matmul)
    # ky[b] shape: (m, M), full[b] shape: (M, N)
    # (a + ib) @ (c + id) = (a@c - b@d) + i(a@d + b@c)
    out1_real = mx.matmul(ky_real, full_real) - mx.matmul(ky_imag, full_imag)
    out1_imag = mx.matmul(ky_real, full_imag) + mx.matmul(ky_imag, full_real)
    # out1_real, out1_imag shape: [batch, m, N]

    # Second matrix multiplication: patch = out1 @ kx.T (complex matmul)
    # Keep this consistent with the CuPy reference path, which does:
    #   einsum("b m N, b n N -> b m n", out1, kx)
    # i.e., no conjugation on kx.
    kx_real_T = mx.transpose(kx_real, (0, 2, 1))
    kx_imag_T = mx.transpose(kx_imag, (0, 2, 1))

    patch_real = mx.matmul(out1_real, kx_real_T) - mx.matmul(out1_imag, kx_imag_T)
    patch_imag = mx.matmul(out1_imag, kx_real_T) + mx.matmul(out1_real, kx_imag_T)
    # patch_real, patch_imag shape: [batch, m, n]

    # Extract real part and reshape back to original batch shape
    result = patch_real
    result = mx.reshape(result, batch_shape + [m, n])

    return result


def zoom_chop_pad(
    arr, target_shape=None, scale=(1, 1, 1), soft_edge=(0, 0, 0), shift=(0, 0, 0), flip=(False, False, False), cval=0
):
    """Zooms, softens, flips, shifts, and pads/crops a 3D array to match the target shape.

    The conceptual order is as follows: zoom, soften, flip, shift, crop/pad.

    Args:
        arr (np.ndarray or mx.array): The input array to be transformed
        target_shape (tuple of int): The desired target shape to pad/crop to. Defaults to the shape of the input array.
        scale (tuple): Zoom factors for each axis. Default: (1, 1, 1).
        soft_edge (tuple of int): The size of the soft edge (Tukey envelope) to be applied to the input array, in voxels. Default: (0, 0, 0).
        shift (tuple): Shifts for each axis, in voxels. Default: (0, 0, 0).
        flip (tuple of bool): Whether to flip each axis. Default: (False, False, False).
        cval (int, float): The value to use for padding. Default: 0.

    Returns:
        np.ndarray or mx.array: The transformed array. Dtype is float32.
    """

    was_numpy = not isinstance(arr, mx.array)

    if target_shape is None:
        target_shape = arr.shape

    arr_np = np.array(arr, dtype="float32", copy=False)

    if any(s > 0 for s in soft_edge):
        scaled_edge = np.array(soft_edge) / np.array(scale)
        arr_np = soften_edges(arr_np, soft_edge=scaled_edge, copy=True)

    coords = np.indices(target_shape, dtype=np.float32)
    for i in range(len(coords)):
        coords[i] -= target_shape[i] / 2
        coords[i] /= scale[i]
        coords[i] += arr_np.shape[i] / 2
        if flip[i]:
            coords[i] *= -1
            coords[i] += arr_np.shape[i] - 1
        coords[i] -= shift[i]
    result = scipy.ndimage.map_coordinates(arr_np, coords, order=1, mode="constant", cval=cval)

    if not was_numpy:
        result = mx.array(result, dtype=mx.float32)
    return result


def soften_edges(arr, soft_edge=(0, 0, 0), copy=True):
    """Apply a soft Tukey edge to the input array.

    Args:
        arr (np.ndarray or mx.array): The input array
        soft_edge (tuple of int): The size of the soft edge (Tukey envelope) to be applied to the input array, in voxels. Default: (0, 0, 0).
        copy (bool): If True, a copy of the array is made. Default: True.

    Returns:
        np.ndarray or mx.array: The transformed array. Dtype is float32.
    """
    was_numpy = not isinstance(arr, mx.array)
    input_dtype = arr.dtype
    arr_np = np.array(arr, dtype="float32", copy=copy)

    if isinstance(soft_edge, (int, float)):
        soft_edge = (soft_edge,) * arr_np.ndim
    soft_edge = np.clip(soft_edge, 0, np.array(arr_np.shape) / 2)

    for i in range(arr_np.ndim):
        if soft_edge[i] > 0:
            alpha = 2 * soft_edge[i] / arr_np.shape[i]
            alpha = np.clip(alpha, 0, 1)
            win = scipy.signal.windows.tukey(arr_np.shape[i], alpha)
            arr_np *= np.moveaxis(win[:, None, None], 0, i)

    arr_np = arr_np.astype(input_dtype, copy=False)
    if not was_numpy:
        arr_np = mx.array(arr_np)
    return arr_np


def zoom(arr, zoom_factors, order=1, mode="constant"):
    """Zooms an array by given factors along each axis.

    Args:
        arr (np.ndarray or mx.array): The input array to be zoomed.
        zoom_factors (tuple of float): Zoom factors for each axis. Values greater than 1 result in a larger output array,
            while values less than 1 result in a smaller array. Divide the physical voxel size of the input array by these values to get the physical voxel size of the output array.
        order (int): The order of the spline interpolation. Default is 1 (linear).

    Returns:
        np.ndarray or mx.array: The zoomed array.
    """
    was_numpy = not isinstance(arr, mx.array)
    arr_np = np.array(arr, dtype="float32", copy=False)
    out = scipy.ndimage.zoom(arr_np, zoom_factors, order=order)
    if not was_numpy:
        out = mx.array(out, dtype=mx.float32)
    return out


def match_volumes(fixed, fixed_res, moving, moving_res, order=1, soft_edge=(0, 0, 0), cval=0, res_method="fixed"):
    """
    Rescale and pad the fixed and moving volumes so both have the same physical size and resolution.

    Args:
        fixed (ndarray): Fixed/reference volume.
        fixed_res (tuple): Pixel size (resolution) for fixed, in physical units.
        moving (ndarray): Moving volume.
        moving_res (tuple): Pixel size (resolution) for moving, in physical units.
        order (int): Interpolation order for zooming.
        soft_edge (tuple): Soft edge parameter for padding.
        cval (float): Constant value for padding.
        res_method (str): How to choose the target resolution.
            "fixed" (default): use fixed_res,
            "min": use the finest (smallest) resolution,
            "max": use the coarsest (largest) resolution,
            "mean": use the mean of fixed_res and moving_res.

    Returns:
        fixed_out (ndarray): The fixed volume, rescaled and padded to the target resolution and physical size.
        moving_out (ndarray): The moving volume, rescaled and padded to the target resolution and physical size.
        target_res (tuple): The target resolution used for both volumes.
    """
    fixed_res = np.array(fixed_res)
    moving_res = np.array(moving_res)
    fixed_shape = np.array(fixed.shape)
    moving_shape = np.array(moving.shape)

    # Compute physical sizes
    fixed_phys = fixed_shape * fixed_res
    moving_phys = moving_shape * moving_res

    # Target: match the larger physical size along each axis
    target_phys = np.maximum(fixed_phys, moving_phys)

    # Determine target resolution
    if res_method == "fixed":
        target_res = fixed_res
    elif res_method == "min":
        target_res = np.minimum(fixed_res, moving_res)
    elif res_method == "max":
        target_res = np.maximum(fixed_res, moving_res)
    elif res_method == "mean":
        target_res = (fixed_res + moving_res) / 2
    else:
        raise ValueError(f"Unknown target_res_type: {res_method}")

    # Compute the target shape
    target_shape = np.ceil(target_phys / target_res).astype(int)

    # Rescale fixed
    scale_fixed = fixed_res / target_res
    fixed_out = zoom_chop_pad(fixed, target_shape=target_shape, scale=scale_fixed, soft_edge=soft_edge, cval=cval)

    # Rescale moving
    scale_moving = moving_res / target_res
    moving_out = zoom_chop_pad(moving, target_shape=target_shape, scale=scale_moving, soft_edge=soft_edge, cval=cval)

    return fixed_out, moving_out, tuple(target_res)


def richardson_lucy_generic(
    img, convolve_psf, correlate_psf=None, num_iter=5, epsilon=1e-3, beta=0.0, initial_guess=None
):
    """Richardson-Lucy deconvolution using arbitrary convolution operations, with optional Biggs acceleration.

    Args:
        img (ArrayLike): input image or volume
        convolve_psf (Callable): function to convolve with PSF. Should take an image and return a convolved image. Ensure that the PSF is non-negative and normalized.
        correlate_psf (Callable): function to correlate with PSF. Defaults to convolve_psf (if PSF is symmetric)
        num_iter (int): number of iterations. Default is 5.
        epsilon (float): small constant to prevent divide-by-zero. Default is 1e-3.
        beta (float): acceleration parameter. Default is 0.0 (no acceleration). Typically, beta is in the range [0, 0.5].

    Returns:
        ndarray: deconvolved image
    """
    was_mlx = isinstance(img, mx.array)
    img_np = np.clip(np.array(img, dtype="float32"), 0, None)

    if num_iter < 1:
        result = mx.array(img_np) if was_mlx else img_np
        return result
    if correlate_psf is None:
        correlate_psf = convolve_psf

    if initial_guess is not None:
        assert initial_guess.shape == img_np.shape, "Initial guess must have the same shape as the input image."
        img_decon = np.clip(np.array(initial_guess, dtype="float32"), 0, None)
    else:
        img_decon = img_np.copy()
    img_decon += epsilon

    for i in range(num_iter):
        img_decon *= correlate_psf(img_np / (convolve_psf(img_decon) + epsilon))

        if beta > 0:
            if i == 0:
                img_decon_prev = img_decon.copy()
            else:
                img_decon_new = img_decon.copy()
                img_decon += beta * (img_decon - img_decon_prev)
                img_decon = np.clip(img_decon, epsilon, None)
                img_decon_prev = img_decon_new

    if was_mlx:
        img_decon = mx.array(img_decon, dtype=mx.float32)
    return img_decon


def richardson_lucy_fft(img, psf, num_iter=5, epsilon=1e-3, beta=0.0, initial_guess=None):
    """Richardson-Lucy deconvolution using FFT-based convolution and optional Biggs acceleration.

    Args:
        img (ndarray): input image or volume
        psf (ndarray): point spread function (before fftshift)
        num_iter (int): number of iterations
        epsilon (float): small constant to prevent divide-by-zero
        beta (float): acceleration parameter for Biggs acceleration
        initial_guess (ndarray): initial guess for the deconvolved image

    Returns:
        ndarray: deconvolved image
    """
    was_mlx = isinstance(img, mx.array)
    img_np = np.array(img, dtype="float32")
    psf_np = np.array(psf, dtype="float32")

    # Prepare PSF for convolution (normalize)
    psf_np = psf_np / psf_np.sum()
    psf_flip = psf_np[tuple(slice(None, None, -1) for _ in psf_np.shape)]

    def convolve_psf(x):
        return scipy.signal.convolve(x, psf_np, mode="same")

    def correlate_psf(x):
        return scipy.signal.correlate(x, psf_flip, mode="same")

    result = richardson_lucy_generic(img_np, convolve_psf, correlate_psf, num_iter, epsilon, beta, initial_guess)
    return result
