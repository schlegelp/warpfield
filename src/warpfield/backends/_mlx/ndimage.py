import scipy.ndimage

import numpy as np
import mlx.core as mx


def dogfilter(vol, sigma_low=1, sigma_high=4, mode="reflect"):
    """
    GPU-accelerated Difference of Gaussians filter with reflect padding.

    Args:
        vol (mx.array or np.ndarray): Input volume.
        sigma_low (scalar or sequence): Standard deviation(s) for low-pass Gaussian.
        sigma_high (scalar or sequence): Standard deviation(s) for high-pass Gaussian.
        mode (str): Padding mode for Gaussian filter ('reflect' supported).

    Returns:
        mx.array: DoG filtered volume.
    """
    if mode not in {"reflect", "constant"}:
        raise ValueError(f"Unsupported mode '{mode}'. Currently 'reflect' and 'constant' are supported.")

    if not isinstance(vol, mx.array):
        vol = mx.array(vol, dtype=mx.float32)

    # Match CuPy/SciPy DoG behavior: same mode and default Gaussian truncate.
    out_low = gaussian_filter(vol, sigma_low, mode=mode)
    out_high = gaussian_filter(vol, sigma_high, mode=mode)

    return out_low - out_high


def gaussian_filter(input_arr, sigma, axes=None, mode="reflect", truncate=4.0):
    """
    Apply Gaussian filter using separable convolution in MLX.

    Args:
        input_arr (mx.array): Input array.
        sigma (float or list): Standard deviation(s) for Gaussian kernel.
        axes (list of int, optional): Axes along which to apply filter.
            If None, apply to all axes with non-zero sigma.
        mode (str): Padding mode ('reflect' or 'constant').
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

    # Apply separable convolution with requested padding mode
    result = convolve_separable_mlx(input_arr, kernels, axes, mode=mode)

    return result


def gaussian_filter_mlx_reflect(input_arr, sigma, axes=None, truncate=4.0):
    """Backward-compatible wrapper for reflect-mode Gaussian filter."""
    return gaussian_filter(input_arr, sigma, axes=axes, mode="reflect", truncate=truncate)


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
        mode (str): Padding mode ('constant' for zero-padding, 'reflect' for mirror padding).

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

        # Apply mode-specific padding
        if mode == "constant":
            padded = mx.pad(result, pad_list, constant_values=0.0)
        elif mode == "reflect":
            padded = pad_reflect_mlx(result, pad_list)
        else:
            raise ValueError(f"Unsupported padding mode '{mode}'. Use 'constant' or 'reflect'.")

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
        kernel_size = len(kernel)

        # Create output array shape
        out_len = original_shape[0] - kernel_size + 1
        rest_shape = original_shape[1:]

        # Build a contiguous-layout element-stride model for the moved axis view.
        elem_strides = mx.zeros(len(original_shape), dtype=mx.int64)
        running = 1
        for d in range(len(original_shape) - 1, -1, -1):
            elem_strides[d] = running
            running *= int(original_shape[d])

        # windows shape: (out_len, kernel_size, *rest_shape)
        win_shape = [int(out_len), int(kernel_size), *[int(s) for s in rest_shape]]
        win_strides = [int(elem_strides[0]), int(elem_strides[0]), *[int(s) for s in elem_strides[1:]]]
        windows = mx.as_strided(padded, shape=win_shape, strides=win_strides)

        # Broadcast kernel over all non-window dimensions and reduce over window axis.
        # Chunk over out_len to avoid materializing a huge weighted tensor at once.
        kernel_shape = [1, int(kernel_size)] + [1] * len(rest_shape)
        kernel_broadcast = mx.reshape(kernel_flipped, kernel_shape)

        # Heuristic target for temporary weighted tensor size per chunk.
        # weighted chunk shape: (chunk, kernel_size, *rest_shape)
        target_chunk_bytes = 256 * 1024 * 1024
        rest_elems = 1
        for s in rest_shape:
            rest_elems *= int(s)
        bytes_per_chunk_row = int(kernel_size) * rest_elems * 4  # float32
        chunk_len = max(1, min(int(out_len), target_chunk_bytes // max(1, bytes_per_chunk_row)))

        conv_chunks = []
        for start in range(0, int(out_len), int(chunk_len)):
            stop = min(int(out_len), start + int(chunk_len))
            conv_chunk = mx.sum(windows[start:stop] * kernel_broadcast, axis=1)
            mx.eval(conv_chunk)  # <- DO NOT REMOVE THIS EVAL - otherwise we see huge memory spikes here due to lazy evaluation of the large intermediate weighted tensor
            conv_chunks.append(conv_chunk)

        result = conv_chunks[0] if len(conv_chunks) == 1 else mx.concatenate(conv_chunks, axis=0)

        # Move axis back to original position
        result = mx.moveaxis(result, 0, axis)
    return result


def pad_reflect_mlx(arr, pad_width):
    """
    Reflect-pad an MLX array using MLX operations.

    In reflect mode, the input is mirrored with edge duplication (SciPy/CuPy ndimage semantics).
    Example for 1D: [a, b, c, d] with pad=(2,2) becomes [b, a | a, b, c, d | d, c]

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

            # SciPy/CuPy ndimage reflect mode: half-sample symmetric with
            # edge duplication. Period for index mapping is 2 * axis_size.
            period = 2 * axis_size

            if pad_before > 0:
                # Positions left of the array: -pad_before..-1
                left_pos = mx.arange(-pad_before, 0, dtype=mx.int64)
                left_mod = left_pos % period
                left_idx = mx.where(left_mod < axis_size, left_mod, period - 1 - left_mod)
                parts.append(mx.take(result, left_idx.astype(mx.int32), axis=axis))

            parts.append(result)

            if pad_after > 0:
                # Positions right of the array: axis_size..axis_size+pad_after-1
                right_pos = mx.arange(axis_size, axis_size + pad_after, dtype=mx.int64)
                right_mod = right_pos % period
                right_idx = mx.where(right_mod < axis_size, right_mod, period - 1 - right_mod)
                parts.append(mx.take(result, right_idx.astype(mx.int32), axis=axis))

            result = mx.concatenate(parts, axis=axis)

    return result


def periodic_smooth_decomposition_nd_rfft(img):
    """
    Decompose ND arrays of 2D images into periodic plus smooth components. This can help with edge artifacts in
    Fourier transforms.

    Args:
        img (mlx.array): input image or volume. The last two axes are treated as the image dimensions.

    Returns:
        mlx.array: periodic component
    """
    if not isinstance(img, mx.array):
        img_mx = mx.array(img, dtype=mx.float32)
    else:
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


def gausswin(shape, sigma):
    """Create Gaussian window of a given shape and sigma

    Args:
        shape (list or tuple): shape along each dimension
        sigma (list or tuple): sigma along each dimension

    Returns:
        (array_like): Gauss window
    """
    grid = np.indices(shape).astype("float32")

    if isinstance(grid, np.ndarray):
        grid = mx.array(grid)
    else:
        grid = [mx.array(g) for g in grid]

    for dim in range(len(grid)):
        grid[dim] -= shape[dim] // 2
        grid[dim] /= sigma[dim]
    out = mx.exp(-(grid**2).sum(0) / 2)
    out /= out.sum()
    return out


def gausskernel_sheared(sigma, shear=0, truncate=3):
    """Create Gaussian window of a given shape and sigma. The window is sheared along the first two axes.

    Args:
        sigma (float or tuple of float): Standard deviation for Gaussian kernel.
        shear (float): Shear factor in d_axis0 / d_axis1
        truncate (float): Truncate the filter at this many standard deviations.

    Returns:
        window (array_like): n-dimensional window
    """
    # TODO: consider moving to .unshear

    shape = (np.r_[sigma] * truncate * 2).astype("int")
    shape[0] = np.maximum(shape[0], int(np.ceil(shape[1] * np.abs(shear))))
    shape = (shape // 2) * 2 + 1
    grid = np.indices(shape).astype("float32")
    if isinstance(grid, np.ndarray):
        grid = mx.array(grid)
    else:
        grid = [mx.array(g) for g in grid]
    for dim in range(len(grid)):
        grid[dim] -= shape[dim] // 2
        grid[dim] /= sigma[dim]
    grid[0] = grid[0] + shear * grid[1] * sigma[1] / sigma[0]
    out = mx.exp(-(grid**2).sum(0) / 2)
    out /= out.sum()
    return out


def ndwindow(shape, window_func):
    """Create a n-dimensional window function

    Args:
        shape (tuple): shape of the window
        window_func (function): window function to be applied to each dimension

    Returns:
        window (array_like): n-dimensional window
    """
    out = 1
    for i in range(len(shape)):
        newshape = np.ones(len(shape), dtype="int")
        newshape[i] = shape[i]
        w = mx.array(window_func(shape[i]))
        out = out * mx.reshape(w, newshape)
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


def fftfreq(n, d=1.0):
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

    This implements the Guizar-Sicairos local DFT upsampling: no full zero-padding,
    just a small mxn patch at subpixel resolution.

    GPU-accelerated version using MLX for all computations.

    Please note that because MLX does not support float64 on GPU, we have 2-3e-3 relative
    error compared to the double-precision CuPy implementation!

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
    # Copy the rfft part and fill conjugate symmetric part directly
    full_real = mx.real(data_mx)
    full_imag = mx.imag(data_mx)

    # Fill in the conjugate symmetric part [Nf:N] via Hermitian symmetry
    if Nf > 1:
        # Extract interior (excluding DC at 0 and Nyquist at Nf-1)
        tail = data_mx[:, :, 1:-1]  # shape: [batch, M, Nf-2]
        # Flip both dimensions and conjugate
        tail_flipped = tail[:, ::-1, ::-1]

        # Concatenate to form full spectrum and immediately force
        # evaluation for numerical stability and to avoid memory peaks
        full_real = mx.concatenate([full_real, mx.real(tail_flipped)], axis=2)
        full_imag = mx.concatenate([full_imag, -mx.imag(tail_flipped)], axis=2)
    else:
        # No conjugate symmetric part needed
        pass

    # Frequency coordinates (on GPU)
    fy = fftfreq(M, d=1.0)  # shape (M,)
    fx = fftfreq(N, d=1.0)  # shape (N,)

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
    # patch_imag = mx.matmul(out1_imag, kx_real_T) + mx.matmul(out1_real, kx_imag_T)
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

    if was_numpy:
        input_dtype = arr.dtype
    else:
        input_dtype = str(arr.dtype).split(".")[-1]  # e.g., "float32"

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
    was_numpy = isinstance(img, np.ndarray)

    # Keep everything on GPU with MLX arrays
    if was_numpy:
        img_mx = mx.array(img, dtype=mx.float32)
    else:
        img_mx = img
    img_mx = mx.clip(img_mx, 0, None)

    if num_iter < 1:
        result = img_mx if not was_numpy else np.array(img_mx)
        return result

    if correlate_psf is None:
        correlate_psf = convolve_psf

    if initial_guess is not None:
        assert initial_guess.shape == img_mx.shape, "Initial guess must have the same shape as the input image."
        if not isinstance(initial_guess, mx.array):
            img_decon = mx.array(initial_guess, dtype=mx.float32)
        else:
            img_decon = initial_guess
        img_decon = mx.clip(img_decon, 0, None)
    else:
        img_decon = mx.array(img_mx)
    img_decon = img_decon + epsilon

    for i in range(num_iter):
        img_decon = img_decon * correlate_psf(img_mx / (convolve_psf(img_decon) + epsilon))

        if beta > 0:
            if i == 0:
                img_decon_prev = mx.array(img_decon)
            else:
                img_decon_new = mx.array(img_decon)
                img_decon = img_decon + beta * (img_decon - img_decon_prev)
                img_decon = mx.clip(img_decon, epsilon, None)
                img_decon_prev = img_decon_new

    if was_numpy:
        img_decon = np.array(img_decon)
    return img_decon


def richardson_lucy_fft(img, psf, num_iter=5, epsilon=1e-3, beta=0.0, initial_guess=None):
    """Richardson-Lucy deconvolution using FFT-based convolution and optional Biggs acceleration.

    Args:
        img (ndarray): input image or volume
        psf (ndarray): point spread function (before fftshift)
        num_iter (int): number of iterations
        epsilon (float): small constant to avoid divide-by-zero
        beta (float): Biggs acceleration parameter (0 = no acceleration)

    Returns:
        ndarray: deconvolved image
    """
    psf = mx.array(psf, dtype=mx.float32)
    psf = mx.clip(psf, 0, None)
    psf = psf / psf.sum()

    shape = img.shape
    ndim = len(shape)
    axes = tuple(range(ndim))
    psf_ft = mx.fft.rfftn(mx.fft.ifftshift(psf), s=shape, axes=axes)
    psf_ft_conj = mx.conj(psf_ft)

    def convolve(x):
        return mx.fft.irfftn(mx.fft.rfftn(x, s=shape, axes=axes) * psf_ft, s=shape, axes=axes)

    def correlate(x):
        return mx.fft.irfftn(mx.fft.rfftn(x, s=shape, axes=axes) * psf_ft_conj, s=shape, axes=axes)

    out = richardson_lucy_generic(
        img, convolve, correlate, num_iter=num_iter, epsilon=epsilon, beta=beta, initial_guess=initial_guess
    )
    return out


def richardson_lucy_gaussian(img, sigmas, num_iter=5, epsilon=1e-3, beta=0.0, initial_guess=None):
    """Richardson-Lucy deconvolution using Gaussian convolution operations

    Args:
        img (ndarray): input image or volume
        sigmas (list or ndarray): list of Gaussian sigmas along each dimension
        num_iter (int): number of iterations
        epsilon (float): small constant to prevent divide-by-zero
        beta (float): acceleration parameter (0 = no acceleration)

    Returns:
        ndarray: deconvolved image
    """
    conv_with_gauss = lambda x: gaussian_filter(x, sigmas)
    out = richardson_lucy_generic(
        img, conv_with_gauss, num_iter=num_iter, epsilon=epsilon, beta=beta, initial_guess=initial_guess
    )
    return out


def richardson_lucy_gaussian_shear(img, sigmas, shear, num_iter=5, epsilon=1e-3, beta=0.0, initial_guess=None):
    """Richardson-Lucy deconvolution using a sheared Gaussian psf

    Args:
        img (ndarray): input image or volume
        sigmas (list or ndarray): list of Gaussian sigmas along each dimension
        shear (scalar): shear ratio
        num_iter (int): number of iterations
        epsilon (float): small constant to prevent divide-by-zero
        beta (float): acceleration parameter (0 = no acceleration)

    Returns:
        ndarray: deconvolved image
    """
    if shear == 0:
        return richardson_lucy_gaussian(img, sigmas, num_iter, epsilon, beta, initial_guess)

    sigmas = np.array(sigmas)
    gw = mx.array(gausskernel_sheared(sigmas, shear=shear, truncate=4), dtype=mx.float32)
    gw01 = gw.sum(2)[:, :, None]
    gw01 = gw01 / gw01.sum()
    gw2 = gw.sum(axis=(0, 1))[None, None, :]
    gw2 = gw2 / gw2.sum()

    # Convert kernels to numpy for spatial convolution.
    # Note: We use scipy.ndimage.convolve here (matches CuPy reference) rather than FFT
    # because scipy's reflect boundary mode is difficult to replicate exactly with FFT.
    # The Richardson-Lucy iteration loop stays GPU-bound; only the PSF convolution uses CPU.
    gw01_np = np.array(gw01)
    gw2_np = np.array(gw2)

    def conv_shear(vol):
        """Apply sheared convolution using scipy (matches CuPy reference exactly)."""
        if isinstance(vol, mx.array):
            vol_np = np.array(vol, dtype="float32")
        else:
            vol_np = np.array(vol, dtype="float32")

        # Apply two-stage separable convolution with reflect mode
        vol_np = scipy.ndimage.convolve(vol_np, gw01_np, mode='reflect')
        vol_np = scipy.ndimage.convolve(vol_np, gw2_np, mode='reflect')

        return mx.array(vol_np, dtype=mx.float32)
    out = richardson_lucy_generic(
        img, conv_shear, num_iter=num_iter, epsilon=epsilon, beta=beta, initial_guess=initial_guess
    )
    return out
