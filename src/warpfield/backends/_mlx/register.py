import warnings
import scipy.signal

import numpy as np
import mlx.core as mx

from tqdm.auto import tqdm
from typing import List, Union
from pydantic import BaseModel

from .warp import warp_volume_mlx
from .ndimage import (
    accumarray,
    dogfilter,
    infill_nans,
    ndwindow,
    periodic_smooth_decomposition_nd_rfft,
    sliding_block,
    soften_edges,
    gaussian_filter_mlx_reflect
)
from .utils import (
    map_coordinates,
    median_filter,
    smooth_func,
    _get_sample_displacement_channels_vmapped,
    _compute_displacement_for_one_projection,
    _get_compiled_vmapped_displacement_computer,
)
from ...base import WarpMapBase

_ArrayType = Union[np.ndarray, mx.array]


class WarpMapMlx(WarpMapBase):
    """Represents a 3D displacement field

    Args:
        warp_field (numpy.array): the displacement field data (3-x-y-z)
        block_size (3-element list or numpy.array):
        block_stride (3-element list or numpy.array):
        ref_shape (tuple): shape of the reference volume
        mov_shape (tuple): shape of the moving volume
    """

    @property
    def warp_field(self):
        return self._warp_field

    @warp_field.setter
    def warp_field(self, value):
        self._warp_field = mx.array(value, dtype=mx.float32)

    @property
    def block_size(self):
        return self._block_size

    @block_size.setter
    def block_size(self, value):
        self._block_size = mx.array(value, dtype=mx.float32)

    @property
    def block_stride(self):
        return self._block_stride

    @block_stride.setter
    def block_stride(self, value):
        self._block_stride = mx.array(value, dtype=mx.float32)

    def warp(self, vol, out=None):
        """Apply the warp to a volume. Can be thought of as pulling the moving volume to the fixed volume space.

        Args:
            vol (mx.array): the volume to be warped

        Returns:
            mx.array: warped volume
        """
        if np.any(vol.shape != np.array(self.mov_shape)):
            warnings.warn(f"Volume shape {vol.shape} does not match the expected shape {self.mov_shape}.")
        if out is None:
            out = mx.zeros(self.ref_shape, dtype=mx.float32)
        vol_out = warp_volume_mlx(
            vol, self.warp_field, self.block_stride, mx.array(-self.block_size / self.block_stride / 2), out=out
        )
        return vol_out

    def fit_affine(self, target=None):
        """Fit affine transformation and return new fitted WarpMap (fully GPU-bound)

        Args:
            target (dict): dict with keys "blocks_shape", "block_size", and "block_stride"

        Returns:
            WarpMap:
            numpy.array: affine tranformation coefficients
        """
        if target is None:
            warp_field_shape = self.warp_field.shape
            block_size_mx = self.block_size
            block_stride_mx = self.block_stride
        else:
            warp_field_shape = target["warp_field_shape"]
            block_size_mx = mx.array(target["block_size"], dtype=mx.float32)
            block_stride_mx = mx.array(target["block_stride"], dtype=mx.float32)

        # Create coordinate grid on GPU using MLX
        shape = self.warp_field.shape[1:]
        z = mx.arange(shape[0], dtype=mx.float32)
        y = mx.arange(shape[1], dtype=mx.float32)
        x = mx.arange(shape[2], dtype=mx.float32)
        zz, yy, xx = mx.meshgrid(z, y, x, indexing="ij")
        ix = mx.stack([zz.reshape(-1), yy.reshape(-1), xx.reshape(-1)], axis=1)  # (N, 3)
        ix = ix * self.block_stride[None, :] + self.block_size[None, :] / 2

        # Build linear system: A @ coeff = b (on GPU)
        ones = mx.ones((ix.shape[0], 1), dtype=mx.float32)
        a_mx = mx.concatenate([ix, ones], axis=1)  # (N, 4)

        # Target coordinates: original coords + displacement
        warp_field_flat = self.warp_field.reshape(3, -1).T  # (N, 3)
        b_mx = ix + warp_field_flat

        # Solve least squares using normal equations: coeff = (A.T @ A)^-1 @ A.T @ b
        # Matrix multiplications happen on GPU, solve uses CPU stream (MLX limitation)
        ata = a_mx.T @ a_mx  # (4, 4) - on GPU
        atb = a_mx.T @ b_mx  # (4, 3) - on GPU
        coeff_mx = mx.linalg.solve(ata, atb, stream=mx.cpu)  # (4, 3) - on CPU

        # Create output coordinate grid on GPU
        z_out = mx.arange(warp_field_shape[1], dtype=mx.float32)
        y_out = mx.arange(warp_field_shape[2], dtype=mx.float32)
        x_out = mx.arange(warp_field_shape[3], dtype=mx.float32)
        zz_out, yy_out, xx_out = mx.meshgrid(z_out, y_out, x_out, indexing="ij")
        ix_out = mx.stack([zz_out.reshape(-1), yy_out.reshape(-1), xx_out.reshape(-1)], axis=1)
        ix_out = ix_out * block_stride_mx[None, :] + block_size_mx[None, :] / 2

        # Apply affine transformation: displacement = (coords @ (coeff[:3] - I)) + coeff[3]
        eye = mx.eye(3, dtype=mx.float32)
        linfit_mx = (ix_out @ (coeff_mx[:3] - eye)) + coeff_mx[3:4]  # (N, 3)
        linfit_mx = linfit_mx.T.reshape(warp_field_shape)

        return (WarpMapMlx(linfit_mx, block_size_mx, block_stride_mx, self.ref_shape, self.mov_shape), coeff_mx)

    def median_filter(self):
        """Apply median filter to the displacement field

        Args:
            WarpMap: new WarpMap with median filtered displacement field
        """
        # Use GPU-accelerated median filter
        filtered = median_filter(self.warp_field, size=[1, 3, 3, 3], mode="nearest")
        return WarpMapMlx(filtered, self.block_size, self.block_stride, self.ref_shape, self.mov_shape)

    def resize_to(self, target):
        """Resize to target WarpMap, using linear interpolation

        Args:
            target (WarpMap or WarpMapper): target to resize to
                or a dict with keys "shape", "block_size", and "block_stride"

        Returns:
            WarpMap: resized WarpMap
        """
        if isinstance(target, WarpMapMlx):
            t_sh, t_bsz, t_bst = target.warp_field.shape[1:], target.block_size, target.block_stride
        elif isinstance(target, WarpMapperMlx):
            t_sh, t_bsz, t_bst = target.blocks_shape[:3], mx.array(target.block_size), mx.array(target.block_stride)
        elif isinstance(target, dict):
            t_sh, t_bsz, t_bst = (
                target["warp_field_shape"][1:],
                mx.array(target["block_size"]),
                mx.array(target["block_stride"]),
            )
        else:
            raise ValueError("target must be a WarpMap, WarpMapper, or dict")

        # Create coordinate grids directly on GPU using MLX (avoid CPU roundtrip via np.indices)
        z_idx = mx.arange(t_sh[0], dtype=mx.float32)
        y_idx = mx.arange(t_sh[1], dtype=mx.float32)
        x_idx = mx.arange(t_sh[2], dtype=mx.float32)
        zz, yy, xx = mx.meshgrid(z_idx, y_idx, x_idx, indexing="ij")
        ix = mx.stack([zz.reshape(-1), yy.reshape(-1), xx.reshape(-1)], axis=0)  # (3, N)
        ix = (ix * t_bst[:, None] + (t_bsz - self.block_size)[:, None] / 2) / self.block_stride[:, None]

        # Fast path: sample all displacement channels at once via vmap/compile.
        dm_r = _get_sample_displacement_channels_vmapped()(self.warp_field, ix).reshape((3, *t_sh))

        return WarpMapMlx(dm_r, t_bsz, t_bst, self.ref_shape, self.mov_shape)

    def invert_fast(self, sigma=0.5, truncate=20):
        """Invert the displacement field using accumulation and Gaussian basis interpolation.

        Args:
            sigma (float): standard deviation for Gaussian basis interpolation
            truncate (float): truncate parameter for Gaussian basis interpolation

        Returns:
            WarpMap: inverted WarpMap
        """
        warp_field = np.array(self.warp_field)
        target_coords = np.indices(warp_field.shape[1:]) + warp_field / np.array(self.block_stride).reshape(3, 1, 1, 1)
        wf_shape = np.ceil(np.array(self.mov_shape) / np.array(self.block_stride) + 1).astype("int")
        num_coords = accumarray(target_coords, wf_shape)
        inv_field = np.zeros((3, *wf_shape), dtype=warp_field.dtype)
        for i in range(3):
            inv_field[i] = -accumarray(target_coords, wf_shape, weights=warp_field[i].ravel())
            with np.errstate(invalid="ignore"):
                inv_field[i] /= num_coords
            inv_field[i][num_coords == 0] = np.nan
            inv_field[i] = infill_nans(inv_field[i], sigma=sigma, truncate=truncate)
        return WarpMapMlx(inv_field, self.block_size, self.block_stride, self.mov_shape, self.ref_shape)

    def push_coordinates(self, coords, negative_shifts=False):
        """Push voxel coordinates from fixed to moving space.

        Args:
            coords (numpy.array): 3D *voxel* coordinates to be warped (3-by-n array)

        Returns:
            array-like: transformed voxel coordinates
        """
        assert coords.shape[0] == 3
        was_numpy = isinstance(coords, np.ndarray)

        coords = mx.array(coords, dtype=mx.float32)
        coords_blocked = coords / self.block_stride[:, None] - (self.block_size / (2 * self.block_stride))[:, None]

        shifts = mx.zeros_like(coords)

        for idim in range(3):
            shifts[idim] = map_coordinates(self.warp_field[idim], coords_blocked, order=1, mode="nearest")

        if negative_shifts:
            shifts = -shifts

        if was_numpy:
            return np.array(coords + shifts)
        return coords + shifts

    def jacobian_det(self, units_per_voxel=[1, 1, 1], edge_order=1):
        """
        Compute det J = det(∇φ) for φ(x)=x+u(x), using np.indices for the identity grid.

        Args:
            edge_order : passed to np.gradient (1 or 2)

        Returns:
            detJ: mx.array of shape spatial
        """
        scaling = mx.array(units_per_voxel, dtype=mx.float32) * self.block_stride
        coords = mx.array(np.indices(self.warp_field.shape[1:]), dtype=mx.float32) * scaling[:, None, None, None]
        phi = coords + self.warp_field
        J = mx.empty(self.warp_field.shape[1:] + (3, 3), dtype=mx.float32)

        phi_np = np.array(phi)
        scaling_np = np.array(scaling)

        for i in range(3):
            grads = np.gradient(phi_np[i], *scaling_np, edge_order=edge_order)
            for j in range(3):
                J[..., i, j] = mx.array(grads[j], dtype=mx.float32)

        return mx.linalg.det(J)


class WarpMapperMlx:
    """Class that estimates warp field using cross-correlation, based on a piece-wise rigid model.

    Args:
        ref_vol (numpy.array): The reference volume
        block_size (3-element list or numpy.array): shape of blocks, whose rigid displacement is estimated
        block_stride (3-element list or numpy.array): stride (usually identical to block_size)
        proj_method (str or callable): Projection method
    """

    def __init__(
        self, ref_vol, block_size, block_stride=None, proj_method=None, subpixel=4, epsilon=1e-6, tukey_alpha=0.5
    ):
        if np.any(block_size > np.array(ref_vol.shape)):
            raise ValueError(
                f"Block size (currently: {block_size}) must be smaller than the volume shape ({np.array(ref_vol.shape)})."
            )
        self.proj_method = proj_method
        self.subpixel = subpixel
        self.epsilon = epsilon
        self.tukey_alpha = tukey_alpha
        self.update_reference(ref_vol, block_size, block_stride)
        self.ref_shape = np.array(ref_vol.shape)

    def update_reference(self, ref_vol, block_size, block_stride=None):
        block_size = np.array(block_size)
        block_stride = block_size if block_stride is None else np.array(block_stride)
        ref_blocks = sliding_block(
            mx.array(ref_vol, dtype=mx.float32), block_size=block_size, block_stride=block_stride
        )
        self.blocks_shape = ref_blocks.shape
        ref_blocks_proj = [self.proj_method(backend="mlx", vol_blocks=ref_blocks, axis=iax) for iax in [-3, -2, -1]]

        if self.tukey_alpha < 1:
            # ndwindow now returns MLX arrays directly, no conversion needed
            ref_blocks_proj = [
                ref_blocks_proj[i]
                * ndwindow(
                    [1, 1, 1, *ref_blocks_proj[i].shape[-2:]], lambda n: scipy.signal.windows.tukey(n, alpha=0.5)
                )
                for i in range(3)
            ]

        # Compute FFT on GPU using MLX
        self.ref_blocks_proj_ft_conj = []
        for i in range(3):
            # Use mx.fft.rfft2 for GPU FFT computation
            ft = mx.fft.rfft2(ref_blocks_proj[i], axes=(-2, -1))
            self.ref_blocks_proj_ft_conj.append(mx.conjugate(ft))

        self.block_size = block_size
        self.block_stride = block_stride

    def get_displacement(self, vol, smooth_func=None):
        """Estimate the displacement of vol with the reference volume, via piece-wise rigid cross-correlation with the pre-saved blocks.

        Args:
            vol (numpy.array): Input volume
            smooth_func (callable): Smoothing function to be applied to the cross-correlation volume

        Returns:
            WarpMapMlx
        """
        vol = mx.array(vol, dtype=mx.float32) if not isinstance(vol, mx.array) else vol
        vol_blocks = sliding_block(vol, block_size=self.block_size, block_stride=self.block_stride)
        vol_blocks_proj = [self.proj_method(backend="mlx", vol_blocks=vol_blocks, axis=iax) for iax in [-3, -2, -1]]
        del vol_blocks

        epsilon_mx = mx.array(self.epsilon, dtype=mx.float32)
        compiled_vmapped_computer = _get_compiled_vmapped_displacement_computer(self.subpixel)

        # Pre-compute FFT cross-correlations and xcorr projections for all 3 axes
        xcorr_proj_list = []
        R_list = []
        for i in range(3):
            R = mx.fft.rfft2(vol_blocks_proj[i], axes=(-2, -1)) * self.ref_blocks_proj_ft_conj[i]
            xcorr = mx.fft.fftshift(mx.fft.irfft2(R, axes=(-2, -1)), axes=(-2, -1))
            if smooth_func is not None:
                # smooth_func is a Smoother instance that routes to backend-specific smoother
                xcorr = smooth_func(backend="mlx", xcorr_proj=xcorr, block_size=self.block_size)
            xcorr_proj_list.append(xcorr)
            R_list.append(R)

        # Check if all xcorr projections have the same shape (cubic blocks)
        # If not, process each axis separately
        shapes_match = all(xcorr_proj_list[0].shape == xcorr_proj_list[i].shape for i in range(1, 3))

        if shapes_match:
            # Fast path: refine displacements via vmap+compile (cubic blocks)
            xcorr_proj_stacked = mx.stack(xcorr_proj_list, axis=0)
            R_stacked = mx.stack(R_list, axis=0)
            disp_field_stacked = compiled_vmapped_computer(xcorr_proj_stacked, R_stacked, epsilon_mx)

            # Combine displacement fields from the 3 projections
            disp_field = (
                mx.stack(
                    [
                        disp_field_stacked[1, 0] + disp_field_stacked[2, 0],
                        disp_field_stacked[0, 0] + disp_field_stacked[2, 1],
                        disp_field_stacked[0, 1] + disp_field_stacked[1, 1],
                    ],
                    axis=0,
                )
                / 2.0
            )
        else:
            # Slow path: process each axis separately (non-cubic blocks)
            disp_field_list = []
            for i in range(3):
                # Refine each projection independently.
                # Using the 3-projection vmapped helper here is incorrect for non-cubic
                # blocks because each projection has a different 2D shape.
                disp_single = _compute_displacement_for_one_projection(
                    xcorr_proj_list[i], R_list[i], epsilon_mx, self.subpixel
                )
                disp_field_list.append(disp_single)

            # Combine displacement fields from the 3 projections
            disp_field = (
                mx.stack(
                    [
                        disp_field_list[1][0] + disp_field_list[2][0],
                        disp_field_list[0][0] + disp_field_list[2][1],
                        disp_field_list[0][1] + disp_field_list[1][1],
                    ],
                    axis=0,
                )
                / 2.0
            )

        return WarpMapMlx(disp_field, self.block_size, self.block_stride, self.ref_shape, vol.shape)


class RegistrationPyramidMlx:
    """A class for performing multi-resolution registration.

    Args:
        ref_vol (numpy.array): Reference volume
        settings (pandas.DataFrame): Settings for each level of the pyramid.
            IMPORTANT: the block sizea in the last level cannot be larger than the block_size in any previous level.
        reg_mask (numpy.array): Mask for registration
        clip_thresh (float): Threshold for clipping the reference volume
    """

    def __init__(self, ref_vol, recipe, reg_mask=1):
        recipe.model_validate(recipe.model_dump())
        self.recipe = recipe
        self.reg_mask = mx.array(reg_mask, dtype=mx.float32)
        self.mappers = []
        ref_vol = mx.array(ref_vol, dtype=mx.float32)
        self.ref_shape = ref_vol.shape
        if self.recipe.pre_filter is not None:
            ref_vol = self.recipe.pre_filter(backend="mlx", vol=ref_vol, reg_mask=self.reg_mask)
        self.mapper_ix = []
        for i in range(len(recipe.levels)):
            if recipe.levels[i].repeats < 1:
                continue
            block_size = np.array(recipe.levels[i].block_size)
            tmp = np.r_[ref_vol.shape] // -block_size
            block_size[block_size < 0] = tmp[block_size < 0]
            if isinstance(recipe.levels[i].block_stride, (int, float)):
                block_stride = (block_size * recipe.levels[i].block_stride).astype("int")
            else:
                block_stride = np.array(recipe.levels[i].block_stride)
            self.mappers.append(
                WarpMapperMlx(
                    ref_vol,
                    block_size,
                    block_stride=block_stride,
                    proj_method=recipe.levels[i].project,
                    tukey_alpha=recipe.levels[i].tukey_ref,
                )
            )
            self.mapper_ix.append(i)
        assert len(self.mappers) > 0, "At least one level of registration is required"

    def register_single(self, vol, callback=None, verbose=False):
        """Register a single volume to the reference volume.

        Args:
            vol (array_like): Volume to be registered (numpy or mlx array)
            callback (function): Callback function to be called after each level of registration

        Returns:
            - vol (array_like): Registered volume (numpy or mlx array, depending on input)
            - warp_map (WarpMapMlx): Displacement field
            - callback_output (list): List of outputs from the callback function
        """
        was_numpy = isinstance(vol, np.ndarray)
        vol = mx.array(vol, dtype=mx.float32)
        offsets = (mx.array(vol.shape, dtype=mx.float32) - mx.array(self.ref_shape, dtype=mx.float32)) / 2
        warp_map = WarpMapMlx(
            offsets[:, None, None, None],
            mx.ones(3, dtype=mx.float32),
            mx.ones(3, dtype=mx.float32),
            self.ref_shape,
            vol.shape,
        )
        warp_map = warp_map.resize_to(self.mappers[-1])
        callback_output = []
        vol_tmp0 = (
            self.recipe.pre_filter(backend="mlx", vol=vol, reg_mask=self.reg_mask)
            if self.recipe.pre_filter is not None
            else vol
        )
        vol_tmp = mx.zeros(self.ref_shape, dtype=mx.float32)
        vol_tmp = warp_map.warp(vol_tmp0, out=vol_tmp)
        min_block_stride = np.min([mapper.block_stride for mapper in self.mappers], axis=0)
        if callback is not None:
            callback_output.append(callback(vol_tmp))

        if np.any(self.mappers[-1].block_stride > min_block_stride[0]):
            warnings.warn(
                "The block stride (in voxels) in the last level should not be larger than the block stride in any previous level (along any axis)."
            )
        for k, mapper in enumerate(tqdm(self.mappers, desc="Levels", disable=not verbose)):
            for _ in tqdm(
                range(self.recipe.levels[self.mapper_ix[k]].repeats), leave=False, desc="Repeats", disable=not verbose
            ):
                wm = mapper.get_displacement(vol_tmp, smooth_func=self.recipe.levels[self.mapper_ix[k]].smooth)
                wm.warp_field *= self.recipe.levels[self.mapper_ix[k]].update_rate
                if self.recipe.levels[self.mapper_ix[k]].median_filter:
                    wm = wm.median_filter()
                if self.recipe.levels[self.mapper_ix[k]].affine:
                    if (np.array(mapper.blocks_shape[:3]) < 2).sum() > 1:
                        raise ValueError(
                            f"Affine fit needs at least two axes with at least 2 blocks! Volume shape: {self.ref_shape}; block size: {mapper.block_size}"
                        )
                    wm, _ = wm.fit_affine(
                        target=dict(
                            warp_field_shape=(3, *self.mappers[-1].blocks_shape[:3]),
                            block_size=self.mappers[-1].block_size,
                            block_stride=self.mappers[-1].block_stride,
                        )
                    )
                else:
                    wm = wm.resize_to(self.mappers[-1])

                warp_map = warp_map.chain(wm)
                vol_tmp = warp_map.warp(vol_tmp0, out=vol_tmp)
                if callback is not None:
                    callback_output.append(callback(vol_tmp))

                # To update the progress bar, we need to evaluate the results
                if verbose:
                    mx.eval(vol_tmp)
        vol_tmp = warp_map.warp(vol, out=vol_tmp)
        if was_numpy:
            vol_tmp = np.array(vol_tmp)
        return vol_tmp, warp_map, callback_output

    def clean_up(self):
        """No-op cleanup hook for API compatibility with other backends."""
        return None


class ProjectorMlx(BaseModel):
    """A class to apply a 2D projection and filters to a volume block

    Parameters:
        max: if True, apply a max filter to the volume block. Default is True
        normalize: if True, normalize projections by the L2 norm (to get correlations, not covariances). Default is False
        dog: if True, apply a DoG filter to the volume block. Default is True
        low: the lower sigma value for the DoG filter. Default is 0.5
        high: the higher sigma value for the DoG filter. Default is 10.0
        periodic_smooth: bool = False
    """

    max: bool = True
    normalize: Union[bool, float] = False
    dog: bool = True
    low: Union[Union[int, float], List[Union[int, float]]] = 0.5
    high: Union[Union[int, float], List[Union[int, float]]] = 10.0
    periodic_smooth: bool = False

    def __call__(self, vol_blocks, axis):
        """Apply a 2D projection and filters to a volume block
        Args:
            vol_blocks (mx.array or np.ndarray): Blocked volume to be projected (6D dataset, with the first 3 dimensions being blocks and the last 3 dimensions being voxels)
            axis (int): Axis along which to project
        Returns:
            mx.array: Projected volume block (5D dataset, with the first 3 dimensions being blocks and the last 2 dimensions being 2D projections)
        """
        # Keep the full projector path on MLX/GPU (optimized type check)
        vol_blocks = mx.array(vol_blocks, dtype=mx.float32) if not isinstance(vol_blocks, mx.array) else vol_blocks.astype(mx.float32)

        # Projection.
        if self.max:
            out = mx.max(vol_blocks, axis=axis)
        else:
            out = mx.mean(vol_blocks, axis=axis)

        # Optional periodic smoothing (MLX path in ndimage helper).
        if self.periodic_smooth:
            out = periodic_smooth_decomposition_nd_rfft(out)

        low = np.delete(np.r_[1, 1, 1] * self.low, axis)
        high = np.delete(np.r_[1, 1, 1] * self.high, axis)

        # Filtering entirely on MLX.
        if self.dog:
            out = dogfilter(out, [0, 0, 0, *low], [0, 0, 0, *high])
        elif not np.all(np.array(self.low) == 0):
            out = gaussian_filter_mlx_reflect(out, [0, 0, 0, *low], truncate=5.0)

        # Optional projection normalization.
        if self.normalize > 0:
            out_sum = mx.sum(out * out, axis=(-2, -1), keepdims=True)
            out = out / (mx.sqrt(out_sum) ** self.normalize + 1e-9)

        return out.astype(mx.float32)


class SmootherMlx(BaseModel):
    """Smooth blocks with a Gaussian kernel
    Args:
        sigmas (list): [sigma0, sigma1, sigma2]. If None, no smoothing is applied.
        truncate (float): truncate parameter for gaussian kernel. Default is 5.
        shear (float): shear parameter for gaussian kernel. Default is None.
        long_range_ratio (float): long range ratio for double gaussian kernel. Default is None.
    """

    sigmas: Union[float, List[float]] = [1.0, 1.0, 1.0]
    shear: Union[float, None] = None
    long_range_ratio: Union[float, None] = 0.05

    def __call__(self, xcorr_proj, block_size=None):
        """Apply a Gaussian filter to the cross-correlation data
        Args:
            xcorr_proj (mx.array): cross-correlation data (5D array, with the first 3 dimensions being the blocks and the last 2 dimensions being the 2D projection)
            block_size (list): shape of blocks, whose rigid displacement is estimated
        Returns:
            mx.array: smoothed cross-correlation volume
        """
        return smooth_func(
            xcorr_proj=xcorr_proj,
            block_size=block_size,
            sigmas=self.sigmas,
            shear=self.shear,
            long_range_ratio=self.long_range_ratio,
        )


class RegFilterMlx(BaseModel):
    """A class to apply a filter to the volume before registration

    Parameters:
        clip_thresh: threshold for clipping the reference volume. Default is 0
        dog: if True, apply a DoG filter to the volume. Default is True
        low: the lower sigma value for the DoG filter. Default is 0.5
        high: the higher sigma value for the DoG filter. Default is 10.0
    """

    clip_thresh: float = 0
    dog: bool = True
    low: float = 0.5
    high: float = 10.0
    soft_edge: Union[Union[int, float], List[Union[int, float]]] = 0.0

    def __call__(self, vol, reg_mask=None):
        """Apply the filter to the volume (fully GPU-bound)
        Args:
            vol (mx or numpy array): 3D volume to be filtered
            reg_mask (array): Mask for registration
        Returns:
            mx.array: Filtered volume
        """
        # Convert to MLX array (optimized single-line conversion)
        vol = mx.array(vol, dtype=mx.float32) if not isinstance(vol, mx.array) else vol.astype(mx.float32)

        # Clip on GPU
        vol = mx.maximum(vol - self.clip_thresh, 0.0)

        # Soften edges on GPU
        if np.any(np.array(self.soft_edge) > 0):
            vol = soften_edges(vol, self.soft_edge)

        # Apply registration mask on GPU
        if reg_mask is not None:
            if not isinstance(reg_mask, mx.array):
                reg_mask = mx.array(reg_mask, dtype=mx.float32)
            vol = vol * reg_mask

        # Apply DoG filter on GPU (using compiled dogfilter_mlx)
        if self.dog:
            vol = dogfilter(vol, self.low, self.high)

        return vol
