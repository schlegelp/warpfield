from __future__ import annotations

import gc
import os
import pathlib
import warnings

import numpy as np

from pydantic import BaseModel
from typing import List, Union, Callable, Any, TYPE_CHECKING

from .backends import registry
from .utils import create_rgb_video

if TYPE_CHECKING:
    _ArrayType = Union[np.ndarray, "cupy.ndarray", "mlx.ndarray"]  # faq:
else:
    _ArrayType = Any


class WarpMap:
    """Factory class for creating WarpMap instances with automatic backend selection.

    This class uses the __new__ method to instantiate the appropriate backend
    implementation (e.g., WarpMapCupy for CUDA GPU acceleration or WarpMapMlx for Apple Silicon GPU).

    Args:
        warp_field: the displacement field data (3-x-y-z)
        block_size: block size in voxels (3-element array-like)
        block_stride: stride between blocks (3-element array-like)
        ref_shape: shape of the reference volume (tuple)
        mov_shape: shape of the moving volume (tuple)
        backend (str): which backend to use. Options are:
            - "auto" (default): automatically selects the best available backend
            - "cupy": forces use of CuPy backend (requires GPU)
            - "mlx": forces use of MLX backend (requires Apple Silicon GPU)
            - "cpu": forces use of CPU backend (not yet implemented)

    Returns:
        An instance of the appropriate WarpMap backend implementation.

    Example:
        >>> wm = WarpMap(warp_field, block_size, block_stride, ref_shape, mov_shape)
        >>> # Automatically uses the best available backend
    """

    def __new__(cls, warp_field, block_size, block_stride, ref_shape, mov_shape, backend="auto"):
        """Factory method that returns the appropriate backend implementation.

        Args:
            warp_field: displacement field data
            block_size: block size
            block_stride: block stride
            ref_shape: reference shape
            mov_shape: moving shape
            backend (str): backend to use ("auto", "cupy", "mlx", "cpu", etc.)

        Returns:
            Instance of a WarpMap backend implementation
        """
        if cls is WarpMap:
            # We're being called on the WarpMap class itself, not a subclass
            return registry.get_backend(backend).warpmap_cls(warp_field, block_size, block_stride, ref_shape, mov_shape)
        else:
            # Being called on a subclass, just create normal instance
            return super().__new__(cls)

    @classmethod
    def from_h5(cls, h5_path, group="warp_map", backend="auto"):
        """Load a WarpMap from an HDF5 file.

        Args:
            h5_path (str or os.PathLike): Path to the HDF5 file.
            group (str): Group path inside the HDF5 file where the WarpMap is stored.
            backend (str): Backend to use for the loaded WarpMap.

        Returns:
            WarpMap: The loaded WarpMap object (with appropriate backend).
        """
        # Delegate to the appropriate backend's from_h5 method
        return registry.get_backend(backend).warpmap_cls.from_h5(h5_path, group)


def register_volumes(
    ref, vol, recipe, reg_mask=1, callback=None, verbose=True, video_path=None, vmax=None, backend="auto"
):
    """Register a volume to a reference volume using a registration pyramid.

    Args:
        ref (numpy, cupy or mlx array): Reference volume
        vol (numpy, cupy or mlx array): Volume to be registered
        recipe (Recipe): Registration recipe
        reg_mask (numpy.array): Mask to be multiplied with the reference volume. Default is 1 (no mask)
        callback (function): Callback function to be called on the volume after each iteration. Default is None.
            Can be used to monitor and optimize registration. Example: `callback = lambda vol: vol.mean(1).get()`
            (note that `vol` is a 3D cupy array. Use `.get()` to turn the output into a numpy array and save GPU memory).
            Callback outputs for each registration step will be returned as a list.
        verbose (bool): If True, show progress bars. Default is True
        video_path (str): Save a video of the registration process, using callback outputs. The callback has to return 2D frames. Default is None.
        vmax (float): Maximum pixel value (to scale video brightness). If none, set to 99.9 percentile of pixel values.

    Returns:
        - numpy, cupy or mlx array (depending on vol input): Registered volume
        - WarpMap: Displacement field
        - list: List of outputs from the callback function
    """
    # Select the appropriate registration pyramid class based on the backend
    pyramid = registry.get_backend(backend).registration_pyramid_cls

    recipe.model_validate(recipe.model_dump())
    reg = pyramid(ref, recipe, reg_mask=reg_mask)
    registered_vol, warp_map, cbout = reg.register_single(vol, callback=callback, verbose=verbose)
    reg.clean_up()  # this might free GPU memory or garbage collect
    del reg
    gc.collect()

    if video_path is not None:
        try:
            assert cbout[0].ndim == 2, "Callback output must be a 2D array"
            ref = callback(recipe.pre_filter(backend=backend, vol=ref))
            vmax = np.percentile(ref, 99.9).item() if vmax is None else vmax
            create_rgb_video(video_path, ref / vmax, np.array(cbout) / vmax, fps=10)
        except (ValueError, AssertionError) as e:
            warnings.warn(f"Video generation failed with error: {e}")
    return registered_vol, warp_map, cbout


class Projector(BaseModel):
    """A class to apply a 2D projection and filters to a volume block

    Parameters:
        max: if True, apply a max filter to the volume block. Default is True
        normalize: if True, normalize projections by the L2 norm (to get correlations, not covariances). Default is False
        dog: if True, apply a DoG filter to the volume block. Default is True
        low: the lower sigma value for the DoG filter. Default is 0.5
        high: the higher sigma value for the DoG filter. Default is 10.0
        tukey_env: if True, apply a Tukey window to the output. Default is False
        gauss_env: if True, apply a Gaussian window to the output. Default is False
    """

    max: bool = True
    normalize: Union[bool, float] = False
    dog: bool = True
    low: Union[Union[int, float], List[Union[int, float]]] = 0.5
    high: Union[Union[int, float], List[Union[int, float]]] = 10.0
    periodic_smooth: bool = False

    def __call__(self, backend, vol_blocks, axis):
        """Apply a 2D projection and filters to a volume block
        Args:
            backend (str): Name of the backend to use (e.g., "cupy", "mlx", "cpu").
            vol_blocks (array-like): Blocked volume to be projected (6D dataset, with the first 3 dimensions being blocks and the last 3 dimensions being voxels)
            axis (int): Axis along which to project
        Returns:
            array: Projected volume block (5D dataset, with the first 3 dimensions being blocks and the last 2 dimensions being 2D projections)
        """
        projector = registry.get_backend(backend).projector_cls()
        projector.max = self.max
        projector.normalize = self.normalize
        projector.dog = self.dog
        projector.low = self.low
        projector.high = self.high
        projector.periodic_smooth = self.periodic_smooth
        return projector(vol_blocks, axis)


class Smoother(BaseModel):
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

    def __call__(self, backend, xcorr_proj, block_size=None):
        smoother = registry.get_backend(backend).smoother_cls()
        smoother.sigmas = self.sigmas
        smoother.shear = self.shear
        smoother.long_range_ratio = self.long_range_ratio
        return smoother(xcorr_proj, block_size)


class RegFilter(BaseModel):
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

    def __call__(self, backend, vol, reg_mask=None):
        """Apply the filter to the volume
        Args:
            vol (cupy, numpy or mlx array): 3D volume to be filtered
            reg_mask (array): Mask for registration
        Returns:
            array: Filtered volume
        """
        filter = registry.get_backend(backend).reg_filter_cls()
        filter.clip_thresh = self.clip_thresh
        filter.dog = self.dog
        filter.low = self.low
        filter.high = self.high
        filter.soft_edge = self.soft_edge
        return filter(vol, reg_mask)


class LevelConfig(BaseModel):
    """Configuration for each level of the registration pyramid

    Args:
        block_size (list): shape of blocks, whose rigid displacement is estimated
        block_stride (list): stride (usually identical to block_size)
        repeats (int): number of iterations for this level (deisable level by setting repeats to 0)
        smooth (Smoother or None): Smoother object
        project (Projector, callable or None): Projector object. The callable should take a volume block and an axis as input and return a projected volume block.
        tukey_ref (float): if not None, apply a Tukey window to the reference volume (alpha = tukey_ref). Default is 0.5
        affine (bool): if True, apply affine transformation to the displacement field
        median_filter (bool): if True, apply median filter to the displacement field
        update_rate (float): update rate for the displacement field. Default is 1.0. Can be lowered to dampen oscillations.
    """

    block_size: Union[List[int]]
    block_stride: Union[List[int], float] = 1.0
    project: Union[Projector, Callable[[_ArrayType, int], _ArrayType]] = Projector()
    tukey_ref: Union[float, None] = 0.5
    smooth: Union[Smoother, None] = Smoother()
    affine: bool = False
    median_filter: bool = True
    update_rate: float = 1.0
    repeats: int = 5


class Recipe(BaseModel):
    """Configuration for the registration recipe. Recipe is initialized with a single affine level.

    Args:
        reg_filter (RegFilter, callable or None): Filter to be applied to the reference volume
        levels (list): List of LevelConfig objects
    """

    pre_filter: Union[RegFilter, Callable[[_ArrayType], _ArrayType], None] = RegFilter()
    levels: List[LevelConfig] = [
        LevelConfig(block_size=[-1, -1, -1], repeats=3),  # translation level
        LevelConfig(  # affine level
            block_size=[-2, -2, -2],
            block_stride=0.5,
            repeats=10,
            affine=True,
            median_filter=False,
            smooth=Smoother(sigmas=[0.5, 0.5, 0.5]),
        ),
    ]

    def add_level(self, block_size, **kwargs):
        """Add a level to the registration recipe

        Args:
            block_size (list): shape of blocks, whose rigid displacement is estimated
            **kwargs: additional arguments for LevelConfig
        """
        if isinstance(block_size, (int, float)):
            block_size = [block_size] * 3
        if len(block_size) != 3:
            raise ValueError("block_size must be a list of 3 integers")
        self.levels.append(LevelConfig(block_size=block_size, **kwargs))

    def insert_level(self, index, block_size, **kwargs):
        """Insert a level to the registration recipe

        Args:
            index (int): A number specifying in which position to insert the level
            block_size (list): shape of blocks, whose rigid displacement is estimated
            **kwargs: additional arguments for LevelConfig
        """
        if isinstance(block_size, (int, float)):
            block_size = [block_size] * 3
        if len(block_size) != 3:
            raise ValueError("block_size must be a list of 3 integers")
        self.levels.insert(index, LevelConfig(block_size=block_size, **kwargs))

    @classmethod
    def from_yaml(cls, yaml_path):
        """Load a recipe from a YAML file

        Args:
            yaml_path (str): path to the YAML file

        Returns:
            Recipe: Recipe object
        """
        import yaml

        this_file_dir = pathlib.Path(__file__).resolve().parent
        if os.path.isfile(yaml_path):
            yaml_path = yaml_path
        else:
            yaml_path = os.path.join(this_file_dir, "recipes", yaml_path)

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        return cls.model_validate(data)

    def to_yaml(self, yaml_path):
        """Save the recipe to a YAML file

        Args:
            yaml_path (str): path to the YAML file
        """
        import yaml

        with open(yaml_path, "w") as f:
            yaml.dump(self.model_dump(), f)
        print(f"Recipe saved to {yaml_path}")
