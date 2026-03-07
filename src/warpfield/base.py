import h5py

import numpy as np

from abc import ABC, abstractmethod


class WarpMapBase(ABC):
    """Represents a 3D displacement field

    Args:
        warp_field (numpy.array): the displacement field data (3-x-y-z)
        block_size (3-element list or numpy.array):
        block_stride (3-element list or numpy.array):
        ref_shape (tuple): shape of the reference volume
        mov_shape (tuple): shape of the moving volume
    """

    def __init__(self, warp_field, block_size, block_stride, ref_shape, mov_shape):
        self.warp_field = warp_field
        self.block_size = block_size
        self.block_stride = block_stride
        self.ref_shape = ref_shape
        self.mov_shape = mov_shape

    @property
    def warp_field(self):
        return self._warp_field

    @warp_field.setter
    def warp_field(self, value):
        self._warp_field = value

    @property
    def warp_field_numpy(self):
        """Return warp field guaranteed to be a numpy array."""
        return np.array(self.warp_field)

    @property
    def block_size(self):
        return self._block_size

    @block_size.setter
    def block_size(self, value):
        self._block_size = value

    @property
    def block_size_numpy(self):
        """Return block size guaranteed to be a numpy array."""
        return np.array(self.block_size)

    @property
    def block_stride(self):
        return self._block_stride

    @block_stride.setter
    def block_stride(self, value):
        self._block_stride = value

    @property
    def block_stride_numpy(self):
        """Return block stride guaranteed to be a numpy array."""
        return np.array(self.block_stride)

    @abstractmethod
    def warp(self, vol, out=None):
        """Apply the warp to a volume. Can be thought of as pulling the moving volume to the fixed volume space."""
        pass

    def apply(self, *args, **kwargs):
        """Alias of warp method"""
        return self.warp(*args, **kwargs)

    @abstractmethod
    def fit_affine(self, target=None):
        """Fit affine transformation and return new fitted WarpMap"""
        pass

    @abstractmethod
    def median_filter(self):
        """Apply median filter to the displacement field"""
        pass

    @abstractmethod
    def resize_to(self, target):
        """Resize to target WarpMap, using linear interpolation"""
        pass

    def chain(self, target):
        """Chain displacement maps

        Args:
            target (WarpMapCupy): WarpMap to be added to existing map

        Returns:
            WarpMapCupy: new WarpMap with chained displacement field
        """
        warp_field = self.warp_field.copy()
        warp_field += target.warp_field
        return type(self)(warp_field, target.block_size, target.block_stride, self.ref_shape, self.mov_shape)

    def invert(self, **kwargs):
        """alias for invert_fast method"""
        return self.invert_fast(**kwargs)

    @abstractmethod
    def invert_fast(self, sigma=0.5, truncate=20):
        """Invert the displacement field using accumulation and Gaussian basis interpolation."""
        pass

    @abstractmethod
    def push_coordinates(self, coords, negative_shifts=False):
        """Push voxel coordinates from fixed to moving space."""
        pass

    def pull_coordinates(self, coords):
        """Pull voxel coordinates through the warp field. Involves inversion, followed by pushing coordinates.

        Args:
            coords (numpy.array): 3D *voxel* coordinates to be warped (3-by-n array)

        Returns:
            numpy.array: transformed voxel coordinates
        """
        return self.invert().push_coordinates(coords, negative_shifts=True)

    @abstractmethod
    def jacobian_det(self, units_per_voxel=[1, 1, 1], edge_order=1):
        """Compute det J = det(∇φ) for φ(x)=x+u(x), using np.indices for the identity grid."""
        pass

    def as_ants_image(self, voxel_size_um=1):
        """Convert to ANTsImage.

        Args:
            voxel_size_um (scalar or array): voxel size (default is 1)

        Returns:
            ants.core.ants_image.ANTsImage:
        """
        try:
            import ants
        except ImportError:
            raise ImportError("ANTs is not installed. Please install it using 'pip install ants'")

        ants_image = ants.from_numpy(
            self.warp_field_numpy.transpose(1, 2, 3, 0),
            origin=list((self.block_size_numpy - 1) / 2 * voxel_size_um),
            spacing=list(self.block_stride_numpy * voxel_size_um),
            has_components=True,
        )
        return ants_image

    def __repr__(self):
        """String representation of the WarpMap object."""
        info = (
            f"WarpMap("
            f"warp_field_shape={self.warp_field.shape}, "
            f"block_size={self.block_size_numpy}, "
            f"block_stride={self.block_stride_numpy}, "
            f"transformation: {str(self.mov_shape)} --> {str(self.ref_shape)}"
        )
        return info

    def to_h5(self, h5_path, group="warp_map", compression="gzip", overwrite=True):
        """
        Save this WarpMap to an HDF5 file.

        Args:
            h5_path (str or os.PathLike): Path to the HDF5 file.
            group (str): Group path inside the HDF5 file to store the WarpMap (created if missing).
            compression (str or None): Dataset compression (e.g., 'gzip', None).
            overwrite (bool): If True, overwrite existing datasets/attrs inside the group.
        """
        with h5py.File(h5_path, "a") as f:
            if overwrite and (group not in (None, "", "/")) and (group in f):
                del f[group]
            if (not overwrite) and (group in f):
                raise ValueError(f"Group '{group}' already exists in {h5_path}. Set 'overwrite=True' to overwrite it.")
            grp = f.require_group(group) if group not in (None, "", "/") else f
            grp.create_dataset("warp_field", data=self.warp_field.get(), compression=compression)
            grp.create_dataset("block_size", data=self.block_size.get())
            grp.create_dataset("block_stride", data=self.block_stride.get())
            grp.create_dataset("ref_shape", data=np.array(self.ref_shape, dtype="int64"))
            grp.create_dataset("mov_shape", data=np.array(self.mov_shape, dtype="int64"))
            grp.attrs["class"] = "WarpMap"

    @classmethod
    def from_h5(cls, h5_path, group="warp_map", backend="auto"):
        """
        Load a WarpMap from an HDF5 file.

        Args:
            h5_path (str or os.PathLike): Path to the HDF5 file.
            group (str): Group path inside the HDF5 file where the WarpMap is stored.

        Returns:
            WarpMap: The loaded WarpMap object.
        """
        with h5py.File(h5_path, "r") as f:
            grp = f[group]
            warp_field = grp["warp_field"][:]
            block_size = grp["block_size"][:]
            block_stride = grp["block_stride"][:]
            ref_shape = tuple(grp["ref_shape"][:].tolist())
            mov_shape = tuple(grp["mov_shape"][:].tolist())
        return cls(warp_field, block_size, block_stride, ref_shape, mov_shape)