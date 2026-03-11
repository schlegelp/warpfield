from importlib.util import find_spec

from .core import Backend, registry


def register_available_backends():
    """Collect available backends based on installed packages."""
    # Note: the order of checks determines the default priority of backends
    # (e.g., prefer GPU if available)
    if find_spec("cupy") is not None:
        from ._cupy.warp import warp_volume_cupy
        from ._cupy.register import (
            RegistrationPyramidCupy,
            WarpMapCupy,
            SmootherCupy,
            ProjectorCupy,
            RegFilterCupy,
        )
        from ._cupy.ndimage import zoom as cupy_zoom

        registry.register_backend(
            Backend(
                name="cupy",
                warpmap_cls=WarpMapCupy,
                warp_volume_func=warp_volume_cupy,
                registration_pyramid_cls=RegistrationPyramidCupy,
                smoother_cls=SmootherCupy,
                projector_cls=ProjectorCupy,
                reg_filter_cls=RegFilterCupy,
                zoom_func=cupy_zoom,
            )
        )

    if find_spec("mlx") is not None:
        from ._mlx.warp import warp_volume_mlx
        from ._mlx.register import (
            RegistrationPyramidMlx,
            WarpMapMlx,
            SmootherMlx,
            ProjectorMlx,
            RegFilterMlx,
        )
        from ._mlx.ndimage import zoom as mlx_zoom

        registry.register_backend(
            Backend(
                name="mlx",
                warpmap_cls=WarpMapMlx,
                warp_volume_func=warp_volume_mlx,
                registration_pyramid_cls=RegistrationPyramidMlx,
                smoother_cls=SmootherMlx,
                projector_cls=ProjectorMlx,
                reg_filter_cls=RegFilterMlx,
                zoom_func=mlx_zoom,
            )
        )

register_available_backends()
