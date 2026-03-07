from importlib.util import find_spec

from .core import Backend, registry


def register_available_backends():
    """Collect available backends based on installed packages."""
    # Note: the order of checks determines the default priority of backends
    # (e.g., prefer GPU if available)
    #if find_spec("cupy") is not None:
    if True:
        from ._cupy.warp import warp_volume_cupy
        from ._cupy.register import (
            RegistrationPyramidCupy,
            WarpMapCupy,
            SmootherCupy,
            ProjectorCupy,
            RegFilterCupy,
        )

        registry.register_backend(
            Backend(
                name="cupy",
                warpmap_cls=WarpMapCupy,
                warp_volume_func=warp_volume_cupy,
                registration_pyramid_cls=RegistrationPyramidCupy,
                smoother_cls=SmootherCupy,
                projector_cls=ProjectorCupy,
                reg_filter_cls=RegFilterCupy,
            )
        )

register_available_backends()
