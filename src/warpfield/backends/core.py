from dataclasses import dataclass


@dataclass
class Backend:
    """A backend for warping and registration operations."""

    name: str  # name of the backend (e.g., "cupy", "numpy")
    warpmap_cls: type  # class implementing the WarpMap for this backend
    warp_volume_func: callable  # function to warp volumes using this backend
    registration_pyramid_cls: type  # class implementing the RegistrationPyramid for this backend
    smoother_cls: type  # class implementing the Smoother for this backend
    projector_cls: type  # class implementing the Projector for this backend
    reg_filter_cls: type  # class implementing the RegFilter for this backend

    def __post_init__(self):
        self.name = self.name.lower()


class BackendRegistry:
    """Manages available backends and their components."""

    def __init__(self):
        self.backends = {}
        self.default_backend = None

    def __repr__(self):
        return (
            f"BackendRegistry(available_backends={list(self.backends.keys())}, default_backend={self.default_backend})"
        )

    def register_backend(self, backend: Backend):
        """Register a new backend."""
        assert isinstance(backend, Backend), "backend must be an instance of Backend dataclass."
        assert backend.name not in self.backends, f"Backend '{backend.name}' is already registered."

        self.backends[backend.name] = backend
        if self.default_backend is None:
            self.default_backend = backend.name

    def get_backend(self, name=None):
        """Get a backend by name, or return the default backend if name is None."""
        if not len(self.backends):
            raise ValueError(
                "No backends found. Please make sure the requirements for at least one of the supported backends are installed."
            )

        if name in (None, "auto", "default"):
            name = self.default_backend if self.default_backend is not None else next(iter(self.backends))

        if name not in self.backends:
            raise ValueError(f"Backend '{name}' not found. Available backends: {list(self.backends.keys())}")
        return self.backends[name]


registry = BackendRegistry()
