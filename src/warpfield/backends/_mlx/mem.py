import os

import mlx.core as mx

_MLX_MEM_PROFILE = os.getenv("WARPFIELD_MLX_MEM_PROFILE", "0") == "1"

def _to_mib(n_bytes):
    return float(n_bytes) / (1024.0 * 1024.0)


def _mlx_get_active_memory():
    if hasattr(mx, "get_active_memory"):
        return mx.get_active_memory()
    # Fallback for older MLX versions
    if hasattr(mx, "metal") and hasattr(mx.metal, "get_active_memory"):
        return mx.metal.get_active_memory()
    return 0


def _mlx_get_cache_memory():
    if hasattr(mx, "get_cache_memory"):
        return mx.get_cache_memory()
    # Fallback for older MLX versions
    if hasattr(mx, "metal") and hasattr(mx.metal, "get_cache_memory"):
        return mx.metal.get_cache_memory()
    return 0


def _mlx_get_peak_memory():
    if hasattr(mx, "get_peak_memory"):
        return mx.get_peak_memory()
    # Fallback for older MLX versions
    if hasattr(mx, "metal") and hasattr(mx.metal, "get_peak_memory"):
        return mx.metal.get_peak_memory()
    return 0


def _mlx_clear_cache():
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
        return
    # Fallback for older MLX versions
    if hasattr(mx, "metal"):
        if hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()


def _mlx_reset_peak_memory_impl():
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
        return
    # Fallback for older MLX versions
    if hasattr(mx, "metal"):
        if hasattr(mx.metal, "reset_peak_memory"):
            mx.metal.reset_peak_memory()


def _mlx_mem_log(label):
    """Log MLX Metal allocator stats when profiling is enabled."""
    if not _MLX_MEM_PROFILE or not hasattr(mx, "metal"):
        return
    try:
        active = _mlx_get_active_memory()
        cache = _mlx_get_cache_memory()
        peak = _mlx_get_peak_memory()
        print(
            f"[mlx-mem] {label}: "
            f"active={_to_mib(active):8.1f} MiB "
            f"cache={_to_mib(cache):8.1f} MiB "
            f"peak={_to_mib(peak):8.1f} MiB"
        )
    except Exception:
        # Never let diagnostics affect registration logic.
        return


def _mlx_reset_peak_memory():
    if not _MLX_MEM_PROFILE or not hasattr(mx, "metal"):
        return
    try:
        _mlx_reset_peak_memory_impl()
    except Exception:
        return