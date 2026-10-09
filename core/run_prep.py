"""Getting the machine ready for a training run."""
from core import models

# What an Anima (Kohya) run commits in system memory (RAM + pagefile) while it loads
# the DiT, text encoder and VAE and caches latents. Krea 2's figure is far higher
# (core/engines/onetrainer/engine.MIN_FREE_COMMIT_GB) because of int8 quantizing.
ANIMA_MIN_FREE_COMMIT_GB = 24


def room_target_gb(engine: str) -> float:
    """System memory (GB) a run on `engine` wants free to commit."""
    if engine == models.ONETRAINER:
        from core.engines.onetrainer.engine import MIN_FREE_COMMIT_GB
        return MIN_FREE_COMMIT_GB
    return ANIMA_MIN_FREE_COMMIT_GB
