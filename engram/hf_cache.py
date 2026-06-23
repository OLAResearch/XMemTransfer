import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SCRATCH_CACHE_BASE = Path("/scratch/project_462001050/cache")


def default_hf_cache_base() -> Path:
    if os.environ.get("ENGRAM_FORCE_LOCAL") == "1" or os.environ.get("ENGRAM_RUNTIME") == "local":
        return PROJECT_ROOT / ".cache" / "huggingface"
    if _DEFAULT_SCRATCH_CACHE_BASE.is_dir() and os.access(_DEFAULT_SCRATCH_CACHE_BASE, os.W_OK):
        return _DEFAULT_SCRATCH_CACHE_BASE / "huggingface"
    return PROJECT_ROOT / ".cache" / "huggingface"


def configure_hf_cache() -> Path:
    cache_base = Path(os.environ.get("HF_CACHE_BASE", str(default_hf_cache_base()))).expanduser()
    cache_base.mkdir(parents=True, exist_ok=True)

    hub_cache = cache_base / "hub"
    datasets_cache = cache_base / "datasets"
    modules_cache = cache_base / "modules"
    assets_cache = cache_base / "assets"
    for path in (hub_cache, datasets_cache, modules_cache, assets_cache):
        path.mkdir(parents=True, exist_ok=True)

    os.environ["HF_CACHE_BASE"] = str(cache_base)
    os.environ["HF_HOME"] = str(cache_base)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(hub_cache)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    os.environ["HF_MODULES_CACHE"] = str(modules_cache)
    os.environ["HF_ASSETS_CACHE"] = str(assets_cache)
    return cache_base


HF_CACHE_BASE = configure_hf_cache()
