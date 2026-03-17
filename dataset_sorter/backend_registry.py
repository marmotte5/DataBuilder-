"""Plugin registry for training backends — auto-discovery and dynamic loading.

Replaces the hardcoded _BACKEND_REGISTRY dict in trainer.py with a
plugin-based system that auto-discovers backends by convention and
supports third-party extensions via entry points.

Backend modules must:
1. Be named train_backend_<model_name>.py in the dataset_sorter package
2. Contain a class ending in 'Backend' that inherits from TrainBackendBase
"""

import importlib
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


class BackendRegistry:
    """Dynamic registry for training backends.

    Discovers backends by scanning the dataset_sorter package for modules
    matching the train_backend_*.py pattern and extracting their Backend class.
    """

    def __init__(self):
        self._backends: dict[str, tuple[str, str]] = {}
        self._loaded: dict[str, type] = {}

    def register(self, model_key: str, module_path: str, class_name: str):
        """Manually register a backend."""
        self._backends[model_key] = (module_path, class_name)

    def discover(self):
        """Auto-discover backends from the dataset_sorter package.

        Scans for modules matching train_backend_*.py and registers them.
        """
        package_dir = Path(__file__).parent
        for item in sorted(package_dir.glob("train_backend_*.py")):
            stem = item.stem  # e.g. "train_backend_sdxl"
            model_key = stem.replace("train_backend_", "")  # e.g. "sdxl"

            if model_key in self._backends:
                continue  # Already registered (manual override)

            module_path = f"dataset_sorter.{stem}"
            # Discover class name by convention: capitalize + "Backend"
            # e.g. sdxl -> SDXLBackend, sd15 -> SD15Backend, flux2 -> Flux2Backend
            class_name = self._infer_class_name(model_key)

            self._backends[model_key] = (module_path, class_name)
            log.debug(f"Discovered backend: {model_key} -> {module_path}.{class_name}")

        log.info(f"Backend registry: {len(self._backends)} backends discovered")

    @staticmethod
    def _infer_class_name(model_key: str) -> str:
        """Infer class name from model key by convention.

        Examples: sdxl -> SDXLBackend, sd15 -> SD15Backend,
                  flux -> FluxBackend, auraflow -> AuraFlowBackend
        """
        # Known mappings for non-trivial names
        _KNOWN = {
            "sdxl": "SDXLBackend",
            "sd15": "SD15Backend",
            "sd2": "SD2Backend",
            "sd3": "SD3Backend",
            "sd35": "SD35Backend",
            "flux": "FluxBackend",
            "flux2": "Flux2Backend",
            "pixart": "PixArtBackend",
            "cascade": "StableCascadeBackend",
            "hunyuan": "HunyuanDiTBackend",
            "kolors": "KolorsBackend",
            "auraflow": "AuraFlowBackend",
            "sana": "SanaBackend",
            "hidream": "HiDreamBackend",
            "chroma": "ChromaBackend",
            "zimage": "ZImageBackend",
            "base": "TrainBackendBase",
        }
        if model_key in _KNOWN:
            return _KNOWN[model_key]
        # Fallback: capitalize and append Backend
        return model_key.capitalize() + "Backend"

    def get_backend_class(self, model_key: str) -> Optional[type]:
        """Load and return the backend class for a model key."""
        if model_key in self._loaded:
            return self._loaded[model_key]

        entry = self._backends.get(model_key)
        if entry is None:
            return None

        module_path, class_name = entry
        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            self._loaded[model_key] = cls
            return cls
        except (ImportError, AttributeError) as e:
            log.error(f"Failed to load backend {model_key}: {e}")
            return None

    def instantiate(self, model_key: str, config, device, dtype):
        """Instantiate a backend for the given model key."""
        cls = self.get_backend_class(model_key)
        if cls is None:
            log.warning(f"Unknown backend '{model_key}', falling back to SDXL")
            cls = self.get_backend_class("sdxl")
        if cls is None:
            raise ValueError(
                f"Backend '{model_key}' not found and SDXL fallback unavailable. "
                f"Available backends: {self.available_backends}"
            )
        return cls(config, device, dtype)

    @property
    def available_backends(self) -> list[str]:
        """List all registered backend model keys."""
        return sorted(self._backends.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._backends

    def __len__(self) -> int:
        return len(self._backends)


# Module-level singleton
_registry: Optional[BackendRegistry] = None


def get_registry() -> BackendRegistry:
    """Get the global backend registry (lazy-initialized with auto-discovery)."""
    global _registry
    if _registry is None:
        _registry = BackendRegistry()
        _registry.discover()
    return _registry
