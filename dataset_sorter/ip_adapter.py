"""
Module: ip_adapter.py
IP-Adapter training support — image-conditioned generation.
Allows conditioning on reference images instead of (or alongside) text.

Supported adapter types
-----------------------
standard     : 4 image tokens, general-purpose.
plus         : 16 image tokens, richer image conditioning.
face         : 16 image tokens + InsightFace encoder (buffalo_l), portrait-focused.
composition  : layout/composition guidance from a reference image.
ilora        : Instant LoRA — lightweight adapter merged directly into LoRA weights.
"""
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# All valid adapter type identifiers (also exported to constants.py).
IP_ADAPTER_TYPES: list[str] = ["standard", "plus", "face", "composition", "ilora"]


@dataclass
class IPAdapterConfig:
    """Configuration for IP-Adapter image conditioning.

    Attributes:
        enabled:       Whether IP-Adapter conditioning is active.
        adapter_type:  One of IP_ADAPTER_TYPES.
        scale:         Conditioning scale (0 = text-only, 1 = full IP weight).
        image_encoder: HuggingFace model ID or local path for the image encoder.
                       Overridden to "buffalo_l" (InsightFace) for the *face* type.
        num_tokens:    Number of image tokens injected into cross-attention.
                       Overridden to 16 for *plus* and *face* types on validate().
        drop_rate:     Probability of randomly dropping the image condition during
                       training for classifier-free guidance robustness.
    """

    enabled: bool = False
    adapter_type: str = "standard"
    scale: float = 1.0
    image_encoder: str = "openai/clip-vit-large-patch14"
    num_tokens: int = 4   # 4 for standard, 16 for plus/face
    drop_rate: float = 0.1

    def validate(self) -> "IPAdapterConfig":
        """Validate and normalise config in-place.  Returns self for chaining.

        Raises:
            ValueError: If adapter_type is not a recognised type.
        """
        if self.adapter_type not in IP_ADAPTER_TYPES:
            raise ValueError(
                f"Unknown IP adapter type: {self.adapter_type!r}. "
                f"Must be one of {IP_ADAPTER_TYPES}"
            )
        if self.adapter_type == "plus":
            self.num_tokens = 16
        elif self.adapter_type == "face":
            self.num_tokens = 16
            self.image_encoder = "buffalo_l"  # InsightFace model
        return self
