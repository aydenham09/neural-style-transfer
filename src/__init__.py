"""Neural Style Transfer (NST) Package
Core modules for custom NST implementation using PyTorch and frozen pretrained VGG-19.
"""

from src import utils
from src.utils import (
    get_device,
    load_image,
    image_to_tensor,
    normalize_imagenet,
    denormalize_imagenet,
    is_normalized,
    tensor_to_image,
    save_image,
)
from src import model
from src.model import (
    VGG19FeatureExtractor,
    build_vgg19_extractor,
    DEFAULT_CONTENT_LAYER,
    DEFAULT_STYLE_LAYERS,
    VGG19_LAYER_INDEX_MAP,
    is_tensor_imagenet_normalized,
)
from src import loss
from src.loss import (
    gram_matrix,
    ContentLoss,
    SingleLayerStyleLoss,
    StyleLoss,
    TotalVariationLoss,
    NSTLoss,
)
from src import transfer
from src.transfer import (
    StyleTransferOptimizer,
    run_single_transfer,
)

__version__ = "0.1.0"
__all__ = [
    "utils",
    "model",
    "loss",
    "transfer",
    "get_device",
    "load_image",
    "image_to_tensor",
    "normalize_imagenet",
    "denormalize_imagenet",
    "is_normalized",
    "tensor_to_image",
    "save_image",
    "VGG19FeatureExtractor",
    "build_vgg19_extractor",
    "DEFAULT_CONTENT_LAYER",
    "DEFAULT_STYLE_LAYERS",
    "VGG19_LAYER_INDEX_MAP",
    "is_tensor_imagenet_normalized",
    "gram_matrix",
    "ContentLoss",
    "SingleLayerStyleLoss",
    "StyleLoss",
    "TotalVariationLoss",
    "NSTLoss",
    "StyleTransferOptimizer",
    "run_single_transfer",
]
