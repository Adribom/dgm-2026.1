from .base import BaseFrequencyExtractor
from .fft import FFTExtractor

_EXTRACTORS = {
    "fft": FFTExtractor,
}


def get_extractor(name, **kwargs):
    if name not in _EXTRACTORS:
        raise ValueError(
            f"Unknown extractor: {name}. Available: {list(_EXTRACTORS.keys())}"
        )
    return _EXTRACTORS[name](**kwargs)
