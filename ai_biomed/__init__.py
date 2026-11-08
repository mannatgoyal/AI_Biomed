"""AI_Biomed: Chronic Kidney Disease Risk Stratification using Biomedical NLP and Machine Learning.

This package provides an end-to-end multimodal research framework that combines structured
clinical data, pharmacological and molecular properties, drug label narratives, and contextual
BioBERT representations for precision nephrology and CKD risk prediction.
"""

__version__ = "1.0.0"
__author__ = "Mannat Goyal"

from ai_biomed.config import PathConfig, PreprocessingConfig, BioBERTConfig, ModelConfig

__all__ = [
    "PathConfig",
    "PreprocessingConfig",
    "BioBERTConfig",
    "ModelConfig",
]
