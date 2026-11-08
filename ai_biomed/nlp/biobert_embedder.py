"""BioBERT Contextual Embedding Extraction Module.

Leverages pretrained dmis-lab/biobert-v1.1 (BERT-base-cased fine-tuned on PubMed abstracts
and PMC full texts) to extract 768-dimensional contextual biomedical embeddings from clinical
notes, drug monographs, and FDA labels.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
from sklearn.decomposition import PCA

from ai_biomed.config import BioBERTConfig, PathConfig

# Configure environment before importing transformers to prevent TensorFlow initialization issues
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
if "tensorflow" not in sys.modules:
    sys.modules["tensorflow"] = None

logger = logging.getLogger(__name__)


class BioBERTEmbedder:
    """Extracts contextual representations using BioBERT v1.1."""

    def __init__(
        self,
        config: Optional[BioBERTConfig] = None,
        path_config: Optional[PathConfig] = None,
    ):
        self.config = config or BioBERTConfig()
        self.path_config = path_config or PathConfig()
        self.tokenizer = None
        self.model = None
        self.pca = None
        self._is_loaded = False

    def _load_model(self):
        """Loads BioBERT tokenizer and PyTorch model offline from local cache."""
        if self._is_loaded:
            return

        import torch
        from transformers import AutoTokenizer, BertModel

        logger.info(f"Loading BioBERT model '{self.config.model_name}' offline...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                local_files_only=True
            )
            self.model = BertModel.from_pretrained(
                self.config.model_name,
                local_files_only=True
            )
        except Exception as e:
            logger.warning(f"Local-only load encountered {e}. Attempting standard load...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = BertModel.from_pretrained(self.config.model_name)

        self.model.eval()
        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        self._is_loaded = True
        logger.info(f"BioBERT successfully initialized on {self.device}.")

    def encode_texts(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        cache_path: Optional[Union[str, Path]] = None,
        force_recompute: bool = False,
    ) -> np.ndarray:
        """Encodes a list of biomedical text strings into 768-dimensional contextual vectors.

        Args:
            texts: List of text strings to embed.
            batch_size: Batch size for tokenization and model inference.
            cache_path: Optional path to save/load cached embeddings.
            force_recompute: Whether to ignore cache and recompute.

        Returns:
            np.ndarray of shape (len(texts), 768).
        """
        if cache_path is not None:
            cpath = Path(cache_path)
            if cpath.exists() and not force_recompute:
                logger.info(f"Loading cached BioBERT embeddings from: {cpath}")
                cached = np.load(cpath)
                if cached.shape[0] == len(texts) and cached.shape[1] == self.config.embedding_dim:
                    return cached
                else:
                    logger.warning(f"Cached shape {cached.shape} does not match expected ({len(texts)}, {self.config.embedding_dim}). Recomputing...")

        self._load_model()
        import torch

        bs = batch_size or self.config.batch_size
        all_embeddings = []
        logger.info(f"Extracting BioBERT embeddings for {len(texts)} texts (batch_size={bs})...")

        with torch.no_grad():
            for i in range(0, len(texts), bs):
                batch_texts = texts[i : i + bs]
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_seq_length,
                    return_tensors="pt"
                ).to(self.device)

                outputs = self.model(**inputs)
                
                # Contextual pooling
                if self.config.pooling_strategy == "mean":
                    # Mean pooling over non-padded tokens using attention mask
                    attention_mask = inputs["attention_mask"].unsqueeze(-1)
                    token_embeddings = outputs.last_hidden_state
                    sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
                    sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                    pooled = (sum_embeddings / sum_mask).cpu().numpy()
                else:
                    # CLS token representation
                    pooled = outputs.last_hidden_state[:, 0, :].cpu().numpy()

                all_embeddings.append(pooled)

        embeddings = np.vstack(all_embeddings)
        assert embeddings.shape == (len(texts), self.config.embedding_dim)

        if cache_path is not None:
            cpath = Path(cache_path)
            cpath.parent.mkdir(parents=True, exist_ok=True)
            np.save(cpath, embeddings)
            logger.info(f"Saved {embeddings.shape} embeddings to cache: {cpath}")

        return embeddings

    def project_embeddings(
        self,
        embeddings: np.ndarray,
        n_components: int = 16,
        fit: bool = True
    ) -> np.ndarray:
        """Optionally projects high-dimensional embeddings using PCA to prevent feature explosion.

        Args:
            embeddings: Array of shape (N, 768).
            n_components: Target projection dimensionality.
            fit: Whether to fit a new PCA or transform using previously fitted PCA.

        Returns:
            np.ndarray of shape (N, n_components).
        """
        if fit or self.pca is None:
            self.pca = PCA(n_components=n_components, random_state=42)
            return self.pca.fit_transform(embeddings)
        return self.pca.transform(embeddings)
