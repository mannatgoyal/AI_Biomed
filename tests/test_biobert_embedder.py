"""Tests for BioBERT contextual embedding extraction and PCA projection."""

import numpy as np
from ai_biomed.nlp.biobert_embedder import BioBERTEmbedder
from ai_biomed.config import BioBERTConfig


def test_biobert_embedder_extraction():
    config = BioBERTConfig(max_seq_length=32, batch_size=2)
    embedder = BioBERTEmbedder(config=config)
    
    sample_texts = [
        "Patient presents with renal dysfunction and acute tubular necrosis.",
        "Gentamicin induced nephrotoxicity requiring serum creatinine monitoring."
    ]
    
    embeddings = embedder.encode_texts(sample_texts)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, 768)
    assert not np.isnan(embeddings).any()


def test_biobert_pca_projection():
    config = BioBERTConfig()
    embedder = BioBERTEmbedder(config=config)
    
    dummy_embs = np.random.randn(20, 768)
    projected = embedder.project_embeddings(dummy_embs, n_components=16)
    
    assert projected.shape == (20, 16)
