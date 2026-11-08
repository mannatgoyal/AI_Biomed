"""PyTorch Deep Multimodal Fusion Network for Biomedical Risk Prediction.

Implements a deep neural classifier with residual connections, batch normalization,
and dropout for comparative deep learning benchmarking against XGBoost.
"""

import logging
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class MultimodalFusionMLP(nn.Module):
    """Deep residual MLP network for fusing clinical features and BioBERT embeddings."""

    def __init__(self, input_dim: int, num_classes: int = 2, hidden_dim: int = 128, dropout_rate: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.dropout(self.relu(self.bn1(self.fc1(x))))
        # Residual connection
        h2 = self.dropout(self.relu(self.bn2(self.fc2(h1)))) + h1
        logits = self.out(h2)
        return logits


class MultimodalNeuralClassifier:
    """Scikit-learn compatible wrapper for MultimodalFusionMLP."""

    def __init__(
        self,
        num_classes: int = 2,
        hidden_dim: int = 128,
        epochs: int = 25,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        random_state: int = 42,
    ):
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.model: Optional[MultimodalFusionMLP] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        torch.manual_seed(self.random_state)
        input_dim = X.shape[1]
        self.model = MultimodalFusionMLP(input_dim=input_dim, num_classes=self.num_classes, hidden_dim=self.hidden_dim)
        self.model.train()

        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            x_t = torch.tensor(X, dtype=torch.float32)
            logits = self.model(x_t)
            preds = torch.argmax(logits, dim=1).numpy()
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            x_t = torch.tensor(X, dtype=torch.float32)
            logits = self.model(x_t)
            probs = torch.softmax(logits, dim=1).numpy()
        return probs

