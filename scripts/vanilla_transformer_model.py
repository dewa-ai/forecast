#!/usr/bin/env python3
"""
vanilla_transformer_model.py — Standard (temporal self-attention) Transformer baseline.

Added per committee feedback requesting a baseline that isolates the effect of
iTransformer's variate-wise attention mechanism. This model applies self-attention
across the TIME dimension (the original formulation from Vaswani et al., 2017),
in contrast to iTransformer, which applies attention across the VARIATE dimension.

Interface mirrors iTransformerForecaster exactly (.fit(data) / .predict(data, horizon))
so it can be dropped into run_full_ablation.py with minimal changes.

Since each currency pair is modeled as a univariate series (n_vars=1, same as the
iTransformer setup in this study), "temporal attention" here means the model attends
across the 30 time steps of the lookback window, encoding each time step as a token.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

# Reuse the same windowing dataset used by iTransformer for a fair, matched comparison
from itransformer_model import TimeSeriesDataset


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017, Section 3.5)."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        return x + self.pe[:, : x.size(1), :]


class VanillaTransformer(nn.Module):
    """
    Standard Transformer encoder for time series forecasting.

    Applies self-attention across the TEMPORAL dimension: each of the `seq_len`
    time steps is treated as a token, with a learned scalar-to-d_model embedding
    plus sinusoidal positional encoding, following the original Transformer
    architecture (Vaswani et al., 2017), in contrast to iTransformer's
    variate-wise attention.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_vars: int = 1,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_vars = n_vars
        self.d_model = d_model

        # Each time step (scalar value per variate) is embedded to d_model
        self.value_embedding = nn.Linear(n_vars, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_len + 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Flatten the encoded sequence and project to (pred_len * n_vars)
        self.projection = nn.Linear(d_model * seq_len, pred_len * n_vars)

    def forward(self, x):
        # x shape: (batch, seq_len, n_vars)
        batch_size = x.shape[0]

        x_emb = self.value_embedding(x)          # (batch, seq_len, d_model)
        x_emb = self.pos_encoding(x_emb)          # add temporal position info
        encoded = self.encoder(x_emb)             # (batch, seq_len, d_model) -- temporal self-attention

        flat = encoded.reshape(batch_size, -1)    # (batch, seq_len * d_model)
        out = self.projection(flat)               # (batch, pred_len * n_vars)
        out = out.view(batch_size, self.pred_len, self.n_vars)
        return out


class VanillaTransformerForecaster:
    """Wrapper for VanillaTransformer, interface-matched to iTransformerForecaster."""

    def __init__(
        self,
        seq_len: int = 30,
        pred_len: int = 1,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        batch_size: int = 32,
        epochs: int = 30,
        learning_rate: float = 0.001,
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_params = {
            "seq_len": seq_len,
            "pred_len": pred_len,
            "n_vars": 1,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "d_ff": d_ff,
            "dropout": dropout,
        }

    def fit(self, data: np.ndarray):
        """Train vanilla Transformer on historical data. Same procedure as iTransformerForecaster.fit."""
        data_scaled = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()

        dataset = TimeSeriesDataset(data_scaled, self.seq_len, self.pred_len)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = VanillaTransformer(**self.model_params).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        print(f"Training Vanilla Transformer on {self.device}...")
        self.model.train()

        for epoch in range(self.epochs):
            total_loss = 0
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.unsqueeze(-1).to(self.device)  # (batch, seq_len, 1)
                y_batch = y_batch.unsqueeze(-1).to(self.device)  # (batch, pred_len, 1)

                optimizer.zero_grad()
                output = self.model(x_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % 2 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.6f}")

        print("Training completed!")
        return self

    def predict(self, data: np.ndarray, horizon: int) -> np.ndarray:
        """Generate multi-step forecasts. Same rolling procedure as iTransformerForecaster.predict."""
        if self.model is None:
            raise ValueError("Model must be trained before prediction")

        self.model.eval()
        predictions = []

        current_data = data[-self.seq_len :].copy()
        current_data_scaled = self.scaler.transform(current_data.reshape(-1, 1)).flatten()

        with torch.no_grad():
            n_steps = (horizon + self.pred_len - 1) // self.pred_len

            for _ in range(n_steps):
                x = torch.FloatTensor(current_data_scaled).unsqueeze(0).unsqueeze(-1).to(self.device)
                pred_scaled = self.model(x).cpu().numpy()[0, :, 0]  # (pred_len,)

                pred = self.scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
                predictions.extend(pred)

                current_data_scaled = np.concatenate(
                    [current_data_scaled[self.pred_len :], pred_scaled]
                )

        return np.array(predictions[:horizon])


# Example usage / smoke test
if __name__ == "__main__":
    np.random.seed(42)
    n = 500
    data = np.cumsum(np.random.randn(n)) + 100

    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]
    horizon = len(test_data)

    print("=" * 60)
    print("Testing Vanilla Transformer Model")
    print("=" * 60)

    m = VanillaTransformerForecaster(seq_len=30, pred_len=5, d_model=64, n_heads=4, n_layers=2, epochs=5)
    m.fit(train_data)
    pred = m.predict(train_data, horizon=horizon)
    print(f"Vanilla Transformer predictions shape: {pred.shape}")
    print("Smoke test passed!")