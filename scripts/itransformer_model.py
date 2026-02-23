#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler


class TimeSeriesDataset(Dataset):
    """Dataset for time series forecasting."""
    
    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        return torch.FloatTensor(x), torch.FloatTensor(y)


class InvertedAttention(nn.Module):
    """Attention mechanism applied on variate (channel) dimension."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.qkv_linear = nn.Linear(d_model, 3 * d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch, n_vars, seq_len)
        batch_size, n_vars, seq_len = x.shape
        
        # Linear projection
        qkv = self.qkv_linear(x.transpose(1, 2))  # (batch, seq_len, 3*d_model)
        qkv = qkv.transpose(1, 2)  # (batch, 3*d_model, seq_len)
        
        # Split into Q, K, V
        q, k, v = torch.chunk(qkv, 3, dim=1)  # Each: (batch, d_model, seq_len)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, self.n_heads, self.d_k, seq_len)
        k = k.view(batch_size, self.n_heads, self.d_k, seq_len)
        v = v.view(batch_size, self.n_heads, self.d_k, seq_len)
        
        # Attention scores on variate dimension
        scores = torch.matmul(q.transpose(-2, -1), k) / np.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = torch.matmul(attn, v.transpose(-2, -1))  # (batch, n_heads, seq_len, d_k)
        out = out.transpose(-2, -1)  # (batch, n_heads, d_k, seq_len)
        out = out.contiguous().view(batch_size, self.d_model, seq_len)
        
        # Output projection
        out = out.transpose(1, 2)  # (batch, seq_len, d_model)
        out = self.out_linear(out)
        out = out.transpose(1, 2)  # (batch, d_model, seq_len)
        
        return out


class InvertedTransformerBlock(nn.Module):
    """Inverted Transformer block with attention on variate dimension."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = InvertedAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch, n_vars, seq_len)
        
        # Self-attention with residual
        attn_out = self.attention(x)
        x = x + self.dropout(attn_out)
        x = x.transpose(1, 2)  # (batch, seq_len, n_vars)
        x = self.norm1(x)
        x = x.transpose(1, 2)  # (batch, n_vars, seq_len)
        
        # Feed-forward with residual
        x_t = x.transpose(1, 2)  # (batch, seq_len, n_vars)
        ff_out = self.feed_forward(x_t)
        x_t = x_t + ff_out
        x_t = self.norm2(x_t)
        x = x_t.transpose(1, 2)  # (batch, n_vars, seq_len)
        
        return x


class iTransformer(nn.Module):
    """
    iTransformer: Inverted Transformer for time series forecasting.
    
    Applies attention on the variate dimension instead of temporal dimension.
    """
    
    def __init__(self, seq_len: int, pred_len: int, n_vars: int = 1,
                 d_model: int = 128, n_heads: int = 8, n_layers: int = 3,
                 d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_vars = n_vars
        self.d_model = d_model
        
        # Input embedding
        self.embedding = nn.Linear(seq_len, d_model)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            InvertedTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.projection = nn.Linear(d_model, pred_len)
        
    def forward(self, x):
        # x shape: (batch, seq_len, n_vars)
        
        # Transpose to (batch, n_vars, seq_len) for inverted attention
        x = x.transpose(1, 2)
        
        # Embedding
        x_t = x.transpose(1, 2)  # (batch, seq_len, n_vars)
        x_emb = self.embedding(x)  # (batch, n_vars, d_model)
        
        # Apply transformer blocks
        for layer in self.layers:
            x_emb = layer(x_emb)
        
        # Project to prediction length
        out = self.projection(x_emb)  # (batch, n_vars, pred_len)
        
        # Transpose back to (batch, pred_len, n_vars)
        out = out.transpose(1, 2)
        
        return out


class iTransformerForecaster:
    """
    Wrapper for iTransformer model for easy training and prediction.
    """
    
    def __init__(self, seq_len: int = 96, pred_len: int = 96,
                 d_model: int = 128, n_heads: int = 8, n_layers: int = 3,
                 d_ff: int = 256, dropout: float = 0.1,
                 batch_size: int = 32, epochs: int = 10, learning_rate: float = 0.001):
        """
        Args:
            seq_len: Input sequence length (lookback window)
            pred_len: Prediction length (forecast horizon)
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            batch_size: Training batch size
            epochs: Number of training epochs
            learning_rate: Learning rate
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model hyperparameters
        self.model_params = {
            'seq_len': seq_len,
            'pred_len': pred_len,
            'n_vars': 1,  # Univariate time series
            'd_model': d_model,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'd_ff': d_ff,
            'dropout': dropout
        }
    
    def fit(self, data: np.ndarray):
        """
        Train iTransformer on historical data.
        
        Args:
            data: 1D array of time series values
        """
        # Normalize data
        data_scaled = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        # Create dataset
        dataset = TimeSeriesDataset(data_scaled, self.seq_len, self.pred_len)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        self.model = iTransformer(**self.model_params).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        print(f"Training iTransformer on {self.device}...")
        self.model.train()
        
        for epoch in range(self.epochs):
            total_loss = 0
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(self.device).unsqueeze(-1)  # Add var dimension
                y_batch = y_batch.to(self.device).unsqueeze(-1)
                
                optimizer.zero_grad()
                output = self.model(x_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % 2 == 0:
                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.6f}')
        
        print("Training completed!")
        return self
    
    def predict(self, data: np.ndarray, horizon: int) -> np.ndarray:
        """
        Generate multi-step forecasts.
        
        Args:
            data: Historical data (at least seq_len values)
            horizon: Number of steps ahead to forecast
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        self.model.eval()
        predictions = []
        
        # Normalize input
        current_data = data[-self.seq_len:].copy()
        current_data_scaled = self.scaler.transform(current_data.reshape(-1, 1)).flatten()
        
        with torch.no_grad():
            n_steps = (horizon + self.pred_len - 1) // self.pred_len
            
            for _ in range(n_steps):
                # Prepare input
                x = torch.FloatTensor(current_data_scaled).unsqueeze(0).unsqueeze(-1).to(self.device)
                
                # Predict
                pred_scaled = self.model(x).cpu().numpy()[0, :, 0]
                
                # Inverse transform
                pred = self.scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
                predictions.extend(pred)
                
                # Update sequence for next prediction
                current_data_scaled = np.concatenate([current_data_scaled[self.pred_len:], pred_scaled])
        
        # Return only requested horizon
        return np.array(predictions[:horizon])


# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    n = 1000
    t = np.linspace(0, 100, n)
    data = np.sin(0.1 * t) + 0.1 * np.random.randn(n) + 0.01 * t
    
    # Split train/test
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    print("="*60)
    print("Testing iTransformer")
    print("="*60)
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
    
    # Train model
    forecaster = iTransformerForecaster(
        seq_len=96,
        pred_len=24,
        d_model=128,
        n_heads=8,
        n_layers=2,
        epochs=5,
        batch_size=32
    )
    
    forecaster.fit(train_data)
    
    # Make predictions
    horizon = len(test_data)
    predictions = forecaster.predict(train_data, horizon)
    
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"First 10 predictions: {predictions[:10]}")
    print(f"First 10 actual: {test_data[:10]}")
    
    # Calculate error
    from evaluation_metrics import calculate_all_metrics
    metrics = calculate_all_metrics(test_data, predictions)
    print(f"\nMetrics:")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  DA: {metrics['da']:.2f}%")
    
    print("\n" + "="*60)
    print("iTransformer test completed!")
    print("="*60)