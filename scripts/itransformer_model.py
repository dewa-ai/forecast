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
        # x shape: (batch, n_vars, d_model)
        batch_size, n_vars, d_model = x.shape
        
        # Linear projection
        qkv = self.qkv_linear(x)  # (batch, n_vars, 3*d_model)
        
        # Split into Q, K, V
        q, k, v = torch.chunk(qkv, 3, dim=-1)  # Each: (batch, n_vars, d_model)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, n_vars, self.n_heads, self.d_k).transpose(1, 2)  # (batch, heads, n_vars, d_k)
        k = k.view(batch_size, n_vars, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, n_vars, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores on variate dimension
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)  # (batch, heads, n_vars, n_vars)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = torch.matmul(attn, v)  # (batch, heads, n_vars, d_k)
        out = out.transpose(1, 2).contiguous().view(batch_size, n_vars, d_model)  # (batch, n_vars, d_model)
        
        # Output projection
        out = self.out_linear(out)
        
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
        # x shape: (batch, n_vars, d_model)
        
        # Self-attention with residual
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
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
        
        # Input embedding: project each variate's time series (seq_len) to d_model
        self.embedding = nn.Linear(seq_len, d_model)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            InvertedTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection: d_model -> pred_len
        self.projection = nn.Linear(d_model, pred_len)
        
    def forward(self, x):
        # x shape: (batch, seq_len, n_vars)
        
        # Transpose to (batch, n_vars, seq_len) for embedding
        x = x.transpose(1, 2)  # (batch, n_vars, seq_len)
        
        # Embed each variate's time series
        x_emb = self.embedding(x)  # (batch, n_vars, d_model)
        
        # Apply transformer blocks
        for layer in self.layers:
            x_emb = layer(x_emb)  # (batch, n_vars, d_model)
        
        # Project to prediction length
        out = self.projection(x_emb)  # (batch, n_vars, pred_len)
        
        # Transpose back to (batch, pred_len, n_vars)
        out = out.transpose(1, 2)
        
        return out


class iTransformerForecaster:
    """Wrapper for iTransformer model for easy training and prediction."""
    
    def __init__(self, seq_len: int = 96, pred_len: int = 96,
                 d_model: int = 128, n_heads: int = 8, n_layers: int = 3,
                 d_ff: int = 256, dropout: float = 0.1,
                 batch_size: int = 32, epochs: int = 10, learning_rate: float = 0.001):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model_params = {
            'seq_len': seq_len,
            'pred_len': pred_len,
            'n_vars': 1,
            'd_model': d_model,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'd_ff': d_ff,
            'dropout': dropout
        }
    
    def fit(self, data: np.ndarray):
        """Train iTransformer on historical data."""
        data_scaled = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        dataset = TimeSeriesDataset(data_scaled, self.seq_len, self.pred_len)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model = iTransformer(**self.model_params).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        print(f"Training iTransformer on {self.device}...")
        self.model.train()
        
        for epoch in range(self.epochs):
            total_loss = 0
            for x_batch, y_batch in dataloader:
                # x_batch: (batch, seq_len) -> (batch, seq_len, 1)
                x_batch = x_batch.unsqueeze(-1).to(self.device)
                # y_batch: (batch, pred_len) -> (batch, pred_len, 1)
                y_batch = y_batch.unsqueeze(-1).to(self.device)
                
                optimizer.zero_grad()
                output = self.model(x_batch)  # (batch, pred_len, 1)
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
        """Generate multi-step forecasts."""
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        self.model.eval()
        predictions = []
        
        current_data = data[-self.seq_len:].copy()
        current_data_scaled = self.scaler.transform(current_data.reshape(-1, 1)).flatten()
        
        with torch.no_grad():
            n_steps = (horizon + self.pred_len - 1) // self.pred_len
            
            for _ in range(n_steps):
                # (1, seq_len, 1)
                x = torch.FloatTensor(current_data_scaled).unsqueeze(0).unsqueeze(-1).to(self.device)
                
                pred_scaled = self.model(x).cpu().numpy()[0, :, 0]  # (pred_len,)
                
                pred = self.scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
                predictions.extend(pred)
                
                current_data_scaled = np.concatenate([current_data_scaled[self.pred_len:], pred_scaled])
        
        return np.array(predictions[:horizon])