#!/usr/bin/env python3
"""
Includes:
- ARIMA (classical statistical)
- LSTM (neural time-series)
- XGBoost (feature-based ML)

Example:
    from baseline_models import ARIMAForecaster, LSTMForecaster, XGBoostForecaster
    
    model = ARIMAForecaster(order=(1,1,1))
    model.fit(train_data)
    predictions = model.predict(horizon=30)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Statistical models
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
except ImportError:
    print("Warning: statsmodels not installed. ARIMA will not work.")
    print("Install with: pip install statsmodels")

# Deep learning
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("Warning: torch not installed. LSTM will not work.")
    print("Install with: pip install torch")

# ML models
try:
    import xgboost as xgb
except ImportError:
    print("Warning: xgboost not installed. XGBoost will not work.")
    print("Install with: pip install xgboost")

from sklearn.preprocessing import MinMaxScaler


class ARIMAForecaster:
    """
    ARIMA (AutoRegressive Integrated Moving Average) model.
    
    Classical statistical baseline for time series forecasting.
    """
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)):
        """
        Args:
            order: (p, d, q) where:
                - p: AR order (autoregressive)
                - d: Integration order (differencing)
                - q: MA order (moving average)
        """
        self.order = order
        self.model = None
        self.fitted_model = None
        
    def fit(self, data: np.ndarray):
        """
        Fit ARIMA model to training data.
        
        Args:
            data: 1D array of time series values
        """
        self.model = ARIMA(data, order=self.order)
        self.fitted_model = self.model.fit()
        return self
    
    def predict(self, horizon: int) -> np.ndarray:
        """
        Generate forecasts for specified horizon.
        
        Args:
            horizon: Number of steps ahead to forecast
            
        Returns:
            Array of predictions
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction")
        
        forecast = self.fitted_model.forecast(steps=horizon)
        return np.array(forecast)
    
    def find_optimal_order(self, data: np.ndarray, max_p: int = 5, 
                          max_d: int = 2, max_q: int = 5) -> Tuple[int, int, int]:
        """
        Find optimal ARIMA order using AIC criterion.
        
        Args:
            data: Time series data
            max_p, max_d, max_q: Maximum values to try for each parameter
            
        Returns:
            Optimal (p, d, q) order
        """
        best_aic = np.inf
        best_order = None
        
        print(f"Searching for optimal ARIMA order...")
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(data, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        print(f"Optimal order: {best_order}, AIC: {best_aic:.2f}")
        self.order = best_order
        return best_order


class LSTMModel(nn.Module):
    """LSTM neural network for time series forecasting."""
    
    def __init__(self, input_size: int = 1, hidden_size: int = 50, 
                 num_layers: int = 2, output_size: int = 1, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Take output from last time step
        out = self.fc(lstm_out[:, -1, :])
        return out


class LSTMForecaster:
    """
    LSTM (Long Short-Term Memory) forecaster.
    
    Neural time-series baseline with memory of long-term dependencies.
    """
    
    def __init__(self, seq_length: int = 30, hidden_size: int = 50, 
                 num_layers: int = 2, epochs: int = 50, batch_size: int = 32,
                 learning_rate: float = 0.001):
        """
        Args:
            seq_length: Number of time steps to look back
            hidden_size: LSTM hidden dimension size
            num_layers: Number of LSTM layers
            epochs: Training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
        """
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.model = None
        self.scaler = MinMaxScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create input-output sequences for LSTM."""
        X, y = [], []
        for i in range(len(data) - self.seq_length):
            X.append(data[i:i + self.seq_length])
            y.append(data[i + self.seq_length])
        return np.array(X), np.array(y)
    
    def fit(self, data: np.ndarray):
        """
        Fit LSTM model to training data.
        
        Args:
            data: 1D array of time series values
        """
        # Normalize data
        data_scaled = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y = self._create_sequences(data_scaled)
        X = X.reshape(-1, self.seq_length, 1)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Create model
        self.model = LSTMModel(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=1
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs.squeeze(), y_tensor)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.6f}')
        
        return self
    
    def predict(self, data: np.ndarray, horizon: int) -> np.ndarray:
        """
        Generate forecasts using the last seq_length values.
        
        Args:
            data: Historical data (at least seq_length values)
            horizon: Number of steps ahead to forecast
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        self.model.eval()
        predictions = []
        
        # Use last seq_length values as initial input
        current_seq = data[-self.seq_length:].copy()
        current_seq_scaled = self.scaler.transform(current_seq.reshape(-1, 1)).flatten()
        
        with torch.no_grad():
            for _ in range(horizon):
                # Prepare input
                x = torch.FloatTensor(current_seq_scaled).reshape(1, self.seq_length, 1).to(self.device)
                
                # Predict next value
                pred_scaled = self.model(x).cpu().numpy()[0, 0]
                
                # Inverse transform
                pred = self.scaler.inverse_transform([[pred_scaled]])[0, 0]
                predictions.append(pred)
                
                # Update sequence (roll forward)
                current_seq_scaled = np.roll(current_seq_scaled, -1)
                current_seq_scaled[-1] = pred_scaled
        
        return np.array(predictions)


class XGBoostForecaster:
    """
    XGBoost forecaster with feature engineering.
    
    Feature-based ML baseline using gradient boosting.
    """
    
    def __init__(self, n_lags: int = 30, n_estimators: int = 100, 
                 learning_rate: float = 0.1, max_depth: int = 5):
        """
        Args:
            n_lags: Number of lag features to create
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate
            max_depth: Maximum tree depth
        """
        self.n_lags = n_lags
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )
        self.scaler = MinMaxScaler()
        
    def _create_features(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create lag features and rolling statistics.
        
        Features:
        - Lag values (t-1, t-2, ..., t-n_lags)
        - Rolling mean (7, 14, 30 days)
        - Rolling std (7, 14, 30 days)
        """
        df = pd.DataFrame({'value': data})
        
        # Lag features
        for i in range(1, self.n_lags + 1):
            df[f'lag_{i}'] = df['value'].shift(i)
        
        # Rolling statistics
        for window in [7, 14, 30]:
            df[f'rolling_mean_{window}'] = df['value'].shift(1).rolling(window).mean()
            df[f'rolling_std_{window}'] = df['value'].shift(1).rolling(window).std()
        
        # Drop NaN rows
        df = df.dropna()
        
        # Separate features and target
        X = df.drop('value', axis=1).values
        y = df['value'].values
        
        return X, y
    
    def fit(self, data: np.ndarray):
        """
        Fit XGBoost model to training data.
        
        Args:
            data: 1D array of time series values
        """
        X, y = self._create_features(data)
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y, verbose=False)
        
        # Store last values for prediction
        self.last_data = data.copy()
        
        return self
    
    def predict(self, horizon: int) -> np.ndarray:
        """
        Generate forecasts for specified horizon.
        
        Args:
            horizon: Number of steps ahead to forecast
            
        Returns:
            Array of predictions
        """
        predictions = []
        current_data = self.last_data.copy()
        
        for _ in range(horizon):
            # Create features from current data
            X, _ = self._create_features(current_data)
            X_last = X[-1:].copy()
            
            # Scale and predict
            X_scaled = self.scaler.transform(X_last)
            pred = self.model.predict(X_scaled)[0]
            predictions.append(pred)
            
            # Update data with prediction
            current_data = np.append(current_data, pred)
        
        return np.array(predictions)


# Example usage
if __name__ == "__main__":
    # Generate example data
    np.random.seed(42)
    n = 500
    data = np.cumsum(np.random.randn(n)) + 100
    
    # Split train/test
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]
    horizon = len(test_data)
    
    print("="*60)
    print("Testing Baseline Models")
    print("="*60)
    
    # ARIMA
    print("\n1. ARIMA Model")
    arima = ARIMAForecaster(order=(1, 1, 1))
    arima.fit(train_data)
    arima_pred = arima.predict(horizon)
    print(f"ARIMA predictions shape: {arima_pred.shape}")
    
    # LSTM
    print("\n2. LSTM Model")
    lstm = LSTMForecaster(seq_length=30, epochs=20)
    lstm.fit(train_data)
    lstm_pred = lstm.predict(train_data, horizon)
    print(f"LSTM predictions shape: {lstm_pred.shape}")
    
    # XGBoost
    print("\n3. XGBoost Model")
    xgb_forecaster = XGBoostForecaster(n_lags=30)
    xgb_forecaster.fit(train_data)
    xgb_pred = xgb_forecaster.predict(horizon)
    print(f"XGBoost predictions shape: {xgb_pred.shape}")
    
    print("\n" + "="*60)
    print("All models trained and tested successfully!")
    print("="*60)