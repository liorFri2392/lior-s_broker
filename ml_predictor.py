#!/usr/bin/env python3
"""
Machine Learning Predictor - Advanced LSTM models for price forecasting
Uses deep learning for more accurate predictions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import yfinance as yf
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.preprocessing import MinMaxScaler
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. LSTM predictions will use fallback methods.")

class MLPredictor:
    """Advanced ML-based price prediction using LSTM."""
    
    def __init__(self):
        self.scaler = None
        self.model = None
        self.use_lstm = TENSORFLOW_AVAILABLE
    
    def prepare_data(self, data: pd.DataFrame, lookback: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM training."""
        if data.empty or len(data) < lookback + 1:
            return None, None
        
        # Use closing prices
        prices = data['Close'].values.reshape(-1, 1)
        
        # Normalize
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_prices = scaler.fit_transform(prices)
        
        X, y = [], []
        for i in range(lookback, len(scaled_prices)):
            X.append(scaled_prices[i-lookback:i, 0])
            y.append(scaled_prices[i, 0])
        
        return np.array(X), np.array(y), scaler
    
    def build_lstm_model(self, input_shape: Tuple) -> Sequential:
        """Build LSTM model architecture."""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def predict_with_lstm(
        self,
        ticker: str,
        periods: int = 30,
        lookback: int = 60,
        training_period: str = "2y"
    ) -> Dict:
        """
        Predict future prices using LSTM.
        
        Args:
            ticker: Stock/ETF ticker
            periods: Number of days to predict
            lookback: Number of days to look back for training
            training_period: Historical period for training
        
        Returns:
            Dict with predictions and metrics
        """
        if not self.use_lstm:
            return self._fallback_prediction(ticker, periods)
        
        try:
            # Get historical data
            stock = yf.Ticker(ticker)
            data = stock.history(period=training_period)
            
            if data.empty or len(data) < lookback + periods:
                logger.warning(f"Insufficient data for {ticker}, using fallback")
                return self._fallback_prediction(ticker, periods)
            
            # Prepare data
            X, y, scaler = self.prepare_data(data, lookback)
            if X is None:
                return self._fallback_prediction(ticker, periods)
            
            # Reshape for LSTM
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            # Split train/test
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Build and train model
            model = self.build_lstm_model((X.shape[1], 1))
            
            # Train (with early stopping to avoid overfitting)
            try:
                from tensorflow.keras.callbacks import EarlyStopping
                early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                
                history = model.fit(
                    X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stop],
                    verbose=0
                )
            except Exception as e:
                logger.debug(f"Training error: {e}, using simpler training")
                model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
            
            # Predict future
            last_sequence = data['Close'].tail(lookback).values
            last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1))
            
            predictions = []
            current_sequence = last_sequence_scaled[-lookback:].reshape(1, lookback, 1)
            
            for _ in range(periods):
                next_pred = model.predict(current_sequence, verbose=0)[0, 0]
                predictions.append(next_pred)
                
                # Update sequence
                current_sequence = np.append(current_sequence[:, 1:, :], [[[next_pred]]], axis=1)
            
            # Inverse transform
            predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
            
            # Calculate expected return
            current_price = data['Close'].iloc[-1]
            predicted_price = predictions[-1]
            expected_return = (predicted_price / current_price - 1) * 100
            
            # Calculate confidence (based on training accuracy)
            train_pred = model.predict(X_train, verbose=0)
            train_mae = np.mean(np.abs(train_pred.flatten() - y_train))
            confidence = max(0, min(100, 100 - (train_mae * 100)))
            
            return {
                "ticker": ticker,
                "current_price": current_price,
                "predicted_price": predicted_price,
                "expected_return": expected_return,
                "predictions": predictions.tolist(),
                "confidence": confidence,
                "method": "LSTM",
                "periods": periods
            }
            
        except Exception as e:
            logger.warning(f"LSTM prediction failed for {ticker}: {e}")
            return self._fallback_prediction(ticker, periods)
    
    def _fallback_prediction(self, ticker: str, periods: int) -> Dict:
        """Fallback to statistical prediction if LSTM unavailable."""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="1y")
            
            if data.empty:
                return {"ticker": ticker, "error": "No data available"}
            
            # Simple trend-based prediction
            returns = data['Close'].pct_change().dropna()
            avg_return = returns.mean()
            current_price = data['Close'].iloc[-1]
            
            # Project forward
            predicted_price = current_price * (1 + avg_return) ** periods
            expected_return = (predicted_price / current_price - 1) * 100
            
            return {
                "ticker": ticker,
                "current_price": current_price,
                "predicted_price": predicted_price,
                "expected_return": expected_return,
                "confidence": 60,  # Lower confidence for fallback
                "method": "Statistical (fallback)",
                "periods": periods
            }
        except Exception as e:
            logger.error(f"Fallback prediction failed: {e}")
            return {"ticker": ticker, "error": str(e)}
    
    def predict_portfolio_returns(
        self,
        tickers: List[str],
        periods: int = 30,
        weights: Dict[str, float] = None
    ) -> Dict:
        """Predict returns for entire portfolio."""
        if weights is None:
            weights = {ticker: 1.0 / len(tickers) for ticker in tickers}
        
        predictions = {}
        total_expected_return = 0
        
        for ticker in tickers:
            pred = self.predict_with_lstm(ticker, periods)
            if "error" not in pred:
                predictions[ticker] = pred
                weight = weights.get(ticker, 0)
                total_expected_return += pred["expected_return"] * weight
        
        return {
            "portfolio_expected_return": total_expected_return,
            "individual_predictions": predictions,
            "periods": periods
        }

if __name__ == "__main__":
    predictor = MLPredictor()
    
    # Test prediction
    result = predictor.predict_with_lstm("SPY", periods=30)
    print(f"Prediction for SPY: {result}")

