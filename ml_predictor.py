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
    
    def prepare_data(
        self, data: pd.DataFrame, lookback: int = 60, train_frac: float = 0.8
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[dict]]:
        """Prepare LSTM training data targeting LOG-RETURNS, not raw prices.

        Why log-returns:
          1. Stationarity. Prices are I(1) random walks; their level has no
             stable distribution. Log-returns are approximately stationary with
             mean ≈ 0 and σ ≈ 1% daily — far easier for the network to model.
          2. No scaler leakage. We standardize by σ estimated on the TRAINING
             split only.
          3. No saturation. The previous price-target model with MinMaxScaler
             could not predict above its training-period max because outputs
             were learned in [0, 1]; the model under-shot every bull move.
          4. Scale-invariance. Multiplying prices by a constant leaves the
             target distribution unchanged.

        Reconstruction at inference time:  Ŝ_{t+h} = S_t · exp(Σ_{k≤h} r̂_{t+k}·σ_train).
        """
        if data.empty or len(data) < lookback + 2:
            return None, None, None

        closes = data['Close'].to_numpy(dtype=float)
        if (closes <= 0).any():
            return None, None, None

        # Log-returns; index 0 is the first available return r_1 = log S_1/S_0.
        log_rets = np.diff(np.log(closes))
        n = len(log_rets)

        # Chronological split BEFORE estimating σ → no peeking at future.
        split_idx = max(lookback + 1, int(n * train_frac))
        if split_idx >= n:
            split_idx = n - 1
        sigma_train = float(np.std(log_rets[:split_idx], ddof=1))
        if sigma_train <= 0 or not np.isfinite(sigma_train):
            return None, None, None

        scaled = log_rets / sigma_train

        X, y = [], []
        for i in range(lookback, n):
            X.append(scaled[i - lookback:i])
            y.append(scaled[i])

        meta = {
            "sigma_train": sigma_train,
            "last_close": float(closes[-1]),
            "split_idx": int(split_idx),
        }
        return np.array(X), np.array(y), meta
    
    def build_lstm_model(self, input_shape: Tuple) -> "Sequential":
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
            stock = yf.Ticker(ticker)
            data = stock.history(period=training_period, auto_adjust=True)

            if data.empty or len(data) < lookback + periods + 2:
                logger.warning(f"Insufficient data for {ticker}, using fallback")
                return self._fallback_prediction(ticker, periods)

            X, y, meta = self.prepare_data(data, lookback)
            if X is None:
                return self._fallback_prediction(ticker, periods)

            X = X.reshape((X.shape[0], X.shape[1], 1))

            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            model = self.build_lstm_model((X.shape[1], 1))

            try:
                from tensorflow.keras.callbacks import EarlyStopping
                early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                model.fit(
                    X_train, y_train,
                    epochs=50, batch_size=32,
                    validation_data=(X_test, y_test) if len(X_test) > 0 else None,
                    callbacks=[early_stop] if len(X_test) > 0 else None,
                    verbose=0,
                )
            except Exception as e:
                logger.debug(f"Training error: {e}, using simpler training")
                model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

            # Build the seed window from the LAST `lookback` log-returns in the
            # training-period series (scaled by σ_train, the same units the
            # model was trained on).
            log_rets = np.diff(np.log(data['Close'].to_numpy(dtype=float)))
            sigma_train = meta["sigma_train"]
            seed_scaled = (log_rets[-lookback:] / sigma_train).reshape(1, lookback, 1)

            # Autoregressive rollout. Errors compound — the model's
            # uncertainty grows roughly like √h. We don't pretend otherwise.
            preds_scaled = []
            current = seed_scaled.copy()
            for _ in range(periods):
                next_scaled = float(model.predict(current, verbose=0)[0, 0])
                preds_scaled.append(next_scaled)
                current = np.append(current[:, 1:, :], [[[next_scaled]]], axis=1)

            # Reconstruct prices: S_{t+h} = S_t · exp( Σ r̂_{t+k} · σ_train )
            preds_log_rets = np.array(preds_scaled) * sigma_train
            cum_log = np.cumsum(preds_log_rets)
            current_price = float(data['Close'].iloc[-1])
            predicted_prices = current_price * np.exp(cum_log)
            predicted_price = float(predicted_prices[-1])
            expected_return = (predicted_price / current_price - 1.0) * 100.0

            # Held-out MAE in σ-units. Convert to *daily-return RMSE* for an
            # interpretable confidence score.
            if len(X_test) > 0:
                val_pred = model.predict(X_test, verbose=0).flatten()
                val_mae_sigma = float(np.mean(np.abs(val_pred - y_test)))
                # MAE/σ ≈ 0.8 is "random walk baseline" (E|Z| = 0.798); below
                # that the model adds some skill, above it the model is worse
                # than a constant-σ random walk.
                skill = max(0.0, 0.8 - val_mae_sigma) / 0.8
                confidence = float(min(100.0, max(0.0, skill * 100.0)))
            else:
                confidence = 0.0

            return {
                "ticker": ticker,
                "current_price": current_price,
                "predicted_price": predicted_price,
                "expected_return": expected_return,
                "predictions": predicted_prices.tolist(),
                "confidence": confidence,
                "method": "LSTM_log_returns",
                "periods": periods,
                "horizon_uncertainty_pct": float(sigma_train * np.sqrt(periods) * 100),
            }

        except Exception as e:
            logger.warning(f"LSTM prediction failed for {ticker}: {e}")
            return self._fallback_prediction(ticker, periods)
    
    def _fallback_prediction(self, ticker: str, periods: int) -> Dict:
        """Fallback to a GBM-style statistical projection.

        Reports the LogNormal MEDIAN as the point forecast (S_0 · exp(m·t),
        m = mean log-return), which is the right central tendency under GBM.
        The previous version used (1 + arithmetic_mean)^N which conflated
        compounding regimes and gave a Jensen-biased projection.
        """
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="1y", auto_adjust=True)

            if data.empty or len(data) < 30:
                return {"ticker": ticker, "error": "No data available"}

            closes = data['Close'].to_numpy(dtype=float)
            if (closes <= 0).any():
                return {"ticker": ticker, "error": "Invalid price series"}
            log_rets = np.diff(np.log(closes))
            m_daily = float(np.mean(log_rets))
            sigma_daily = float(np.std(log_rets, ddof=1))
            current_price = float(closes[-1])

            # GBM median (mode of the log-return path)
            predicted_price = current_price * float(np.exp(m_daily * periods))
            # 1-σ band on log price at horizon
            sd_log = sigma_daily * np.sqrt(periods)
            band_low = float(current_price * np.exp(m_daily * periods - sd_log))
            band_high = float(current_price * np.exp(m_daily * periods + sd_log))
            expected_return = (predicted_price / current_price - 1.0) * 100.0

            return {
                "ticker": ticker,
                "current_price": current_price,
                "predicted_price": predicted_price,
                "expected_return": expected_return,
                "one_sigma_band": (band_low, band_high),
                "horizon_uncertainty_pct": float(sd_log * 100),
                "confidence": 60.0,
                "method": "GBM_median (fallback)",
                "periods": periods,
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

