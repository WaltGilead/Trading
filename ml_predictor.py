import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import logging

logger = logging.getLogger(__name__)

class MLPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def _add_features(self, df):
        """
        Internal method to add derived features to the dataframe.
        """
        df = df.copy()
        
        # --- Enhanced Feature Engineering ---
        
        # 1. Distance from SMA 200 (Mean Reversion)
        if 'SMA_200' in df.columns:
            df['Dist_SMA200'] = (df['Close'] - df['SMA_200']) / df['SMA_200']
            
        # 2. Bollinger Band Width (Volatility)
        if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']
            
        # 3. RSI Slope (Momentum Shift)
        if 'RSI' in df.columns:
            df['RSI_Slope'] = df['RSI'].diff(3)
            
        # 4. Lagged Returns (Autocorrelation)
        df['Lagged_Return_1'] = df['Close'].pct_change().shift(1)
        
        return df

    def prepare_features(self, df):
        """
        Creates features for the ML model.
        Assumes df already has technical indicators (RSI, MACD, etc.)
        """
        if df is None or df.empty:
            return None
            
        # Add derived features
        df = self._add_features(df)
        
        # Create Target: Next period return
        df['Target_Return'] = df['Close'].shift(-1).pct_change()
        
        # Drop NaNs created by lag/shift
        df.dropna(inplace=True)
        
        feature_cols = [
            'RSI', 'SMA_50', 'SMA_200', 'MACD_Line', 'MACD_Signal', 'MACD_Hist',
            'BB_Upper', 'BB_Lower', 'BB_Mid', 'StochRSI_K', 'StochRSI_D',
            'Dist_SMA200', 'BB_Width', 'RSI_Slope', 'Lagged_Return_1',
            'ADX', 'OBV_Slope' # New indicators from SignalGenerator
        ]
        
        # Filter only available columns
        available_features = [c for c in feature_cols if c in df.columns]
        
        if not available_features:
            return None
            
        return df, available_features

    def train_and_predict(self, df):
        """
        Trains the model on historical data and predicts the next period's return.
        Returns: predicted_return (float) - kept for backward compatibility
        """
        result = self.get_detailed_prediction(df)
        return result['prediction'] if result else 0.0
    
    def get_detailed_prediction(self, df):
        """
        Enhanced prediction with comprehensive ML insights.
        """
        # Get features list from prepare_features (which also validates data)
        data = self.prepare_features(df)
        if data is None:
            return None
            
        _, features = data
        
        # Re-doing prep to keep the prediction row
        # We must apply the SAME feature engineering to df_full
        df_full = self._add_features(df)
        df_full['Target_Return'] = df_full['Close'].shift(-1).pct_change()
        
        # Training set: Rows where we know the future (all except last)
        train_df = df_full.dropna(subset=['Target_Return'] + features)
        
        if train_df.empty:
            return None
            
        X = train_df[features]
        y = train_df['Target_Return']
        
        # Train
        self.model.fit(X, y)
        
        # Calculate Model Performance Metrics
        y_pred_train = self.model.predict(X)
        mse = mean_squared_error(y, y_pred_train)
        
        # R² Score (coefficient of determination)
        from sklearn.metrics import r2_score
        r2 = r2_score(y, y_pred_train)
        
        # Feature Importance
        feature_importance_values = self.model.feature_importances_
        feature_importance = {
            feature: float(importance) 
            for feature, importance in zip(features, feature_importance_values)
        }
        
        # Predict for the LATEST available data point
        latest_row = df_full.iloc[[-1]][features]
        
        # Handle NaNs in latest row
        if latest_row.isnull().values.any():
            latest_row = df_full.iloc[[-2]][features]
        
        prediction = self.model.predict(latest_row)[0]
        
        # Calculate Prediction Confidence (based on tree prediction variance)
        # Get predictions from all trees
        tree_predictions = np.array([tree.predict(latest_row)[0] for tree in self.model.estimators_])
        prediction_std = np.std(tree_predictions)
        
        # Convert std to confidence score (0-100)
        # Lower std = higher confidence
        # We'll use an inverse relationship: confidence = 100 * (1 - normalized_std)
        # Normalize by a reasonable std threshold (e.g., 0.05 = 5% std)
        max_std = 0.05
        normalized_std = min(prediction_std / max_std, 1.0)
        confidence = 100 * (1 - normalized_std)
        
        return {
            'prediction': float(prediction),
            'confidence': float(confidence),
            'feature_importance': feature_importance,
            'metrics': {
                'mse': float(mse),
                'r2_score': float(r2)
            },
            'features_used': features
        }
