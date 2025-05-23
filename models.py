#!/usr/bin/env python3
"""
Clean Models for Figure 8: IMDB Histogram Prediction

Simple, robust models using Keras and scikit-learn.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class HistogramPredictor:
    """Simple histogram prediction model using Keras."""
    
    def __init__(self, n_features, n_outputs=10):
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.scaler = StandardScaler()
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build a simple neural network for histogram prediction."""
        
        # Simple architecture: 
        # Input -> Dense(32) -> Dropout -> Dense(16) -> Dense(10)
        model = keras.Sequential([
            layers.Dense(32, activation='relu', input_shape=(self.n_features,)),
            layers.Dropout(0.3),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.n_outputs, activation='linear')  # Linear output for regression
        ])
        
        # Compile with MSE loss for regression
        model.compile(
            optimizer='adam',
            loss='mse',  # Mean squared error for regression
            metrics=['mae', 'mse']
        )
        
        self.model = model
        return model
    
    def normalize_histograms(self, y):
        """Keep histogram percentages as-is for MSE loss."""
        # For MSE loss, we can work directly with percentages
        return y
    
    def fit(self, X_train, y_train, X_val, y_val, epochs=100, verbose=1):
        """Train the model."""
        print("Training Keras neural network...")
        
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Normalize histograms to proper probability distributions
        y_train_norm = self.normalize_histograms(y_train)
        y_val_norm = self.normalize_histograms(y_val)
        
        # Build model if not already built
        if self.model is None:
            self.build_model()
        
        # Callbacks for training
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-6
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            X_train_scaled, y_train_norm,
            validation_data=(X_val_scaled, y_val_norm),
            epochs=epochs,
            batch_size=8,  # Small batch size for small dataset
            callbacks=callbacks_list,
            verbose=verbose
        )
        
        print(f"  Training completed after {len(self.history.history['loss'])} epochs")
        
    def predict(self, X):
        """Make predictions."""
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled, verbose=0)
        
        # Return predictions directly (already in percentage scale)
        return predictions
    
    def evaluate(self, X, y, set_name="Test"):
        """Evaluate model performance."""
        y_pred = self.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Per-output R² scores
        per_output_r2 = [r2_score(y[:, i], y_pred[:, i]) for i in range(y.shape[1])]
        
        print(f"\n{set_name} Set Performance (Keras NN):")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  R² range per output: {min(per_output_r2):.3f} - {max(per_output_r2):.3f}")
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'per_output_r2': per_output_r2,
            'predictions': y_pred
        }

class RandomForestPredictor:
    """Simple Random Forest baseline model."""
    
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
    def fit(self, X_train, y_train):
        """Train the Random Forest model."""
        print("Training Random Forest baseline...")
        self.model.fit(X_train, y_train)
        print(f"  Training completed with {self.model.n_estimators} trees")
        
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)
    
    def evaluate(self, X, y, set_name="Test"):
        """Evaluate model performance."""
        y_pred = self.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Per-output R² scores
        per_output_r2 = [r2_score(y[:, i], y_pred[:, i]) for i in range(y.shape[1])]
        
        print(f"\n{set_name} Set Performance (Random Forest):")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  R² range per output: {min(per_output_r2):.3f} - {max(per_output_r2):.3f}")
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'per_output_r2': per_output_r2,
            'predictions': y_pred
        }
    
    def get_feature_importance(self):
        """Get feature importance scores."""
        return self.model.feature_importances_

class ModelEvaluator:
    """Utility class for comparing multiple models."""
    
    @staticmethod
    def compare_models(results_dict):
        """Compare results from multiple models."""
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        
        for model_name, results in results_dict.items():
            print(f"\n{model_name.upper()}:")
            print("-" * 30)
            for set_name, metrics in results.items():
                if isinstance(metrics, dict):
                    print(f"  {set_name.title()}: R²={metrics['r2']:.3f}, MAE={metrics['mae']:.3f}")
    
    @staticmethod
    def print_training_summary(data, keras_model, rf_model):
        """Print a comprehensive training summary."""
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        
        print(f"Dataset:")
        print(f"  Features: {len(data['feature_names'])}")
        print(f"  Train episodes: {data['X_train'].shape[0]}")
        print(f"  Validation episodes: {data['X_val'].shape[0]}")
        print(f"  Test episodes: {data['X_test'].shape[0]}")
        
        print(f"\nFeature names:")
        for i, feature in enumerate(data['feature_names']):
            print(f"  {i+1}. {feature}")
        
        if hasattr(keras_model, 'history') and keras_model.history:
            final_loss = keras_model.history.history['val_loss'][-1]
            print(f"\nKeras model final validation loss: {final_loss:.4f}")

if __name__ == "__main__":
    # Test the models
    from data_loader import HistogramDataLoader
    
    loader = HistogramDataLoader()
    data = loader.load_complete_dataset()
    
    # Test Keras model
    keras_model = HistogramPredictor(n_features=len(data['feature_names']))
    keras_model.fit(data['X_train'], data['y_train'], data['X_val'], data['y_val'])
    
    # Test Random Forest
    rf_model = RandomForestPredictor()
    rf_model.fit(data['X_train'], data['y_train'])
    
    print("Models trained successfully!") 