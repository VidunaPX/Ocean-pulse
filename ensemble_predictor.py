# src/ensemble_predictor.py
# Enhanced Ensemble Prediction System for OceanPulse
# Advanced ML with multiple models and uncertainty quantification

import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
from datetime import datetime, timedelta

class EnsemblePredictor:
    """
    Advanced ensemble prediction system combining multiple ML models
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.uncertainty_estimators = {}
        self.ensemble_weights = {}
        
    def initialize_models(self):
        """
        Initialize multiple prediction models
        """
        print("🤖 Initializing ensemble models...")
        
        # 1. LSTM-based model
        self.models['lstm'] = self._create_lstm_model()
        
        # 2. Random Forest
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42
        )
        
        # 3. Gradient Boosting
        self.models['gradient_boosting'] = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
        )
        
        # 4. Neural Network
        self.models['neural_network'] = MLPRegressor(
            hidden_layer_sizes=(100, 50, 25), max_iter=500, random_state=42
        )
        
        # 5. ConvLSTM (for spatial-temporal data)
        self.models['convlstm'] = self._create_convlstm_model()
        
        # Initialize scalers for each model
        for model_name in self.models.keys():
            self.scalers[model_name] = StandardScaler()
        
        print("✅ All models initialized successfully!")
    
    def _create_lstm_model(self):
        """
        Create LSTM model for time series prediction
        """
        class LSTMModel(nn.Module):
            def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=3):
                super().__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                                  batch_first=True, dropout=0.2)
                self.fc = nn.Linear(hidden_size, output_size)
                self.dropout = nn.Dropout(0.2)
                
            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                
                out, _ = self.lstm(x, (h0, c0))
                out = self.dropout(out[:, -1, :])
                out = self.fc(out)
                return out
        
        return LSTMModel()
    
    def _create_convlstm_model(self):
        """
        Create ConvLSTM model for spatial-temporal prediction
        """
        class ConvLSTMModel(nn.Module):
            def __init__(self, input_channels=1, hidden_channels=32, kernel_size=3, output_channels=3):
                super().__init__()
                self.hidden_channels = hidden_channels
                
                # ConvLSTM layers
                self.convlstm1 = nn.ConvLSTM2d(input_channels, hidden_channels, kernel_size, 
                                             padding=1, batch_first=True)
                self.convlstm2 = nn.ConvLSTM2d(hidden_channels, hidden_channels//2, kernel_size, 
                                             padding=1, batch_first=True)
                
                # Output convolution
                self.conv_out = nn.Conv2d(hidden_channels//2, output_channels, kernel_size=1)
                
            def forward(self, x):
                # x shape: (batch, time, channels, height, width)
                batch_size, seq_len, C, H, W = x.size()
                
                # Initialize hidden states
                h1 = torch.zeros(batch_size, self.hidden_channels, H, W)
                c1 = torch.zeros(batch_size, self.hidden_channels, H, W)
                h2 = torch.zeros(batch_size, self.hidden_channels//2, H, W)
                c2 = torch.zeros(batch_size, self.hidden_channels//2, H, W)
                
                # Process sequence
                for t in range(seq_len):
                    h1, c1 = self.convlstm1(x[:, t:t+1], (h1, c1))
                    h2, c2 = self.convlstm2(h1, (h2, c2))
                
                # Generate output
                out = self.conv_out(h2)
                return out
        
        return ConvLSTMModel()
    
    def prepare_training_data(self, sst_data, input_length=12, horizon=3):
        """
        Prepare training data for ensemble models
        """
        print("📊 Preparing training data...")
        
        # Convert to numpy arrays
        if hasattr(sst_data, 'values'):
            data = sst_data.values
        else:
            data = sst_data
        
        # Create sequences
        X, y = [], []
        for i in range(len(data) - input_length - horizon + 1):
            X.append(data[i:i+input_length])
            y.append(data[i+input_length:i+input_length+horizon])
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"   Training samples: {X.shape[0]}")
        print(f"   Input shape: {X.shape}")
        print(f"   Output shape: {y.shape}")
        
        return X, y
    
    def train_ensemble_models(self, X, y):
        """
        Train all ensemble models
        """
        print("🎯 Training ensemble models...")
        
        # Split data for training and validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        for model_name, model in self.models.items():
            print(f"   Training {model_name}...")
            
            if model_name in ['lstm', 'convlstm']:
                # Train PyTorch models
                self._train_pytorch_model(model_name, model, X, y)
            else:
                # Train scikit-learn models
                self._train_sklearn_model(model_name, model, X, y, tscv)
        
        print("✅ All models trained successfully!")
    
    def _train_pytorch_model(self, model_name, model, X, y):
        """
        Train PyTorch-based models (LSTM, ConvLSTM)
        """
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        # Training loop
        model.train()
        for epoch in range(50):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                if model_name == 'lstm':
                    # Reshape for LSTM: (batch, time, features)
                    batch_X = batch_X.reshape(batch_X.size(0), batch_X.size(1), -1)
                    outputs = model(batch_X)
                else:  # convlstm
                    # Reshape for ConvLSTM: (batch, time, channels, height, width)
                    batch_X = batch_X.reshape(batch_X.size(0), batch_X.size(1), 1, 
                                           batch_X.size(2), batch_X.size(3))
                    outputs = model(batch_X)
                    outputs = outputs.reshape(outputs.size(0), -1)
                
                loss = criterion(outputs, batch_y.reshape(batch_y.size(0), -1))
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            scheduler.step(total_loss)
            
            if epoch % 10 == 0:
                print(f"     Epoch {epoch}, Loss: {total_loss:.6f}")
    
    def _train_sklearn_model(self, model_name, model, X, y, tscv):
        """
        Train scikit-learn models
        """
        # Reshape data for sklearn models
        X_reshaped = X.reshape(X.shape[0], -1)
        y_reshaped = y.reshape(y.shape[0], -1)
        
        # Scale features
        X_scaled = self.scalers[model_name].fit_transform(X_reshaped)
        
        # Cross-validation scores
        scores = []
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y_reshaped[train_idx], y_reshaped[val_idx]
            
            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            scores.append(score)
        
        # Train on full dataset
        model.fit(X_scaled, y_reshaped)
        
        print(f"     Cross-validation scores: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    def predict_ensemble(self, X_test, uncertainty=True):
        """
        Make ensemble predictions with uncertainty quantification
        """
        print("🔮 Making ensemble predictions...")
        
        predictions = {}
        uncertainties = {}
        
        for model_name, model in self.models.items():
            print(f"   Predicting with {model_name}...")
            
            if model_name in ['lstm', 'convlstm']:
                # PyTorch model prediction
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_test)
                    
                    if model_name == 'lstm':
                        X_tensor = X_tensor.reshape(X_tensor.size(0), X_tensor.size(1), -1)
                        pred = model(X_tensor)
                    else:  # convlstm
                        X_tensor = X_tensor.reshape(X_tensor.size(0), X_tensor.size(1), 1,
                                                  X_tensor.size(2), X_tensor.size(3))
                        pred = model(X_tensor)
                        pred = pred.reshape(pred.size(0), -1)
                    
                    predictions[model_name] = pred.numpy()
            else:
                # Scikit-learn model prediction
                X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
                X_test_scaled = self.scalers[model_name].transform(X_test_reshaped)
                pred = model.predict(X_test_scaled)
                predictions[model_name] = pred.reshape(X_test.shape[0], -1)
        
        # Calculate ensemble prediction
        ensemble_pred = self._calculate_ensemble_prediction(predictions)
        
        # Calculate uncertainty if requested
        if uncertainty:
            uncertainties = self._calculate_uncertainty(predictions, ensemble_pred)
        
        return ensemble_pred, uncertainties, predictions
    
    def _calculate_ensemble_prediction(self, predictions):
        """
        Calculate weighted ensemble prediction
        """
        # Initialize weights (can be learned or set manually)
        weights = {
            'lstm': 0.25,
            'random_forest': 0.20,
            'gradient_boosting': 0.20,
            'neural_network': 0.15,
            'convlstm': 0.20
        }
        
        # Weighted average
        ensemble_pred = np.zeros_like(list(predictions.values())[0])
        for model_name, pred in predictions.items():
            ensemble_pred += weights[model_name] * pred
        
        return ensemble_pred
    
    def _calculate_uncertainty(self, predictions, ensemble_pred):
        """
        Calculate prediction uncertainty using multiple methods
        """
        uncertainties = {}
        
        # 1. Model disagreement (standard deviation across models)
        pred_array = np.array(list(predictions.values()))
        model_disagreement = np.std(pred_array, axis=0)
        uncertainties['model_disagreement'] = model_disagreement
        
        # 2. Ensemble variance
        ensemble_variance = np.var(pred_array, axis=0)
        uncertainties['ensemble_variance'] = ensemble_variance
        
        # 3. Confidence intervals (95%)
        confidence_lower = ensemble_pred - 1.96 * np.sqrt(ensemble_variance)
        confidence_upper = ensemble_pred + 1.96 * np.sqrt(ensemble_variance)
        uncertainties['confidence_interval'] = (confidence_lower, confidence_upper)
        
        return uncertainties
    
    def create_uncertainty_visualization(self, predictions, uncertainties, lat, lon):
        """
        Create visualization showing predictions with uncertainty
        """
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Ensemble Prediction
        ax1 = plt.subplot(2, 3, 1, projection=ccrs.PlateCarree())
        ax1.set_global()
        ax1.coastlines(resolution='50m', linewidth=0.8, color='#2d3748')
        ax1.add_feature(cfeature.BORDERS, linewidth=0.5, color='#4a5568')
        ax1.add_feature(cfeature.OCEAN, color='#1a365d', alpha=0.1)
        ax1.add_feature(cfeature.LAND, color='#2d3748', alpha=0.1)
        
        # Plot ensemble prediction
        ensemble_pred = predictions['ensemble']
        im1 = ax1.pcolormesh(lon, lat, ensemble_pred, cmap='RdYlBu_r', 
                            shading='auto', transform=ccrs.PlateCarree())
        ax1.set_title('Ensemble Prediction\n(SST Anomaly °C)', fontsize=14, fontweight='bold')
        plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.05, shrink=0.8)
        
        # 2. Model Disagreement
        ax2 = plt.subplot(2, 3, 2, projection=ccrs.PlateCarree())
        ax2.set_global()
        ax2.coastlines(resolution='50m', linewidth=0.8, color='#2d3748')
        ax2.add_feature(cfeature.BORDERS, linewidth=0.5, color='#4a5568')
        ax2.add_feature(cfeature.OCEAN, color='#1a365d', alpha=0.1)
        ax2.add_feature(cfeature.LAND, color='#2d3748', alpha=0.1)
        
        disagreement = uncertainties['model_disagreement']
        im2 = ax2.pcolormesh(lon, lat, disagreement, cmap='Reds', 
                           shading='auto', transform=ccrs.PlateCarree())
        ax2.set_title('Model Disagreement\n(Uncertainty)', fontsize=14, fontweight='bold')
        plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.05, shrink=0.8)
        
        # 3. Confidence Intervals
        ax3 = plt.subplot(2, 3, 3, projection=ccrs.PlateCarree())
        ax3.set_global()
        ax3.coastlines(resolution='50m', linewidth=0.8, color='#2d3748')
        ax3.add_feature(cfeature.BORDERS, linewidth=0.5, color='#4a5568')
        ax3.add_feature(cfeature.OCEAN, color='#1a365d', alpha=0.1)
        ax3.add_feature(cfeature.LAND, color='#2d3748', alpha=0.1)
        
        # Plot confidence interval width
        conf_lower, conf_upper = uncertainties['confidence_interval']
        conf_width = conf_upper - conf_lower
        im3 = ax3.pcolormesh(lon, lat, conf_width, cmap='Oranges', 
                           shading='auto', transform=ccrs.PlateCarree())
        ax3.set_title('Confidence Interval Width\n(95% CI)', fontsize=14, fontweight='bold')
        plt.colorbar(im3, ax=ax3, orientation='horizontal', pad=0.05, shrink=0.8)
        
        # 4. Individual Model Predictions
        model_names = ['lstm', 'random_forest', 'gradient_boosting', 'neural_network', 'convlstm']
        for i, model_name in enumerate(model_names[:3]):  # Show first 3 models
            ax = plt.subplot(2, 3, 4+i, projection=ccrs.PlateCarree())
            ax.set_global()
            ax.coastlines(resolution='50m', linewidth=0.8, color='#2d3748')
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, color='#4a5568')
            ax.add_feature(cfeature.OCEAN, color='#1a365d', alpha=0.1)
            ax.add_feature(cfeature.LAND, color='#2d3748', alpha=0.1)
            
            if model_name in predictions:
                model_pred = predictions[model_name]
                im = ax.pcolormesh(lon, lat, model_pred, cmap='RdYlBu_r', 
                                 shading='auto', transform=ccrs.PlateCarree())
                ax.set_title(f'{model_name.replace("_", " ").title()}\nPrediction', 
                           fontsize=12, fontweight='bold')
                plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        
        plt.suptitle('🤖 OceanPulse: Ensemble Prediction System\n'
                    'Advanced ML with Uncertainty Quantification', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        return fig
    
    def evaluate_ensemble_performance(self, X_test, y_test, predictions, uncertainties):
        """
        Evaluate ensemble performance with multiple metrics
        """
        print("📊 Evaluating ensemble performance...")
        
        # Calculate metrics
        ensemble_pred = predictions['ensemble']
        
        # Reshape for evaluation
        y_test_flat = y_test.reshape(y_test.shape[0], -1)
        ensemble_pred_flat = ensemble_pred.reshape(ensemble_pred.shape[0], -1)
        
        # Calculate metrics
        mse = np.mean((y_test_flat - ensemble_pred_flat) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test_flat - ensemble_pred_flat))
        
        # R-squared
        ss_res = np.sum((y_test_flat - ensemble_pred_flat) ** 2)
        ss_tot = np.sum((y_test_flat - np.mean(y_test_flat)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Uncertainty calibration
        uncertainty_scores = uncertainties['model_disagreement']
        avg_uncertainty = np.mean(uncertainty_scores)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'avg_uncertainty': avg_uncertainty
        }
        
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   R²: {r2:.4f}")
        print(f"   Average Uncertainty: {avg_uncertainty:.4f}")
        
        return metrics

def main():
    """
    Main execution function for ensemble predictor
    """
    print("🤖 Initializing OceanPulse Ensemble Predictor...")
    
    # Initialize ensemble predictor
    predictor = EnsemblePredictor()
    predictor.initialize_models()
    
    # Generate synthetic training data
    print("📊 Generating synthetic training data...")
    # This would normally load from actual datasets
    lat = np.linspace(-90, 90, 90)
    lon = np.linspace(-180, 180, 180)
    time = np.arange(0, 1000)  # 1000 time steps
    
    # Generate synthetic SST data
    sst_data = np.random.normal(0, 2, (len(time), len(lat), len(lon)))
    
    # Add some realistic patterns
    for t in range(len(time)):
        # Seasonal pattern
        seasonal = 2 * np.sin(2 * np.pi * t / 365)
        # Spatial pattern
        for i in range(len(lat)):
            for j in range(len(lon)):
                lat_effect = 0.1 * lat[i] / 90  # Latitude effect
                lon_effect = 0.05 * np.sin(2 * np.pi * lon[j] / 360)  # Longitude effect
                sst_data[t, i, j] += seasonal + lat_effect + lon_effect
    
    # Prepare training data
    X, y = predictor.prepare_training_data(sst_data)
    
    # Split into train/test
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train ensemble models
    predictor.train_ensemble_models(X_train, y_train)
    
    # Make predictions
    predictions, uncertainties, individual_preds = predictor.predict_ensemble(X_test)
    
    # Evaluate performance
    metrics = predictor.evaluate_ensemble_performance(X_test, y_test, 
                                                    {'ensemble': predictions}, uncertainties)
    
    # Create visualizations
    print("📊 Creating uncertainty visualizations...")
    os.makedirs("data/output", exist_ok=True)
    
    # Add ensemble prediction to individual predictions
    individual_preds['ensemble'] = predictions
    
    fig = predictor.create_uncertainty_visualization(individual_preds, uncertainties, lat, lon)
    plt.savefig("Frontend/public", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Ensemble prediction analysis completed!")
    print("📁 Output saved to: data/output/ensemble_predictions.png")
    
    return predictor, metrics

if __name__ == "__main__":
    predictor, metrics = main()
