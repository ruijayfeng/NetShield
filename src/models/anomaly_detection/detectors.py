"""
Anomaly detection algorithms for network monitoring data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os
import yaml
from dataclasses import dataclass
from datetime import datetime
import networkx as nx


@dataclass
class AnomalyDetectionConfig:
    """Configuration for anomaly detection"""
    methods: List[str] = None
    ensemble: bool = True
    contamination: float = 0.1
    n_estimators: int = 100
    random_state: int = 42
    nu: float = 0.1
    kernel: str = "rbf"
    gamma: str = "scale"
    
    def __post_init__(self):
        if self.methods is None:
            self.methods = ["isolation_forest", "one_class_svm", "local_outlier_factor"]
    
    @classmethod
    def from_config(cls, config_path: str = None):
        """Load configuration from YAML file"""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "../../../config/config.yaml")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        ad_config = config.get('anomaly_detection', {})
        return cls(
            methods=ad_config.get('methods', ["isolation_forest", "one_class_svm", "local_outlier_factor"]),
            ensemble=ad_config.get('ensemble', True),
            contamination=ad_config.get('contamination', 0.1),
            n_estimators=ad_config.get('n_estimators', 100),
            random_state=ad_config.get('random_state', 42),
            nu=ad_config.get('nu', 0.1),
            kernel=ad_config.get('kernel', 'rbf'),
            gamma=ad_config.get('gamma', 'scale')
        )


class BaseAnomalyDetector(ABC):
    """Abstract base class for anomaly detectors"""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.scaler = None
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, X: np.ndarray) -> 'BaseAnomalyDetector':
        """Fit the anomaly detection model"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies (1 for anomaly, -1 for normal)"""
        pass
    
    @abstractmethod
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores"""
        pass
    
    def preprocess_data(self, X: np.ndarray, fit_scaler: bool = False) -> np.ndarray:
        """Preprocess data using scaling"""
        if fit_scaler or self.scaler is None:
            self.scaler = RobustScaler()
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'name': self.name,
            'is_fitted': self.is_fitted
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.name = model_data['name']
        self.is_fitted = model_data['is_fitted']


class IsolationForestDetector(BaseAnomalyDetector):
    """Isolation Forest anomaly detector"""
    
    def __init__(self, contamination: float = 0.1, n_estimators: int = 100, 
                 random_state: int = 42):
        super().__init__("IsolationForest")
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state
        )
    
    def fit(self, X: np.ndarray) -> 'IsolationForestDetector':
        """Fit the Isolation Forest model"""
        X_scaled = self.preprocess_data(X, fit_scaler=True)
        self.model.fit(X_scaled)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.preprocess_data(X)
        return self.model.predict(X_scaled)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.preprocess_data(X)
        return self.model.decision_function(X_scaled)


class OneClassSVMDetector(BaseAnomalyDetector):
    """One-Class SVM anomaly detector"""
    
    def __init__(self, nu: float = 0.1, kernel: str = "rbf", gamma: str = "scale"):
        super().__init__("OneClassSVM")
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    
    def fit(self, X: np.ndarray) -> 'OneClassSVMDetector':
        """Fit the One-Class SVM model"""
        X_scaled = self.preprocess_data(X, fit_scaler=True)
        self.model.fit(X_scaled)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.preprocess_data(X)
        return self.model.predict(X_scaled)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.preprocess_data(X)
        return self.model.decision_function(X_scaled)


class LocalOutlierFactorDetector(BaseAnomalyDetector):
    """Local Outlier Factor anomaly detector"""
    
    def __init__(self, n_neighbors: int = 20, contamination: float = 0.1):
        super().__init__("LocalOutlierFactor")
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True  # Enable prediction on new data
        )
    
    def fit(self, X: np.ndarray) -> 'LocalOutlierFactorDetector':
        """Fit the LOF model"""
        X_scaled = self.preprocess_data(X, fit_scaler=True)
        self.model.fit(X_scaled)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.preprocess_data(X)
        return self.model.predict(X_scaled)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.preprocess_data(X)
        return self.model.decision_function(X_scaled)


class EnsembleAnomalyDetector:
    """Ensemble anomaly detector combining multiple methods"""
    
    def __init__(self, config: AnomalyDetectionConfig = None):
        self.config = config or AnomalyDetectionConfig()
        self.detectors = {}
        self.is_fitted = False
        self._initialize_detectors()
    
    def _initialize_detectors(self):
        """Initialize individual detectors"""
        if "isolation_forest" in self.config.methods:
            self.detectors["isolation_forest"] = IsolationForestDetector(
                contamination=self.config.contamination,
                n_estimators=self.config.n_estimators,
                random_state=self.config.random_state
            )
        
        if "one_class_svm" in self.config.methods:
            self.detectors["one_class_svm"] = OneClassSVMDetector(
                nu=self.config.nu,
                kernel=self.config.kernel,
                gamma=self.config.gamma
            )
        
        if "local_outlier_factor" in self.config.methods:
            self.detectors["local_outlier_factor"] = LocalOutlierFactorDetector(
                contamination=self.config.contamination
            )
    
    def fit(self, X: np.ndarray) -> 'EnsembleAnomalyDetector':
        """Fit all detectors"""
        print(f"Training ensemble with {len(self.detectors)} detectors...")
        
        for name, detector in self.detectors.items():
            print(f"  Training {name}...")
            detector.fit(X)
        
        self.is_fitted = True
        print("Ensemble training completed.")
        return self
    
    def predict(self, X: np.ndarray, method: str = "voting") -> np.ndarray:
        """Predict anomalies using ensemble method"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        predictions = {}
        
        for name, detector in self.detectors.items():
            predictions[name] = detector.predict(X)
        
        if method == "voting":
            # Majority voting
            pred_array = np.array(list(predictions.values()))
            # Convert -1/1 to 0/1 for easier voting
            pred_binary = (pred_array == 1).astype(int)
            ensemble_pred = (pred_binary.mean(axis=0) > 0.5).astype(int)
            # Convert back to -1/1 format
            ensemble_pred = 2 * ensemble_pred - 1
            
        elif method == "unanimous":
            # All detectors must agree for anomaly
            pred_array = np.array(list(predictions.values()))
            ensemble_pred = (pred_array == 1).all(axis=0).astype(int)
            ensemble_pred = 2 * ensemble_pred - 1
            
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        return ensemble_pred
    
    def decision_function(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Return anomaly scores from all detectors"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        scores = {}
        for name, detector in self.detectors.items():
            scores[name] = detector.decision_function(X)
        
        # Calculate ensemble score as average
        scores['ensemble'] = np.mean(list(scores.values()), axis=0)
        
        return scores
    
    def get_anomaly_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly probabilities (0-1 scale)"""
        scores = self.decision_function(X)
        ensemble_scores = scores['ensemble']
        
        # Normalize scores to 0-1 probability scale
        # Use sigmoid transformation
        probabilities = 1 / (1 + np.exp(ensemble_scores))
        
        return probabilities
    
    def save_ensemble(self, dirpath: str):
        """Save all detectors in the ensemble"""
        os.makedirs(dirpath, exist_ok=True)
        
        for name, detector in self.detectors.items():
            filepath = os.path.join(dirpath, f"{name}.pkl")
            detector.save_model(filepath)
        
        # Save ensemble metadata
        metadata = {
            'methods': self.config.methods,
            'is_fitted': self.is_fitted,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(dirpath, 'ensemble_metadata.yaml'), 'w') as f:
            yaml.dump(metadata, f)
    
    def load_ensemble(self, dirpath: str):
        """Load all detectors in the ensemble"""
        # Load metadata
        with open(os.path.join(dirpath, 'ensemble_metadata.yaml'), 'r') as f:
            metadata = yaml.safe_load(f)
        
        self.config.methods = metadata['methods']
        self.is_fitted = metadata['is_fitted']
        
        # Reinitialize and load detectors
        self._initialize_detectors()
        
        for name, detector in self.detectors.items():
            filepath = os.path.join(dirpath, f"{name}.pkl")
            if os.path.exists(filepath):
                detector.load_model(filepath)


class NetworkAnomalyAnalyzer:
    """High-level analyzer for network anomaly detection"""
    
    def __init__(self, config: AnomalyDetectionConfig = None):
        self.config = config or AnomalyDetectionConfig()
        self.ensemble = EnsembleAnomalyDetector(self.config)
        self.feature_columns = None
        self.training_stats = {}
    
    def prepare_features(self, data: pd.DataFrame, 
                        feature_columns: List[str] = None) -> np.ndarray:
        """Prepare features from DataFrame"""
        if feature_columns is None:
            # Auto-select numeric columns excluding timestamps and labels
            exclude_cols = ['timestamp', 'is_anomaly', 'anomaly_score', 'node_id']
            feature_columns = [col for col in data.columns 
                             if col not in exclude_cols and 
                             pd.api.types.is_numeric_dtype(data[col])]
        
        self.feature_columns = feature_columns
        
        # Handle missing values
        features = data[feature_columns].fillna(data[feature_columns].median())
        
        return features.values
    
    def train(self, training_data: pd.DataFrame, 
             feature_columns: List[str] = None) -> Dict[str, Any]:
        """Train anomaly detection models"""
        print("Preparing training features...")
        X_train = self.prepare_features(training_data, feature_columns)
        
        # Store training statistics
        self.training_stats = {
            'n_samples': X_train.shape[0],
            'n_features': X_train.shape[1],
            'feature_means': np.mean(X_train, axis=0),
            'feature_stds': np.std(X_train, axis=0),
            'training_timestamp': datetime.now().isoformat()
        }
        
        # Train ensemble
        self.ensemble.fit(X_train)
        
        # Evaluate on training data (for reference)
        train_predictions = self.ensemble.predict(X_train)
        train_scores = self.ensemble.decision_function(X_train)
        
        # Calculate training metrics
        if 'is_anomaly' in training_data.columns:
            y_true = training_data['is_anomaly'].astype(int)
            y_true = 2 * y_true - 1  # Convert 0/1 to -1/1
            
            training_metrics = self._calculate_metrics(y_true, train_predictions)
            self.training_stats['metrics'] = training_metrics
        
        return self.training_stats
    
    def detect_anomalies(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies in test data"""
        if not self.ensemble.is_fitted:
            raise ValueError("Model must be trained before detection")
        
        X_test = self.prepare_features(test_data, self.feature_columns)
        
        # Get predictions and scores
        predictions = self.ensemble.predict(X_test)
        scores = self.ensemble.decision_function(X_test)
        probabilities = self.ensemble.get_anomaly_probabilities(X_test)
        
        # Create results DataFrame
        results = test_data.copy()
        results['predicted_anomaly'] = (predictions == 1)
        results['anomaly_probability'] = probabilities
        results['ensemble_score'] = scores['ensemble']
        
        # Add individual detector scores
        for detector_name, detector_scores in scores.items():
            if detector_name != 'ensemble':
                results[f'{detector_name}_score'] = detector_scores
        
        # Summary statistics
        anomaly_summary = {
            'total_samples': len(results),
            'predicted_anomalies': int((predictions == 1).sum()),
            'anomaly_rate': float((predictions == 1).mean()),
            'mean_anomaly_probability': float(probabilities.mean()),
            'max_anomaly_probability': float(probabilities.max()),
            'high_confidence_anomalies': int((probabilities > 0.8).sum())
        }
        
        return {
            'results': results,
            'summary': anomaly_summary,
            'predictions': predictions,
            'scores': scores,
            'probabilities': probabilities
        }
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        # Convert to binary format for sklearn metrics
        y_true_binary = (y_true == 1).astype(int)
        y_pred_binary = (y_pred == 1).astype(int)
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        try:
            auc = roc_auc_score(y_true_binary, y_pred_binary)
        except ValueError:
            auc = 0.5  # Default AUC when only one class present
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'auc': auc,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        }
    
    def evaluate_model(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate model performance on labeled test data"""
        if 'is_anomaly' not in test_data.columns:
            raise ValueError("Test data must contain 'is_anomaly' column for evaluation")
        
        detection_results = self.detect_anomalies(test_data)
        
        y_true = test_data['is_anomaly'].astype(int)
        y_true = 2 * y_true - 1  # Convert to -1/1 format
        y_pred = detection_results['predictions']
        
        metrics = self._calculate_metrics(y_true, y_pred)
        
        return {
            'metrics': metrics,
            'detection_results': detection_results,
            'classification_report': classification_report(
                (y_true == 1).astype(int), 
                (y_pred == 1).astype(int),
                target_names=['Normal', 'Anomaly']
            )
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance (simplified implementation)"""
        if not self.feature_columns:
            return {}
        
        # For now, return uniform importance
        # In a more advanced implementation, this could analyze model internals
        importance = 1.0 / len(self.feature_columns)
        return {col: importance for col in self.feature_columns}
    
    def save_analyzer(self, dirpath: str):
        """Save the complete analyzer"""
        os.makedirs(dirpath, exist_ok=True)
        
        # Save ensemble
        ensemble_path = os.path.join(dirpath, 'ensemble')
        self.ensemble.save_ensemble(ensemble_path)
        
        # Save analyzer metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'training_stats': self.training_stats,
            'config': {
                'methods': self.config.methods,
                'ensemble': self.config.ensemble,
                'contamination': self.config.contamination
            }
        }
        
        with open(os.path.join(dirpath, 'analyzer_metadata.yaml'), 'w') as f:
            yaml.dump(metadata, f)
    
    def load_analyzer(self, dirpath: str):
        """Load a saved analyzer"""
        # Load metadata
        with open(os.path.join(dirpath, 'analyzer_metadata.yaml'), 'r') as f:
            metadata = yaml.safe_load(f)
        
        self.feature_columns = metadata['feature_columns']
        self.training_stats = metadata['training_stats']
        
        # Load ensemble
        ensemble_path = os.path.join(dirpath, 'ensemble')
        self.ensemble.load_ensemble(ensemble_path)


# Example usage and testing
if __name__ == "__main__":
    # Test the anomaly detection system
    from ..data.generators.network_generator import NetworkGenerator, NetworkConfig
    from ..data.generators.data_generator import NetworkDataGenerator, DataConfig
    
    # Generate test data
    print("Generating test data...")
    network_config = NetworkConfig(node_count=20)
    data_config = DataConfig(time_steps=500, anomaly_ratio=0.08)
    
    network_gen = NetworkGenerator(network_config)
    data_gen = NetworkDataGenerator(data_config)
    
    network = network_gen.generate_network('small_world')
    test_node = list(network.nodes())[0]
    data = data_gen.generate_node_timeseries(network, test_node)
    
    print(f"Generated data shape: {data.shape}")
    print(f"True anomaly rate: {data['is_anomaly'].mean():.3f}")
    
    # Split data
    split_idx = int(0.7 * len(data))
    train_data = data[:split_idx].copy()
    test_data = data[split_idx:].copy()
    
    # Initialize and train analyzer
    print("\nTraining anomaly detection models...")
    config = AnomalyDetectionConfig()
    analyzer = NetworkAnomalyAnalyzer(config)
    
    training_stats = analyzer.train(train_data)
    print(f"Training completed. Features used: {len(analyzer.feature_columns)}")
    
    # Detect anomalies
    print("\nDetecting anomalies in test data...")
    detection_results = analyzer.detect_anomalies(test_data)
    print(f"Detection completed. Summary: {detection_results['summary']}")
    
    # Evaluate performance
    if 'is_anomaly' in test_data.columns:
        print("\nEvaluating model performance...")
        evaluation = analyzer.evaluate_model(test_data)
        print("Metrics:", evaluation['metrics'])
        print("\nClassification Report:")
        print(evaluation['classification_report'])