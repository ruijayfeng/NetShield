"""
Data generation module for creating synthetic network traffic and performance data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import networkx as nx
from dataclasses import dataclass
import yaml
import os
import random


@dataclass
class DataConfig:
    """Configuration for data generation"""
    time_steps: int = 1000
    features: List[str] = None
    anomaly_ratio: float = 0.05
    noise_level: float = 0.1
    
    def __post_init__(self):
        if self.features is None:
            self.features = ["traffic", "latency", "packet_loss", "cpu_usage", "memory_usage"]
    
    @classmethod
    def from_config(cls, config_path: str = None):
        """Load configuration from YAML file"""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "../../../config/config.yaml")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        data_config = config.get('data', {})
        return cls(
            time_steps=data_config.get('time_steps', 1000),
            features=data_config.get('features', ["traffic", "latency", "packet_loss", "cpu_usage", "memory_usage"]),
            anomaly_ratio=data_config.get('anomaly_ratio', 0.05),
            noise_level=data_config.get('noise_level', 0.1)
        )


class NetworkDataGenerator:
    """Generate synthetic network monitoring data"""
    
    def __init__(self, config: DataConfig = None):
        self.config = config or DataConfig()
        self.random_state = np.random.RandomState(42)
        
    def generate_node_timeseries(self, network: nx.Graph, node_id: Any = None) -> pd.DataFrame:
        """Generate time series data for a specific node or all nodes"""
        
        if node_id is not None:
            # Generate data for specific node
            return self._generate_single_node_data(network, node_id)
        else:
            # Generate data for all nodes
            all_data = []
            for node in network.nodes():
                node_data = self._generate_single_node_data(network, node)
                node_data['node_id'] = node
                all_data.append(node_data)
            
            return pd.concat(all_data, ignore_index=True)
    
    def _generate_single_node_data(self, network: nx.Graph, node_id: Any) -> pd.DataFrame:
        """Generate time series data for a single node"""
        
        # Get node properties
        node_attrs = network.nodes[node_id]
        degree = node_attrs.get('degree', 1)
        node_type = node_attrs.get('node_type', 'leaf')
        
        # Base patterns based on node characteristics
        base_patterns = self._get_base_patterns(degree, node_type)
        
        # Generate timestamps
        start_time = datetime.now() - timedelta(hours=self.config.time_steps // 60)
        timestamps = [start_time + timedelta(minutes=i) for i in range(self.config.time_steps)]
        
        # Generate feature data
        data = {'timestamp': timestamps}
        
        for feature in self.config.features:
            # Generate base signal with patterns
            signal = self._generate_feature_signal(feature, base_patterns)
            
            # Add seasonal patterns
            signal = self._add_seasonal_patterns(signal, feature)
            
            # Add noise
            noise = self.random_state.normal(0, self.config.noise_level, len(signal))
            signal += noise
            
            # Ensure non-negative values for certain features
            if feature in ['traffic', 'cpu_usage', 'memory_usage']:
                signal = np.maximum(signal, 0)
            
            # Add to data
            data[feature] = signal
        
        # Inject anomalies
        data = self._inject_anomalies(data)
        
        return pd.DataFrame(data)
    
    def _get_base_patterns(self, degree: int, node_type: str) -> Dict[str, float]:
        """Get base patterns based on node characteristics"""
        patterns = {
            'traffic_base': degree * 10,
            'latency_base': 20 + (5 if node_type == 'hub' else 0),
            'packet_loss_base': 0.01,
            'cpu_base': 0.3 + (0.2 if node_type == 'hub' else 0),
            'memory_base': 0.4 + (0.3 if node_type == 'hub' else 0)
        }
        return patterns
    
    def _generate_feature_signal(self, feature: str, patterns: Dict[str, float]) -> np.ndarray:
        """Generate base signal for a specific feature"""
        
        time_points = np.arange(self.config.time_steps)
        
        if feature == 'traffic':
            # Traffic pattern: base level with periodic variations
            base = patterns['traffic_base']
            signal = base + base * 0.3 * np.sin(2 * np.pi * time_points / 144)  # Daily pattern
            
        elif feature == 'latency':
            # Latency pattern: generally stable with occasional spikes
            base = patterns['latency_base']
            signal = np.full(len(time_points), base, dtype=np.float64)
            # Add random spikes
            spike_indices = self.random_state.choice(len(time_points), size=int(0.02 * len(time_points)), replace=False)
            signal[spike_indices] *= self.random_state.uniform(2, 5, len(spike_indices))
            
        elif feature == 'packet_loss':
            # Packet loss: usually low with occasional increases
            base = patterns['packet_loss_base']
            signal = self.random_state.exponential(base, len(time_points))
            signal = np.minimum(signal, 0.1)  # Cap at 10%
            
        elif feature == 'cpu_usage':
            # CPU usage: base level with work-related patterns
            base = patterns['cpu_base']
            work_pattern = 0.2 * np.sin(2 * np.pi * time_points / 144) + 0.1 * np.sin(2 * np.pi * time_points / 24)
            signal = base + work_pattern
            signal = np.clip(signal, 0, 1)
            
        elif feature == 'memory_usage':
            # Memory usage: gradually increasing with occasional drops
            base = patterns['memory_base']
            trend = np.linspace(0, 0.1, len(time_points))
            # Add occasional memory releases
            release_points = self.random_state.choice(len(time_points), size=int(0.01 * len(time_points)), replace=False)
            release_signal = np.zeros(len(time_points))
            release_signal[release_points] = -0.2
            
            signal = base + trend + np.cumsum(release_signal) * 0.1
            signal = np.clip(signal, 0, 1)
            
        else:
            # Default pattern for unknown features
            signal = self.random_state.normal(1, 0.2, len(time_points))
        
        return signal
    
    def _add_seasonal_patterns(self, signal: np.ndarray, feature: str) -> np.ndarray:
        """Add seasonal and daily patterns to the signal"""
        time_points = np.arange(len(signal))
        
        # Daily pattern (24 hour cycle, assuming 1 data point per minute)
        daily_pattern = 0.1 * np.sin(2 * np.pi * time_points / (24 * 60))
        
        # Weekly pattern (7 day cycle)
        weekly_pattern = 0.05 * np.sin(2 * np.pi * time_points / (7 * 24 * 60))
        
        # Apply patterns differently based on feature type
        if feature in ['traffic', 'cpu_usage']:
            # More pronounced patterns for traffic and CPU
            signal = signal * (1 + daily_pattern + weekly_pattern)
        elif feature in ['latency', 'memory_usage']:
            # Subtle patterns for latency and memory
            signal = signal + signal * 0.5 * (daily_pattern + weekly_pattern)
        
        return signal
    
    def _inject_anomalies(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Inject anomalies into the data"""
        
        num_anomalies = int(self.config.anomaly_ratio * self.config.time_steps)
        anomaly_indices = self.random_state.choice(
            self.config.time_steps, 
            size=num_anomalies, 
            replace=False
        )
        
        # Add anomaly labels
        data['is_anomaly'] = np.zeros(self.config.time_steps, dtype=bool)
        data['is_anomaly'][anomaly_indices] = True
        
        # Add anomaly scores (0-1, where 1 is most anomalous)
        data['anomaly_score'] = np.zeros(self.config.time_steps)
        
        # Modify feature values at anomaly points
        for idx in anomaly_indices:
            anomaly_type = self.random_state.choice(['spike', 'drop', 'shift'])
            intensity = self.random_state.uniform(0.7, 1.0)
            data['anomaly_score'][idx] = intensity
            
            for feature in self.config.features:
                if anomaly_type == 'spike':
                    # Sudden spike
                    multiplier = 1 + intensity * self.random_state.uniform(2, 5)
                    data[feature][idx] *= multiplier
                elif anomaly_type == 'drop':
                    # Sudden drop
                    multiplier = max(0.1, 1 - intensity * self.random_state.uniform(0.5, 0.9))
                    data[feature][idx] *= multiplier
                elif anomaly_type == 'shift':
                    # Value shift
                    shift = intensity * self.random_state.uniform(-2, 2)
                    data[feature][idx] += shift
                
                # Ensure realistic bounds
                if feature in ['cpu_usage', 'memory_usage']:
                    data[feature][idx] = np.clip(data[feature][idx], 0, 1)
                elif feature == 'packet_loss':
                    data[feature][idx] = np.clip(data[feature][idx], 0, 1)
                elif feature in ['traffic', 'latency']:
                    data[feature][idx] = max(0, data[feature][idx])
        
        return data
    
    def generate_network_events(self, network: nx.Graph, num_events: int = 50) -> pd.DataFrame:
        """Generate network-level events (failures, maintenance, etc.)"""
        
        events = []
        event_types = ['node_failure', 'link_failure', 'maintenance', 'overload', 'recovery']
        
        for _ in range(num_events):
            event_time = datetime.now() - timedelta(
                minutes=self.random_state.randint(0, self.config.time_steps)
            )
            
            event_type = self.random_state.choice(event_types)
            
            if event_type in ['node_failure', 'maintenance', 'overload', 'recovery']:
                affected_node = self.random_state.choice(list(network.nodes()))
                events.append({
                    'timestamp': event_time,
                    'event_type': event_type,
                    'affected_element': f'node_{affected_node}',
                    'severity': self.random_state.choice(['low', 'medium', 'high', 'critical']),
                    'duration_minutes': self.random_state.randint(5, 180),
                    'description': f'{event_type.replace("_", " ").title()} on node {affected_node}'
                })
            elif event_type == 'link_failure':
                edge = self.random_state.choice(list(network.edges()))
                events.append({
                    'timestamp': event_time,
                    'event_type': event_type,
                    'affected_element': f'edge_{edge[0]}_{edge[1]}',
                    'severity': self.random_state.choice(['low', 'medium', 'high']),
                    'duration_minutes': self.random_state.randint(10, 120),
                    'description': f'Link failure between nodes {edge[0]} and {edge[1]}'
                })
        
        return pd.DataFrame(events).sort_values('timestamp')
    
    def generate_correlated_features(self, base_data: pd.DataFrame) -> pd.DataFrame:
        """Generate additional features that are correlated with existing ones"""
        
        correlated_data = base_data.copy()
        
        # Response time correlated with traffic and latency
        if 'traffic' in base_data.columns and 'latency' in base_data.columns:
            response_time = (
                0.7 * base_data['latency'] + 
                0.3 * base_data['traffic'] / base_data['traffic'].max() * 100 +
                self.random_state.normal(0, 5, len(base_data))
            )
            correlated_data['response_time'] = np.maximum(response_time, 0)
        
        # Throughput inversely related to packet loss and latency
        if 'packet_loss' in base_data.columns and 'latency' in base_data.columns:
            max_throughput = 1000  # Mbps
            throughput = max_throughput * (
                1 - base_data['packet_loss'] - base_data['latency'] / 200
            ) + self.random_state.normal(0, 50, len(base_data))
            correlated_data['throughput'] = np.maximum(throughput, 0)
        
        # Error rate correlated with CPU usage and packet loss
        if 'cpu_usage' in base_data.columns and 'packet_loss' in base_data.columns:
            error_rate = (
                0.01 * base_data['cpu_usage'] + 
                0.5 * base_data['packet_loss'] +
                self.random_state.exponential(0.001, len(base_data))
            )
            correlated_data['error_rate'] = np.maximum(error_rate, 0)
        
        return correlated_data
    
    def save_data(self, data: pd.DataFrame, filepath: str, format: str = 'csv'):
        """Save generated data to file"""
        if format.lower() == 'csv':
            data.to_csv(filepath, index=False)
        elif format.lower() == 'json':
            data.to_json(filepath, orient='records', date_format='iso')
        elif format.lower() == 'parquet':
            data.to_parquet(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def load_real_data(self, filepath: str) -> pd.DataFrame:
        """Load real network monitoring data"""
        file_extension = filepath.split('.')[-1].lower()
        
        if file_extension == 'csv':
            return pd.read_csv(filepath)
        elif file_extension == 'json':
            return pd.read_json(filepath)
        elif file_extension == 'parquet':
            return pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate the quality of generated or loaded data"""
        
        validation_results = {
            'is_valid': True,
            'issues': [],
            'statistics': {},
            'data_info': {}
        }
        
        # Basic data info
        validation_results['data_info'] = {
            'shape': data.shape,
            'columns': list(data.columns),
            'missing_values': data.isnull().sum().to_dict(),
            'data_types': data.dtypes.to_dict()
        }
        
        # Check for missing values
        missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        if missing_ratio > 0.05:  # More than 5% missing
            validation_results['issues'].append(f"High missing value ratio: {missing_ratio:.2%}")
        
        # Check for duplicate timestamps
        if 'timestamp' in data.columns:
            duplicate_timestamps = data['timestamp'].duplicated().sum()
            if duplicate_timestamps > 0:
                validation_results['issues'].append(f"Found {duplicate_timestamps} duplicate timestamps")
        
        # Statistical validation
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            col_stats = {
                'mean': data[col].mean(),
                'std': data[col].std(),
                'min': data[col].min(),
                'max': data[col].max(),
                'outliers': len(data[col][np.abs(data[col] - data[col].mean()) > 3 * data[col].std()])
            }
            validation_results['statistics'][col] = col_stats
            
            # Check for unrealistic values based on column type
            if col in ['cpu_usage', 'memory_usage']:
                if (data[col] < 0).any() or (data[col] > 1).any():
                    validation_results['issues'].append(f"Unrealistic values in {col}")
            elif col.endswith('_mbps') or 'traffic' in col.lower():
                if (data[col] < 0).any() or (data[col] > 100000).any():  # Very high traffic threshold
                    validation_results['issues'].append(f"Unrealistic values in {col}")
            elif 'latency' in col.lower() or col.endswith('_ms'):
                if (data[col] < 0).any() or (data[col] > 10000).any():  # Very high latency threshold
                    validation_results['issues'].append(f"Unrealistic values in {col}")
            elif 'temperature' in col.lower():
                if (data[col] < -50).any() or (data[col] > 150).any():  # Temperature range
                    validation_results['issues'].append(f"Unrealistic values in {col}")
            elif 'error' in col.lower():
                if (data[col] < 0).any():  # Errors should be non-negative
                    validation_results['issues'].append(f"Unrealistic values in {col}")
        
        validation_results['is_valid'] = len(validation_results['issues']) == 0
        
        return validation_results


# Example usage and testing
if __name__ == "__main__":
    # Test data generation
    from .network_generator import NetworkGenerator, NetworkConfig
    
    # Generate a test network
    network_config = NetworkConfig()
    generator = NetworkGenerator(network_config)
    network = generator.generate_network('small_world')
    
    # Generate monitoring data
    data_config = DataConfig()
    data_generator = NetworkDataGenerator(data_config)
    
    # Generate data for a single node
    node_data = data_generator.generate_node_timeseries(network, list(network.nodes())[0])
    print("Generated node data shape:", node_data.shape)
    print("Columns:", node_data.columns.tolist())
    print("Anomaly ratio:", node_data['is_anomaly'].mean())
    
    # Generate network events
    events = data_generator.generate_network_events(network, 20)
    print("\nGenerated events shape:", events.shape)
    print("Event types:", events['event_type'].value_counts().to_dict())
    
    # Validate data quality
    validation = data_generator.validate_data_quality(node_data)
    print("\nData validation:", validation['is_valid'])
    if validation['issues']:
        print("Issues found:", validation['issues'])