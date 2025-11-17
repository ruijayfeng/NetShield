"""
Network topology generation module for creating various types of complex networks.
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import random
from dataclasses import dataclass
import yaml
import os


@dataclass
class NetworkConfig:
    """Configuration for network generation"""
    node_count: int = 50
    edge_probability: float = 0.1
    network_type: str = "small_world"
    k_neighbors: int = 6
    rewiring_prob: float = 0.3
    
    @classmethod
    def from_config(cls, config_path: str = None):
        """Load configuration from YAML file"""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "../../../config/config.yaml")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        network_config = config.get('network', {})
        return cls(
            node_count=network_config.get('node_count', 50),
            edge_probability=network_config.get('edge_probability', 0.1),
            network_type=network_config.get('network_type', 'small_world'),
            k_neighbors=network_config.get('k_neighbors', 6),
            rewiring_prob=network_config.get('rewiring_prob', 0.3)
        )


class NetworkGenerator:
    """Generate various types of complex networks"""
    
    def __init__(self, config: NetworkConfig = None):
        self.config = config or NetworkConfig()
        self.network = None
        
    def generate_network(self, network_type: str = None) -> nx.Graph:
        """Generate network based on specified type"""
        network_type = network_type or self.config.network_type
        
        if network_type == "small_world":
            return self._generate_small_world()
        elif network_type == "scale_free":
            return self._generate_scale_free()
        elif network_type == "erdos_renyi":
            return self._generate_erdos_renyi()
        elif network_type == "grid":
            return self._generate_grid()
        else:
            raise ValueError(f"Unsupported network type: {network_type}")
    
    def _generate_small_world(self) -> nx.Graph:
        """Generate small world network (Watts-Strogatz model)"""
        network = nx.watts_strogatz_graph(
            n=self.config.node_count,
            k=self.config.k_neighbors,
            p=self.config.rewiring_prob,
            seed=42
        )
        
        # Add node attributes
        self._add_node_attributes(network)
        return network
    
    def _generate_scale_free(self) -> nx.Graph:
        """Generate scale-free network (Barabási-Albert model)"""
        m = max(1, self.config.k_neighbors // 2)
        network = nx.barabasi_albert_graph(
            n=self.config.node_count,
            m=m,
            seed=42
        )
        
        self._add_node_attributes(network)
        return network
    
    def _generate_erdos_renyi(self) -> nx.Graph:
        """Generate Erdős-Rényi random network"""
        network = nx.erdos_renyi_graph(
            n=self.config.node_count,
            p=self.config.edge_probability,
            seed=42
        )
        
        self._add_node_attributes(network)
        return network
    
    def _generate_grid(self) -> nx.Graph:
        """Generate 2D grid network"""
        # Calculate grid dimensions
        side_length = int(np.sqrt(self.config.node_count))
        network = nx.grid_2d_graph(side_length, side_length)
        
        # Convert to simple graph with integer node labels
        mapping = {node: i for i, node in enumerate(network.nodes())}
        network = nx.relabel_nodes(network, mapping)
        
        self._add_node_attributes(network)
        return network
    
    def _add_node_attributes(self, network: nx.Graph):
        """Add attributes to network nodes"""
        for node in network.nodes():
            degree = network.degree(node)
            # Initialize node attributes
            network.nodes[node].update({
                'degree': degree,
                'capacity': degree * 1.2 + np.random.normal(0, 0.1),  # Base capacity on degree
                'initial_load': degree * 0.8 + np.random.normal(0, 0.05),
                'status': 'active',
                'node_type': self._assign_node_type(degree),
                'coordinates': (np.random.random(), np.random.random())  # For visualization
            })
    
    def _assign_node_type(self, degree: int) -> str:
        """Assign node type based on degree"""
        if degree > 8:
            return 'hub'
        elif degree > 4:
            return 'intermediate'
        else:
            return 'leaf'
    
    def add_edge_attributes(self, network: nx.Graph):
        """Add attributes to network edges"""
        for edge in network.edges():
            network.edges[edge].update({
                'weight': np.random.uniform(0.5, 2.0),
                'capacity': np.random.uniform(10, 100),
                'latency': np.random.uniform(1, 50),  # milliseconds
                'reliability': np.random.uniform(0.9, 0.99)
            })
    
    def get_network_statistics(self, network: nx.Graph) -> Dict[str, Any]:
        """Calculate network topology statistics"""
        if len(network) == 0:
            return {}
        
        # Basic statistics
        stats = {
            'nodes': network.number_of_nodes(),
            'edges': network.number_of_edges(),
            'density': nx.density(network),
            'average_degree': sum(dict(network.degree()).values()) / network.number_of_nodes()
        }
        
        # Connectivity statistics
        if nx.is_connected(network):
            stats.update({
                'diameter': nx.diameter(network),
                'average_path_length': nx.average_shortest_path_length(network),
                'radius': nx.radius(network)
            })
        else:
            # For disconnected graphs
            largest_cc = max(nx.connected_components(network), key=len)
            largest_subgraph = network.subgraph(largest_cc)
            stats.update({
                'connected_components': nx.number_connected_components(network),
                'largest_cc_size': len(largest_cc),
                'diameter': nx.diameter(largest_subgraph),
                'average_path_length': nx.average_shortest_path_length(largest_subgraph)
            })
        
        # Centrality measures (sample of nodes for large networks)
        sample_nodes = list(network.nodes())
        if len(sample_nodes) > 100:
            sample_nodes = random.sample(sample_nodes, 100)
        
        degree_centrality = nx.degree_centrality(network)
        betweenness_centrality = nx.betweenness_centrality(network, k=len(sample_nodes))
        closeness_centrality = nx.closeness_centrality(network)
        
        stats.update({
            'average_degree_centrality': np.mean(list(degree_centrality.values())),
            'average_betweenness_centrality': np.mean(list(betweenness_centrality.values())),
            'average_closeness_centrality': np.mean(list(closeness_centrality.values())),
            'clustering_coefficient': nx.average_clustering(network)
        })
        
        return stats
    
    def save_network(self, network: nx.Graph, filepath: str, format: str = 'gml'):
        """Save network to file"""
        if format.lower() == 'gml':
            nx.write_gml(network, filepath)
        elif format.lower() == 'graphml':
            nx.write_graphml(network, filepath)
        elif format.lower() == 'edgelist':
            nx.write_edgelist(network, filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def load_network(self, filepath: str, format: str = 'gml') -> nx.Graph:
        """Load network from file"""
        if format.lower() == 'gml':
            return nx.read_gml(filepath)
        elif format.lower() == 'graphml':
            return nx.read_graphml(filepath)
        elif format.lower() == 'edgelist':
            return nx.read_edgelist(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")


class RealNetworkInterface:
    """Interface for integrating real network data"""
    
    def __init__(self):
        self.supported_formats = ['csv', 'json', 'gml', 'graphml', 'edgelist']
    
    def load_from_file(self, filepath: str, format: str = None) -> nx.Graph:
        """Load network from various file formats"""
        if format is None:
            format = filepath.split('.')[-1].lower()
        
        if format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format}")
        
        if format == 'csv':
            return self._load_from_csv(filepath)
        elif format == 'json':
            return self._load_from_json(filepath)
        elif format in ['gml', 'graphml', 'edgelist']:
            generator = NetworkGenerator()
            return generator.load_network(filepath, format)
    
    def _load_from_csv(self, filepath: str) -> nx.Graph:
        """Load network from CSV file (edge list format)"""
        df = pd.read_csv(filepath)
        
        # Assume CSV has columns: source, target, [weight]
        if 'source' not in df.columns or 'target' not in df.columns:
            raise ValueError("CSV must contain 'source' and 'target' columns")
        
        network = nx.Graph()
        
        for _, row in df.iterrows():
            source, target = row['source'], row['target']
            weight = row.get('weight', 1.0)
            
            network.add_edge(source, target, weight=weight)
        
        return network
    
    def _load_from_json(self, filepath: str) -> nx.Graph:
        """Load network from JSON file"""
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Expect format: {"nodes": [...], "edges": [...]}
        network = nx.Graph()
        
        # Add nodes
        for node_data in data.get('nodes', []):
            if isinstance(node_data, dict):
                node_id = node_data.pop('id')
                network.add_node(node_id, **node_data)
            else:
                network.add_node(node_data)
        
        # Add edges
        for edge_data in data.get('edges', []):
            if isinstance(edge_data, dict):
                source = edge_data.pop('source')
                target = edge_data.pop('target')
                network.add_edge(source, target, **edge_data)
            elif isinstance(edge_data, (list, tuple)) and len(edge_data) >= 2:
                network.add_edge(edge_data[0], edge_data[1])
        
        return network
    
    def validate_network_data(self, network: nx.Graph) -> Dict[str, Any]:
        """Validate loaded network data"""
        validation_results = {
            'is_valid': True,
            'issues': [],
            'statistics': {}
        }
        
        # Check for isolated nodes
        isolated_nodes = list(nx.isolates(network))
        if isolated_nodes:
            validation_results['issues'].append(f"Found {len(isolated_nodes)} isolated nodes")
        
        # Check for self-loops
        self_loops = list(nx.selfloop_edges(network))
        if self_loops:
            validation_results['issues'].append(f"Found {len(self_loops)} self-loops")
        
        # Check connectivity
        if not nx.is_connected(network):
            validation_results['issues'].append("Network is not connected")
        
        # Basic statistics
        generator = NetworkGenerator()
        validation_results['statistics'] = generator.get_network_statistics(network)
        
        validation_results['is_valid'] = len(validation_results['issues']) == 0
        
        return validation_results


# Example usage and testing
if __name__ == "__main__":
    # Test network generation
    config = NetworkConfig.from_config()
    generator = NetworkGenerator(config)
    
    # Generate different types of networks
    networks = {
        'small_world': generator.generate_network('small_world'),
        'scale_free': generator.generate_network('scale_free'),
        'erdos_renyi': generator.generate_network('erdos_renyi')
    }
    
    # Print statistics for each network
    for name, network in networks.items():
        print(f"\n{name.upper()} Network Statistics:")
        stats = generator.get_network_statistics(network)
        for key, value in stats.items():
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")