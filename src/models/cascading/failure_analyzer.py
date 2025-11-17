"""
Cascading failure analysis module for complex networks.
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import random
from collections import defaultdict, deque
import yaml
import os
from datetime import datetime
import copy


class NodeState(Enum):
    """Node states in cascading failure simulation"""
    ACTIVE = "active"
    FAILED = "failed"
    OVERLOADED = "overloaded"
    RECOVERING = "recovering"


@dataclass
class CascadingFailureConfig:
    """Configuration for cascading failure analysis"""
    initial_capacity_ratio: float = 1.5
    failure_threshold: float = 0.8
    max_iterations: int = 50
    failure_probability: float = 0.05
    num_simulations: int = 100
    critical_nodes_count: int = 10
    recovery_probability: float = 0.1
    load_redistribution_method: str = "capacity_based"  # capacity_based, equal, degree_based
    
    @classmethod
    def from_config(cls, config_path: str = None):
        """Load configuration from YAML file"""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "../../../config/config.yaml")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        cf_config = config.get('cascading_failure', {})
        return cls(
            initial_capacity_ratio=cf_config.get('initial_capacity_ratio', 1.5),
            failure_threshold=cf_config.get('failure_threshold', 0.8),
            max_iterations=cf_config.get('max_iterations', 50),
            failure_probability=cf_config.get('failure_probability', 0.05),
            num_simulations=cf_config.get('num_simulations', 100),
            critical_nodes_count=cf_config.get('critical_nodes_count', 10)
        )


@dataclass
class NodeInfo:
    """Information about a node in the network"""
    node_id: str
    capacity: float
    load: float
    state: NodeState
    failure_threshold: float = 0.8
    failure_probability: float = 0.05
    recovery_time: int = 0
    original_capacity: float = 0.0
    
    def __post_init__(self):
        if self.original_capacity == 0.0:
            self.original_capacity = self.capacity


@dataclass
class SimulationSnapshot:
    """Snapshot of the network state at a specific iteration"""
    iteration: int
    failed_nodes: Set[str]
    overloaded_nodes: Set[str]
    total_failures: int
    total_load: float
    average_load_ratio: float
    network_connectivity: float
    new_failures: Set[str]


class CascadingFailureSimulator:
    """Simulator for cascading failure in networks"""
    
    def __init__(self, network: nx.Graph, config: CascadingFailureConfig = None):
        self.original_network = network.copy()
        self.network = network.copy()
        self.config = config or CascadingFailureConfig()
        self.node_info = {}
        self.simulation_history = []
        self.random_state = random.Random(42)
        self._initialize_nodes()
    
    def _initialize_nodes(self):
        """Initialize node information for simulation"""
        self.node_info = {}
        degrees = dict(self.network.degree())
        max_degree = max(degrees.values()) if degrees else 1
        
        for node in self.network.nodes():
            degree = degrees[node]
            
            # Base load proportional to degree
            initial_load = degree / max_degree
            
            # Capacity based on load with some buffer
            capacity = initial_load * self.config.initial_capacity_ratio
            
            # Add some randomization to thresholds
            failure_threshold = self.config.failure_threshold + self.random_state.uniform(-0.1, 0.1)
            failure_threshold = max(0.5, min(0.95, failure_threshold))
            
            self.node_info[str(node)] = NodeInfo(
                node_id=str(node),
                capacity=capacity,
                load=initial_load,
                state=NodeState.ACTIVE,
                failure_threshold=failure_threshold,
                failure_probability=self.config.failure_probability,
                original_capacity=capacity
            )
    
    def simulate_cascading_failure(self, initial_failures: List[str]) -> Dict[str, Any]:
        """Simulate cascading failure starting from initial failures"""
        # Reset network state
        self._reset_network_state()
        
        # Set initial failures
        for node_id in initial_failures:
            if node_id in self.node_info:
                self.node_info[node_id].state = NodeState.FAILED
        
        # Simulation loop
        iteration = 0
        self.simulation_history = []
        
        while iteration < self.config.max_iterations:
            # Create snapshot of current state
            snapshot = self._create_snapshot(iteration)
            self.simulation_history.append(snapshot)
            
            # Redistribute load
            self._redistribute_load()
            
            # Check for new failures
            new_failures = self._check_for_new_failures()
            
            # Update node states
            self._update_node_states(new_failures)
            
            # Check for recovery (optional)
            if self.config.recovery_probability > 0:
                self._attempt_recovery()
            
            # Check stopping condition
            if not new_failures:
                break
            
            iteration += 1
        
        # Final snapshot
        final_snapshot = self._create_snapshot(iteration)
        self.simulation_history.append(final_snapshot)
        
        # Calculate results
        return self._calculate_simulation_results()
    
    def _reset_network_state(self):
        """Reset all nodes to active state"""
        for node_info in self.node_info.values():
            node_info.state = NodeState.ACTIVE
            node_info.recovery_time = 0
            node_info.capacity = node_info.original_capacity
            # Reset load to initial state
            node_info.load = node_info.original_capacity / self.config.initial_capacity_ratio
    
    def _create_snapshot(self, iteration: int) -> SimulationSnapshot:
        """Create a snapshot of the current network state"""
        failed_nodes = {node_id for node_id, info in self.node_info.items() 
                       if info.state == NodeState.FAILED}
        overloaded_nodes = {node_id for node_id, info in self.node_info.items() 
                           if info.state == NodeState.OVERLOADED}
        
        # Calculate total load and average load ratio
        active_nodes = [info for info in self.node_info.values() 
                       if info.state == NodeState.ACTIVE]
        
        total_load = sum(info.load for info in self.node_info.values())
        avg_load_ratio = np.mean([info.load / info.capacity for info in active_nodes]) if active_nodes else 0
        
        # Calculate network connectivity
        temp_network = self.network.copy()
        temp_network.remove_nodes_from(failed_nodes)
        connectivity = self._calculate_connectivity(temp_network)
        
        # Determine new failures from previous iteration
        if iteration > 0 and self.simulation_history:
            prev_failed = self.simulation_history[-1].failed_nodes
            new_failures = failed_nodes - prev_failed
        else:
            new_failures = failed_nodes
        
        return SimulationSnapshot(
            iteration=iteration,
            failed_nodes=failed_nodes.copy(),
            overloaded_nodes=overloaded_nodes.copy(),
            total_failures=len(failed_nodes),
            total_load=total_load,
            average_load_ratio=avg_load_ratio,
            network_connectivity=connectivity,
            new_failures=new_failures.copy()
        )
    
    def _redistribute_load(self):
        """Redistribute load from failed nodes to active neighbors"""
        failed_nodes = {node_id for node_id, info in self.node_info.items() 
                       if info.state == NodeState.FAILED}
        
        for failed_node in failed_nodes:
            failed_load = self.node_info[failed_node].load
            
            if failed_load > 0:
                # Find active neighbors
                neighbors = list(self.network.neighbors(int(failed_node)))
                active_neighbors = [n for n in neighbors 
                                  if str(n) in self.node_info and 
                                  self.node_info[str(n)].state == NodeState.ACTIVE]
                
                if active_neighbors:
                    # Redistribute load based on method
                    self._distribute_load_to_neighbors(failed_load, active_neighbors)
                
                # Set failed node load to zero
                self.node_info[failed_node].load = 0
    
    def _distribute_load_to_neighbors(self, load_to_distribute: float, 
                                    active_neighbors: List[str]):
        """Distribute load to active neighbors based on redistribution method"""
        method = self.config.load_redistribution_method
        
        if method == "equal":
            # Equal distribution
            load_per_neighbor = load_to_distribute / len(active_neighbors)
            for neighbor in active_neighbors:
                self.node_info[str(neighbor)].load += load_per_neighbor
        
        elif method == "capacity_based":
            # Distribution based on remaining capacity
            remaining_capacities = []
            for neighbor in active_neighbors:
                neighbor_info = self.node_info[str(neighbor)]
                remaining_capacity = max(0, neighbor_info.capacity - neighbor_info.load)
                remaining_capacities.append(remaining_capacity)
            
            total_remaining = sum(remaining_capacities)
            
            if total_remaining > 0:
                for i, neighbor in enumerate(active_neighbors):
                    proportion = remaining_capacities[i] / total_remaining
                    additional_load = load_to_distribute * proportion
                    self.node_info[str(neighbor)].load += additional_load
            else:
                # Fallback to equal distribution
                load_per_neighbor = load_to_distribute / len(active_neighbors)
                for neighbor in active_neighbors:
                    self.node_info[str(neighbor)].load += load_per_neighbor
        
        elif method == "degree_based":
            # Distribution based on node degree
            degrees = [self.network.degree(neighbor) for neighbor in active_neighbors]  # neighbors are already int
            total_degree = sum(degrees)
            
            if total_degree > 0:
                for i, neighbor in enumerate(active_neighbors):
                    proportion = degrees[i] / total_degree
                    additional_load = load_to_distribute * proportion
                    self.node_info[str(neighbor)].load += additional_load
            else:
                # Fallback to equal distribution
                load_per_neighbor = load_to_distribute / len(active_neighbors)
                for neighbor in active_neighbors:
                    self.node_info[str(neighbor)].load += load_per_neighbor
    
    def _check_for_new_failures(self) -> Set[str]:
        """Check for nodes that should fail based on load and threshold"""
        new_failures = set()
        
        for node_id, node_info in self.node_info.items():
            if node_info.state == NodeState.ACTIVE:
                load_ratio = node_info.load / node_info.capacity
                
                if load_ratio > node_info.failure_threshold:
                    # Calculate failure probability based on overload
                    overload_factor = load_ratio - node_info.failure_threshold
                    failure_prob = min(0.9, node_info.failure_probability * (1 + overload_factor * 3))
                    
                    if self.random_state.random() < failure_prob:
                        new_failures.add(node_id)
                    else:
                        # Mark as overloaded but not failed
                        node_info.state = NodeState.OVERLOADED
                elif node_info.state == NodeState.OVERLOADED:
                    # Node was overloaded but load decreased, return to active
                    node_info.state = NodeState.ACTIVE
        
        return new_failures
    
    def _update_node_states(self, new_failures: Set[str]):
        """Update node states with new failures"""
        for node_id in new_failures:
            self.node_info[node_id].state = NodeState.FAILED
    
    def _attempt_recovery(self):
        """Attempt to recover some failed nodes"""
        failed_nodes = [node_id for node_id, info in self.node_info.items() 
                       if info.state == NodeState.FAILED]
        
        for node_id in failed_nodes:
            if self.random_state.random() < self.config.recovery_probability:
                self.node_info[node_id].state = NodeState.RECOVERING
                # Partial capacity restoration
                self.node_info[node_id].capacity = self.node_info[node_id].original_capacity * 0.7
                self.node_info[node_id].load = 0  # Start with no load
    
    def _calculate_connectivity(self, network: nx.Graph) -> float:
        """Calculate network connectivity (size of largest connected component)"""
        if network.number_of_nodes() == 0:
            return 0.0
        
        if nx.is_connected(network):
            return 1.0
        
        # Find largest connected component
        largest_cc = max(nx.connected_components(network), key=len)
        return len(largest_cc) / self.original_network.number_of_nodes()
    
    def _calculate_simulation_results(self) -> Dict[str, Any]:
        """Calculate comprehensive simulation results"""
        if not self.simulation_history:
            return {}
        
        final_snapshot = self.simulation_history[-1]
        
        # Basic statistics
        results = {
            'initial_nodes': self.original_network.number_of_nodes(),
            'final_failures': final_snapshot.total_failures,
            'failure_ratio': final_snapshot.total_failures / self.original_network.number_of_nodes(),
            'total_iterations': final_snapshot.iteration,
            'final_connectivity': final_snapshot.network_connectivity,
            'robustness_score': 1.0 - (final_snapshot.total_failures / self.original_network.number_of_nodes()),
            'simulation_history': [asdict(snapshot) for snapshot in self.simulation_history]
        }
        
        # Analyze failure progression
        failure_progression = [snapshot.total_failures for snapshot in self.simulation_history]
        results['failure_progression'] = failure_progression
        
        # Calculate failure velocity (failures per iteration)
        if len(failure_progression) > 1:
            velocities = [failure_progression[i+1] - failure_progression[i] 
                         for i in range(len(failure_progression)-1)]
            results['max_failure_velocity'] = max(velocities) if velocities else 0
            results['average_failure_velocity'] = np.mean(velocities) if velocities else 0
        
        # Connectivity evolution
        connectivity_evolution = [snapshot.network_connectivity for snapshot in self.simulation_history]
        results['connectivity_evolution'] = connectivity_evolution
        results['connectivity_drop'] = 1.0 - min(connectivity_evolution) if connectivity_evolution else 0
        
        return results


class CascadingFailureAnalyzer:
    """High-level analyzer for cascading failure analysis"""
    
    def __init__(self, config: CascadingFailureConfig = None):
        self.config = config or CascadingFailureConfig()
        self.simulator = None
        self.analysis_results = {}
    
    def analyze_network_robustness(self, network: nx.Graph) -> Dict[str, Any]:
        """Comprehensive robustness analysis of the network"""
        self.simulator = CascadingFailureSimulator(network, self.config)
        
        analysis_results = {
            'network_info': self._get_network_info(network),
            'single_node_failures': self._analyze_single_node_failures(network),
            'multiple_node_failures': self._analyze_multiple_node_failures(network),
            'critical_nodes': self._identify_critical_nodes(network),
            'failure_patterns': self._analyze_failure_patterns(network),
            'robustness_metrics': self._calculate_robustness_metrics(network)
        }
        
        self.analysis_results = analysis_results
        return analysis_results
    
    def _get_network_info(self, network: nx.Graph) -> Dict[str, Any]:
        """Get basic network information"""
        return {
            'nodes': network.number_of_nodes(),
            'edges': network.number_of_edges(),
            'density': nx.density(network),
            'is_connected': nx.is_connected(network),
            'diameter': nx.diameter(network) if nx.is_connected(network) else None,
            'average_clustering': nx.average_clustering(network),
            'average_path_length': nx.average_shortest_path_length(network) if nx.is_connected(network) else None
        }
    
    def _analyze_single_node_failures(self, network: nx.Graph) -> Dict[str, Any]:
        """Analyze impact of single node failures"""
        nodes = list(network.nodes())
        failure_impacts = {}
        
        print(f"Analyzing single node failures for {len(nodes)} nodes...")
        
        for i, node in enumerate(nodes):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(nodes)}")
            
            # Simulate failure of single node
            result = self.simulator.simulate_cascading_failure([str(node)])
            
            failure_impacts[str(node)] = {
                'final_failures': result['final_failures'],
                'failure_ratio': result['failure_ratio'],
                'iterations': result['total_iterations'],
                'connectivity_drop': result.get('connectivity_drop', 0),
                'robustness_score': result['robustness_score']
            }
        
        # Calculate statistics
        impacts = list(failure_impacts.values())
        
        return {
            'individual_impacts': failure_impacts,
            'statistics': {
                'mean_final_failures': np.mean([imp['final_failures'] for imp in impacts]),
                'max_final_failures': np.max([imp['final_failures'] for imp in impacts]),
                'mean_failure_ratio': np.mean([imp['failure_ratio'] for imp in impacts]),
                'max_failure_ratio': np.max([imp['failure_ratio'] for imp in impacts]),
                'mean_iterations': np.mean([imp['iterations'] for imp in impacts]),
                'max_iterations': np.max([imp['iterations'] for imp in impacts])
            }
        }
    
    def _analyze_multiple_node_failures(self, network: nx.Graph) -> Dict[str, Any]:
        """Analyze impact of multiple simultaneous node failures"""
        nodes = list(network.nodes())
        results = {}
        
        # Test different numbers of simultaneous failures
        failure_counts = [2, 3, 5, min(10, len(nodes)//2)]
        
        for count in failure_counts:
            if count >= len(nodes):
                continue
            
            print(f"Analyzing {count} simultaneous failures...")
            
            # Multiple random selections
            impacts = []
            for _ in range(min(20, 100 // count)):  # Limit simulations for larger counts
                failed_nodes = self.simulator.random_state.sample(nodes, count)
                result = self.simulator.simulate_cascading_failure([str(n) for n in failed_nodes])
                impacts.append(result)
            
            # Calculate statistics
            results[f'{count}_failures'] = {
                'mean_final_failures': np.mean([r['final_failures'] for r in impacts]),
                'max_final_failures': np.max([r['final_failures'] for r in impacts]),
                'mean_failure_ratio': np.mean([r['failure_ratio'] for r in impacts]),
                'max_failure_ratio': np.max([r['failure_ratio'] for r in impacts]),
                'mean_robustness': np.mean([r['robustness_score'] for r in impacts]),
                'min_robustness': np.min([r['robustness_score'] for r in impacts])
            }
        
        return results
    
    def _identify_critical_nodes(self, network: nx.Graph) -> Dict[str, Any]:
        """Identify most critical nodes in the network"""
        # Get single node failure impacts
        single_failures = self._analyze_single_node_failures(network)
        impacts = single_failures['individual_impacts']
        
        # Sort nodes by impact
        sorted_nodes = sorted(impacts.items(), 
                            key=lambda x: (x[1]['final_failures'], x[1]['connectivity_drop']), 
                            reverse=True)
        
        critical_nodes = sorted_nodes[:self.config.critical_nodes_count]
        
        # Additional centrality measures
        centralities = {
            'degree': nx.degree_centrality(network),
            'betweenness': nx.betweenness_centrality(network, k=min(100, len(network))),
            'closeness': nx.closeness_centrality(network),
            'eigenvector': nx.eigenvector_centrality(network, max_iter=1000)
        }
        
        # Combine failure impact with centrality
        critical_analysis = {}
        for node_id, impact in critical_nodes:
            critical_analysis[node_id] = {
                'failure_impact': impact,
                'centralities': {
                    centrality_type: centralities[centrality_type].get(node_id, 0)
                    for centrality_type in centralities
                },
                'degree': network.degree(int(node_id)),
                'neighbors': list(network.neighbors(int(node_id)))
            }
        
        return {
            'critical_nodes_ranking': [(node, impact['final_failures']) for node, impact in critical_nodes],
            'detailed_analysis': critical_analysis,
            'centrality_correlation': self._calculate_centrality_correlation(impacts, centralities)
        }
    
    def _calculate_centrality_correlation(self, impacts: Dict, centralities: Dict) -> Dict:
        """Calculate correlation between centrality measures and failure impact"""
        correlations = {}
        
        # Extract impact values
        impact_values = [impacts[node]['final_failures'] for node in impacts.keys() if node in impacts]
        
        for centrality_type, centrality_values in centralities.items():
            # Extract centrality values in the same order
            centrality_list = [centrality_values.get(node, 0) for node in impacts.keys()]
            
            # Calculate correlation
            if len(impact_values) > 1 and len(centrality_list) > 1:
                correlation = np.corrcoef(impact_values, centrality_list)[0, 1]
                correlations[centrality_type] = correlation if not np.isnan(correlation) else 0
            else:
                correlations[centrality_type] = 0
        
        return correlations
    
    def _analyze_failure_patterns(self, network: nx.Graph) -> Dict[str, Any]:
        """Analyze common failure patterns and propagation paths"""
        # Sample multiple failure scenarios
        nodes = list(network.nodes())
        pattern_data = []
        
        print("Analyzing failure propagation patterns...")
        
        # Analyze 50 random single-node failures for pattern detection
        for i in range(min(50, len(nodes))):
            initial_node = self.simulator.random_state.choice(nodes)
            result = self.simulator.simulate_cascading_failure([str(initial_node)])
            
            pattern_data.append({
                'initial_node': str(initial_node),
                'initial_degree': network.degree(initial_node),
                'result': result
            })
        
        # Analyze patterns
        patterns = {
            'degree_impact_correlation': self._analyze_degree_impact_correlation(pattern_data),
            'failure_velocity_patterns': self._analyze_failure_velocity_patterns(pattern_data),
            'propagation_characteristics': self._analyze_propagation_characteristics(pattern_data)
        }
        
        return patterns
    
    def _analyze_degree_impact_correlation(self, pattern_data: List[Dict]) -> Dict:
        """Analyze correlation between initial node degree and failure impact"""
        degrees = [data['initial_degree'] for data in pattern_data]
        impacts = [data['result']['final_failures'] for data in pattern_data]
        
        if len(degrees) > 1:
            correlation = np.corrcoef(degrees, impacts)[0, 1]
            correlation = correlation if not np.isnan(correlation) else 0
        else:
            correlation = 0
        
        return {
            'correlation': correlation,
            'mean_impact_by_degree': self._group_by_degree_ranges(pattern_data)
        }
    
    def _group_by_degree_ranges(self, pattern_data: List[Dict]) -> Dict:
        """Group impacts by degree ranges"""
        degree_groups = {'low': [], 'medium': [], 'high': []}
        degrees = [data['initial_degree'] for data in pattern_data]
        
        if degrees:
            low_threshold = np.percentile(degrees, 33)
            high_threshold = np.percentile(degrees, 67)
            
            for data in pattern_data:
                degree = data['initial_degree']
                impact = data['result']['final_failures']
                
                if degree <= low_threshold:
                    degree_groups['low'].append(impact)
                elif degree >= high_threshold:
                    degree_groups['high'].append(impact)
                else:
                    degree_groups['medium'].append(impact)
        
        return {
            group: {'mean': np.mean(impacts) if impacts else 0, 'count': len(impacts)}
            for group, impacts in degree_groups.items()
        }
    
    def _analyze_failure_velocity_patterns(self, pattern_data: List[Dict]) -> Dict:
        """Analyze failure propagation velocity patterns"""
        velocities = []
        
        for data in pattern_data:
            result = data['result']
            if 'max_failure_velocity' in result:
                velocities.append(result['max_failure_velocity'])
        
        if velocities:
            return {
                'mean_velocity': np.mean(velocities),
                'max_velocity': np.max(velocities),
                'velocity_distribution': np.histogram(velocities, bins=5)[0].tolist()
            }
        else:
            return {'mean_velocity': 0, 'max_velocity': 0, 'velocity_distribution': []}
    
    def _analyze_propagation_characteristics(self, pattern_data: List[Dict]) -> Dict:
        """Analyze general propagation characteristics"""
        characteristics = {
            'quick_failures': 0,  # Failures that complete in < 5 iterations
            'slow_failures': 0,   # Failures that take > 10 iterations
            'limited_impact': 0,  # Failures affecting < 10% of nodes
            'severe_impact': 0    # Failures affecting > 50% of nodes
        }
        
        total_nodes = len(list(pattern_data[0]['result']['simulation_history'][0]['failed_nodes']) + 
                         list(pattern_data[0]['result']['simulation_history'][0]['overloaded_nodes'])) \
                     if pattern_data else 0
        
        for data in pattern_data:
            result = data['result']
            
            iterations = result['total_iterations']
            failure_ratio = result['failure_ratio']
            
            if iterations < 5:
                characteristics['quick_failures'] += 1
            if iterations > 10:
                characteristics['slow_failures'] += 1
            if failure_ratio < 0.1:
                characteristics['limited_impact'] += 1
            if failure_ratio > 0.5:
                characteristics['severe_impact'] += 1
        
        # Convert to ratios
        total = len(pattern_data) if pattern_data else 1
        return {key: value / total for key, value in characteristics.items()}
    
    def _calculate_robustness_metrics(self, network: nx.Graph) -> Dict[str, Any]:
        """Calculate comprehensive robustness metrics"""
        # Network structural metrics
        structural_metrics = {
            'algebraic_connectivity': self._calculate_algebraic_connectivity(network),
            'node_connectivity': nx.node_connectivity(network),
            'edge_connectivity': nx.edge_connectivity(network),
            'assortativity': nx.degree_assortativity_coefficient(network)
        }
        
        # Failure-based metrics (from previous analyses)
        if hasattr(self, 'analysis_results') and 'single_node_failures' in self.analysis_results:
            single_failure_stats = self.analysis_results['single_node_failures']['statistics']
            failure_metrics = {
                'average_cascade_size': single_failure_stats['mean_final_failures'],
                'worst_case_cascade': single_failure_stats['max_final_failures'],
                'robustness_index': 1.0 - single_failure_stats['mean_failure_ratio']
            }
        else:
            failure_metrics = {}
        
        return {
            'structural_metrics': structural_metrics,
            'failure_metrics': failure_metrics,
            'overall_robustness_score': self._calculate_overall_robustness_score(
                structural_metrics, failure_metrics
            )
        }
    
    def _calculate_algebraic_connectivity(self, network: nx.Graph) -> float:
        """Calculate algebraic connectivity (second smallest eigenvalue of Laplacian)"""
        if network.number_of_nodes() < 2:
            return 0.0
        
        try:
            L = nx.laplacian_matrix(network).toarray()
            eigenvalues = np.linalg.eigvals(L)
            eigenvalues.sort()
            return float(eigenvalues[1])  # Second smallest eigenvalue
        except:
            return 0.0
    
    def _calculate_overall_robustness_score(self, structural: Dict, failure: Dict) -> float:
        """Calculate overall robustness score combining multiple metrics"""
        score = 0.0
        weight_sum = 0.0
        
        # Structural component (40% weight)
        if 'algebraic_connectivity' in structural and structural['algebraic_connectivity'] > 0:
            score += 0.2 * min(1.0, structural['algebraic_connectivity'] / 0.1)
            weight_sum += 0.2
        
        if 'node_connectivity' in structural and structural['node_connectivity'] > 0:
            score += 0.1 * min(1.0, structural['node_connectivity'] / 3.0)
            weight_sum += 0.1
        
        if 'edge_connectivity' in structural and structural['edge_connectivity'] > 0:
            score += 0.1 * min(1.0, structural['edge_connectivity'] / 3.0)
            weight_sum += 0.1
        
        # Failure-based component (60% weight)
        if 'robustness_index' in failure:
            score += 0.6 * failure['robustness_index']
            weight_sum += 0.6
        
        return score / weight_sum if weight_sum > 0 else 0.5
    
    def generate_report(self) -> str:
        """Generate a comprehensive analysis report"""
        if not self.analysis_results:
            return "No analysis results available. Please run analyze_network_robustness() first."
        
        report = []
        report.append("=" * 60)
        report.append("CASCADING FAILURE ANALYSIS REPORT")
        report.append("=" * 60)
        
        # Network info
        network_info = self.analysis_results.get('network_info', {})
        report.append(f"\nNetwork Information:")
        report.append(f"  Nodes: {network_info.get('nodes', 'N/A')}")
        report.append(f"  Edges: {network_info.get('edges', 'N/A')}")
        report.append(f"  Density: {network_info.get('density', 0):.4f}")
        report.append(f"  Connected: {network_info.get('is_connected', False)}")
        
        # Critical nodes
        critical = self.analysis_results.get('critical_nodes', {})
        if 'critical_nodes_ranking' in critical:
            report.append(f"\nTop 5 Most Critical Nodes:")
            for i, (node, impact) in enumerate(critical['critical_nodes_ranking'][:5]):
                report.append(f"  {i+1}. Node {node}: {impact} cascade failures")
        
        # Robustness metrics
        robustness = self.analysis_results.get('robustness_metrics', {})
        if 'overall_robustness_score' in robustness:
            score = robustness['overall_robustness_score']
            report.append(f"\nOverall Robustness Score: {score:.3f}")
            
            if score > 0.8:
                report.append("  Assessment: HIGHLY ROBUST")
            elif score > 0.6:
                report.append("  Assessment: MODERATELY ROBUST")
            elif score > 0.4:
                report.append("  Assessment: VULNERABLE")
            else:
                report.append("  Assessment: HIGHLY VULNERABLE")
        
        # Single failure analysis
        single = self.analysis_results.get('single_node_failures', {})
        if 'statistics' in single:
            stats = single['statistics']
            report.append(f"\nSingle Node Failure Analysis:")
            report.append(f"  Average cascade size: {stats.get('mean_final_failures', 0):.1f}")
            report.append(f"  Worst case cascade: {stats.get('max_final_failures', 0)}")
            report.append(f"  Average failure ratio: {stats.get('mean_failure_ratio', 0):.3f}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def save_results(self, filepath: str):
        """Save analysis results to file"""
        import json
        
        # Convert any non-serializable objects to strings
        serializable_results = self._make_serializable(self.analysis_results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def _make_serializable(self, obj):
        """Make object JSON serializable"""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        else:
            return obj


# Example usage
if __name__ == "__main__":
    # Test cascading failure analysis
    from ..data.generators.network_generator import NetworkGenerator, NetworkConfig
    
    # Generate test network
    print("Generating test network...")
    config = NetworkConfig(node_count=30, network_type='small_world')
    generator = NetworkGenerator(config)
    network = generator.generate_network()
    
    print(f"Network generated: {network.number_of_nodes()} nodes, {network.number_of_edges()} edges")
    
    # Analyze cascading failures
    print("\nStarting cascading failure analysis...")
    cf_config = CascadingFailureConfig(num_simulations=20)  # Reduced for testing
    analyzer = CascadingFailureAnalyzer(cf_config)
    
    results = analyzer.analyze_network_robustness(network)
    
    # Print report
    print("\n" + analyzer.generate_report())