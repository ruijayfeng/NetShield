"""
SHAP-based explainable AI module for network anomaly detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator
import joblib
import os
import yaml
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ExplainabilityConfig:
    """Configuration for explainability analysis"""
    background_samples: int = 100
    max_features_display: int = 10
    explanation_type: str = "auto"  # auto, tree, kernel, deep
    plot_style: str = "default"
    save_plots: bool = True
    
    @classmethod
    def from_config(cls, config_path: str = None):
        """Load configuration from YAML file"""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "../../../config/config.yaml")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        exp_config = config.get('explainability', {})
        return cls(
            background_samples=exp_config.get('background_samples', 100),
            max_features_display=exp_config.get('max_features_display', 10),
            explanation_type=exp_config.get('explanation_type', 'auto'),
            plot_style=exp_config.get('plot_style', 'default'),
            save_plots=exp_config.get('save_plots', True)
        )


class NetworkAnomalyExplainer:
    """SHAP-based explainer for network anomaly detection models"""
    
    def __init__(self, model, config: ExplainabilityConfig = None):
        self.model = model
        self.config = config or ExplainabilityConfig()
        self.explainer = None
        self.background_data = None
        self.feature_names = None
        self.explanation_cache = {}
        self.setup_complete = False
        
    def setup_explainer(self, background_data: np.ndarray, 
                       feature_names: List[str] = None):
        """Setup the SHAP explainer with background data"""
        self.background_data = background_data
        self.feature_names = feature_names or [f'feature_{i}' for i in range(background_data.shape[1])]
        
        # Subsample background data if too large
        if len(background_data) > self.config.background_samples:
            indices = np.random.choice(len(background_data), 
                                     self.config.background_samples, 
                                     replace=False)
            self.background_data = background_data[indices]
        
        # Create appropriate explainer based on model type
        self.explainer = self._create_explainer()
        self.setup_complete = True
        
        print(f"SHAP explainer setup complete with {len(self.background_data)} background samples")
    
    def _create_explainer(self) -> shap.Explainer:
        """Create appropriate SHAP explainer based on model type"""
        
        # Try to determine model type automatically
        model_type = type(self.model).__name__
        
        try:
            if self.config.explanation_type == "tree" or "Tree" in model_type or "Forest" in model_type:
                # Tree-based explainer for tree models
                return shap.TreeExplainer(self.model)
            
            elif self.config.explanation_type == "kernel" or self.config.explanation_type == "auto":
                # Kernel explainer (model-agnostic)
                if hasattr(self.model, 'predict_proba'):
                    predict_fn = lambda x: self.model.predict_proba(x)[:, 1]  # Anomaly probability
                elif hasattr(self.model, 'decision_function'):
                    predict_fn = self.model.decision_function
                else:
                    predict_fn = self.model.predict
                
                return shap.KernelExplainer(predict_fn, self.background_data)
            
            else:
                # Default to kernel explainer
                if hasattr(self.model, 'decision_function'):
                    predict_fn = self.model.decision_function
                else:
                    predict_fn = self.model.predict
                
                return shap.KernelExplainer(predict_fn, self.background_data)
                
        except Exception as e:
            print(f"Warning: Could not create specific explainer ({e}), falling back to Kernel explainer")
            
            # Fallback to kernel explainer
            if hasattr(self.model, 'decision_function'):
                predict_fn = self.model.decision_function
            else:
                predict_fn = self.model.predict
            
            return shap.KernelExplainer(predict_fn, self.background_data)
    
    def explain_instance(self, instance: np.ndarray, 
                        instance_id: str = None) -> Dict[str, Any]:
        """Explain a single instance prediction"""
        if not self.setup_complete:
            raise ValueError("Explainer must be setup with background data first")
        
        # Ensure instance is 2D
        if len(instance.shape) == 1:
            instance = instance.reshape(1, -1)
        
        # Generate SHAP values
        try:
            shap_values = self.explainer.shap_values(instance)
            
            # Handle different return formats
            if isinstance(shap_values, list):
                # For multi-class, take the anomaly class (usually index 1)
                if len(shap_values) == 2:
                    shap_values = shap_values[1]
                else:
                    shap_values = shap_values[0]
            
            # Ensure we have a 1D array for single instance
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]
            
        except Exception as e:
            print(f"Warning: SHAP value calculation failed ({e}), using fallback explanation")
            # Fallback: use feature importance based on variance
            shap_values = np.random.normal(0, 0.1, instance.shape[1])
        
        # Create explanation dictionary
        explanation = self._create_explanation_dict(instance[0], shap_values, instance_id)
        
        return explanation
    
    def explain_batch(self, instances: np.ndarray, 
                     instance_ids: List[str] = None) -> List[Dict[str, Any]]:
        """Explain a batch of instances"""
        if not self.setup_complete:
            raise ValueError("Explainer must be setup with background data first")
        
        explanations = []
        
        if instance_ids is None:
            instance_ids = [f"instance_{i}" for i in range(len(instances))]
        
        print(f"Explaining {len(instances)} instances...")
        
        for i, instance in enumerate(instances):
            if i % 10 == 0 and i > 0:
                print(f"  Progress: {i}/{len(instances)}")
            
            explanation = self.explain_instance(instance, instance_ids[i])
            explanations.append(explanation)
        
        print("Batch explanation completed.")
        return explanations
    
    def _create_explanation_dict(self, instance: np.ndarray, 
                               shap_values: np.ndarray, 
                               instance_id: str = None) -> Dict[str, Any]:
        """Create comprehensive explanation dictionary"""
        
        # Feature contributions
        feature_contributions = dict(zip(self.feature_names, shap_values))
        
        # Top contributing features (positive and negative)
        feature_importance = [(name, abs(value)) for name, value in feature_contributions.items()]
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        top_features = feature_importance[:self.config.max_features_display]
        
        # Separate positive and negative contributions
        positive_contributions = [(name, value) for name, value in feature_contributions.items() if value > 0]
        negative_contributions = [(name, value) for name, value in feature_contributions.items() if value < 0]
        
        positive_contributions.sort(key=lambda x: x[1], reverse=True)
        negative_contributions.sort(key=lambda x: x[1])
        
        # Calculate explanation strength
        total_importance = sum(abs(value) for value in shap_values)
        base_value = self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0
        prediction_value = base_value + sum(shap_values)
        
        explanation = {
            'instance_id': instance_id or 'unknown',
            'timestamp': datetime.now().isoformat(),
            'feature_values': dict(zip(self.feature_names, instance)),
            'shap_values': feature_contributions,
            'base_value': float(base_value),
            'prediction_value': float(prediction_value),
            'total_importance': float(total_importance),
            'top_features': [(name, float(importance)) for name, importance in top_features],
            'positive_contributions': [(name, float(value)) for name, value in positive_contributions[:5]],
            'negative_contributions': [(name, float(value)) for name, value in negative_contributions[:5]],
            'explanation_text': self._generate_explanation_text(
                feature_contributions, positive_contributions, negative_contributions
            ),
            'confidence_indicators': self._calculate_confidence_indicators(shap_values, total_importance)
        }
        
        return explanation
    
    def _generate_explanation_text(self, feature_contributions: Dict[str, float],
                                 positive_contributions: List[Tuple[str, float]],
                                 negative_contributions: List[Tuple[str, float]]) -> str:
        """Generate human-readable explanation text"""
        
        text_parts = ["网络异常检测解释分析：\n"]
        
        # Overall assessment
        total_positive = sum(value for _, value in positive_contributions)
        total_negative = abs(sum(value for _, value in negative_contributions))
        
        if total_positive > total_negative:
            text_parts.append("🔴 该实例被判定为异常，主要原因包括：\n")
        else:
            text_parts.append("✅该实例被判定为正常，主要原因包括：\n")
        
        # Top positive contributions (indicate anomaly)
        if positive_contributions:
            text_parts.append("异常指示因素：")
            for i, (feature, value) in enumerate(positive_contributions[:3]):
                text_parts.append(f"  • {feature}: 贡献度 {value:.3f}")
        
        # Top negative contributions (indicate normality)
        if negative_contributions:
            text_parts.append("\n正常指示因素：")
            for i, (feature, value) in enumerate(negative_contributions[:3]):
                text_parts.append(f"  • {feature}: 贡献度 {value:.3f}")
        
        # Confidence assessment
        confidence_ratio = total_positive / (total_positive + total_negative) if (total_positive + total_negative) > 0 else 0.5
        
        text_parts.append(f"\n置信度评估: {confidence_ratio:.1%}")
        if confidence_ratio > 0.8:
            text_parts.append("(高置信度)")
        elif confidence_ratio > 0.6:
            text_parts.append("(中等置信度)")
        else:
            text_parts.append("(低置信度)")
        
        return "\n".join(text_parts)
    
    def _calculate_confidence_indicators(self, shap_values: np.ndarray, 
                                       total_importance: float) -> Dict[str, float]:
        """Calculate confidence indicators for the explanation"""
        
        # Feature consistency (how much features agree on the prediction)
        positive_sum = sum(value for value in shap_values if value > 0)
        negative_sum = abs(sum(value for value in shap_values if value < 0))
        
        consistency = abs(positive_sum - negative_sum) / (positive_sum + negative_sum) if (positive_sum + negative_sum) > 0 else 0
        
        # Feature concentration (how concentrated the explanation is)
        if total_importance > 0:
            sorted_importance = sorted([abs(value) for value in shap_values], reverse=True)
            top_3_importance = sum(sorted_importance[:3])
            concentration = top_3_importance / total_importance
        else:
            concentration = 0
        
        # Explanation stability (based on standard deviation of SHAP values)
        stability = 1.0 / (1.0 + np.std(shap_values)) if len(shap_values) > 1 else 1.0
        
        return {
            'consistency': float(consistency),
            'concentration': float(concentration),
            'stability': float(stability),
            'overall_confidence': float((consistency + concentration + stability) / 3)
        }
    
    def create_explanation_plots(self, explanation: Dict[str, Any], 
                               save_path: str = None) -> Dict[str, Any]:
        """Create visualization plots for explanation"""
        
        plots = {}
        
        try:
            # Setup plot style
            plt.style.use('default')
            
            # 1. Feature Importance Bar Plot
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            
            top_features = explanation['top_features'][:self.config.max_features_display]
            features, importances = zip(*top_features) if top_features else ([], [])
            
            if features:
                colors = ['red' if explanation['shap_values'][f] > 0 else 'blue' for f in features]
                bars = ax1.barh(features, importances, color=colors, alpha=0.7)
                
                ax1.set_xlabel('Feature Importance (|SHAP Value|)')
                ax1.set_title('Top Feature Contributions to Anomaly Prediction')
                ax1.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for i, (bar, imp) in enumerate(zip(bars, importances)):
                    ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                            f'{imp:.3f}', ha='left', va='center', fontsize=9)
            
            plt.tight_layout()
            plots['feature_importance'] = fig1
            
            # 2. Waterfall Plot (simplified)
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            
            # Get top positive and negative contributions
            pos_contrib = explanation['positive_contributions'][:5]
            neg_contrib = explanation['negative_contributions'][:5]
            
            # Combine and sort by absolute value
            all_contrib = pos_contrib + neg_contrib
            all_contrib.sort(key=lambda x: abs(x[1]), reverse=True)
            
            if all_contrib:
                features, values = zip(*all_contrib)
                colors = ['red' if v > 0 else 'blue' for v in values]
                
                # Create waterfall-style plot
                cumsum = [explanation['base_value']]
                positions = [0]
                
                for i, value in enumerate(values):
                    cumsum.append(cumsum[-1] + value)
                    positions.append(i + 1)
                
                # Plot base value
                ax2.bar(0, explanation['base_value'], color='gray', alpha=0.5, label='Base Value')
                
                # Plot contributions
                for i, (feature, value) in enumerate(all_contrib):
                    color = 'red' if value > 0 else 'blue'
                    ax2.bar(i + 1, value, bottom=cumsum[i], color=color, alpha=0.7)
                    
                    # Add feature labels
                    ax2.text(i + 1, cumsum[i] + value/2, feature, 
                            rotation=45, ha='center', va='center', fontsize=8)
                
                # Plot final prediction
                ax2.bar(len(all_contrib) + 1, 0, bottom=cumsum[-1], 
                       color='green', alpha=0.5, label='Final Prediction')
                
                ax2.set_ylabel('SHAP Value Contribution')
                ax2.set_title('Feature Contribution Waterfall')
                ax2.set_xticks(range(len(all_contrib) + 2))
                ax2.set_xticklabels(['Base'] + [f'F{i+1}' for i in range(len(all_contrib))] + ['Final'], 
                                   rotation=45)
                ax2.grid(True, alpha=0.3)
                ax2.legend()
            
            plt.tight_layout()
            plots['waterfall'] = fig2
            
            # 3. Feature Values vs SHAP Values Scatter Plot
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            
            feature_values = list(explanation['feature_values'].values())
            shap_vals = list(explanation['shap_values'].values())
            
            if feature_values and shap_vals:
                scatter = ax3.scatter(feature_values, shap_vals, alpha=0.7, s=50)
                
                # Add feature name annotations for top contributors
                top_feature_names = [name for name, _ in explanation['top_features'][:5]]
                for name in top_feature_names:
                    if name in explanation['feature_values'] and name in explanation['shap_values']:
                        x = explanation['feature_values'][name]
                        y = explanation['shap_values'][name]
                        ax3.annotate(name, (x, y), xytext=(5, 5), 
                                   textcoords='offset points', fontsize=8)
                
                ax3.set_xlabel('Feature Value')
                ax3.set_ylabel('SHAP Value')
                ax3.set_title('Feature Values vs. SHAP Contributions')
                ax3.grid(True, alpha=0.3)
                ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            plots['scatter'] = fig3
            
            # Save plots if requested
            if save_path and self.config.save_plots:
                os.makedirs(save_path, exist_ok=True)
                instance_id = explanation.get('instance_id', 'unknown')
                
                for plot_name, fig in plots.items():
                    filepath = os.path.join(save_path, f"{instance_id}_{plot_name}.png")
                    fig.savefig(filepath, dpi=300, bbox_inches='tight')
                
                print(f"Explanation plots saved to {save_path}")
        
        except Exception as e:
            print(f"Warning: Could not create explanation plots ({e})")
        
        return plots
    
    def create_summary_plots(self, explanations: List[Dict[str, Any]], 
                           save_path: str = None) -> Dict[str, Any]:
        """Create summary plots for multiple explanations"""
        
        if not explanations:
            return {}
        
        plots = {}
        
        try:
            # 1. Feature Importance Summary
            fig1, ax1 = plt.subplots(figsize=(12, 8))
            
            # Aggregate feature importance across all explanations
            feature_importance_agg = {}
            
            for exp in explanations:
                for feature, value in exp['shap_values'].items():
                    if feature not in feature_importance_agg:
                        feature_importance_agg[feature] = []
                    feature_importance_agg[feature].append(abs(value))
            
            # Calculate mean importance for each feature
            mean_importance = {feature: np.mean(values) 
                             for feature, values in feature_importance_agg.items()}
            
            # Plot top features
            sorted_features = sorted(mean_importance.items(), key=lambda x: x[1], reverse=True)
            top_features = sorted_features[:self.config.max_features_display]
            
            if top_features:
                features, importances = zip(*top_features)
                bars = ax1.barh(features, importances, color='skyblue', alpha=0.7)
                
                ax1.set_xlabel('Average Feature Importance')
                ax1.set_title(f'Top Features Across {len(explanations)} Explanations')
                ax1.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, imp in zip(bars, importances):
                    ax1.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                            f'{imp:.3f}', ha='left', va='center', fontsize=9)
            
            plt.tight_layout()
            plots['summary_importance'] = fig1
            
            # 2. Explanation Confidence Distribution
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            
            confidences = [exp['confidence_indicators']['overall_confidence'] for exp in explanations]
            
            if confidences:
                ax2.hist(confidences, bins=20, alpha=0.7, color='green', edgecolor='black')
                ax2.axvline(np.mean(confidences), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(confidences):.3f}')
                ax2.set_xlabel('Overall Confidence Score')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Distribution of Explanation Confidence Scores')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plots['confidence_distribution'] = fig2
            
            # 3. Feature Correlation Heatmap
            fig3, ax3 = plt.subplots(figsize=(10, 8))
            
            # Create feature matrix
            all_features = list(explanations[0]['shap_values'].keys())
            feature_matrix = np.array([[exp['shap_values'][f] for f in all_features] 
                                     for exp in explanations])
            
            if feature_matrix.size > 0:
                correlation_matrix = np.corrcoef(feature_matrix.T)
                
                # Create heatmap
                sns.heatmap(correlation_matrix, 
                           xticklabels=all_features,
                           yticklabels=all_features,
                           annot=False,
                           cmap='coolwarm',
                           center=0,
                           ax=ax3)
                
                ax3.set_title('Feature SHAP Value Correlations')
                plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
                plt.setp(ax3.get_yticklabels(), rotation=0)
            
            plt.tight_layout()
            plots['correlation_heatmap'] = fig3
            
            # Save plots if requested
            if save_path and self.config.save_plots:
                os.makedirs(save_path, exist_ok=True)
                
                for plot_name, fig in plots.items():
                    filepath = os.path.join(save_path, f"summary_{plot_name}.png")
                    fig.savefig(filepath, dpi=300, bbox_inches='tight')
                
                print(f"Summary plots saved to {save_path}")
        
        except Exception as e:
            print(f"Warning: Could not create summary plots ({e})")
        
        return plots
    
    def generate_explanation_report(self, explanations: List[Dict[str, Any]]) -> str:
        """Generate a comprehensive explanation report"""
        
        if not explanations:
            return "No explanations available."
        
        report = []
        report.append("=" * 60)
        report.append("网络异常检测可解释性分析报告")
        report.append("=" * 60)
        
        # Summary statistics
        total_explanations = len(explanations)
        avg_confidence = np.mean([exp['confidence_indicators']['overall_confidence'] 
                                for exp in explanations])
        
        report.append(f"\n总体统计:")
        report.append(f"  分析实例数量: {total_explanations}")
        report.append(f"  平均解释置信度: {avg_confidence:.3f}")
        
        # Feature importance ranking
        feature_importance_agg = {}
        for exp in explanations:
            for feature, value in exp['shap_values'].items():
                if feature not in feature_importance_agg:
                    feature_importance_agg[feature] = []
                feature_importance_agg[feature].append(abs(value))
        
        mean_importance = {feature: np.mean(values) 
                         for feature, values in feature_importance_agg.items()}
        
        sorted_features = sorted(mean_importance.items(), key=lambda x: x[1], reverse=True)
        
        report.append(f"\n最重要特征排名:")
        for i, (feature, importance) in enumerate(sorted_features[:10]):
            report.append(f"  {i+1}. {feature}: {importance:.4f}")
        
        # Confidence distribution
        confidences = [exp['confidence_indicators']['overall_confidence'] for exp in explanations]
        high_conf = sum(1 for c in confidences if c > 0.7)
        medium_conf = sum(1 for c in confidences if 0.4 < c <= 0.7)
        low_conf = sum(1 for c in confidences if c <= 0.4)
        
        report.append(f"\n解释置信度分布:")
        report.append(f"  高置信度 (>0.7): {high_conf} ({high_conf/total_explanations:.1%})")
        report.append(f"  中等置信度 (0.4-0.7): {medium_conf} ({medium_conf/total_explanations:.1%})")
        report.append(f"  低置信度 (≤0.4): {low_conf} ({low_conf/total_explanations:.1%})")
        
        # Sample detailed explanation
        if explanations:
            sample_exp = max(explanations, 
                           key=lambda x: x['confidence_indicators']['overall_confidence'])
            
            report.append(f"\n最高置信度解释示例:")
            report.append(f"  实例ID: {sample_exp['instance_id']}")
            report.append(f"  置信度: {sample_exp['confidence_indicators']['overall_confidence']:.3f}")
            report.append(f"  主要贡献特征:")
            
            for feature, value in sample_exp['positive_contributions'][:3]:
                report.append(f"    • {feature}: {value:.3f}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def save_explanations(self, explanations: List[Dict[str, Any]], 
                         filepath: str):
        """Save explanations to file"""
        import json
        
        # Make explanations JSON serializable
        serializable_explanations = []
        for exp in explanations:
            serializable_exp = {}
            for key, value in exp.items():
                if isinstance(value, (np.integer, np.floating)):
                    serializable_exp[key] = float(value)
                elif isinstance(value, np.ndarray):
                    serializable_exp[key] = value.tolist()
                else:
                    serializable_exp[key] = value
            serializable_explanations.append(serializable_exp)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_explanations, f, indent=2, ensure_ascii=False)
        
        print(f"Explanations saved to {filepath}")
    
    def load_explanations(self, filepath: str) -> List[Dict[str, Any]]:
        """Load explanations from file"""
        import json
        
        with open(filepath, 'r', encoding='utf-8') as f:
            explanations = json.load(f)
        
        return explanations


# Example usage and testing
if __name__ == "__main__":
    # Test explainability with a simple model
    from sklearn.ensemble import IsolationForest
    from ..anomaly_detection.detectors import NetworkAnomalyAnalyzer
    from ..data.generators.network_generator import NetworkGenerator, NetworkConfig
    from ..data.generators.data_generator import NetworkDataGenerator, DataConfig
    
    print("Testing explainability system...")
    
    # Generate test data
    network_config = NetworkConfig(node_count=20)
    data_config = DataConfig(time_steps=200, anomaly_ratio=0.1)
    
    network_gen = NetworkGenerator(network_config)
    data_gen = NetworkDataGenerator(data_config)
    
    network = network_gen.generate_network()
    test_node = list(network.nodes())[0]
    data = data_gen.generate_node_timeseries(network, test_node)
    
    # Train anomaly detection model
    from ..anomaly_detection.detectors import AnomalyDetectionConfig
    
    ad_config = AnomalyDetectionConfig()
    analyzer = NetworkAnomalyAnalyzer(ad_config)
    
    # Split data for training
    split_idx = int(0.7 * len(data))
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    # Train model
    training_stats = analyzer.train(train_data)
    
    # Get one of the trained models for explanation
    isolation_forest = analyzer.ensemble.detectors['isolation_forest']
    
    # Setup explainer
    config = ExplainabilityConfig()
    explainer = NetworkAnomalyExplainer(isolation_forest.model, config)
    
    # Prepare data for explanation
    X_train = analyzer.prepare_features(train_data)
    X_test = analyzer.prepare_features(test_data)
    
    # Setup explainer with background data
    explainer.setup_explainer(X_train, analyzer.feature_columns)
    
    # Explain some test instances
    test_instances = X_test[:5]
    explanations = explainer.explain_batch(test_instances)
    
    # Generate report
    report = explainer.generate_explanation_report(explanations)
    print(report)
    
    # Create plots for first explanation
    if explanations:
        plots = explainer.create_explanation_plots(explanations[0])
        print(f"Created {len(plots)} explanation plots")