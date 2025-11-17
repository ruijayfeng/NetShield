"""
Main entry point for the Network Anomaly Detection and Cascading Failure Analysis System.
"""

import sys
import os
import argparse
import asyncio
from datetime import datetime
import logging
import networkx as nx

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.generators.network_generator import NetworkGenerator, NetworkConfig
from src.data.generators.data_generator import NetworkDataGenerator, DataConfig
from src.models.anomaly_detection.detectors import NetworkAnomalyAnalyzer, AnomalyDetectionConfig
from src.models.cascading.failure_analyzer import CascadingFailureAnalyzer, CascadingFailureConfig
from src.models.explainable.shap_explainer import NetworkAnomalyExplainer, ExplainabilityConfig
from src.alerts.alert_manager import AlertManager, AlertLevel, AlertCategory


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('fenxi.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


class NetworkAnalysisSystem:
    """Main system controller"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or "config/config.yaml"
        
        # Initialize components
        self.network_generator = None
        self.data_generator = None
        self.anomaly_analyzer = None
        self.cascade_analyzer = None
        self.explainer = None
        self.alert_manager = None
        
        self.network = None
        self.data = None
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Network Analysis System initialized")
    
    def initialize_system(self):
        """Initialize all system components"""
        self.logger.info("Initializing system components...")
        
        try:
            # Load configurations
            network_config = NetworkConfig.from_config(self.config_path)
            data_config = DataConfig.from_config(self.config_path)
            ad_config = AnomalyDetectionConfig.from_config(self.config_path)
            cf_config = CascadingFailureConfig.from_config(self.config_path)
            exp_config = ExplainabilityConfig.from_config(self.config_path)
            
            # Initialize generators
            self.network_generator = NetworkGenerator(network_config)
            self.data_generator = NetworkDataGenerator(data_config)
            
            # Initialize analyzers
            self.anomaly_analyzer = NetworkAnomalyAnalyzer(ad_config)
            self.cascade_analyzer = CascadingFailureAnalyzer(cf_config)
            
            # Initialize alert manager
            self.alert_manager = AlertManager()
            
            self.logger.info("System components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize system: {e}")
            raise
    
    def generate_network_and_data(self, network_file=None, data_file=None):
        """Generate network topology and monitoring data or load from files"""
        if network_file or data_file:
            self.logger.info("Loading real network and monitoring data...")
        else:
            self.logger.info("Generating network topology and data...")
        
        try:
            # Load or generate network
            if network_file:
                from src.data.generators.network_generator import RealNetworkInterface
                interface = RealNetworkInterface()
                self.network = interface.load_from_file(network_file)
                self.logger.info(f"Network loaded from {network_file}: {self.network.number_of_nodes()} nodes, "
                               f"{self.network.number_of_edges()} edges")
            else:
                # Generate network
                self.network = self.network_generator.generate_network()
                self.logger.info(f"Network generated: {self.network.number_of_nodes()} nodes, "
                               f"{self.network.number_of_edges()} edges")
            
            # Load or generate monitoring data
            if data_file:
                self.data = self.data_generator.load_real_data(data_file)
                self.logger.info(f"Monitoring data loaded from {data_file}: {len(self.data)} time points")
            else:
                # Generate monitoring data for a sample node
                sample_node = list(self.network.nodes())[0]
                self.data = self.data_generator.generate_node_timeseries(self.network, sample_node)
                self.logger.info(f"Monitoring data generated: {len(self.data)} time points")
            
            # Validate data quality
            validation = self.data_generator.validate_data_quality(self.data)
            if not validation['is_valid']:
                self.logger.warning(f"Data quality issues: {validation['issues']}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate/load network and data: {e}")
            raise
    
    def train_anomaly_detection(self):
        """Train anomaly detection models"""
        self.logger.info("Training anomaly detection models...")
        
        try:
            if self.data is None:
                raise ValueError("No data available for training")
            
            # Split data for training
            split_idx = int(0.7 * len(self.data))
            train_data = self.data[:split_idx].copy()
            
            # Train models
            training_stats = self.anomaly_analyzer.train(train_data)
            self.logger.info(f"Anomaly detection models trained. Features: {len(self.anomaly_analyzer.feature_columns)}")
            
            # Setup explainer
            if self.anomaly_analyzer.ensemble.detectors:
                isolation_forest = self.anomaly_analyzer.ensemble.detectors['isolation_forest']
                self.explainer = NetworkAnomalyExplainer(isolation_forest.model, ExplainabilityConfig())
                
                # Setup with background data
                X_train = self.anomaly_analyzer.prepare_features(train_data)
                self.explainer.setup_explainer(X_train, self.anomaly_analyzer.feature_columns)
                self.logger.info("Explainer initialized successfully")
            
            return training_stats
            
        except Exception as e:
            self.logger.error(f"Failed to train anomaly detection: {e}")
            raise
    
    async def perform_anomaly_detection(self):
        """Perform anomaly detection on test data"""
        self.logger.info("Performing anomaly detection...")
        
        try:
            if self.data is None or self.anomaly_analyzer is None:
                raise ValueError("Data or analyzer not available")
            
            # Use latter part of data for testing
            test_data = self.data.tail(int(len(self.data) * 0.3)).copy()
            
            # Detect anomalies
            detection_results = self.anomaly_analyzer.detect_anomalies(test_data)
            
            # Log results
            summary = detection_results['summary']
            self.logger.info(f"Anomaly detection completed. "
                           f"Detected {summary['predicted_anomalies']} anomalies "
                           f"out of {summary['total_samples']} samples "
                           f"({summary['anomaly_rate']:.1%} rate)")
            
            # Create alerts for detected anomalies
            await self._create_anomaly_alerts(detection_results)
            
            return detection_results
            
        except Exception as e:
            self.logger.error(f"Failed to perform anomaly detection: {e}")
            raise
    
    async def perform_cascading_failure_analysis(self):
        """Perform cascading failure analysis"""
        self.logger.info("Performing cascading failure analysis...")
        
        try:
            if self.network is None or self.cascade_analyzer is None:
                raise ValueError("Network or analyzer not available")
            
            # Analyze network robustness
            analysis_results = self.cascade_analyzer.analyze_network_robustness(self.network)
            
            # Log results
            robustness_score = analysis_results.get('robustness_metrics', {}).get('overall_robustness_score', 0)
            self.logger.info(f"Cascading failure analysis completed. "
                           f"Network robustness score: {robustness_score:.3f}")
            
            # Create alerts if network is vulnerable
            await self._create_cascading_failure_alerts(analysis_results)
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Failed to perform cascading failure analysis: {e}")
            raise
    
    def generate_explanations(self, num_instances: int = 5):
        """Generate explanations for anomaly predictions"""
        self.logger.info(f"Generating explanations for {num_instances} instances...")
        
        try:
            if (self.data is None or 
                self.anomaly_analyzer is None or 
                self.explainer is None):
                raise ValueError("Required components not available")
            
            # Get test instances
            test_data = self.data.tail(100)
            sample_instances = test_data.sample(num_instances)
            
            # Prepare features
            X_instances = self.anomaly_analyzer.prepare_features(
                sample_instances, self.anomaly_analyzer.feature_columns
            )
            
            # Generate explanations
            explanations = self.explainer.explain_batch(
                X_instances,
                [f"instance_{i}" for i in range(num_instances)]
            )
            
            self.logger.info(f"Generated {len(explanations)} explanations")
            
            # Generate report
            report = self.explainer.generate_explanation_report(explanations)
            self.logger.info("Explanation report generated")
            
            return explanations, report
            
        except Exception as e:
            self.logger.error(f"Failed to generate explanations: {e}")
            raise
    
    async def _create_anomaly_alerts(self, detection_results):
        """Create alerts for anomaly detection results"""
        summary = detection_results['summary']
        
        if summary['predicted_anomalies'] > 0:
            await self.alert_manager.evaluate_rules({
                'is_anomaly': True,
                'confidence': summary['mean_anomaly_probability'],
                'node_id': 'sample_node',
                'predicted_count': summary['predicted_anomalies'],
                'total_samples': summary['total_samples']
            })
    
    async def _create_cascading_failure_alerts(self, analysis_results):
        """Create alerts for cascading failure analysis results"""
        robustness_metrics = analysis_results.get('robustness_metrics', {})
        robustness_score = robustness_metrics.get('overall_robustness_score', 1.0)
        
        # Create alert if robustness is low
        if robustness_score < 0.7:
            await self.alert_manager.create_alert(
                title="网络鲁棒性警告",
                description=f"网络鲁棒性评分较低: {robustness_score:.3f}",
                level=AlertLevel.WARNING if robustness_score > 0.5 else AlertLevel.CRITICAL,
                category=AlertCategory.CASCADING_FAILURE,
                source="cascading_failure_analyzer",
                metadata={'robustness_score': robustness_score}
            )
    
    def generate_comprehensive_report(self):
        """Generate comprehensive system analysis report"""
        self.logger.info("Generating comprehensive report...")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("复杂网络异常行为检测与级联失效分析系统 - 综合报告")
        report_lines.append("=" * 80)
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Network information
        if self.network:
            report_lines.append("网络拓扑信息:")
            report_lines.append(f"  节点数量: {self.network.number_of_nodes()}")
            report_lines.append(f"  边数量: {self.network.number_of_edges()}")
            report_lines.append(f"  网络密度: {nx.density(self.network):.4f}")
            report_lines.append(f"  连通性: {'是' if nx.is_connected(self.network) else '否'}")
            report_lines.append("")
        
        # Data information
        if self.data is not None:
            report_lines.append("监控数据信息:")
            report_lines.append(f"  数据点数量: {len(self.data)}")
            report_lines.append(f"  特征数量: {len([col for col in self.data.columns if col not in ['timestamp', 'is_anomaly', 'anomaly_score']])}")
            report_lines.append(f"  真实异常率: {self.data['is_anomaly'].mean():.3f}")
            report_lines.append("")
        
        # Alert statistics
        if self.alert_manager:
            stats = self.alert_manager.get_alert_statistics()
            report_lines.append("告警统计:")
            report_lines.append(f"  总告警数: {stats['total_alerts']}")
            report_lines.append(f"  活跃告警数: {stats['active_alerts']}")
            report_lines.append(f"  已解决告警数: {stats['resolved_alerts']}")
            
            if stats['by_level']:
                report_lines.append("  按级别分布:")
                for level, count in stats['by_level'].items():
                    report_lines.append(f"    {level}: {count}")
            report_lines.append("")
        
        # System status
        report_lines.append("系统组件状态:")
        report_lines.append(f"  网络生成器: {'OK' if self.network_generator else 'FAIL'}")
        report_lines.append(f"  数据生成器: {'OK' if self.data_generator else 'FAIL'}")
        report_lines.append(f"  异常检测器: {'OK' if self.anomaly_analyzer else 'FAIL'}")
        report_lines.append(f"  级联失效分析器: {'OK' if self.cascade_analyzer else 'FAIL'}")
        report_lines.append(f"  可解释性分析器: {'OK' if self.explainer else 'FAIL'}")
        report_lines.append(f"  告警管理器: {'OK' if self.alert_manager else 'FAIL'}")
        
        report_lines.append("=" * 80)
        
        report = "\n".join(report_lines)
        self.logger.info("Comprehensive report generated")
        
        return report
    
    def save_results(self, output_dir: str = "output"):
        """Save analysis results to files"""
        self.logger.info(f"Saving results to {output_dir}...")
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save comprehensive report
            report = self.generate_comprehensive_report()
            with open(os.path.join(output_dir, "comprehensive_report.txt"), "w", encoding="utf-8") as f:
                f.write(report)
            
            # Save network
            if self.network:
                import networkx as nx
                nx.write_gml(self.network, os.path.join(output_dir, "network.gml"))
            
            # Save data
            if self.data is not None:
                self.data.to_csv(os.path.join(output_dir, "monitoring_data.csv"), index=False)
            
            # Export alerts
            if self.alert_manager:
                self.alert_manager.export_alerts(os.path.join(output_dir, "alerts.json"))
            
            self.logger.info("Results saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            raise


async def run_full_analysis(config_path: str = None, output_dir: str = "output", 
                           network_file: str = None, data_file: str = None):
    """Run complete analysis pipeline"""
    system = NetworkAnalysisSystem(config_path)
    
    try:
        # Initialize system
        system.initialize_system()
        
        # Generate or load data
        system.generate_network_and_data(network_file, data_file)
        
        # Train models
        system.train_anomaly_detection()
        
        # Perform analyses
        anomaly_results = await system.perform_anomaly_detection()
        cascade_results = system.perform_cascading_failure_analysis()
        
        # Generate explanations
        explanations, exp_report = system.generate_explanations()
        
        # Generate and save results
        report = system.generate_comprehensive_report()
        print(report)
        
        system.save_results(output_dir)
        
        return system
        
    except Exception as e:
        logging.error(f"Analysis pipeline failed: {e}")
        raise


def main():
    """Main entry point with command line interface"""
    parser = argparse.ArgumentParser(
        description="复杂网络异常行为检测与级联失效分析系统"
    )
    
    parser.add_argument(
        "--mode",
        choices=["dashboard", "analysis", "test"],
        default="dashboard",
        help="运行模式: dashboard(仪表板), analysis(批量分析), test(测试)"
    )
    
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="配置文件路径"
    )
    
    parser.add_argument(
        "--output",
        default="output",
        help="输出目录"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别"
    )
    
    parser.add_argument(
        "--network-file",
        help="真实网络拓扑文件路径 (支持CSV, JSON, GML, GraphML格式)"
    )
    
    parser.add_argument(
        "--data-file", 
        help="真实监控数据文件路径 (支持CSV, JSON, Parquet格式)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    if args.mode == "dashboard":
        print("启动Streamlit仪表板...")
        print("请运行: streamlit run src/visualization/dashboard.py")
        
    elif args.mode == "analysis":
        print("开始完整分析...")
        if args.network_file or args.data_file:
            print(f"使用真实数据: 网络文件={args.network_file}, 数据文件={args.data_file}")
        system = asyncio.run(run_full_analysis(args.config, args.output, args.network_file, args.data_file))
        print(f"分析完成，结果保存在 {args.output} 目录")
        
    elif args.mode == "test":
        print("运行系统测试...")
        system = NetworkAnalysisSystem(args.config)
        system.initialize_system()
        system.generate_network_and_data()
        
        print("Network generation successful")
        print("Data generation successful")
        print("System test completed")


if __name__ == "__main__":
    main()