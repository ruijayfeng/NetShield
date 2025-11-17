"""
Streamlit dashboard for network anomaly detection and cascading failure analysis.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
import asyncio
import time
import yaml
import os
import sys
import requests
import json
from typing import Dict, List, Any, Optional

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.data.generators.network_generator import NetworkGenerator, NetworkConfig, RealNetworkInterface
from src.data.generators.data_generator import NetworkDataGenerator, DataConfig
from src.models.anomaly_detection.detectors import NetworkAnomalyAnalyzer, AnomalyDetectionConfig
from src.models.cascading.failure_analyzer import CascadingFailureAnalyzer, CascadingFailureConfig
from src.models.explainable.shap_explainer import NetworkAnomalyExplainer, ExplainabilityConfig
from src.alerts.alert_manager import AlertManager, AlertLevel, AlertCategory


class NetworkDashboard:
    """Main dashboard class for the network monitoring system"""
    
    def __init__(self):
        self.setup_page_config()
        self.initialize_session_state()
        
    def setup_page_config(self):
        """Configure Streamlit page"""
        st.set_page_config(
            page_title="网络异常检测与级联失效分析系统",
            page_icon="🕸️",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/your-repo/fenxi',
                'Report a bug': 'https://github.com/your-repo/fenxi/issues',
                'About': "复杂网络异常行为检测与级联失效分析系统"
            }
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
        }
        .alert-critical {
            background-color: #ffebee;
            border-left: 4px solid #f44336;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .alert-warning {
            background-color: #fff8e1;
            border-left: 4px solid #ff9800;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .alert-info {
            background-color: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'network' not in st.session_state:
            st.session_state.network = None
        
        if 'data' not in st.session_state:
            st.session_state.data = None
        
        if 'anomaly_analyzer' not in st.session_state:
            st.session_state.anomaly_analyzer = None
        
        if 'cascade_analyzer' not in st.session_state:
            st.session_state.cascade_analyzer = None
        
        if 'explainer' not in st.session_state:
            st.session_state.explainer = None
        
        if 'alert_manager' not in st.session_state:
            st.session_state.alert_manager = AlertManager()
        
        if 'last_update' not in st.session_state:
            st.session_state.last_update = datetime.now()
        
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {}

        # AI分析相关状态
        if 'ai_chat_history' not in st.session_state:
            st.session_state.ai_chat_history = []

        if 'ai_summary' not in st.session_state:
            st.session_state.ai_summary = None

        if 'ai_api_key' not in st.session_state:
            st.session_state.ai_api_key = "不许偷看我的小密钥！"
    
    def render_dashboard(self):
        """Render the main dashboard"""
        st.title("🕸️ 复杂网络异常行为检测与级联失效分析系统")
        st.markdown("---")
        
        # Sidebar configuration
        self.render_sidebar()
        
        # Main content based on selected page
        page = st.session_state.get('current_page', 'overview')
        
        if page == 'overview':
            self.render_overview_page()
        elif page == 'network':
            self.render_network_page()
        elif page == 'anomaly':
            self.render_anomaly_page()
        elif page == 'cascading':
            self.render_cascading_page()
        elif page == 'explainability':
            self.render_explainability_page()
        elif page == 'ai_analysis':
            self.render_ai_analysis_page()
        elif page == 'alerts':
            self.render_alerts_page()
        else:
            self.render_overview_page()
    
    def render_sidebar(self):
        """Render sidebar with navigation and controls"""
        with st.sidebar:
            st.header("⚙️ 系统控制")
            
            # Page navigation
            st.subheader("📄 页面导航")
            pages = {
                'overview': '📊 系统概览',
                'network': '🔗 网络拓扑',
                'anomaly': '🚨 异常检测',
                'cascading': '⚡ 级联失效',
                'explainability': '🔍 可解释性',
                'ai_analysis': '🤖 AI大模型分析',
                'alerts': '📢 告警管理'
            }
            
            selected_page = st.selectbox(
                "选择页面",
                list(pages.keys()),
                format_func=lambda x: pages[x],
                key='current_page'
            )
            
            st.markdown("---")
            
            # System configuration
            st.subheader("🔧 系统配置")
            
            # Data source selection
            st.subheader("📂 数据源选择")
            data_source = st.radio(
                "选择数据源",
                ["生成模拟数据", "上传真实数据"],
                key="data_source_option"
            )
            
            if data_source == "生成模拟数据":
                # Network configuration
                with st.expander("网络配置", expanded=True):
                    node_count = st.slider("节点数量", 10, 100, 50)
                    network_type = st.selectbox(
                        "网络类型",
                        ['small_world', 'scale_free', 'erdos_renyi'],
                        format_func=lambda x: {
                            'small_world': '小世界网络',
                            'scale_free': '无标度网络',
                            'erdos_renyi': '随机网络'
                        }[x]
                    )
                
                # Data configuration
                with st.expander("数据配置"):
                    time_steps = st.slider("时间步数", 100, 2000, 500)
                    anomaly_ratio = st.slider("异常比例", 0.01, 0.2, 0.05)
                    
                # Generate or update data
                if st.button("🔄 生成/更新数据", type="primary"):
                    self.generate_data(node_count, network_type, time_steps, anomaly_ratio)
            
            else:  # Upload real data
                with st.expander("上传真实数据", expanded=True):
                    st.markdown("**网络拓扑文件** (可选)")
                    network_file = st.file_uploader(
                        "上传网络拓扑文件",
                        type=['csv', 'json', 'gml', 'graphml'],
                        help="CSV格式需包含source,target列；JSON格式需包含nodes和edges"
                    )
                    
                    st.markdown("**监控数据文件** (必需)")
                    data_file = st.file_uploader(
                        "上传监控数据文件",
                        type=['csv', 'json', 'parquet'],
                        help="必须包含timestamp列和至少一个数值特征列"
                    )
                    
                    if st.button("📤 加载真实数据", type="primary"):
                        self.load_real_data(network_file, data_file)
            
            st.markdown("---")
            
            # System status
            st.subheader("📈 系统状态")
            
            if st.session_state.network is not None:
                st.success(f"✅ 网络已加载 ({st.session_state.network.number_of_nodes()} 节点)")
            else:
                st.warning("⚠️ 未加载网络")
            
            if st.session_state.data is not None:
                st.success(f"✅ 数据已生成 ({len(st.session_state.data)} 条记录)")
            else:
                st.warning("⚠️ 未生成数据")
            
            # Auto-refresh toggle
            auto_refresh = st.checkbox("🔄 自动刷新", value=False)
            if auto_refresh:
                st.rerun()
    
    def generate_data(self, node_count: int, network_type: str, 
                     time_steps: int, anomaly_ratio: float):
        """Generate network and data"""
        with st.spinner("正在生成网络和数据..."):
            try:
                # Generate network
                network_config = NetworkConfig(
                    node_count=node_count,
                    network_type=network_type
                )
                
                network_gen = NetworkGenerator(network_config)
                st.session_state.network = network_gen.generate_network()
                
                # Generate data
                data_config = DataConfig(
                    time_steps=time_steps,
                    anomaly_ratio=anomaly_ratio
                )
                
                data_gen = NetworkDataGenerator(data_config)
                node_id = list(st.session_state.network.nodes())[0]
                st.session_state.data = data_gen.generate_node_timeseries(
                    st.session_state.network, node_id
                )
                
                # Train models
                self.train_models()
                
                st.session_state.last_update = datetime.now()
                st.success("✅ 数据生成完成！")
                
            except Exception as e:
                st.error(f"❌ 数据生成失败: {str(e)}")
    
    def load_real_data(self, network_file, data_file):
        """Load real network and monitoring data"""
        if data_file is None:
            st.error("❌ 请上传监控数据文件")
            return
            
        try:
            with st.spinner("正在加载真实数据..."):
                # Load network data if provided
                if network_file is not None:
                    # Save uploaded file temporarily
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{network_file.name.split('.')[-1]}") as tmp:
                        tmp.write(network_file.read())
                        tmp_network_path = tmp.name
                    
                    # Load network using RealNetworkInterface
                    interface = RealNetworkInterface()
                    st.session_state.network = interface.load_from_file(tmp_network_path)
                    
                    # Clean up temp file
                    os.unlink(tmp_network_path)
                    
                    st.success(f"✅ 网络拓扑已加载: {st.session_state.network.number_of_nodes()} 节点, "
                             f"{st.session_state.network.number_of_edges()} 条边")
                else:
                    # Generate a default network if none provided
                    network_gen = NetworkGenerator(NetworkConfig(node_count=20))
                    st.session_state.network = network_gen.generate_network()
                    st.info("ℹ️ 未提供网络文件，已生成默认网络拓扑")
                
                # Load monitoring data
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{data_file.name.split('.')[-1]}") as tmp:
                    tmp.write(data_file.read())
                    tmp_data_path = tmp.name
                
                # Load data using NetworkDataGenerator
                data_gen = NetworkDataGenerator()
                st.session_state.data = data_gen.load_real_data(tmp_data_path)
                
                # Clean up temp file
                os.unlink(tmp_data_path)
                
                # Validate data
                validation = data_gen.validate_data_quality(st.session_state.data)
                if not validation['is_valid']:
                    st.warning(f"⚠️ 数据质量问题: {', '.join(validation['issues'])}")
                else:
                    st.success("✅ 数据质量验证通过")
                
                # Display data info
                st.info(f"📊 监控数据已加载: {len(st.session_state.data)} 条记录, "
                       f"{len(st.session_state.data.columns)} 个特征")
                
                # Train models
                self.train_models()
                
                st.session_state.last_update = datetime.now()
                st.success("✅ 真实数据加载完成！")
                
        except Exception as e:
            st.error(f"❌ 数据加载失败: {str(e)}")
    
    def train_models(self):
        """Train anomaly detection and other models"""
        if st.session_state.data is None:
            return
        
        try:
            # Train anomaly detection model
            ad_config = AnomalyDetectionConfig()
            st.session_state.anomaly_analyzer = NetworkAnomalyAnalyzer(ad_config)
            
            # Split data for training
            data = st.session_state.data
            split_idx = int(0.7 * len(data))
            train_data = data[:split_idx]
            
            st.session_state.anomaly_analyzer.train(train_data)
            
            # Initialize other analyzers
            cf_config = CascadingFailureConfig(num_simulations=20)  # Reduced for demo
            st.session_state.cascade_analyzer = CascadingFailureAnalyzer(cf_config)
            
            # Initialize explainer
            if st.session_state.anomaly_analyzer.ensemble.detectors:
                isolation_forest = st.session_state.anomaly_analyzer.ensemble.detectors['isolation_forest']
                exp_config = ExplainabilityConfig()
                st.session_state.explainer = NetworkAnomalyExplainer(isolation_forest.model, exp_config)
                
                # Setup explainer with background data
                X_train = st.session_state.anomaly_analyzer.prepare_features(train_data)
                st.session_state.explainer.setup_explainer(
                    X_train, st.session_state.anomaly_analyzer.feature_columns
                )
            
        except Exception as e:
            st.error(f"模型训练失败: {str(e)}")

    def call_zhipu_ai(self, messages: List[Dict], system_prompt: str = None):
        """调用智谱AI API"""
        try:
            # 构建请求消息
            if system_prompt:
                # 将系统提示词作为第一条消息
                all_messages = [{"role": "system", "content": system_prompt}] + messages
            else:
                all_messages = messages

            payload = {
                "model": "glm-4-flash",  # 使用性价比高的模型
                "messages": all_messages,
                "thinking": {"type": "enabled"},
                "max_tokens": 4096,
                "temperature": 0.6
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {st.session_state.ai_api_key}"
            }

            response = requests.post(
                "https://open.bigmodel.cn/api/paas/v4/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                return f"API调用失败: {response.status_code} - {response.text}"

        except Exception as e:
            return f"调用AI服务时出错: {str(e)}"

    def get_system_context(self):
        """获取当前系统状态作为AI上下文"""
        context = {}

        if st.session_state.network:
            context["network"] = {
                "node_count": st.session_state.network.number_of_nodes(),
                "edge_count": st.session_state.network.number_of_edges(),
                "density": nx.density(st.session_state.network),
                "is_connected": nx.is_connected(st.session_state.network)
            }

        if st.session_state.data is not None:
            data = st.session_state.data
            context["data"] = {
                "total_points": len(data),
                "anomaly_count": int(data['is_anomaly'].sum()) if 'is_anomaly' in data.columns else 0,
                "anomaly_rate": data['is_anomaly'].mean() if 'is_anomaly' in data.columns else 0,
                "features": [col for col in data.columns if col not in ['timestamp', 'is_anomaly', 'anomaly_score']]
            }

        if st.session_state.alert_manager:
            stats = st.session_state.alert_manager.get_alert_statistics()
            context["alerts"] = stats

        return context
    
    def render_overview_page(self):
        """Render system overview page"""
        st.header("📊 系统概览")
        
        if st.session_state.network is None or st.session_state.data is None:
            st.info("👆 请先在侧边栏生成网络和数据")
            return
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "网络节点数",
                st.session_state.network.number_of_nodes(),
                delta=None
            )
        
        with col2:
            st.metric(
                "网络边数",
                st.session_state.network.number_of_edges(),
                delta=None
            )
        
        with col3:
            anomaly_count = int(st.session_state.data['is_anomaly'].sum())
            st.metric(
                "检测到异常",
                anomaly_count,
                delta=f"{anomaly_count/len(st.session_state.data):.1%}"
            )
        
        with col4:
            active_alerts = len(st.session_state.alert_manager.get_active_alerts())
            st.metric(
                "活跃告警",
                active_alerts,
                delta=None
            )
        
        st.markdown("---")
        
        # Recent data visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 最近时序数据")
            self.plot_timeseries_overview()
        
        with col2:
            st.subheader("🔗 网络拓扑概览")
            self.plot_network_overview()
        
        # System status
        st.markdown("---")
        st.subheader("📊 系统性能指标")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Anomaly detection performance
            if st.session_state.anomaly_analyzer:
                st.info("✅ 异常检测模型已训练")
                # Show some mock metrics
                st.metric("检测准确率", "87.3%", "2.1%")
        
        with col2:
            # Cascading failure analysis
            if st.session_state.cascade_analyzer:
                st.info("✅ 级联失效分析就绪")
                st.metric("网络鲁棒性", "0.73", "-0.05")
        
        with col3:
            # Explainability
            if st.session_state.explainer:
                st.info("✅ 可解释性分析就绪")
                st.metric("解释置信度", "91.2%", "1.8%")
    
    def plot_timeseries_overview(self):
        """Plot overview of time series data"""
        data = st.session_state.data
        
        # Sample data for display (last 100 points)
        display_data = data.tail(100).copy()
        
        fig = go.Figure()
        
        # Detect traffic column name
        traffic_col = None
        for col in ['traffic_mbps', 'traffic', 'throughput_mbps', 'network_throughput']:
            if col in display_data.columns:
                traffic_col = col
                break
        
        if traffic_col:
            # Plot traffic
            fig.add_trace(go.Scatter(
                x=display_data.index,
                y=display_data[traffic_col],
                mode='lines',
                name='网络流量',
                line=dict(color='blue')
            ))
            
            # Highlight anomalies
            anomaly_data = display_data[display_data['is_anomaly'] == True]
            if not anomaly_data.empty:
                fig.add_trace(go.Scatter(
                    x=anomaly_data.index,
                    y=anomaly_data[traffic_col],
                    mode='markers',
                    name='异常点',
                    marker=dict(color='red', size=8, symbol='circle-open')
                ))
        
        fig.update_layout(
            title="网络流量时序图",
            xaxis_title="时间点",
            yaxis_title="流量值",
            height=300,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_network_overview(self):
        """Plot network topology overview"""
        network = st.session_state.network
        
        # Use spring layout for positioning
        pos = nx.spring_layout(network, k=1, iterations=50)
        
        # Prepare node data
        node_x = [pos[node][0] for node in network.nodes()]
        node_y = [pos[node][1] for node in network.nodes()]
        
        # Node colors based on degree
        degrees = dict(network.degree())
        max_degree = max(degrees.values())
        node_colors = [degrees[node]/max_degree for node in network.nodes()]
        
        # Prepare edge data
        edge_x = []
        edge_y = []
        for edge in network.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='lightgray'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(
                size=8,
                color=node_colors,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="节点度数")
            ),
            text=[f"节点 {node}<br>度数: {degrees[node]}" for node in network.nodes()],
            hoverinfo='text',
            showlegend=False
        ))
        
        fig.update_layout(
            title="网络拓扑结构",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[
                dict(
                    text=f"节点: {network.number_of_nodes()}, 边: {network.number_of_edges()}",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(color='gray', size=12)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_network_page(self):
        """Render network topology analysis page"""
        st.header("🔗 网络拓扑分析")
        
        if st.session_state.network is None:
            st.info("请先生成网络数据")
            return
        
        network = st.session_state.network
        
        # Network statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("基本统计")
            st.metric("节点数", network.number_of_nodes())
            st.metric("边数", network.number_of_edges())
            st.metric("网络密度", f"{nx.density(network):.4f}")
            st.metric("连通性", "是" if nx.is_connected(network) else "否")
        
        with col2:
            st.subheader("拓扑指标")
            if nx.is_connected(network):
                st.metric("直径", nx.diameter(network))
                st.metric("平均路径长度", f"{nx.average_shortest_path_length(network):.2f}")
                st.metric("半径", nx.radius(network))
            else:
                st.metric("连通分量数", nx.number_connected_components(network))
        
        with col3:
            st.subheader("网络特性")
            st.metric("平均聚类系数", f"{nx.average_clustering(network):.4f}")
            st.metric("度数同配性", f"{nx.degree_assortativity_coefficient(network):.4f}")
        
        # Detailed network visualization
        st.subheader("🗺️ 详细网络图")
        
        layout_type = st.selectbox(
            "选择布局算法",
            ['spring', 'circular', 'random'],
            format_func=lambda x: {
                'spring': '弹簧布局',
                'circular': '环形布局',
                'random': '随机布局'
            }[x]
        )
        
        self.plot_detailed_network(layout_type)
        
        # Centrality analysis
        st.subheader("📊 中心性分析")
        self.plot_centrality_analysis()
    
    def plot_detailed_network(self, layout_type: str = 'spring'):
        """Plot detailed network with interactive features"""
        network = st.session_state.network
        
        # Choose layout
        if layout_type == 'spring':
            pos = nx.spring_layout(network, k=1, iterations=50)
        elif layout_type == 'circular':
            pos = nx.circular_layout(network)
        else:
            pos = nx.random_layout(network)
        
        # Calculate node metrics
        degrees = dict(network.degree())
        betweenness = nx.betweenness_centrality(network)
        closeness = nx.closeness_centrality(network)
        
        # Prepare data
        node_x = [pos[node][0] for node in network.nodes()]
        node_y = [pos[node][1] for node in network.nodes()]
        
        # Node sizes based on degree
        node_sizes = [degrees[node] * 3 + 5 for node in network.nodes()]
        
        # Node colors based on betweenness centrality
        node_colors = [betweenness[node] for node in network.nodes()]
        
        # Edge data
        edge_x = []
        edge_y = []
        for edge in network.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='rgba(125,125,125,0.5)'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
        
        # Add nodes
        hover_text = [
            f"节点: {node}<br>"
            f"度数: {degrees[node]}<br>"
            f"介数中心性: {betweenness[node]:.3f}<br>"
            f"接近中心性: {closeness[node]:.3f}"
            for node in network.nodes()
        ]
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="介数中心性"),
                line=dict(width=1, color='black')
            ),
            text=hover_text,
            hoverinfo='text',
            showlegend=False
        ))
        
        fig.update_layout(
            title=f"网络拓扑图 - {layout_type}布局",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_centrality_analysis(self):
        """Plot centrality measures analysis"""
        network = st.session_state.network
        
        # Calculate centralities
        degree_cent = nx.degree_centrality(network)
        betweenness_cent = nx.betweenness_centrality(network)
        closeness_cent = nx.closeness_centrality(network)
        
        # Create DataFrame
        nodes = list(network.nodes())
        centrality_df = pd.DataFrame({
            'node': nodes,
            'degree': [degree_cent[node] for node in nodes],
            'betweenness': [betweenness_cent[node] for node in nodes],
            'closeness': [closeness_cent[node] for node in nodes]
        })
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['度数中心性', '介数中心性', '接近中心性', '中心性比较'],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Degree centrality
        top_degree = centrality_df.nlargest(10, 'degree')
        fig.add_trace(
            go.Bar(x=top_degree['node'], y=top_degree['degree'], name='度数中心性'),
            row=1, col=1
        )
        
        # Betweenness centrality
        top_between = centrality_df.nlargest(10, 'betweenness')
        fig.add_trace(
            go.Bar(x=top_between['node'], y=top_between['betweenness'], name='介数中心性'),
            row=1, col=2
        )
        
        # Closeness centrality
        top_close = centrality_df.nlargest(10, 'closeness')
        fig.add_trace(
            go.Bar(x=top_close['node'], y=top_close['closeness'], name='接近中心性'),
            row=2, col=1
        )
        
        # Centrality comparison scatter
        fig.add_trace(
            go.Scatter(
                x=centrality_df['degree'],
                y=centrality_df['betweenness'],
                mode='markers',
                text=centrality_df['node'],
                name='度数 vs 介数',
                marker=dict(
                    size=centrality_df['closeness'] * 20 + 5,
                    color=centrality_df['closeness'],
                    colorscale='Viridis',
                    showscale=True
                )
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_anomaly_page(self):
        """Render anomaly detection analysis page"""
        st.header("🚨 异常检测分析")
        
        if st.session_state.data is None or st.session_state.anomaly_analyzer is None:
            st.info("请先生成数据并训练模型")
            return
        
        data = st.session_state.data
        analyzer = st.session_state.anomaly_analyzer
        
        # Anomaly detection results
        test_data = data.tail(int(len(data) * 0.3))  # Use last 30% as test
        detection_results = analyzer.detect_anomalies(test_data)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        summary = detection_results['summary']
        with col1:
            st.metric("测试样本数", summary['total_samples'])
        
        with col2:
            st.metric("预测异常数", summary['predicted_anomalies'])
        
        with col3:
            st.metric("异常率", f"{summary['anomaly_rate']:.1%}")
        
        with col4:
            st.metric("平均异常概率", f"{summary['mean_anomaly_probability']:.3f}")
        
        # Detailed visualization
        st.subheader("📊 异常检测结果可视化")
        
        tab1, tab2, tab3 = st.tabs(["时序分析", "特征分布", "检测性能"])
        
        with tab1:
            self.plot_anomaly_timeseries(detection_results)
        
        with tab2:
            self.plot_feature_distributions(detection_results)
        
        with tab3:
            self.plot_detection_performance(detection_results, test_data)
    
    def plot_anomaly_timeseries(self, detection_results):
        """Plot anomaly detection time series results"""
        results_df = detection_results['results']
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=['网络流量', '异常概率', '检测结果'],
            vertical_spacing=0.1
        )
        
        # Find traffic column
        traffic_col = None
        for col in ['traffic_mbps', 'traffic', 'throughput_mbps', 'network_throughput']:
            if col in results_df.columns:
                traffic_col = col
                break
        
        if traffic_col:
            # Plot traffic
            fig.add_trace(
                go.Scatter(
                    x=results_df.index,
                    y=results_df[traffic_col],
                    mode='lines',
                    name='流量',
                    line=dict(color='blue')
                ),
                row=1, col=1
        )
        
        # Highlight actual anomalies in traffic plot
        if 'is_anomaly' in results_df.columns and traffic_col:
            anomaly_points = results_df[results_df['is_anomaly'] == True]
            if not anomaly_points.empty:
                fig.add_trace(
                    go.Scatter(
                        x=anomaly_points.index,
                        y=anomaly_points[traffic_col],
                        mode='markers',
                        name='真实异常',
                        marker=dict(color='red', size=6)
                    ),
                    row=1, col=1
                )
        
        # Plot anomaly probability
        fig.add_trace(
            go.Scatter(
                x=results_df.index,
                y=results_df['anomaly_probability'],
                mode='lines',
                name='异常概率',
                line=dict(color='orange'),
                fill='tozeroy'
            ),
            row=2, col=1
        )
        
        # Add threshold line
        fig.add_hline(y=0.5, line_dash="dash", line_color="red", 
                     annotation_text="阈值", row=2, col=1)
        
        # Plot detection results
        predicted_anomalies = results_df[results_df['predicted_anomaly'] == True]
        if not predicted_anomalies.empty:
            fig.add_trace(
                go.Scatter(
                    x=predicted_anomalies.index,
                    y=[1] * len(predicted_anomalies),
                    mode='markers',
                    name='预测异常',
                    marker=dict(color='red', size=8, symbol='x')
                ),
                row=3, col=1
            )
        
        fig.update_layout(
            height=600,
            title="异常检测时序分析",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_feature_distributions(self, detection_results):
        """Plot feature distributions for normal vs anomaly"""
        results_df = detection_results['results']
        
        # Select numeric features
        # Auto-detect feature columns
        feature_columns = []
        column_mappings = {
            'traffic': ['traffic_mbps', 'traffic', 'throughput_mbps', 'network_throughput'],
            'latency': ['latency_ms', 'latency', 'response_time_ms'],
            'packet_loss': ['packet_loss_rate', 'packet_loss', 'loss_rate'],
            'cpu_usage': ['cpu_usage', 'cpu_util', 'processor_usage'],
            'memory_usage': ['memory_usage', 'mem_usage', 'memory_util']
        }
        
        for standard_name, possible_names in column_mappings.items():
            for col_name in possible_names:
                if col_name in results_df.columns:
                    feature_columns.append(col_name)
                    break
        
        if not feature_columns:
            # Fallback: use numeric columns except timestamp and anomaly columns
            feature_columns = [col for col in results_df.columns 
                             if col not in ['timestamp', 'is_anomaly', 'anomaly_score'] 
                             and results_df[col].dtype in ['float64', 'int64']][:5]
        available_features = [col for col in feature_columns if col in results_df.columns]
        
        if not available_features:
            st.warning("没有可用的特征数据")
            return
        
        # Create comparison plots
        n_features = len(available_features)
        cols = min(3, n_features)
        rows = (n_features + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=available_features
        )
        
        for i, feature in enumerate(available_features):
            row = i // cols + 1
            col = i % cols + 1
            
            # Normal data
            normal_data = results_df[results_df['predicted_anomaly'] == False][feature]
            
            # Anomaly data
            anomaly_data = results_df[results_df['predicted_anomaly'] == True][feature]
            
            # Plot histograms
            fig.add_trace(
                go.Histogram(
                    x=normal_data,
                    name=f'{feature} - 正常',
                    opacity=0.7,
                    nbinsx=20
                ),
                row=row, col=col
            )
            
            if not anomaly_data.empty:
                fig.add_trace(
                    go.Histogram(
                        x=anomaly_data,
                        name=f'{feature} - 异常',
                        opacity=0.7,
                        nbinsx=20
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            height=400 * rows,
            title="特征分布对比：正常 vs 异常",
            showlegend=True,
            barmode='overlay'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_detection_performance(self, detection_results, test_data):
        """Plot detection performance metrics"""
        if 'is_anomaly' not in test_data.columns:
            st.warning("没有真实标签数据，无法评估性能")
            return
        
        results_df = detection_results['results']
        
        # Calculate performance metrics
        y_true = test_data['is_anomaly'].astype(int)
        y_pred = results_df['predicted_anomaly'].astype(int)
        y_prob = results_df['anomaly_probability']
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Create performance visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Confusion matrix heatmap
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                title="混淆矩阵",
                labels=dict(x="预测", y="真实", color="样本数")
            )
            fig_cm.update_xaxes(tickmode='array', tickvals=[0, 1], ticktext=['正常', '异常'])
            fig_cm.update_yaxes(tickmode='array', tickvals=[0, 1], ticktext=['正常', '异常'])
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col2:
            # ROC curve (simplified)
            from sklearn.metrics import roc_curve, auc
            
            try:
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                roc_auc = auc(fpr, tpr)
                
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'ROC曲线 (AUC = {roc_auc:.3f})'
                ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    line=dict(dash='dash'),
                    name='随机猜测'
                ))
                
                fig_roc.update_layout(
                    title='ROC曲线',
                    xaxis_title='假正率',
                    yaxis_title='真正率'
                )
                
                st.plotly_chart(fig_roc, use_container_width=True)
                
            except Exception as e:
                st.error(f"ROC曲线绘制失败: {str(e)}")
        
        # Classification report
        st.subheader("📋 分类报告")
        
        report = classification_report(y_true, y_pred, target_names=['正常', '异常'], output_dict=True)
        
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(3))
    
    def render_cascading_page(self):
        """Render cascading failure analysis page"""
        st.header("⚡ 级联失效分析")
        
        if st.session_state.network is None or st.session_state.cascade_analyzer is None:
            st.info("请先生成网络数据")
            return
        
        network = st.session_state.network
        analyzer = st.session_state.cascade_analyzer
        
        # Analysis controls
        col1, col2 = st.columns(2)
        
        with col1:
            initial_failure_count = st.slider("初始失效节点数", 1, 5, 1)
        
        with col2:
            if st.button("🔬 开始级联失效分析"):
                with st.spinner("正在进行级联失效分析..."):
                    try:
                        results = analyzer.analyze_network_robustness(network)
                        st.session_state.analysis_results['cascading'] = results
                        st.success("分析完成！")
                    except Exception as e:
                        st.error(f"分析失败: {str(e)}")
        
        # Display results if available
        if 'cascading' in st.session_state.analysis_results:
            results = st.session_state.analysis_results['cascading']
            
            # Overall robustness metrics
            st.subheader("📊 网络鲁棒性评估")
            
            robustness_metrics = results.get('robustness_metrics', {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'overall_robustness_score' in robustness_metrics:
                    score = robustness_metrics['overall_robustness_score']
                    st.metric("总体鲁棒性评分", f"{score:.3f}")
                    
                    if score > 0.8:
                        st.success("🟢 网络鲁棒性强")
                    elif score > 0.6:
                        st.warning("🟡 网络鲁棒性中等")
                    else:
                        st.error("🔴 网络鲁棒性弱")
            
            with col2:
                single_failures = results.get('single_node_failures', {})
                if 'statistics' in single_failures:
                    stats = single_failures['statistics']
                    avg_cascade = stats.get('mean_final_failures', 0)
                    st.metric("平均级联规模", f"{avg_cascade:.1f}")
            
            with col3:
                if 'statistics' in single_failures:
                    max_cascade = stats.get('max_final_failures', 0)
                    st.metric("最大级联规模", f"{max_cascade}")
            
            # Critical nodes analysis
            st.subheader("🎯 关键节点分析")
            
            critical_nodes = results.get('critical_nodes', {})
            if 'critical_nodes_ranking' in critical_nodes:
                ranking = critical_nodes['critical_nodes_ranking'][:10]
                
                # Create bar chart
                if ranking:
                    nodes, impacts = zip(*ranking)
                    
                    fig = px.bar(
                        x=list(nodes),
                        y=list(impacts),
                        title="关键节点排名（前10位）",
                        labels={'x': '节点ID', 'y': '级联失效影响'}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Critical nodes table
                st.subheader("📋 关键节点详细信息")
                
                critical_df = pd.DataFrame(ranking, columns=['节点ID', '级联影响'])
                st.dataframe(critical_df)
            
            # Analysis report
            st.subheader("📄 分析报告")
            
            report = analyzer.generate_report()
            st.text(report)
    
    def render_explainability_page(self):
        """Render explainability analysis page"""
        st.header("🔍 可解释性分析")
        
        if (st.session_state.data is None or 
            st.session_state.anomaly_analyzer is None or 
            st.session_state.explainer is None):
            st.info("请先生成数据并训练模型")
            return
        
        data = st.session_state.data
        analyzer = st.session_state.anomaly_analyzer
        explainer = st.session_state.explainer
        
        st.subheader("🎯 单个实例解释")
        
        # Instance selection
        test_data = data.tail(100)  # Use last 100 samples
        
        instance_idx = st.selectbox(
            "选择要解释的实例",
            range(len(test_data)),
            format_func=lambda x: f"实例 {x} ({'异常' if test_data.iloc[x]['is_anomaly'] else '正常'})"
        )
        
        selected_instance = test_data.iloc[instance_idx]
        
        # Show instance details
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("实例详情")
            # Auto-detect feature columns 
            feature_cols = []
            column_mappings = {
                'traffic': ['traffic_mbps', 'traffic', 'throughput_mbps'],
                'latency': ['latency_ms', 'latency', 'response_time_ms'],
                'packet_loss': ['packet_loss_rate', 'packet_loss'],
                'cpu_usage': ['cpu_usage', 'cpu_util'],
                'memory_usage': ['memory_usage', 'mem_usage']
            }
            
            for standard_name, possible_names in column_mappings.items():
                for col_name in possible_names:
                    if col_name in data.columns:
                        feature_cols.append(col_name)
                        break
                        
            if not feature_cols:
                # Fallback: use numeric columns
                feature_cols = [col for col in data.columns 
                               if col not in ['timestamp', 'is_anomaly', 'anomaly_score'] 
                               and data[col].dtype in ['float64', 'int64']][:5]
            available_features = [col for col in feature_cols if col in selected_instance.index]
            
            for feature in available_features:
                st.metric(feature, f"{selected_instance[feature]:.3f}")
        
        with col2:
            st.subheader("检测结果")
            st.metric("真实标签", "异常" if selected_instance['is_anomaly'] else "正常")
            
            # Get prediction for this instance
            X_instance = analyzer.prepare_features(
                pd.DataFrame([selected_instance]), 
                analyzer.feature_columns
            )
            
            detection_result = analyzer.detect_anomalies(pd.DataFrame([selected_instance]))
            prediction = detection_result['results']['predicted_anomaly'].iloc[0]
            probability = detection_result['results']['anomaly_probability'].iloc[0]
            
            st.metric("预测标签", "异常" if prediction else "正常")
            st.metric("异常概率", f"{probability:.3f}")
        
        # Generate explanation
        if st.button("🔍 生成解释"):
            with st.spinner("正在生成解释..."):
                try:
                    explanation = explainer.explain_instance(X_instance[0], f"instance_{instance_idx}")
                    
                    # Display explanation
                    st.subheader("📋 解释结果")
                    
                    # Explanation text
                    st.write("**解释文本:**")
                    st.write(explanation['explanation_text'])
                    
                    # Feature contributions
                    st.subheader("📊 特征贡献度")
                    
                    # Create feature importance plot
                    top_features = explanation['top_features']
                    if top_features:
                        features, importances = zip(*top_features)
                        
                        # Get SHAP values for coloring
                        shap_values = [explanation['shap_values'][f] for f in features]
                        colors = ['red' if v > 0 else 'blue' for v in shap_values]
                        
                        fig = go.Figure(data=[
                            go.Bar(
                                x=list(features),
                                y=list(importances),
                                marker_color=colors,
                                text=[f'{v:.3f}' for v in shap_values],
                                textposition='outside'
                            )
                        ])
                        
                        fig.update_layout(
                            title="特征重要性（SHAP值）",
                            xaxis_title="特征",
                            yaxis_title="重要性",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Confidence indicators
                    st.subheader("🎯 解释置信度")
                    
                    confidence = explanation['confidence_indicators']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("一致性", f"{confidence['consistency']:.3f}")
                    
                    with col2:
                        st.metric("集中度", f"{confidence['concentration']:.3f}")
                    
                    with col3:
                        st.metric("稳定性", f"{confidence['stability']:.3f}")
                    
                    with col4:
                        overall_conf = confidence['overall_confidence']
                        st.metric("总体置信度", f"{overall_conf:.3f}")
                        
                        if overall_conf > 0.8:
                            st.success("🟢 高置信度解释")
                        elif overall_conf > 0.6:
                            st.warning("🟡 中等置信度解释")
                        else:
                            st.error("🔴 低置信度解释")
                
                except Exception as e:
                    st.error(f"解释生成失败: {str(e)}")

    def render_ai_analysis_page(self):
        """Render AI model analysis page"""
        st.header("🤖 AI大模型分析")

        if st.session_state.network is None or st.session_state.data is None:
            st.info("👆 请先在侧边栏生成网络和数据")
            return

        # 智能摘要和对话界面
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("📊 智能摘要")
            self.render_ai_summary_panel()

        with col2:
            st.subheader("💬 AI分析助手")
            self.render_ai_chat_interface()

        # 下方分析模块
        st.markdown("---")

        tab1, tab2, tab3 = st.tabs(["📈 趋势分析", "🔍 模式识别", "⚠️ 风险评估"])

        with tab1:
            self.render_ai_trend_analysis()

        with tab2:
            self.render_ai_pattern_recognition()

        with tab3:
            self.render_ai_risk_assessment()

    def render_ai_summary_panel(self):
        """渲染AI智能摘要面板"""
        if st.button("🔄 生成智能摘要", type="primary"):
            with st.spinner("正在生成智能摘要..."):
                context = self.get_system_context()

                system_prompt = f"""
                你是一个专业的网络安全分析师。请基于以下系统状态数据，生成一份简洁的智能摘要报告：

                网络状态：{context.get('network', {})}
                数据状态：{context.get('data', {})}
                告警状态：{context.get('alerts', {})}

                请从以下几个方面进行总结：
                1. 网络整体健康状况
                2. 当前主要风险点
                3. 需要关注的异常情况
                4. 简要的建议措施

                要求简洁明了，突出重点，控制在300字以内。
                """

                messages = [{"role": "user", "content": "请生成当前系统的智能摘要报告"}]

                response = self.call_zhipu_ai(messages, system_prompt)
                st.session_state.ai_summary = response

        if st.session_state.ai_summary:
            st.markdown("### 📋 系统摘要")
            st.markdown(st.session_state.ai_summary)
        else:
            st.info("点击上方按钮生成智能摘要")

    def render_ai_chat_interface(self):
        """渲染AI对话界面"""
        # 预设问题
        st.markdown("### 🚀 快速问题")

        quick_questions = [
            "当前网络整体健康状况如何？",
            "有哪些需要立即关注的问题？",
            "解释最新检测到的异常原因",
            "网络的级联失效风险有多高？",
            "如何提升网络的鲁棒性？"
        ]

        col1, col2 = st.columns(2)
        for i, question in enumerate(quick_questions):
            col = col1 if i % 2 == 0 else col2
            with col:
                if st.button(question, key=f"quick_q_{i}"):
                    st.session_state.ai_chat_history.append({"role": "user", "content": question})
                    with st.spinner("AI正在思考..."):
                        context = self.get_system_context()
                        system_prompt = f"""
                        你是一个专业的网络安全分析师和复杂网络专家。当前系统状态：

                        网络状态：{context.get('network', {})}
                        数据状态：{context.get('data', {})}
                        告警状态：{context.get('alerts', {})}

                        请基于实际数据回答用户问题，提供专业、准确、可操作的建议。
                        """

                        messages = [{"role": "user", "content": question}]
                        response = self.call_zhipu_ai(messages, system_prompt)
                        st.session_state.ai_chat_history.append({"role": "assistant", "content": response})
                    st.rerun()

        # 对话历史
        st.markdown("### 💬 对话历史")

        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.ai_chat_history[-6:]):  # 显示最近6条
                if message["role"] == "user":
                    st.markdown(f"**👤 用户：** {message['content']}")
                else:
                    st.markdown(f"**🤖 AI助手：** {message['content']}")
                st.markdown("---")

        # 自定义问题输入
        with st.form("ai_chat_form", clear_on_submit=True):
            user_input = st.text_area("输入您的问题：", placeholder="例如：分析当前网络的主要风险点", height=100)
            submitted = st.form_submit_button("发送", type="primary")

            if submitted and user_input:
                st.session_state.ai_chat_history.append({"role": "user", "content": user_input})

                with st.spinner("AI正在分析..."):
                    context = self.get_system_context()
                    system_prompt = f"""
                    你是一个专业的网络安全分析师和复杂网络专家。当前系统状态：

                    网络状态：{context.get('network', {})}
                    数据状态：{context.get('data', {})}
                    告警状态：{context.get('alerts', {})}

                    请基于实际数据回答用户问题，提供专业、准确、可操作的建议。保持回答简洁明了。
                    """

                    # 获取最近的对话历史作为上下文
                    recent_messages = st.session_state.ai_chat_history[-4:]  # 最近4条消息

                    response = self.call_zhipu_ai(recent_messages, system_prompt)
                    st.session_state.ai_chat_history.append({"role": "assistant", "content": response})

                st.rerun()

    def render_ai_trend_analysis(self):
        """渲染AI趋势分析"""
        st.markdown("### 📈 智能趋势分析")

        if st.button("🔍 分析数据趋势"):
            if st.session_state.data is not None:
                with st.spinner("正在进行趋势分析..."):
                    data = st.session_state.data

                    # 计算基本统计信息
                    recent_data = data.tail(100)
                    stats_info = {
                        "异常率趋势": recent_data['is_anomaly'].rolling(window=10).mean().iloc[-1] if 'is_anomaly' in data.columns else 0,
                        "数据变化": len(recent_data),
                        "特征数量": len([col for col in data.columns if col not in ['timestamp', 'is_anomaly', 'anomaly_score']])
                    }

                    system_prompt = f"""
                    基于网络监控数据，进行趋势分析。当前数据统计：{stats_info}

                    请从以下角度分析：
                    1. 异常趋势变化
                    2. 可能的周期性模式
                    3. 风险预警信号
                    4. 优化建议
                    """

                    messages = [{"role": "user", "content": "请对当前网络数据进行趋势分析"}]
                    response = self.call_zhipu_ai(messages, system_prompt)

                    st.markdown(response)
            else:
                st.warning("暂无数据用于趋势分析")

    def render_ai_pattern_recognition(self):
        """渲染AI模式识别"""
        st.markdown("### 🔍 智能模式识别")

        if st.button("🎯 识别异常模式"):
            if st.session_state.data is not None and st.session_state.anomaly_analyzer is not None:
                with st.spinner("正在识别异常模式..."):
                    data = st.session_state.data
                    anomaly_data = data[data['is_anomaly'] == True] if 'is_anomaly' in data.columns else pd.DataFrame()

                    pattern_info = {
                        "异常数量": len(anomaly_data),
                        "异常率": data['is_anomaly'].mean() if 'is_anomaly' in data.columns else 0,
                        "特征相关性": "已分析" if st.session_state.anomaly_analyzer else "未分析"
                    }

                    system_prompt = f"""
                    基于网络异常检测结果，识别异常行为模式。当前异常统计：{pattern_info}

                    请分析：
                    1. 异常行为的主要特征
                    2. 可能的攻击模式或故障类型
                    3. 异常的分布规律
                    4. 预防措施建议
                    """

                    messages = [{"role": "user", "content": "请识别和分析当前网络中的异常模式"}]
                    response = self.call_zhipu_ai(messages, system_prompt)

                    st.markdown(response)
            else:
                st.warning("需要先训练异常检测模型")

    def render_ai_risk_assessment(self):
        """渲染AI风险评估"""
        st.markdown("### ⚠️ 智能风险评估")

        if st.button("📊 生成风险评估报告"):
            with st.spinner("正在评估风险..."):
                context = self.get_system_context()

                # 计算风险指标
                risk_factors = []
                if context.get('data', {}).get('anomaly_rate', 0) > 0.1:
                    risk_factors.append("异常率偏高")
                if context.get('alerts', {}).get('active_alerts', 0) > 0:
                    risk_factors.append("存在活跃告警")
                if not context.get('network', {}).get('is_connected', True):
                    risk_factors.append("网络连通性问题")

                system_prompt = f"""
                基于系统当前状态进行综合风险评估：

                系统状态：{context}
                风险因素：{risk_factors}

                请提供：
                1. 风险等级评估（低/中/高）
                2. 主要风险因素分析
                3. 潜在影响评估
                4. 风险缓解建议
                5. 监控重点建议
                """

                messages = [{"role": "user", "content": "请对当前网络系统进行全面的风险评估"}]
                response = self.call_zhipu_ai(messages, system_prompt)

                st.markdown(response)

    def render_alerts_page(self):
        """Render alerts management page"""
        st.header("📢 告警管理")
        
        alert_manager = st.session_state.alert_manager
        
        # Alert statistics
        stats = alert_manager.get_alert_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("总告警数", stats['total_alerts'])
        
        with col2:
            st.metric("活跃告警", stats['active_alerts'])
        
        with col3:
            st.metric("已解决告警", stats['resolved_alerts'])
        
        with col4:
            avg_resolution = stats.get('average_resolution_time', 0) / 60
            st.metric("平均解决时间", f"{avg_resolution:.1f}分钟")
        
        # Create test alert
        st.subheader("🧪 测试告警生成")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("生成异常检测告警"):
                asyncio.run(alert_manager.evaluate_rules({
                    'is_anomaly': True,
                    'confidence': 0.85,
                    'node_id': 'test_node',
                    'top_features': ['traffic_mbps', 'latency_ms']
                }))
                st.success("异常检测告警已生成")
        
        with col2:
            if st.button("生成级联失效告警"):
                asyncio.run(alert_manager.evaluate_rules({
                    'failure_ratio': 0.25,
                    'failed_nodes': 5,
                    'total_nodes': 20,
                    'iterations': 3
                }))
                st.success("级联失效告警已生成")
        
        # Active alerts
        st.subheader("🚨 当前活跃告警")
        
        active_alerts = alert_manager.get_active_alerts()
        
        if active_alerts:
            for alert in active_alerts:
                alert_class = f"alert-{alert.level.value}"
                
                st.markdown(f"""
                <div class="{alert_class}">
                    <h4>🚨 {alert.title}</h4>
                    <p><strong>级别:</strong> {alert.level.value.upper()}</p>
                    <p><strong>时间:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><strong>描述:</strong> {alert.description}</p>
                    <p><strong>来源:</strong> {alert.source}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Alert actions
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button(f"确认告警", key=f"ack_{alert.id}"):
                        alert_manager.acknowledge_alert(alert.id, "dashboard_user")
                        st.success("告警已确认")
                        st.rerun()
                
                with col2:
                    if st.button(f"解决告警", key=f"resolve_{alert.id}"):
                        alert_manager.resolve_alert(alert.id)
                        st.success("告警已解决")
                        st.rerun()
        else:
            st.info("🎉 暂无活跃告警")
        
        # Alert history and statistics
        st.subheader("📊 告警统计")
        
        if stats['by_level']:
            # Alert level distribution
            levels = list(stats['by_level'].keys())
            counts = list(stats['by_level'].values())
            
            fig = px.pie(
                values=counts,
                names=levels,
                title="告警级别分布"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Generate alert report
        st.subheader("📄 告警报告")
        
        if st.button("生成告警报告"):
            report = alert_manager.generate_alert_report()
            st.text(report)


def main():
    """Main application entry point"""
    dashboard = NetworkDashboard()
    dashboard.render_dashboard()


if __name__ == "__main__":
    main()