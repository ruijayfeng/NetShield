# 复杂网络异常行为检测与级联失效分析系统 - 实现指导文档

## 项目概述

本项目旨在构建一套面向复杂网络的可解释异常行为检测体系，集成级联失效分析和告警挖掘机制，通过可视化技术提供直观的异常洞察。

## 核心技术架构

### 系统架构图
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   数据采集层     │    │   分析处理层     │    │   可视化展示层   │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • 网络流量监控   │───▶│ • 异常检测引擎   │───▶│ • 实时监控面板   │
│ • 拓扑结构感知   │    │ • 级联失效分析   │    │ • 告警可视化     │
│ • 设备状态监控   │    │ • 可解释性分析   │    │ • 拓扑图展示     │
│ • 日志数据收集   │    │ • 时序关联挖掘   │    │ • 特征洞察界面   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
           │                      │                      │
           ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                        数据存储与管理层                           │
│  • 时序数据库(InfluxDB)  • 图数据库(Neo4j)  • 关系数据库(PostgreSQL) │
└─────────────────────────────────────────────────────────────────┘
```

## 详细实现方案

### Phase 1: 环境搭建与基础框架 (第1-2周)

#### 1.1 项目结构创建
```bash
fenxi/
├── requirements.txt              # 依赖管理
├── config/                      # 配置文件
│   ├── database.yaml
│   ├── model_config.yaml
│   └── logging.yaml
├── src/                         # 源代码
│   ├── __init__.py
│   ├── data/                    # 数据处理模块
│   │   ├── collectors/          # 数据采集
│   │   ├── preprocessors/       # 数据预处理
│   │   └── storage/            # 数据存储
│   ├── models/                  # 机器学习模型
│   │   ├── anomaly_detection/   # 异常检测
│   │   ├── explainable/        # 可解释性
│   │   └── cascading/          # 级联失效
│   ├── analysis/               # 分析引擎
│   │   ├── topology/           # 拓扑分析
│   │   ├── temporal/           # 时序分析
│   │   └── correlation/        # 关联分析
│   ├── visualization/          # 可视化
│   ├── alerts/                 # 告警系统
│   └── recovery/              # 恢复机制
├── tests/                      # 测试代码
├── docs/                       # 文档
├── notebooks/                  # Jupyter笔记本
└── data/                       # 数据文件
    ├── raw/                    # 原始数据
    ├── processed/              # 处理后数据
    └── models/                 # 训练模型
```

#### 1.2 核心依赖安装
```bash
# 创建虚拟环境
python -m venv fenxi_env
source fenxi_env/bin/activate  # Linux/Mac
# fenxi_env\Scripts\activate   # Windows

# 安装核心依赖
pip install torch torchvision torchaudio
pip install tensorflow keras
pip install networkx graph-tool
pip install pandas numpy scipy
pip install scikit-learn xgboost lightgbm
pip install plotly dash streamlit
pip install fastapi uvicorn
pip install psycopg2-binary pymongo redis
pip install shap lime
pip install influxdb-client neo4j
```

#### 1.3 基础配置文件
创建 `config/database.yaml`:
```yaml
databases:
  postgresql:
    host: localhost
    port: 5432
    database: fenxi_db
    username: fenxi_user
    password: fenxi_pass
  
  neo4j:
    uri: bolt://localhost:7687
    username: neo4j
    password: neo4j_pass
  
  influxdb:
    url: http://localhost:8086
    token: your_influx_token
    org: fenxi_org
    bucket: network_data
  
  redis:
    host: localhost
    port: 6379
    db: 0
```

### Phase 2: 数据采集与预处理系统 (第3-4周)

#### 2.1 网络数据采集器
```python
# src/data/collectors/network_collector.py
import asyncio
import psutil
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass
from datetime import datetime

@dataclass
class NetworkMetrics:
    timestamp: datetime
    bytes_sent: int
    bytes_recv: int
    packets_sent: int
    packets_recv: int
    errors_in: int
    errors_out: int
    drops_in: int
    drops_out: int

class NetworkDataCollector:
    def __init__(self, collection_interval: int = 10):
        self.collection_interval = collection_interval
        self.is_running = False
        
    async def start_collection(self):
        """开始网络数据采集"""
        self.is_running = True
        while self.is_running:
            metrics = self._collect_network_metrics()
            await self._store_metrics(metrics)
            await asyncio.sleep(self.collection_interval)
    
    def _collect_network_metrics(self) -> Dict[str, NetworkMetrics]:
        """采集网络接口指标"""
        interfaces = psutil.net_io_counters(pernic=True)
        metrics = {}
        
        for interface, stats in interfaces.items():
            metrics[interface] = NetworkMetrics(
                timestamp=datetime.now(),
                bytes_sent=stats.bytes_sent,
                bytes_recv=stats.bytes_recv,
                packets_sent=stats.packets_sent,
                packets_recv=stats.packets_recv,
                errors_in=stats.errin,
                errors_out=stats.errout,
                drops_in=stats.dropin,
                drops_out=stats.dropout
            )
        
        return metrics
    
    async def _store_metrics(self, metrics: Dict[str, NetworkMetrics]):
        """存储指标到数据库"""
        # 实现数据存储逻辑
        pass
```

#### 2.2 拓扑发现模块
```python
# src/data/collectors/topology_collector.py
import networkx as nx
from typing import Dict, List, Tuple
import subprocess
import re

class NetworkTopologyCollector:
    def __init__(self):
        self.topology_graph = nx.Graph()
        
    def discover_topology(self) -> nx.Graph:
        """发现网络拓扑结构"""
        # 使用多种方法发现拓扑
        devices = self._discover_devices()
        connections = self._discover_connections(devices)
        
        # 构建网络图
        self.topology_graph.clear()
        self.topology_graph.add_nodes_from(devices)
        self.topology_graph.add_edges_from(connections)
        
        return self.topology_graph
    
    def _discover_devices(self) -> List[str]:
        """发现网络设备"""
        devices = []
        
        # ARP表扫描
        try:
            result = subprocess.run(['arp', '-a'], 
                                  capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                ip_match = re.search(r'(\d+\.\d+\.\d+\.\d+)', line)
                if ip_match:
                    devices.append(ip_match.group(1))
        except:
            pass
            
        return list(set(devices))
    
    def _discover_connections(self, devices: List[str]) -> List[Tuple[str, str]]:
        """发现设备间连接"""
        connections = []
        
        # 实现连接发现逻辑
        # 可以使用traceroute、SNMP等方法
        
        return connections
```

#### 2.3 数据预处理管道
```python
# src/data/preprocessors/data_preprocessor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from typing import Dict, Any, Tuple

class NetworkDataPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.imputers = {}
        
    def preprocess_timeseries(self, data: pd.DataFrame, 
                            window_size: int = 10) -> pd.DataFrame:
        """预处理时序数据"""
        # 1. 处理缺失值
        data_filled = self._handle_missing_values(data)
        
        # 2. 异常值处理
        data_cleaned = self._handle_outliers(data_filled)
        
        # 3. 特征工程
        data_featured = self._create_features(data_cleaned, window_size)
        
        # 4. 数据标准化
        data_normalized = self._normalize_data(data_featured)
        
        return data_normalized
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col not in self.imputers:
                self.imputers[col] = SimpleImputer(strategy='mean')
                
            data[col] = self.imputers[col].fit_transform(
                data[col].values.reshape(-1, 1)).flatten()
        
        return data
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理异常值（使用IQR方法）"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
        
        return data
    
    def _create_features(self, data: pd.DataFrame, 
                        window_size: int) -> pd.DataFrame:
        """创建时序特征"""
        feature_data = data.copy()
        
        # 滑动窗口统计特征
        for col in data.select_dtypes(include=[np.number]).columns:
            feature_data[f'{col}_rolling_mean'] = (
                data[col].rolling(window=window_size).mean()
            )
            feature_data[f'{col}_rolling_std'] = (
                data[col].rolling(window=window_size).std()
            )
            feature_data[f'{col}_rolling_max'] = (
                data[col].rolling(window=window_size).max()
            )
            feature_data[f'{col}_rolling_min'] = (
                data[col].rolling(window=window_size).min()
            )
            
        return feature_data
    
    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """数据标准化"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        normalized_data = data.copy()
        
        for col in numeric_columns:
            if col not in self.scalers:
                self.scalers[col] = StandardScaler()
                
            normalized_data[col] = self.scalers[col].fit_transform(
                data[col].values.reshape(-1, 1)).flatten()
        
        return normalized_data
```

### Phase 3: 异常检测算法实现 (第5-7周)

#### 3.1 基于图神经网络的异常检测
```python
# src/models/anomaly_detection/gnn_detector.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import networkx as nx
import numpy as np

class GraphAnomalyDetector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, 
                 num_classes: int = 2):
        super(GraphAnomalyDetector, self).__init__()
        
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim // 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, edge_index, batch=None):
        # 图卷积层
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        
        x = F.relu(self.conv3(x, edge_index))
        
        # 全局池化（如果是图级别分类）
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        # 分类层
        out = self.classifier(x)
        
        return out

class GNNAnomalyDetectionSystem:
    def __init__(self, input_dim: int, device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = GraphAnomalyDetector(input_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
    def prepare_graph_data(self, topology: nx.Graph, 
                          features: np.ndarray) -> Data:
        """将NetworkX图转换为PyTorch Geometric数据"""
        # 节点特征
        x = torch.tensor(features, dtype=torch.float)
        
        # 边信息
        edge_list = list(topology.edges())
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index)
        return data.to(self.device)
    
    def train(self, train_data: list, labels: list, epochs: int = 100):
        """训练模型"""
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            
            for data, label in zip(train_data, labels):
                self.optimizer.zero_grad()
                
                out = self.model(data.x, data.edge_index)
                loss = self.criterion(out, torch.tensor([label], dtype=torch.long).to(self.device))
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Average Loss: {total_loss/len(train_data):.4f}')
    
    def detect_anomaly(self, graph_data: Data) -> tuple:
        """异常检测"""
        self.model.eval()
        
        with torch.no_grad():
            out = self.model(graph_data.x, graph_data.edge_index)
            probabilities = F.softmax(out, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
            
        return prediction.cpu().numpy(), probabilities.cpu().numpy()
```

#### 3.2 时序异常检测
```python
# src/models/anomaly_detection/temporal_detector.py
import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import pandas as pd

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 50, 
                 num_layers: int = 2, dropout: float = 0.2):
        super(LSTMAutoencoder, self).__init__()
        
        # Encoder
        self.encoder_lstm = nn.LSTM(input_size, hidden_size, 
                                   num_layers, dropout=dropout, 
                                   batch_first=True)
        
        # Decoder
        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, 
                                   num_layers, dropout=dropout, 
                                   batch_first=True)
        self.output_layer = nn.Linear(hidden_size, input_size)
        
    def forward(self, x):
        # Encoder
        encoded, (hidden, cell) = self.encoder_lstm(x)
        
        # Use the last encoded state as context
        context = encoded[:, -1, :].unsqueeze(1)
        
        # Decoder
        decoded, _ = self.decoder_lstm(context.repeat(1, x.size(1), 1))
        
        # Output reconstruction
        output = self.output_layer(decoded)
        
        return output

class TemporalAnomalyDetector:
    def __init__(self, sequence_length: int = 50, 
                 feature_dim: int = 10, method: str = 'lstm'):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.method = method
        
        if method == 'lstm':
            self.model = LSTMAutoencoder(feature_dim)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            self.criterion = nn.MSELoss()
        elif method == 'isolation_forest':
            self.model = IsolationForest(contamination=0.1, random_state=42)
        elif method == 'one_class_svm':
            self.model = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
    
    def create_sequences(self, data: np.ndarray) -> tuple:
        """创建时序序列"""
        sequences = []
        
        for i in range(len(data) - self.sequence_length + 1):
            sequences.append(data[i:i + self.sequence_length])
        
        return np.array(sequences)
    
    def train_lstm(self, train_data: np.ndarray, epochs: int = 100):
        """训练LSTM自编码器"""
        sequences = self.create_sequences(train_data)
        train_loader = torch.utils.data.DataLoader(
            torch.FloatTensor(sequences), 
            batch_size=32, 
            shuffle=True
        )
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            
            for batch in train_loader:
                self.optimizer.zero_grad()
                
                reconstructed = self.model(batch)
                loss = self.criterion(reconstructed, batch)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                avg_loss = total_loss / len(train_loader)
                print(f'Epoch {epoch}, Average Loss: {avg_loss:.4f}')
    
    def train_classical(self, train_data: np.ndarray):
        """训练传统机器学习模型"""
        if self.method in ['isolation_forest', 'one_class_svm']:
            self.model.fit(train_data)
        else:
            raise ValueError(f"Unsupported method: {self.method}")
    
    def detect_anomalies(self, test_data: np.ndarray) -> np.ndarray:
        """异常检测"""
        if self.method == 'lstm':
            return self._detect_lstm_anomalies(test_data)
        else:
            return self._detect_classical_anomalies(test_data)
    
    def _detect_lstm_anomalies(self, test_data: np.ndarray) -> np.ndarray:
        """LSTM异常检测"""
        sequences = self.create_sequences(test_data)
        test_tensor = torch.FloatTensor(sequences)
        
        self.model.eval()
        
        with torch.no_grad():
            reconstructed = self.model(test_tensor)
            reconstruction_errors = torch.mean((test_tensor - reconstructed) ** 2, dim=(1, 2))
        
        # 使用重构误差的阈值判断异常
        threshold = torch.quantile(reconstruction_errors, 0.95)
        anomalies = reconstruction_errors > threshold
        
        return anomalies.numpy().astype(int)
    
    def _detect_classical_anomalies(self, test_data: np.ndarray) -> np.ndarray:
        """传统方法异常检测"""
        predictions = self.model.predict(test_data)
        # 转换为二进制标签（1为异常，0为正常）
        return (predictions == -1).astype(int)
```

### Phase 4: 级联失效分析模块 (第8-10周)

#### 4.1 级联失效模型
```python
# src/models/cascading/failure_model.py
import networkx as nx
import numpy as np
import pandas as pd
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import random

class NodeState(Enum):
    ACTIVE = "active"
    FAILED = "failed"
    OVERLOADED = "overloaded"

@dataclass
class NodeInfo:
    node_id: str
    capacity: float
    load: float
    state: NodeState
    failure_threshold: float = 0.8
    failure_probability: float = 0.0

class CascadingFailureModel:
    def __init__(self, network: nx.Graph, initial_capacity_ratio: float = 1.5):
        self.network = network.copy()
        self.initial_capacity_ratio = initial_capacity_ratio
        self.node_info = {}
        self.failure_history = []
        self._initialize_nodes()
    
    def _initialize_nodes(self):
        """初始化节点信息"""
        for node in self.network.nodes():
            # 基于节点度数设置初始负载
            degree = self.network.degree(node)
            initial_load = degree / max(dict(self.network.degree()).values())
            
            self.node_info[node] = NodeInfo(
                node_id=str(node),
                capacity=initial_load * self.initial_capacity_ratio,
                load=initial_load,
                state=NodeState.ACTIVE,
                failure_threshold=0.8 + random.uniform(-0.1, 0.1),  # 添加随机性
                failure_probability=0.05
            )
    
    def simulate_cascading_failure(self, initial_failures: List[str], 
                                 max_iterations: int = 100) -> Dict:
        """模拟级联失效过程"""
        # 重置网络状态
        self._reset_network()
        
        # 设置初始失效节点
        for node in initial_failures:
            if node in self.node_info:
                self.node_info[node].state = NodeState.FAILED
        
        failure_sequence = []
        iteration = 0
        
        while iteration < max_iterations:
            # 记录当前状态
            current_failed = self._get_failed_nodes()
            failure_sequence.append({
                'iteration': iteration,
                'failed_nodes': current_failed.copy(),
                'total_failures': len(current_failed)
            })
            
            # 重新分配负载
            self._redistribute_load()
            
            # 检查新的失效
            new_failures = self._check_for_failures()
            
            if not new_failures:
                break
            
            # 更新节点状态
            for node in new_failures:
                self.node_info[node].state = NodeState.FAILED
            
            iteration += 1
        
        self.failure_history = failure_sequence
        
        return {
            'total_iterations': iteration,
            'final_failures': len(self._get_failed_nodes()),
            'failure_sequence': failure_sequence,
            'network_robustness': self._calculate_robustness()
        }
    
    def _redistribute_load(self):
        """重新分配网络负载"""
        active_nodes = [node for node, info in self.node_info.items() 
                       if info.state == NodeState.ACTIVE]
        
        # 计算总负载
        total_load = sum(info.load for info in self.node_info.values() 
                        if info.state != NodeState.FAILED)
        
        if not active_nodes:
            return
        
        # 基于节点容量重新分配负载
        total_capacity = sum(self.node_info[node].capacity for node in active_nodes)
        
        for node in active_nodes:
            capacity_ratio = self.node_info[node].capacity / total_capacity
            self.node_info[node].load = total_load * capacity_ratio
    
    def _check_for_failures(self) -> List[str]:
        """检查是否有新的节点失效"""
        new_failures = []
        
        for node, info in self.node_info.items():
            if info.state == NodeState.ACTIVE:
                # 检查负载是否超过阈值
                load_ratio = info.load / info.capacity
                
                if load_ratio > info.failure_threshold:
                    # 根据失效概率决定是否失效
                    failure_prob = min(0.9, info.failure_probability * (load_ratio ** 2))
                    
                    if random.random() < failure_prob:
                        new_failures.append(node)
                    else:
                        info.state = NodeState.OVERLOADED
        
        return new_failures
    
    def _get_failed_nodes(self) -> Set[str]:
        """获取所有失效节点"""
        return {node for node, info in self.node_info.items() 
                if info.state == NodeState.FAILED}
    
    def _reset_network(self):
        """重置网络状态"""
        for info in self.node_info.values():
            info.state = NodeState.ACTIVE
        self.failure_history = []
    
    def _calculate_robustness(self) -> float:
        """计算网络鲁棒性"""
        total_nodes = len(self.node_info)
        failed_nodes = len(self._get_failed_nodes())
        
        return 1.0 - (failed_nodes / total_nodes)
    
    def analyze_critical_nodes(self, num_simulations: int = 100) -> Dict:
        """分析关键节点"""
        node_criticality = {}
        
        for node in self.network.nodes():
            failure_impacts = []
            
            for _ in range(num_simulations):
                result = self.simulate_cascading_failure([str(node)])
                failure_impacts.append(result['final_failures'])
            
            node_criticality[str(node)] = {
                'average_impact': np.mean(failure_impacts),
                'max_impact': np.max(failure_impacts),
                'std_impact': np.std(failure_impacts)
            }
        
        # 根据平均影响排序
        sorted_nodes = sorted(node_criticality.items(), 
                            key=lambda x: x[1]['average_impact'], 
                            reverse=True)
        
        return {
            'criticality_ranking': sorted_nodes,
            'most_critical': sorted_nodes[0][0] if sorted_nodes else None,
            'criticality_scores': node_criticality
        }
```

#### 4.2 失效传播路径分析
```python
# src/analysis/cascading/propagation_analyzer.py
import networkx as nx
import numpy as np
from typing import List, Dict, Tuple
import pandas as pd
from collections import defaultdict, deque

class FailurePropagationAnalyzer:
    def __init__(self, network: nx.Graph):
        self.network = network
        self.propagation_paths = []
        
    def trace_propagation_paths(self, failure_sequence: List[Dict]) -> List[Dict]:
        """追踪失效传播路径"""
        propagation_paths = []
        
        if len(failure_sequence) < 2:
            return propagation_paths
        
        for i in range(1, len(failure_sequence)):
            prev_failures = set(failure_sequence[i-1]['failed_nodes'])
            curr_failures = set(failure_sequence[i]['failed_nodes'])
            
            new_failures = curr_failures - prev_failures
            
            if new_failures:
                # 分析每个新失效节点的传播路径
                for failed_node in new_failures:
                    propagation_info = self._analyze_node_propagation(
                        failed_node, prev_failures, curr_failures
                    )
                    
                    propagation_paths.append({
                        'iteration': i,
                        'failed_node': failed_node,
                        'propagation_info': propagation_info
                    })
        
        self.propagation_paths = propagation_paths
        return propagation_paths
    
    def _analyze_node_propagation(self, failed_node: str, 
                                prev_failures: set, 
                                curr_failures: set) -> Dict:
        """分析单个节点的失效传播"""
        neighbors = list(self.network.neighbors(failed_node))
        
        # 找到可能的传播源
        potential_sources = []
        for neighbor in neighbors:
            if neighbor in prev_failures:
                potential_sources.append(neighbor)
        
        # 计算传播距离和路径
        propagation_info = {
            'direct_neighbors': neighbors,
            'potential_sources': potential_sources,
            'propagation_distance': self._calculate_min_distance_to_failures(
                failed_node, prev_failures),
            'influence_score': self._calculate_influence_score(
                failed_node, prev_failures)
        }
        
        return propagation_info
    
    def _calculate_min_distance_to_failures(self, node: str, 
                                          failed_nodes: set) -> int:
        """计算节点到最近失效节点的距离"""
        if not failed_nodes:
            return float('inf')
        
        min_distance = float('inf')
        
        for failed_node in failed_nodes:
            try:
                distance = nx.shortest_path_length(self.network, 
                                                 source=failed_node, 
                                                 target=node)
                min_distance = min(min_distance, distance)
            except nx.NetworkXNoPath:
                continue
        
        return min_distance
    
    def _calculate_influence_score(self, node: str, failed_nodes: set) -> float:
        """计算失效节点对目标节点的影响分数"""
        influence_score = 0.0
        
        for failed_node in failed_nodes:
            try:
                distance = nx.shortest_path_length(self.network, 
                                                 source=failed_node, 
                                                 target=node)
                # 距离越近，影响越大
                influence_score += 1.0 / (1.0 + distance)
            except nx.NetworkXNoPath:
                continue
        
        return influence_score
    
    def identify_propagation_patterns(self) -> Dict:
        """识别传播模式"""
        if not self.propagation_paths:
            return {}
        
        # 分析传播距离分布
        distances = [path['propagation_info']['propagation_distance'] 
                    for path in self.propagation_paths 
                    if path['propagation_info']['propagation_distance'] != float('inf')]
        
        # 分析传播源分布
        source_counts = defaultdict(int)
        for path in self.propagation_paths:
            for source in path['propagation_info']['potential_sources']:
                source_counts[source] += 1
        
        # 分析传播时间模式
        iteration_failures = defaultdict(int)
        for path in self.propagation_paths:
            iteration_failures[path['iteration']] += 1
        
        return {
            'distance_distribution': {
                'mean': np.mean(distances) if distances else 0,
                'std': np.std(distances) if distances else 0,
                'max': np.max(distances) if distances else 0,
                'histogram': np.histogram(distances, bins=10)[0].tolist() if distances else []
            },
            'top_propagation_sources': dict(sorted(source_counts.items(), 
                                                 key=lambda x: x[1], 
                                                 reverse=True)[:10]),
            'temporal_pattern': dict(iteration_failures),
            'total_propagation_events': len(self.propagation_paths)
        }
    
    def generate_propagation_tree(self, root_failure: str) -> nx.DiGraph:
        """生成失效传播树"""
        propagation_tree = nx.DiGraph()
        propagation_tree.add_node(root_failure, level=0)
        
        # 使用BFS构建传播树
        queue = deque([(root_failure, 0)])
        visited = {root_failure}
        
        for path in self.propagation_paths:
            failed_node = path['failed_node']
            sources = path['propagation_info']['potential_sources']
            
            for source in sources:
                if source in visited and failed_node not in visited:
                    propagation_tree.add_edge(source, failed_node)
                    visited.add(failed_node)
        
        return propagation_tree
```

### Phase 5: 可解释性分析模块 (第11-12周)

#### 5.1 SHAP集成
```python
# src/models/explainable/shap_explainer.py
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import torch

class NetworkAnomalyExplainer:
    def __init__(self, model, background_data: np.ndarray):
        self.model = model
        self.background_data = background_data
        self.explainer = None
        self._setup_explainer()
    
    def _setup_explainer(self):
        """设置SHAP解释器"""
        if hasattr(self.model, 'predict_proba'):
            # 适用于scikit-learn模型
            self.explainer = shap.Explainer(self.model, self.background_data)
        elif isinstance(self.model, torch.nn.Module):
            # 适用于PyTorch模型
            self.explainer = shap.DeepExplainer(self.model, 
                torch.FloatTensor(self.background_data))
        else:
            # 通用解释器
            self.explainer = shap.Explainer(self.model.predict, self.background_data)
    
    def explain_prediction(self, sample: np.ndarray, 
                          feature_names: List[str] = None) -> Dict:
        """解释单个预测"""
        if isinstance(self.model, torch.nn.Module):
            sample_tensor = torch.FloatTensor(sample.reshape(1, -1))
            shap_values = self.explainer.shap_values(sample_tensor)
        else:
            shap_values = self.explainer.shap_values(sample.reshape(1, -1))
        
        # 处理SHAP值格式
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # 通常取异常类的SHAP值
        
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]
        
        # 创建解释结果
        explanation = {
            'shap_values': shap_values,
            'feature_importance': dict(zip(
                feature_names or [f'feature_{i}' for i in range(len(shap_values))],
                shap_values
            )),
            'top_contributing_features': self._get_top_features(shap_values, feature_names),
            'explanation_summary': self._generate_explanation_text(shap_values, feature_names)
        }
        
        return explanation
    
    def _get_top_features(self, shap_values: np.ndarray, 
                         feature_names: List[str] = None, 
                         top_k: int = 5) -> List[Dict]:
        """获取最重要的特征"""
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(shap_values))]
        
        # 按绝对值排序
        indices = np.argsort(np.abs(shap_values))[::-1][:top_k]
        
        top_features = []
        for idx in indices:
            top_features.append({
                'feature_name': feature_names[idx],
                'shap_value': float(shap_values[idx]),
                'importance': abs(float(shap_values[idx])),
                'contribution': 'positive' if shap_values[idx] > 0 else 'negative'
            })
        
        return top_features
    
    def _generate_explanation_text(self, shap_values: np.ndarray, 
                                  feature_names: List[str] = None) -> str:
        """生成解释文本"""
        top_features = self._get_top_features(shap_values, feature_names, 3)
        
        explanation = "异常检测解释：\n"
        
        for i, feature in enumerate(top_features):
            contribution = "增加" if feature['contribution'] == 'positive' else "降低"
            explanation += f"{i+1}. {feature['feature_name']} {contribution}了异常概率 "
            explanation += f"(贡献值: {feature['shap_value']:.3f})\n"
        
        return explanation
    
    def batch_explain(self, samples: np.ndarray, 
                     feature_names: List[str] = None) -> List[Dict]:
        """批量解释"""
        explanations = []
        
        for sample in samples:
            explanation = self.explain_prediction(sample, feature_names)
            explanations.append(explanation)
        
        return explanations
    
    def create_explanation_dashboard(self, samples: np.ndarray, 
                                   feature_names: List[str] = None, 
                                   save_path: str = None):
        """创建解释性仪表板"""
        if isinstance(self.model, torch.nn.Module):
            samples_tensor = torch.FloatTensor(samples)
            shap_values = self.explainer.shap_values(samples_tensor)
        else:
            shap_values = self.explainer.shap_values(samples)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # 创建SHAP图表
        plt.figure(figsize=(15, 10))
        
        # 特征重要性总结图
        plt.subplot(2, 2, 1)
        shap.summary_plot(shap_values, samples, 
                         feature_names=feature_names, 
                         show=False, plot_type="bar")
        plt.title("特征重要性总结")
        
        # 详细特征影响图
        plt.subplot(2, 2, 2)
        shap.summary_plot(shap_values, samples, 
                         feature_names=feature_names, 
                         show=False)
        plt.title("特征影响详情")
        
        # 依赖图（前两个最重要特征）
        if feature_names and len(feature_names) >= 2:
            plt.subplot(2, 2, 3)
            shap.dependence_plot(0, shap_values, samples, 
                               feature_names=feature_names, 
                               show=False)
            
            plt.subplot(2, 2, 4)
            shap.dependence_plot(1, shap_values, samples, 
                               feature_names=feature_names, 
                               show=False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
```

### Phase 6: 告警系统与恢复机制 (第13-14周)

#### 6.1 智能告警系统
```python
# src/alerts/alert_manager.py
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertStatus(Enum):
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SUPPRESSED = "suppressed"

@dataclass
class Alert:
    id: str
    title: str
    description: str
    level: AlertLevel
    status: AlertStatus
    timestamp: datetime
    source: str
    metadata: Dict = None
    resolution_time: Optional[datetime] = None
    acknowledged_by: Optional[str] = None

class AlertManager:
    def __init__(self):
        self.active_alerts = {}
        self.alert_history = []
        self.notification_channels = {}
        self.alert_rules = []
        self.suppression_rules = []
        
    def register_notification_channel(self, channel_name: str, 
                                    notification_func: Callable):
        """注册通知渠道"""
        self.notification_channels[channel_name] = notification_func
    
    def add_alert_rule(self, rule_func: Callable, conditions: Dict):
        """添加告警规则"""
        self.alert_rules.append({
            'function': rule_func,
            'conditions': conditions
        })
    
    async def create_alert(self, title: str, description: str, 
                          level: AlertLevel, source: str, 
                          metadata: Dict = None) -> str:
        """创建新告警"""
        alert_id = f"{source}_{int(datetime.now().timestamp())}"
        
        alert = Alert(
            id=alert_id,
            title=title,
            description=description,
            level=level,
            status=AlertStatus.ACTIVE,
            timestamp=datetime.now(),
            source=source,
            metadata=metadata or {}
        )
        
        # 检查抑制规则
        if not self._is_suppressed(alert):
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            # 发送通知
            await self._send_notifications(alert)
        
        return alert_id
    
    def _is_suppressed(self, alert: Alert) -> bool:
        """检查告警是否被抑制"""
        for rule in self.suppression_rules:
            if rule(alert):
                return True
        return False
    
    async def _send_notifications(self, alert: Alert):
        """发送告警通知"""
        for channel_name, notification_func in self.notification_channels.items():
            try:
                await notification_func(alert)
            except Exception as e:
                print(f"Failed to send notification via {channel_name}: {e}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """确认告警"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].status = AlertStatus.ACKNOWLEDGED
            self.active_alerts[alert_id].acknowledged_by = acknowledged_by
    
    def resolve_alert(self, alert_id: str):
        """解决告警"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].status = AlertStatus.RESOLVED
            self.active_alerts[alert_id].resolution_time = datetime.now()
            # 从活跃告警中移除
            resolved_alert = self.active_alerts.pop(alert_id)
            return resolved_alert
    
    def get_alert_statistics(self, time_range: timedelta = None) -> Dict:
        """获取告警统计信息"""
        if time_range:
            cutoff_time = datetime.now() - time_range
            relevant_alerts = [alert for alert in self.alert_history 
                             if alert.timestamp >= cutoff_time]
        else:
            relevant_alerts = self.alert_history
        
        stats = {
            'total_alerts': len(relevant_alerts),
            'active_alerts': len(self.active_alerts),
            'by_level': {},
            'by_source': {},
            'resolution_times': []
        }
        
        for alert in relevant_alerts:
            # 按级别统计
            level_str = alert.level.value
            stats['by_level'][level_str] = stats['by_level'].get(level_str, 0) + 1
            
            # 按来源统计
            stats['by_source'][alert.source] = stats['by_source'].get(alert.source, 0) + 1
            
            # 解决时间统计
            if alert.resolution_time:
                resolution_time = (alert.resolution_time - alert.timestamp).total_seconds()
                stats['resolution_times'].append(resolution_time)
        
        return stats

class NetworkAnomalyAlerter:
    def __init__(self, alert_manager: AlertManager):
        self.alert_manager = alert_manager
        self._setup_alert_rules()
    
    def _setup_alert_rules(self):
        """设置网络异常告警规则"""
        # 异常检测告警规则
        async def anomaly_detection_rule(detection_result: Dict):
            if detection_result.get('is_anomaly', False):
                confidence = detection_result.get('confidence', 0)
                
                if confidence > 0.9:
                    level = AlertLevel.CRITICAL
                elif confidence > 0.7:
                    level = AlertLevel.WARNING
                else:
                    level = AlertLevel.INFO
                
                await self.alert_manager.create_alert(
                    title="网络异常检测告警",
                    description=f"检测到网络异常，置信度: {confidence:.2f}",
                    level=level,
                    source="anomaly_detector",
                    metadata=detection_result
                )
        
        # 级联失效告警规则
        async def cascading_failure_rule(failure_result: Dict):
            failed_nodes = failure_result.get('final_failures', 0)
            total_nodes = failure_result.get('total_nodes', 1)
            failure_ratio = failed_nodes / total_nodes
            
            if failure_ratio > 0.5:
                level = AlertLevel.EMERGENCY
            elif failure_ratio > 0.2:
                level = AlertLevel.CRITICAL
            elif failure_ratio > 0.1:
                level = AlertLevel.WARNING
            else:
                level = AlertLevel.INFO
            
            await self.alert_manager.create_alert(
                title="级联失效告警",
                description=f"检测到级联失效，失效节点比例: {failure_ratio:.2%}",
                level=level,
                source="cascading_failure_analyzer",
                metadata=failure_result
            )
        
        # 注册规则
        self.anomaly_rule = anomaly_detection_rule
        self.cascading_rule = cascading_failure_rule
```

#### 6.2 恢复机制
```python
# src/recovery/recovery_manager.py
import asyncio
import networkx as nx
from typing import Dict, List, Set, Optional
from dataclasses import dataclass
from enum import Enum
import logging

class RecoveryAction(Enum):
    REROUTE_TRAFFIC = "reroute_traffic"
    INCREASE_CAPACITY = "increase_capacity"
    ISOLATE_NODE = "isolate_node"
    ACTIVATE_BACKUP = "activate_backup"
    LOAD_BALANCING = "load_balancing"

@dataclass
class RecoveryPlan:
    plan_id: str
    target_nodes: List[str]
    actions: List[RecoveryAction]
    priority: int
    estimated_recovery_time: int  # 秒
    success_probability: float

class NetworkRecoveryManager:
    def __init__(self, network: nx.Graph):
        self.network = network
        self.recovery_strategies = {}
        self.active_recovery_plans = {}
        self.recovery_history = []
        self.logger = logging.getLogger(__name__)
        
        self._initialize_recovery_strategies()
    
    def _initialize_recovery_strategies(self):
        """初始化恢复策略"""
        self.recovery_strategies = {
            'traffic_rerouting': self._traffic_rerouting_strategy,
            'capacity_scaling': self._capacity_scaling_strategy,
            'node_isolation': self._node_isolation_strategy,
            'backup_activation': self._backup_activation_strategy,
            'load_redistribution': self._load_redistribution_strategy
        }
    
    async def generate_recovery_plan(self, failed_nodes: Set[str], 
                                   network_state: Dict) -> RecoveryPlan:
        """生成恢复计划"""
        # 分析网络状态
        impact_analysis = self._analyze_failure_impact(failed_nodes, network_state)
        
        # 选择恢复策略
        selected_actions = self._select_recovery_actions(impact_analysis)
        
        # 创建恢复计划
        plan = RecoveryPlan(
            plan_id=f"recovery_{int(asyncio.get_event_loop().time())}",
            target_nodes=list(failed_nodes),
            actions=selected_actions,
            priority=self._calculate_priority(impact_analysis),
            estimated_recovery_time=self._estimate_recovery_time(selected_actions),
            success_probability=self._estimate_success_probability(selected_actions)
        )
        
        return plan
    
    def _analyze_failure_impact(self, failed_nodes: Set[str], 
                               network_state: Dict) -> Dict:
        """分析失效影响"""
        # 创建失效后的网络副本
        recovery_network = self.network.copy()
        recovery_network.remove_nodes_from(failed_nodes)
        
        # 计算连通性影响
        connected_components = list(nx.connected_components(recovery_network))
        largest_component_size = max(len(comp) for comp in connected_components) if connected_components else 0
        
        # 计算路径长度变化
        avg_path_length_before = network_state.get('avg_path_length', 0)
        try:
            avg_path_length_after = nx.average_shortest_path_length(recovery_network)
        except:
            avg_path_length_after = float('inf')
        
        # 计算负载重分布影响
        affected_neighbors = set()
        for failed_node in failed_nodes:
            if failed_node in self.network:
                affected_neighbors.update(self.network.neighbors(failed_node))
        
        impact_analysis = {
            'failed_nodes': failed_nodes,
            'affected_neighbors': affected_neighbors,
            'connectivity_loss': 1 - (largest_component_size / len(self.network)),
            'path_length_increase': avg_path_length_after - avg_path_length_before,
            'isolated_components': len(connected_components) - 1,
            'critical_paths_affected': self._count_critical_paths_affected(failed_nodes)
        }
        
        return impact_analysis
    
    def _select_recovery_actions(self, impact_analysis: Dict) -> List[RecoveryAction]:
        """选择恢复动作"""
        actions = []
        
        # 基于影响分析选择恢复动作
        connectivity_loss = impact_analysis['connectivity_loss']
        path_length_increase = impact_analysis['path_length_increase']
        
        if connectivity_loss > 0.3:
            actions.append(RecoveryAction.ACTIVATE_BACKUP)
            actions.append(RecoveryAction.REROUTE_TRAFFIC)
        elif connectivity_loss > 0.1:
            actions.append(RecoveryAction.REROUTE_TRAFFIC)
            actions.append(RecoveryAction.LOAD_BALANCING)
        
        if path_length_increase > 2:
            actions.append(RecoveryAction.INCREASE_CAPACITY)
        
        if impact_analysis['isolated_components'] > 2:
            actions.append(RecoveryAction.ISOLATE_NODE)
        
        return actions if actions else [RecoveryAction.LOAD_BALANCING]
    
    async def execute_recovery_plan(self, plan: RecoveryPlan) -> Dict:
        """执行恢复计划"""
        self.active_recovery_plans[plan.plan_id] = plan
        execution_results = []
        
        self.logger.info(f"开始执行恢复计划 {plan.plan_id}")
        
        for action in plan.actions:
            try:
                result = await self._execute_recovery_action(action, plan.target_nodes)
                execution_results.append(result)
                
                # 验证恢复效果
                if result['success']:
                    self.logger.info(f"恢复动作 {action.value} 执行成功")
                else:
                    self.logger.warning(f"恢复动作 {action.value} 执行失败: {result.get('error')}")
                
            except Exception as e:
                self.logger.error(f"执行恢复动作 {action.value} 时发生错误: {e}")
                execution_results.append({
                    'action': action.value,
                    'success': False,
                    'error': str(e)
                })
        
        # 计算整体恢复成功率
        successful_actions = sum(1 for result in execution_results if result['success'])
        overall_success = successful_actions / len(execution_results) if execution_results else 0
        
        recovery_result = {
            'plan_id': plan.plan_id,
            'overall_success': overall_success,
            'execution_results': execution_results,
            'recovery_time': sum(result.get('execution_time', 0) for result in execution_results)
        }
        
        # 记录恢复历史
        self.recovery_history.append(recovery_result)
        
        # 从活跃计划中移除
        if plan.plan_id in self.active_recovery_plans:
            del self.active_recovery_plans[plan.plan_id]
        
        return recovery_result
    
    async def _execute_recovery_action(self, action: RecoveryAction, 
                                     target_nodes: List[str]) -> Dict:
        """执行具体的恢复动作"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            if action in self.recovery_strategies:
                success = await self.recovery_strategies[action.value.replace('_', '_') + '_strategy'](target_nodes)
            else:
                success = False
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return {
                'action': action.value,
                'success': success,
                'execution_time': execution_time
            }
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return {
                'action': action.value,
                'success': False,
                'execution_time': execution_time,
                'error': str(e)
            }
    
    async def _traffic_rerouting_strategy(self, target_nodes: List[str]) -> bool:
        """流量重路由策略"""
        # 实现流量重路由逻辑
        await asyncio.sleep(2)  # 模拟执行时间
        self.logger.info(f"执行流量重路由，目标节点: {target_nodes}")
        return True
    
    async def _capacity_scaling_strategy(self, target_nodes: List[str]) -> bool:
        """容量扩展策略"""
        await asyncio.sleep(3)
        self.logger.info(f"执行容量扩展，目标节点: {target_nodes}")
        return True
    
    async def _node_isolation_strategy(self, target_nodes: List[str]) -> bool:
        """节点隔离策略"""
        await asyncio.sleep(1)
        self.logger.info(f"执行节点隔离，目标节点: {target_nodes}")
        return True
    
    async def _backup_activation_strategy(self, target_nodes: List[str]) -> bool:
        """备份激活策略"""
        await asyncio.sleep(5)
        self.logger.info(f"执行备份激活，目标节点: {target_nodes}")
        return True
    
    async def _load_redistribution_strategy(self, target_nodes: List[str]) -> bool:
        """负载重分布策略"""
        await asyncio.sleep(2)
        self.logger.info(f"执行负载重分布，目标节点: {target_nodes}")
        return True
    
    def _calculate_priority(self, impact_analysis: Dict) -> int:
        """计算恢复计划优先级"""
        connectivity_loss = impact_analysis['connectivity_loss']
        isolated_components = impact_analysis['isolated_components']
        
        if connectivity_loss > 0.5 or isolated_components > 5:
            return 1  # 最高优先级
        elif connectivity_loss > 0.2 or isolated_components > 2:
            return 2  # 高优先级
        elif connectivity_loss > 0.1:
            return 3  # 中优先级
        else:
            return 4  # 低优先级
    
    def _estimate_recovery_time(self, actions: List[RecoveryAction]) -> int:
        """估算恢复时间"""
        time_estimates = {
            RecoveryAction.REROUTE_TRAFFIC: 30,
            RecoveryAction.INCREASE_CAPACITY: 60,
            RecoveryAction.ISOLATE_NODE: 10,
            RecoveryAction.ACTIVATE_BACKUP: 120,
            RecoveryAction.LOAD_BALANCING: 20
        }
        
        return sum(time_estimates.get(action, 60) for action in actions)
    
    def _estimate_success_probability(self, actions: List[RecoveryAction]) -> float:
        """估算成功概率"""
        success_rates = {
            RecoveryAction.REROUTE_TRAFFIC: 0.9,
            RecoveryAction.INCREASE_CAPACITY: 0.85,
            RecoveryAction.ISOLATE_NODE: 0.95,
            RecoveryAction.ACTIVATE_BACKUP: 0.8,
            RecoveryAction.LOAD_BALANCING: 0.9
        }
        
        # 计算联合成功概率
        combined_probability = 1.0
        for action in actions:
            combined_probability *= success_rates.get(action, 0.7)
        
        return combined_probability
    
    def _count_critical_paths_affected(self, failed_nodes: Set[str]) -> int:
        """计算受影响的关键路径数量"""
        # 简化实现，实际可根据网络拓扑特征计算
        return len(failed_nodes) * 2
```

### Phase 7: 可视化界面 (第15-16周)

#### 7.1 实时监控仪表板
```python
# src/visualization/dashboard.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
import asyncio

class NetworkMonitoringDashboard:
    def __init__(self):
        self.setup_page_config()
        
    def setup_page_config(self):
        """设置页面配置"""
        st.set_page_config(
            page_title="网络异常检测与级联失效分析系统",
            page_icon="🕸️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def render_dashboard(self, network_data: dict, 
                        anomaly_data: dict, 
                        cascade_data: dict):
        """渲染主仪表板"""
        st.title("🕸️ 复杂网络异常行为检测与级联失效分析系统")
        
        # 侧边栏配置
        self.render_sidebar()
        
        # 主要内容区域
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 网络拓扑可视化
            self.render_network_topology(network_data)
            
            # 异常检测结果
            self.render_anomaly_detection(anomaly_data)
        
        with col2:
            # 实时指标
            self.render_real_time_metrics(network_data)
            
            # 告警信息
            self.render_alerts()
        
        # 底部分析区域
        st.markdown("---")
        
        tab1, tab2, tab3 = st.tabs(["级联失效分析", "可解释性分析", "恢复建议"])
        
        with tab1:
            self.render_cascading_failure_analysis(cascade_data)
        
        with tab2:
            self.render_explainability_analysis(anomaly_data)
        
        with tab3:
            self.render_recovery_recommendations()
    
    def render_sidebar(self):
        """渲染侧边栏"""
        with st.sidebar:
            st.header("⚙️ 系统配置")
            
            # 检测参数配置
            st.subheader("检测参数")
            sensitivity = st.slider("检测敏感度", 0.1, 1.0, 0.7, 0.1)
            window_size = st.selectbox("时间窗口", [5, 10, 15, 30], index=1)
            
            # 可视化配置
            st.subheader("可视化设置")
            update_interval = st.selectbox("更新频率(秒)", [1, 5, 10, 30], index=1)
            show_labels = st.checkbox("显示节点标签", True)
            
            # 导出功能
            st.subheader("数据导出")
            if st.button("导出报告"):
                self.export_report()
    
    def render_network_topology(self, network_data: dict):
        """渲染网络拓扑图"""
        st.subheader("🔗 网络拓扑结构")
        
        # 创建网络图
        if 'graph' in network_data:
            G = network_data['graph']
            
            # 计算节点位置
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # 准备节点数据
            node_x = [pos[node][0] for node in G.nodes()]
            node_y = [pos[node][1] for node in G.nodes()]
            
            # 节点颜色基于状态
            node_colors = []
            node_text = []
            for node in G.nodes():
                status = network_data.get('node_status', {}).get(node, 'normal')
                if status == 'failed':
                    node_colors.append('red')
                elif status == 'anomaly':
                    node_colors.append('orange')
                elif status == 'warning':
                    node_colors.append('yellow')
                else:
                    node_colors.append('lightblue')
                
                # 节点信息
                degree = G.degree(node)
                node_text.append(f"节点: {node}<br>度数: {degree}<br>状态: {status}")
            
            # 准备边数据
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            # 创建图形
            fig = go.Figure()
            
            # 添加边
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, color='gray'),
                hoverinfo='none',
                mode='lines',
                name='连接'
            ))
            
            # 添加节点
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                marker=dict(
                    size=20,
                    color=node_colors,
                    line=dict(width=2, color='darkslategray')
                ),
                text=[str(node) for node in G.nodes()],
                textposition="middle center",
                hovertext=node_text,
                hoverinfo='text',
                name='节点'
            ))
            
            # 更新布局
            fig.update_layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[
                    dict(
                        text="网络拓扑可视化",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor='left', yanchor='bottom',
                        font=dict(color='gray', size=12)
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_anomaly_detection(self, anomaly_data: dict):
        """渲染异常检测结果"""
        st.subheader("🚨 异常检测结果")
        
        if 'timeseries' in anomaly_data:
            # 时序异常检测图
            df = pd.DataFrame(anomaly_data['timeseries'])
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('网络流量趋势', '异常检测结果'),
                shared_xaxis=True
            )
            
            # 流量趋势
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['traffic'],
                    mode='lines',
                    name='正常流量',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
            # 异常点
            anomaly_points = df[df['is_anomaly'] == True]
            fig.add_trace(
                go.Scatter(
                    x=anomaly_points['timestamp'],
                    y=anomaly_points['traffic'],
                    mode='markers',
                    name='异常点',
                    marker=dict(color='red', size=8)
                ),
                row=1, col=1
            )
            
            # 异常分数
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['anomaly_score'],
                    mode='lines',
                    name='异常分数',
                    line=dict(color='orange')
                ),
                row=2, col=1
            )
            
            # 阈值线
            fig.add_hline(
                y=0.5, line_dash="dash", line_color="red",
                annotation_text="异常阈值",
                row=2, col=1
            )
            
            fig.update_layout(height=400, title_text="异常检测时序分析")
            st.plotly_chart(fig, use_container_width=True)
    
    def render_real_time_metrics(self, network_data: dict):
        """渲染实时指标"""
        st.subheader("📊 实时网络指标")
        
        # 指标卡片
        metrics = network_data.get('metrics', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="网络健康度",
                value=f"{metrics.get('health_score', 0):.1%}",
                delta=f"{metrics.get('health_delta', 0):+.1%}"
            )
            
            st.metric(
                label="活跃节点",
                value=metrics.get('active_nodes', 0),
                delta=metrics.get('node_delta', 0)
            )
        
        with col2:
            st.metric(
                label="平均延迟",
                value=f"{metrics.get('avg_latency', 0):.2f}ms",
                delta=f"{metrics.get('latency_delta', 0):+.2f}ms"
            )
            
            st.metric(
                label="吞吐量",
                value=f"{metrics.get('throughput', 0):.1f}Mbps",
                delta=f"{metrics.get('throughput_delta', 0):+.1f}Mbps"
            )
        
        # 历史趋势图
        if 'history' in network_data:
            history_df = pd.DataFrame(network_data['history'])
            
            fig = px.line(
                history_df, 
                x='timestamp', 
                y=['health_score', 'avg_latency', 'throughput'],
                title="历史趋势"
            )
            
            fig.update_layout(height=200)
            st.plotly_chart(fig, use_container_width=True)
    
    def render_alerts(self):
        """渲染告警信息"""
        st.subheader("⚠️ 系统告警")
        
        # 模拟告警数据
        alerts = [
            {
                'level': 'CRITICAL',
                'message': '检测到级联失效，影响30%节点',
                'time': datetime.now() - timedelta(minutes=5),
                'source': '级联失效分析器'
            },
            {
                'level': 'WARNING',
                'message': '节点Node_15异常流量激增',
                'time': datetime.now() - timedelta(minutes=15),
                'source': '异常检测器'
            },
            {
                'level': 'INFO',
                'message': '系统启动自动恢复程序',
                'time': datetime.now() - timedelta(minutes=20),
                'source': '恢复管理器'
            }
        ]
        
        for alert in alerts:
            if alert['level'] == 'CRITICAL':
                st.error(f"🔴 {alert['message']}")
            elif alert['level'] == 'WARNING':
                st.warning(f"🟡 {alert['message']}")
            else:
                st.info(f"🔵 {alert['message']}")
            
            st.caption(f"来源: {alert['source']} | 时间: {alert['time'].strftime('%H:%M:%S')}")
    
    def render_cascading_failure_analysis(self, cascade_data: dict):
        """渲染级联失效分析"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("级联失效传播路径")
            
            if 'failure_sequence' in cascade_data:
                sequence_df = pd.DataFrame(cascade_data['failure_sequence'])
                
                fig = px.line(
                    sequence_df,
                    x='iteration',
                    y='total_failures',
                    title='失效节点数量随时间变化'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("关键节点分析")
            
            if 'critical_nodes' in cascade_data:
                critical_df = pd.DataFrame(
                    cascade_data['critical_nodes']
                ).head(10)
                
                fig = px.bar(
                    critical_df,
                    x='node',
                    y='criticality_score',
                    title='节点关键性排名'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def render_explainability_analysis(self, anomaly_data: dict):
        """渲染可解释性分析"""
        st.subheader("可解释性分析")
        
        if 'explanation' in anomaly_data:
            explanation = anomaly_data['explanation']
            
            # 特征重要性
            col1, col2 = st.columns(2)
            
            with col1:
                if 'feature_importance' in explanation:
                    importance_df = pd.DataFrame([
                        {'feature': k, 'importance': abs(v)}
                        for k, v in explanation['feature_importance'].items()
                    ]).sort_values('importance', ascending=False).head(10)
                    
                    fig = px.bar(
                        importance_df,
                        x='importance',
                        y='feature',
                        orientation='h',
                        title='特征重要性'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'top_contributing_features' in explanation:
                    st.subheader("主要贡献特征")
                    
                    for feature in explanation['top_contributing_features'][:5]:
                        contribution = "增加" if feature['contribution'] == 'positive' else "降低"
                        st.write(f"• **{feature['feature_name']}**: {contribution}异常概率")
                        st.write(f"  贡献值: {feature['shap_value']:.3f}")
    
    def render_recovery_recommendations(self):
        """渲染恢复建议"""
        st.subheader("🔧 恢复建议")
        
        # 模拟恢复建议
        recommendations = [
            {
                'priority': 1,
                'action': '启动备份节点',
                'description': '激活备份节点Node_Backup_01以替代失效节点',
                'estimated_time': '5分钟',
                'success_rate': '85%'
            },
            {
                'priority': 2,
                'action': '流量重路由',
                'description': '将受影响路径的流量重新路由到可用路径',
                'estimated_time': '2分钟',
                'success_rate': '90%'
            },
            {
                'priority': 3,
                'action': '负载重分布',
                'description': '在剩余活跃节点间重新分配负载',
                'estimated_time': '3分钟',
                'success_rate': '95%'
            }
        ]
        
        for rec in recommendations:
            with st.expander(f"优先级 {rec['priority']}: {rec['action']}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**预计时间**: {rec['estimated_time']}")
                
                with col2:
                    st.write(f"**成功率**: {rec['success_rate']}")
                
                with col3:
                    if st.button(f"执行", key=f"exec_{rec['priority']}"):
                        st.success("恢复操作已启动！")
                
                st.write(f"**描述**: {rec['description']}")
    
    def export_report(self):
        """导出分析报告"""
        st.success("报告导出功能开发中...")

# 主应用入口
def main():
    dashboard = NetworkMonitoringDashboard()
    
    # 模拟数据
    network_data = {
        'graph': nx.karate_club_graph(),
        'metrics': {
            'health_score': 0.87,
            'health_delta': -0.05,
            'active_nodes': 32,
            'node_delta': -2,
            'avg_latency': 45.2,
            'latency_delta': 5.3,
            'throughput': 156.7,
            'throughput_delta': -12.4
        }
    }
    
    anomaly_data = {
        'timeseries': [
            {
                'timestamp': datetime.now() - timedelta(minutes=i),
                'traffic': 100 + np.random.normal(0, 10) + (50 if i < 20 and i > 10 else 0),
                'is_anomaly': i < 20 and i > 10,
                'anomaly_score': 0.8 if i < 20 and i > 10 else np.random.uniform(0.1, 0.4)
            }
            for i in range(60, 0, -1)
        ]
    }
    
    cascade_data = {
        'failure_sequence': [
            {'iteration': i, 'total_failures': min(i*2, 15)}
            for i in range(0, 8)
        ]
    }
    
    dashboard.render_dashboard(network_data, anomaly_data, cascade_data)

if __name__ == "__main__":
    main()
```

## 实施时间表

### 第1-2周：环境搭建
- [ ] 项目结构创建
- [ ] 依赖安装配置
- [ ] 基础框架搭建
- [ ] 数据库环境配置

### 第3-4周：数据层实现
- [ ] 网络数据采集器开发
- [ ] 拓扑发现模块实现
- [ ] 数据预处理管道构建
- [ ] 存储系统集成

### 第5-7周：核心算法
- [ ] GNN异常检测模型训练
- [ ] 时序异常检测实现
- [ ] 集成学习方法优化
- [ ] 模型性能调优

### 第8-10周：级联失效分析
- [ ] 级联失效模型实现
- [ ] 传播路径分析算法
- [ ] 关键节点识别
- [ ] 失效预测优化

### 第11-12周：可解释性
- [ ] SHAP集成开发
- [ ] 解释性仪表板
- [ ] 特征重要性分析
- [ ] 决策路径可视化

### 第13-14周：告警与恢复
- [ ] 智能告警系统
- [ ] 恢复策略实现
- [ ] 自动化恢复机制
- [ ] 通知系统集成

### 第15-16周：可视化与集成
- [ ] 实时监控界面
- [ ] 交互式分析工具
- [ ] 系统集成测试
- [ ] 性能优化调整

## 技术要点总结

1. **模块化设计**：每个功能模块独立开发，便于维护和扩展
2. **异步编程**：使用异步框架提高系统并发性能
3. **可扩展架构**：支持新算法和功能的快速集成
4. **实时处理**：支持流式数据处理和实时分析
5. **可解释性**：集成多种解释方法，提供直观的分析结果

这套实现方案提供了完整的技术路线和代码框架，可以根据具体需求进行调整和优化。每个阶段都有明确的交付目标，便于项目管理和进度跟踪。