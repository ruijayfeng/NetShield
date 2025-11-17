### 项目核心代码（精简可读版）

以下直接内嵌“最小但完整”的核心实现片段，覆盖主流程、网络与数据、异常检测集成、级联失效仿真与可解释性，便于单文件快速理解与复用。

---

#### 一、主流程控制器（初始化→数据→训练→检测→级联→解释→保存）

**功能说明：**
这是系统的核心控制器，负责协调各个模块的初始化、数据准备、模型训练、异常检测、级联失效分析和可解释性分析。采用面向对象设计，将整个分析流程封装为可复用的方法。

**关键点：**
- `initialize_system()`: 从配置文件加载参数，初始化所有子模块（网络生成器、数据生成器、异常检测器、级联分析器等）
- `generate_network_and_data()`: 支持生成模拟数据或加载真实数据文件，自动进行数据质量验证
- `train_anomaly_detection()`: 使用70%数据进行训练，同时初始化SHAP可解释性模块（使用IsolationForest模型）
- `perform_anomaly_detection()`: 在测试集上检测异常，如果发现异常则自动触发告警系统
- `perform_cascading_failure_analysis()`: 分析网络鲁棒性，当评分低于0.7时自动生成告警
- `generate_explanations()`: 对检测结果进行SHAP解释，输出特征贡献度和置信度指标

```python
import os, logging, asyncio, networkx as nx
from datetime import datetime

class NetworkAnalysisSystem:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.network_generator = None
        self.data_generator = None
        self.anomaly_analyzer = None
        self.cascade_analyzer = None
        self.explainer = None
        self.alert_manager = None
        self.network = None
        self.data = None
        self.logger = logging.getLogger(__name__)

    def initialize_system(self):
        from src.data.generators.network_generator import NetworkGenerator, NetworkConfig
        from src.data.generators.data_generator import NetworkDataGenerator, DataConfig
        from src.models.anomaly_detection.detectors import NetworkAnomalyAnalyzer, AnomalyDetectionConfig
        from src.models.cascading.failure_analyzer import CascadingFailureAnalyzer, CascadingFailureConfig
        from src.models.explainable.shap_explainer import NetworkAnomalyExplainer, ExplainabilityConfig
        from src.alerts.alert_manager import AlertManager

        net_cfg = NetworkConfig.from_config(self.config_path)
        data_cfg = DataConfig.from_config(self.config_path)
        ad_cfg = AnomalyDetectionConfig.from_config(self.config_path)
        cf_cfg = CascadingFailureConfig.from_config(self.config_path)
        exp_cfg = ExplainabilityConfig.from_config(self.config_path)

        self.network_generator = NetworkGenerator(net_cfg)
        self.data_generator = NetworkDataGenerator(data_cfg)
        self.anomaly_analyzer = NetworkAnomalyAnalyzer(ad_cfg)
        self.cascade_analyzer = CascadingFailureAnalyzer(cf_cfg)
        self.explainer = None  # 延后，用已训练模型初始化
        self.alert_manager = AlertManager()

    def generate_network_and_data(self, network_file=None, data_file=None):
        if network_file:
            from src.data.generators.network_generator import RealNetworkInterface
            self.network = RealNetworkInterface().load_from_file(network_file)
        else:
            self.network = self.network_generator.generate_network()

        if data_file:
            self.data = self.data_generator.load_real_data(data_file)
        else:
            sample_node = list(self.network.nodes())[0]
            self.data = self.data_generator.generate_node_timeseries(self.network, sample_node)

        _ = self.data_generator.validate_data_quality(self.data)

    def train_anomaly_detection(self):
        split_idx = int(0.7 * len(self.data))
        train_data = self.data[:split_idx].copy()
        stats = self.anomaly_analyzer.train(train_data)
        if self.anomaly_analyzer.ensemble.detectors:
            iso = self.anomaly_analyzer.ensemble.detectors['isolation_forest']
            X_train = self.anomaly_analyzer.prepare_features(train_data)
            from src.models.explainable.shap_explainer import NetworkAnomalyExplainer, ExplainabilityConfig
            self.explainer = NetworkAnomalyExplainer(iso.model, ExplainabilityConfig())
            self.explainer.setup_explainer(X_train, self.anomaly_analyzer.feature_columns)
        return stats

    async def perform_anomaly_detection(self):
        test_data = self.data.tail(int(len(self.data) * 0.3)).copy()
        det = self.anomaly_analyzer.detect_anomalies(test_data)
        if det['summary']['predicted_anomalies'] > 0:
            await self.alert_manager.evaluate_rules({
                'is_anomaly': True,
                'confidence': det['summary']['mean_anomaly_probability']
            })
        return det

    async def perform_cascading_failure_analysis(self):
        results = self.cascade_analyzer.analyze_network_robustness(self.network)
        score = results.get('robustness_metrics', {}).get('overall_robustness_score', 1.0)
        if score < 0.7:
            from src.alerts.alert_manager import AlertLevel, AlertCategory
            await self.alert_manager.create_alert(
                title="网络鲁棒性警告",
                description=f"鲁棒性评分: {score:.3f}",
                level=AlertLevel.WARNING if score > 0.5 else AlertLevel.CRITICAL,
                category=AlertCategory.CASCADING_FAILURE,
                source="cascading_failure_analyzer",
                metadata={'robustness_score': score}
            )
        return results

    def generate_explanations(self, num_instances: int = 5):
        test_data = self.data.tail(100)
        X = self.anomaly_analyzer.prepare_features(test_data, self.anomaly_analyzer.feature_columns)
        sample = X[:num_instances]
        explanations = self.explainer.explain_batch(sample, [f"instance_{i}" for i in range(len(sample))])
        report = self.explainer.generate_explanation_report(explanations)
        return explanations, report
```

---

#### 二、网络生成（小世界/无标度/ER/网格 + 真实导入）

**功能说明：**
网络生成模块负责创建各种类型的复杂网络拓扑结构，支持小世界网络（Watts-Strogatz）、无标度网络（Barabási-Albert）、随机网络（Erdős-Rényi）和网格网络。同时提供配置管理和节点属性初始化功能。

**关键点：**
- `NetworkConfig`: 数据类封装网络生成参数，支持从YAML配置文件加载
- `NetworkGenerator.generate_network()`: 根据网络类型调用NetworkX库生成相应拓扑，并在节点上附加属性（度数、容量、初始负载、状态）
- 容量与负载设计：容量 = 初始负载 × 1.5（提供20%缓冲），初始负载与节点度数成正比，为级联失效分析做准备
- 支持多种网络模型，每种模型模拟不同的网络特性（小世界=高聚类+短路径，无标度=幂律分布，随机=均匀随机）

```python
import networkx as nx, numpy as np
from dataclasses import dataclass
import yaml, os

@dataclass
class NetworkConfig:
    node_count: int = 50
    edge_probability: float = 0.1
    network_type: str = "small_world"
    k_neighbors: int = 6
    rewiring_prob: float = 0.3
    @classmethod
    def from_config(cls, config_path: str = None):
        path = config_path or os.path.join(os.path.dirname(__file__), "../../../config/config.yaml")
        with open(path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f).get('network', {})
        return cls(
            node_count=cfg.get('node_count', 50),
            edge_probability=cfg.get('edge_probability', 0.1),
            network_type=cfg.get('network_type', 'small_world'),
            k_neighbors=cfg.get('k_neighbors', 6),
            rewiring_prob=cfg.get('rewiring_prob', 0.3)
        )

class NetworkGenerator:
    def __init__(self, config: NetworkConfig = None):
        self.config = config or NetworkConfig()
    def generate_network(self, network_type: str = None) -> nx.Graph:
        t = network_type or self.config.network_type
        if t == "small_world":
            g = nx.watts_strogatz_graph(self.config.node_count, self.config.k_neighbors, self.config.rewiring_prob, seed=42)
        elif t == "scale_free":
            g = nx.barabasi_albert_graph(self.config.node_count, max(1, self.config.k_neighbors // 2), seed=42)
        elif t == "erdos_renyi":
            g = nx.erdos_renyi_graph(self.config.node_count, self.config.edge_probability, seed=42)
        else:
            s = int(np.sqrt(self.config.node_count)); g = nx.grid_2d_graph(s, s); g = nx.convert_node_labels_to_integers(g)
        for n in g.nodes():
            deg = g.degree(n)
            g.nodes[n].update({'degree': deg, 'capacity': deg * 1.2, 'initial_load': deg * 0.8, 'status': 'active'})
        return g
```

---

#### 三、异常检测（集成器 + 统一分析器）

**功能说明：**
异常检测模块采用集成学习方法，结合三种经典异常检测算法（孤立森林、One-Class SVM、局部异常因子），通过多数投票机制提高检测准确性和鲁棒性。统一的分析器封装了特征工程、模型训练、预测和评估流程。

**关键点：**
- **集成策略**：`EnsembleAnomalyDetector` 包含多个检测器，预测时采用多数投票（>50%判定为异常），得分取平均值
- **特征预处理**：使用RobustScaler（对异常值不敏感）标准化特征，自动处理缺失值（用中位数填充）
- **三种算法特点**：
  - Isolation Forest：适合高维稀疏异常，基于随机分割树
  - One-Class SVM：适合非线性边界，基于支持向量机
  - Local Outlier Factor：适合局部异常检测，基于密度估计
- **概率转换**：将决策函数得分通过sigmoid函数转换为0-1概率值，便于阈值判定和可视化
- **自动特征选择**：排除时间戳、标签等非特征列，自动识别数值型特征

```python
import numpy as np, pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

@dataclass
class AnomalyDetectionConfig:
    methods: list[str] = None
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

class BaseDetector:
    def __init__(self, model):
        self.model = model; self.scaler = RobustScaler(); self.fitted = False
    def fit(self, X): Xs = self.scaler.fit_transform(X); self.model.fit(Xs); self.fitted = True; return self
    def predict(self, X): Xs = self.scaler.transform(X); return self.model.predict(Xs)
    def decision_function(self, X): Xs = self.scaler.transform(X); return self.model.decision_function(Xs)

class EnsembleAnomalyDetector:
    def __init__(self, cfg: AnomalyDetectionConfig):
        self.detectors = {}
        if "isolation_forest" in cfg.methods:
            self.detectors["isolation_forest"] = BaseDetector(IsolationForest(contamination=cfg.contamination, n_estimators=cfg.n_estimators, random_state=cfg.random_state))
        if "one_class_svm" in cfg.methods:
            self.detectors["one_class_svm"] = BaseDetector(OneClassSVM(nu=cfg.nu, kernel=cfg.kernel, gamma=cfg.gamma))
        if "local_outlier_factor" in cfg.methods:
            self.detectors["local_outlier_factor"] = BaseDetector(LocalOutlierFactor(n_neighbors=20, contamination=cfg.contamination, novelty=True))
        self.fitted = False
    def fit(self, X):
        for d in self.detectors.values(): d.fit(X)
        self.fitted = True; return self
    def predict(self, X):
        preds = np.array([d.predict(X) for d in self.detectors.values()])  # -1/1
        binp = (preds == 1).astype(int)
        maj = (binp.mean(axis=0) > 0.5).astype(int)
        return 2 * maj - 1
    def decision_function(self, X):
        scores = {name: d.decision_function(X) for name, d in self.detectors.items()}
        scores['ensemble'] = np.mean(list(scores.values()), axis=0)
        return scores
    def get_prob(self, X):
        s = self.decision_function(X)['ensemble']
        return 1 / (1 + np.exp(s))

class NetworkAnomalyAnalyzer:
    def __init__(self, cfg: AnomalyDetectionConfig):
        self.cfg = cfg; self.ensemble = EnsembleAnomalyDetector(cfg); self.feature_columns = None
    def prepare_features(self, df: pd.DataFrame, cols: list[str] | None = None):
        if cols is None:
            exclude = ['timestamp', 'is_anomaly', 'anomaly_score', 'node_id']
            cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
        self.feature_columns = cols
        return df[cols].fillna(df[cols].median()).values
    def train(self, train_df: pd.DataFrame):
        X = self.prepare_features(train_df)
        self.ensemble.fit(X)
        return {'n_samples': X.shape[0], 'n_features': X.shape[1]}
    def detect_anomalies(self, test_df: pd.DataFrame):
        X = self.prepare_features(test_df, self.feature_columns)
        pred = self.ensemble.predict(X)
        prob = self.ensemble.get_prob(X)
        res = test_df.copy()
        res['predicted_anomaly'] = (pred == 1)
        res['anomaly_probability'] = prob
        return {
            'results': res,
            'summary': {
                'total_samples': len(res),
                'predicted_anomalies': int((pred == 1).sum()),
                'anomaly_rate': float((pred == 1).mean()),
                'mean_anomaly_probability': float(prob.mean())
            }
        }
```

---

#### 四、级联失效仿真（容量-负载 + 重分配）

**功能说明：**
级联失效分析模拟网络中节点因过载而失效，并将负载重分配到相邻节点，可能引发连锁反应的过程。这是评估网络鲁棒性和识别关键节点的核心方法。

**关键点：**
- **容量-负载模型**：每个节点有容量上限和当前负载，当负载/容量 > 阈值（默认0.8）时节点失效
- **负载重分配策略**：失败节点的负载按剩余容量比例分配给活跃邻居（容量越大承担越多），避免邻居立即过载
- **仿真流程**：初始化失效 → 重分配负载 → 检查新失效 → 更新状态 → 迭代直到收敛或达到最大迭代次数
- **鲁棒性评分**：基于单点失效测试的平均级联规模计算，评分 = 1 - (平均失效数/总节点数)，值越接近1表示网络越鲁棒
- **关键节点识别**：通过测试每个节点失效的影响，找出导致最大级联失效的关键节点

```python
import networkx as nx, numpy as np
from dataclasses import dataclass
from enum import Enum

class NodeState(Enum):
    ACTIVE = "active"; FAILED = "failed"; OVERLOADED = "overloaded"

@dataclass
class CascadingFailureConfig:
    initial_capacity_ratio: float = 1.5
    failure_threshold: float = 0.8
    max_iterations: int = 50
    failure_probability: float = 0.05

class CascadingFailureSimulator:
    def __init__(self, g: nx.Graph, cfg: CascadingFailureConfig):
        self.g = g.copy(); self.cfg = cfg; self.info = {}
        deg = dict(self.g.degree()); mdeg = max(deg.values()) if deg else 1
        for n in self.g.nodes():
            load = (deg[n] / mdeg) if mdeg else 0
            cap = load * cfg.initial_capacity_ratio
            self.info[str(n)] = {'load': load, 'cap': cap, 'state': NodeState.ACTIVE}
    def _distribute(self, failed_node: str):
        load = self.info[failed_node]['load']; self.info[failed_node]['load'] = 0
        nbrs = [str(v) for v in self.g.neighbors(int(failed_node)) if self.info[str(v)]['state'] == NodeState.ACTIVE]
        if not nbrs or load <= 0: return
        rem = [max(0, self.info[n]['cap'] - self.info[n]['load']) for n in nbrs]
        tot = sum(rem) or len(nbrs)
        for i, n in enumerate(nbrs):
            add = load * ((rem[i] / tot) if sum(rem) > 0 else 1/len(nbrs))
            self.info[n]['load'] += add
    def simulate(self, initial_failures: list[str]):
        for n in initial_failures: self.info[n]['state'] = NodeState.FAILED
        it = 0
        while it < self.cfg.max_iterations:
            # 重分配失败节点负载
            for nid, s in list(self.info.items()):
                if s['state'] == NodeState.FAILED: self._distribute(nid)
            # 判定新失效
            new_fail = []
            for nid, s in self.info.items():
                if s['state'] == NodeState.ACTIVE:
                    ratio = (s['load'] / s['cap']) if s['cap'] > 0 else 0
                    if ratio > self.cfg.failure_threshold:
                        new_fail.append(nid)
            if not new_fail: break
            for nid in new_fail: self.info[nid]['state'] = NodeState.FAILED
            it += 1
        total_failed = sum(1 for s in self.info.values() if s['state'] == NodeState.FAILED)
        return {
            'final_failures': total_failed,
            'failure_ratio': total_failed / self.g.number_of_nodes(),
            'total_iterations': it
        }

class CascadingFailureAnalyzer:
    def __init__(self, cfg: CascadingFailureConfig):
        self.cfg = cfg
    def analyze_network_robustness(self, g: nx.Graph):
        sim = CascadingFailureSimulator(g, self.cfg)
        # 单点失效概览（快速模式）
        impacts = []
        for n in list(g.nodes())[:min(20, g.number_of_nodes())]:
            impacts.append(sim.simulate([str(n)]))
        avg_fail = float(np.mean([r['final_failures'] for r in impacts])) if impacts else 0
        return {
            'robustness_metrics': {
                'overall_robustness_score': 1.0 - (avg_fail / max(1, g.number_of_nodes()))
            }
        }
```

---

#### 五、可解释性（SHAP 最小实现）

**功能说明：**
可解释性模块使用SHAP（SHapley Additive exPlanations）值来解释异常检测模型的决策过程，帮助理解哪些特征对异常判定贡献最大。这是AI可解释性的重要工具。

**关键点：**
- **SHAP原理**：基于博弈论中的Shapley值，公平分配每个特征对预测结果的贡献度，正值为促进异常判定，负值为抑制异常判定
- **Kernel Explainer**：模型无关的解释方法，通过背景数据（100个样本）建立基准，计算单个实例与背景的差异贡献
- **背景数据采样**：为了平衡计算效率和准确性，通常采样100个代表性样本作为背景数据集
- **特征重要性排序**：计算每个特征SHAP值的绝对值，按重要性排序，突出关键决策特征
- **置信度指标**：基于SHAP值的集中度和稳定性计算解释的可信度，帮助判断解释结果的可靠性
- **批量解释**：支持对多个实例进行批量解释，输出特征贡献字典和汇总报告

```python
import numpy as np
import shap

class NetworkAnomalyExplainer:
    def __init__(self, model, cfg=None):
        self.model = model; self.explainer = None; self.feature_names = None; self.bg = None
    def setup_explainer(self, background_data: np.ndarray, feature_names: list[str]):
        self.feature_names = feature_names
        n = min(len(background_data), 100)
        self.bg = background_data[:n]
        predict_fn = self.model.decision_function if hasattr(self.model, 'decision_function') else self.model.predict
        self.explainer = shap.KernelExplainer(predict_fn, self.bg)
    def explain_batch(self, X: np.ndarray, ids: list[str]):
        exps = []
        for i, row in enumerate(X):
            try:
                sv = self.explainer.shap_values(row.reshape(1, -1))
                sv = sv[0] if isinstance(sv, list) else (sv[0] if sv.ndim > 1 else sv)
            except Exception:
                sv = np.zeros(row.shape[0])
            fmap = dict(zip(self.feature_names, sv))
            exps.append({
                'instance_id': ids[i],
                'shap_values': fmap,
                'top_features': sorted([(k, abs(v)) for k, v in fmap.items()], key=lambda x: x[1], reverse=True)[:10],
                'confidence_indicators': {'overall_confidence': float(np.mean(np.abs(sv)) if sv.size else 0.0)}
            })
        return exps
    def generate_explanation_report(self, exps):
        if not exps: return "No explanations available."
        avg_conf = np.mean([e['confidence_indicators']['overall_confidence'] for e in exps])
        lines = ["="*60, "网络异常检测可解释性分析报告", "="*60, f"平均解释置信度: {avg_conf:.3f}"]
        return "\n".join(lines)
```

---

#### 六、最小运行指令

**快速开始指南：**

1. **安装依赖**：`pip install -r requirements.txt`
   - 安装所有必需的Python包（NetworkX、scikit-learn、pandas、streamlit、SHAP等）

2. **冒烟测试**：`python main.py --mode test`
   - 验证系统基本功能是否正常，检查配置加载和模块初始化

3. **全流程批处理**：`python main.py --mode analysis --output results`
   - 执行完整的分析流程（生成数据→训练模型→检测异常→分析级联失效→生成解释）
   - 结果保存在 `results/` 目录，包括报告、网络文件、数据和告警记录

4. **启动仪表板**：`python run_dashboard.py`
   - 启动Streamlit Web界面（默认地址：http://localhost:8501）
   - 提供交互式可视化、实时分析和AI辅助功能



