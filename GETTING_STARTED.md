# 🚀 快速开始指南

## 复杂网络异常行为检测与级联失效分析系统

> 一个基于机器学习的网络安全分析平台，支持实时监控、智能预警和可视化分析

---

## 📋 环境要求

### 系统要求
- **Python**: 3.9+ (推荐 3.11 或 3.12)
- **操作系统**: Windows 10+, Ubuntu 18.04+, macOS 10.15+
- **内存**: 4GB+ (推荐 8GB+)
- **硬盘**: 1GB+ 可用空间

### 推荐环境
```bash
# 查看Python版本
python --version

# 确保pip是最新版本
python -m pip install --upgrade pip
```

---

## ⚡ 一键安装

### Windows用户

1. **下载项目**
   ```cmd
   git clone <repository-url>
   cd fenxi
   ```

2. **运行安装脚本**
   ```cmd
   python install_deps.py
   ```

3. **启动系统**
   ```cmd
   python run_dashboard.py
   ```

### Linux/Mac用户

1. **下载项目**
   ```bash
   git clone <repository-url>
   cd fenxi
   ```

2. **创建虚拟环境** (推荐)
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # 或 venv\Scripts\activate  # Windows
   ```

3. **安装依赖**
   ```bash
   python install_deps.py
   ```

4. **启动系统**
   ```bash
   python run_dashboard.py
   ```

---

## 🎮 使用方式

### 方式一：Web仪表板 (推荐)

**启动仪表板**
```bash
python run_dashboard.py
```

**访问界面**
- 打开浏览器访问: `http://localhost:8501`
- 系统将自动打开默认浏览器

**界面功能**
- 📊 **系统概览**: 查看网络状态和关键指标
- 🕸️ **网络拓扑**: 分析网络结构和中心性指标
- 🔍 **异常检测**: 训练模型并检测异常行为
- ⚡ **级联失效**: 分析网络鲁棒性和失效传播
- 🎯 **可解释性**: 理解模型决策过程
- 📢 **告警管理**: 监控和管理系统告警

### 方式二：命令行模式

**运行完整分析**
```bash
python main.py --mode analysis --output results
```

**测试系统**
```bash
python main.py --mode test
```

**查看帮助**
```bash
python main.py --help
```

**使用真实数据**
```bash
python main.py --mode analysis \
  --network-file data/my_network.csv \
  --data-file data/my_data.csv \
  --output results
```

---

## 📊 快速演示

### 1. 启动系统
```bash
python run_dashboard.py
```

### 2. 配置网络参数
在侧边栏设置:
- **节点数量**: 50
- **网络类型**: 小世界网络
- **时间步长**: 1000

### 3. 生成数据
点击 **"生成/更新数据"** 按钮

### 4. 探索功能
- 查看网络拓扑可视化
- 运行异常检测
- 分析级联失效
- 查看解释性分析

---

## 🔧 配置说明

### 主配置文件: `config/config.yaml`

```yaml
# 网络配置
network:
  node_count: 50              # 网络节点数量
  network_type: "small_world" # 网络类型
  
# 数据配置  
data:
  time_steps: 1000           # 时序数据长度
  anomaly_ratio: 0.05        # 异常比例
  
# 异常检测配置
anomaly_detection:
  methods: ["isolation_forest", "one_class_svm"]
  contamination: 0.1         # 污染比例
  
# 级联失效配置
cascading_failure:
  failure_threshold: 0.8     # 失效阈值
  max_iterations: 50         # 最大迭代次数
```

### 自定义配置
```bash
# 使用自定义配置文件
python main.py --config my_config.yaml --mode analysis
```

---

## 📁 目录结构

```
fenxi/
├── 📋 README.md              # 项目说明
├── 🚀 GETTING_STARTED.md     # 快速开始(本文件)
├── ⚙️ main.py                # 主程序入口
├── 🎯 run_dashboard.py       # 仪表板启动器
├── 🔧 install_deps.py        # 依赖安装器
├── 📦 requirements.txt       # 依赖列表
├── ⚙️ config/               # 配置文件
│   └── config.yaml          # 主配置文件
├── 💾 src/                  # 源代码
│   ├── data/                # 数据处理模块
│   ├── models/              # 算法模型模块
│   ├── alerts/              # 告警系统模块
│   └── visualization/       # 可视化模块
├── 🧪 tests/                # 测试代码
├── 📊 data/                 # 数据文件
└── 📚 docs/                 # 文档
```

---

## 🛠️ 故障排除

### 常见问题

**1. 依赖安装失败**
```bash
# 更新pip
python -m pip install --upgrade pip

# 使用国内镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 逐个安装核心包
pip install numpy pandas scikit-learn networkx streamlit
```

**2. 仪表板无法启动**
```bash
# 检查streamlit是否安装
python -m streamlit --version

# 手动启动
python -m streamlit run src/visualization/dashboard.py
```

**3. 内存不足错误**
- 减少网络节点数量 (在config.yaml中设置)
- 减少时间步长
- 关闭其他占用内存的程序

**4. Python版本不兼容**
```bash
# 检查Python版本
python --version

# 推荐使用Python 3.11
# 避免使用Python 3.13 (部分依赖可能不兼容)
```

### 系统测试
```bash
# 运行基础测试
python test_basic.py

# 运行系统测试
python main.py --mode test
```

---

## 📖 使用教程

### 基础工作流程

1. **启动系统** → `python run_dashboard.py`
2. **配置参数** → 在侧边栏调整网络和数据参数
3. **生成数据** → 点击"生成/更新数据"按钮
4. **异常检测** → 导航到"异常检测"页面，训练模型
5. **级联分析** → 查看网络鲁棒性分析结果
6. **解释分析** → 理解模型决策过程
7. **查看告警** → 监控系统告警和统计

### 高级功能

**导入真实数据**
```bash
# 支持的格式: CSV, JSON, Parquet, GML, GraphML
python main.py --mode analysis \
  --network-file network.gml \
  --data-file monitoring_data.csv
```

**批量分析**
```bash
# 生成完整分析报告
python main.py --mode analysis --output detailed_results
```

**自定义告警规则**
- 在Web界面的"告警管理"页面配置
- 支持多级别告警和自定义阈值

---

## 🎯 实用技巧

### 性能优化
- **大网络**: 节点数 > 100 时，建议使用命令行模式
- **内存管理**: 定期清理数据，避免长时间运行
- **并行处理**: 多核CPU可以加速模型训练

### 数据准备
- **网络数据**: 支持邻接矩阵、边列表格式
- **监控数据**: 需包含时间戳和特征列
- **数据质量**: 系统会自动检测和报告数据问题

### 可视化技巧
- **网络布局**: 尝试不同的布局算法
- **颜色配置**: 在config.yaml中自定义颜色主题
- **导出结果**: 所有图表支持PNG/PDF导出

---

## 🤝 获取帮助

### 文档资源
- 📚 **详细文档**: 查看 `docs/` 目录
- 💡 **使用示例**: 查看 `notebooks/` 目录
- 📋 **实施指南**: 查看 `IMPLEMENTATION_GUIDE.md`

### 社区支持
- 🐛 **报告问题**: 在GitHub Issues中提交
- 💬 **讨论交流**: 在GitHub Discussions中参与
- 📧 **联系开发**: 通过项目邮箱联系

### 开发调试
```bash
# 启用调试日志
python main.py --log-level DEBUG --mode test

# 查看日志文件
cat fenxi.log

# 查看告警日志
cat alerts.log
```

---

## 🎉 开始探索

现在你已经准备好开始使用这个强大的网络分析系统了！

**下一步建议**:
1. 🎮 启动Web仪表板体验完整功能
2. 📊 尝试不同的网络类型和参数配置
3. 🔍 使用自己的数据进行分析
4. 📈 探索高级功能和自定义选项

**记住**: 这个系统功能强大，不要害怕尝试不同的配置和分析方法！

---

*✨ 祝你使用愉快！如有问题，请查看故障排除部分或联系我们。*