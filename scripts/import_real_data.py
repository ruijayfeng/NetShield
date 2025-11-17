#!/usr/bin/env python3
"""
示例脚本：将真实数据导入到网络异常检测系统进行分析

使用方法:
python scripts/import_real_data.py --network-file data/real/network.csv --data-file data/real/monitoring.csv
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import networkx as nx

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from main import NetworkAnalysisSystem
import asyncio


def create_sample_network_file(filepath: str):
    """创建示例网络拓扑文件"""
    print(f"创建示例网络文件: {filepath}")
    
    # 创建一个简单的网络拓扑
    edges = [
        {'source': 'node_1', 'target': 'node_2', 'weight': 0.8},
        {'source': 'node_1', 'target': 'node_3', 'weight': 0.9},
        {'source': 'node_2', 'target': 'node_3', 'weight': 1.2},
        {'source': 'node_2', 'target': 'node_4', 'weight': 0.7},
        {'source': 'node_3', 'target': 'node_4', 'weight': 1.0},
        {'source': 'node_3', 'target': 'node_5', 'weight': 0.6},
        {'source': 'node_4', 'target': 'node_5', 'weight': 0.9},
    ]
    
    df = pd.DataFrame(edges)
    df.to_csv(filepath, index=False)
    print(f"✅ 网络文件已创建: {len(edges)} 条边")


def create_sample_monitoring_data(filepath: str, duration_hours: int = 24):
    """创建示例监控数据文件"""
    print(f"创建示例监控数据文件: {filepath}")
    
    # 生成时间序列
    start_time = datetime.now() - timedelta(hours=duration_hours)
    timestamps = [start_time + timedelta(minutes=i) for i in range(duration_hours * 60)]
    
    data = []
    for i, timestamp in enumerate(timestamps):
        # 基础模式 + 随机噪声
        base_traffic = 100 + 20 * np.sin(2 * np.pi * i / (24 * 60)) + np.random.normal(0, 5)
        base_latency = 25 + 5 * np.sin(2 * np.pi * i / (12 * 60)) + np.random.normal(0, 2)
        base_cpu = 0.3 + 0.2 * np.sin(2 * np.pi * i / (24 * 60)) + np.random.normal(0, 0.05)
        base_memory = 0.5 + 0.1 * np.sin(2 * np.pi * i / (48 * 60)) + np.random.normal(0, 0.02)
        
        # 注入一些异常
        is_anomaly = False
        if np.random.random() < 0.03:  # 3% 异常率
            is_anomaly = True
            base_traffic *= np.random.uniform(2, 4)  # 流量异常
            base_latency *= np.random.uniform(2, 3)  # 延迟异常
            base_cpu = min(1.0, base_cpu * np.random.uniform(1.5, 2.5))  # CPU异常
        
        record = {
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'traffic': max(0, base_traffic),
            'latency': max(0, base_latency),
            'packet_loss': max(0, np.random.exponential(0.001)),
            'cpu_usage': np.clip(base_cpu, 0, 1),
            'memory_usage': np.clip(base_memory, 0, 1),
            'node_id': 'node_1',
            'is_anomaly': is_anomaly,
            'anomaly_score': 0.8 if is_anomaly else 0.0
        }
        data.append(record)
    
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"✅ 监控数据文件已创建: {len(data)} 条记录, 异常率: {df['is_anomaly'].mean():.3f}")
    return df


def validate_data_files(network_file: str, data_file: str):
    """验证数据文件格式"""
    print("验证数据文件格式...")
    
    # 验证网络文件
    if network_file and os.path.exists(network_file):
        try:
            network_df = pd.read_csv(network_file)
            required_cols = ['source', 'target']
            missing_cols = [col for col in required_cols if col not in network_df.columns]
            if missing_cols:
                print(f"❌ 网络文件缺少必需列: {missing_cols}")
                return False
            print(f"✅ 网络文件格式正确: {len(network_df)} 条边")
        except Exception as e:
            print(f"❌ 网络文件读取失败: {e}")
            return False
    
    # 验证监控数据文件
    if not os.path.exists(data_file):
        print(f"❌ 监控数据文件不存在: {data_file}")
        return False
    
    try:
        data_df = pd.read_csv(data_file)
        required_cols = ['timestamp']
        missing_cols = [col for col in required_cols if col not in data_df.columns]
        if missing_cols:
            print(f"❌ 监控数据文件缺少必需列: {missing_cols}")
            return False
        
        # 检查数值特征列
        numeric_cols = data_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 1:
            print("❌ 监控数据文件至少需要一个数值特征列")
            return False
            
        print(f"✅ 监控数据文件格式正确: {len(data_df)} 条记录, {len(numeric_cols)} 个数值特征")
        return True
        
    except Exception as e:
        print(f"❌ 监控数据文件读取失败: {e}")
        return False


async def run_analysis_with_real_data(network_file: str, data_file: str, output_dir: str):
    """使用真实数据运行完整分析"""
    print("开始使用真实数据进行分析...")
    
    try:
        # 创建分析系统
        system = NetworkAnalysisSystem()
        
        # 初始化系统
        print("初始化系统组件...")
        system.initialize_system()
        
        # 加载真实数据
        print("加载真实数据...")
        system.generate_network_and_data(network_file, data_file)
        
        # 训练模型
        print("训练异常检测模型...")
        training_stats = system.train_anomaly_detection()
        
        # 执行异常检测
        print("执行异常检测...")
        detection_results = await system.perform_anomaly_detection()
        
        # 执行级联失效分析
        print("执行级联失效分析...")
        cascade_results = await system.perform_cascading_failure_analysis()
        
        # 生成解释
        print("生成可解释性分析...")
        try:
            explanations, exp_report = system.generate_explanations(5)
            print("✅ 可解释性分析完成")
        except Exception as e:
            print(f"⚠️ 可解释性分析失败: {e}")
        
        # 生成综合报告
        print("生成综合分析报告...")
        report = system.generate_comprehensive_report()
        print("\n" + "="*80)
        print(report)
        print("="*80)
        
        # 保存结果
        print(f"保存分析结果到 {output_dir}...")
        system.save_results(output_dir)
        
        # 显示关键结果
        summary = detection_results.get('summary', {})
        print(f"\n📊 分析结果摘要:")
        print(f"   异常检测: {summary.get('predicted_anomalies', 0)} 个异常")
        print(f"   异常率: {summary.get('anomaly_rate', 0):.1%}")
        
        robustness_metrics = cascade_results.get('robustness_metrics', {})
        robustness_score = robustness_metrics.get('overall_robustness_score', 0)
        print(f"   网络鲁棒性: {robustness_score:.3f}")
        
        print(f"\n✅ 分析完成！结果已保存到 {output_dir}")
        
    except Exception as e:
        print(f"❌ 分析过程失败: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="导入真实数据进行网络异常检测和级联失效分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用真实数据文件
  python scripts/import_real_data.py --network-file data/network.csv --data-file data/monitoring.csv
  
  # 生成示例数据并分析
  python scripts/import_real_data.py --create-sample --output output/
  
  # 仅验证数据格式
  python scripts/import_real_data.py --data-file data/monitoring.csv --validate-only
        """
    )
    
    parser.add_argument(
        '--network-file',
        help='网络拓扑文件路径 (CSV格式: source,target,weight)'
    )
    
    parser.add_argument(
        '--data-file',
        help='监控数据文件路径 (CSV格式，必须包含timestamp列)'
    )
    
    parser.add_argument(
        '--output',
        default='output_real_data',
        help='输出目录 (默认: output_real_data)'
    )
    
    parser.add_argument(
        '--create-sample',
        action='store_true',
        help='创建示例数据文件用于测试'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='仅验证数据文件格式，不执行分析'
    )
    
    args = parser.parse_args()
    
    # 创建示例数据
    if args.create_sample:
        os.makedirs('data/sample', exist_ok=True)
        network_file = 'data/sample/network.csv'
        data_file = 'data/sample/monitoring.csv'
        
        create_sample_network_file(network_file)
        create_sample_monitoring_data(data_file)
        
        print(f"\n示例数据文件已创建:")
        print(f"  网络文件: {network_file}")
        print(f"  监控数据: {data_file}")
        print(f"\n现在可以运行:")
        print(f"  python scripts/import_real_data.py --network-file {network_file} --data-file {data_file}")
        return
    
    # 检查必需参数
    if not args.data_file:
        print("❌ 请指定监控数据文件 (--data-file)")
        parser.print_help()
        return
    
    # 验证数据文件
    if not validate_data_files(args.network_file, args.data_file):
        print("❌ 数据文件验证失败")
        return
    
    if args.validate_only:
        print("✅ 数据文件验证通过")
        return
    
    # 执行分析
    try:
        asyncio.run(run_analysis_with_real_data(
            args.network_file, 
            args.data_file, 
            args.output
        ))
    except KeyboardInterrupt:
        print("\n❌ 分析被用户中断")
    except Exception as e:
        print(f"❌ 分析失败: {e}")


if __name__ == "__main__":
    main()