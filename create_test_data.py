"""
生成测试数据的脚本 - Generate comprehensive test monitoring data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

def generate_monitoring_data():
    """生成监控时间序列数据"""
    
    # 设置随机种子确保可重现性
    np.random.seed(42)
    
    # 时间范围: 过去30天，每5分钟一个数据点
    start_time = datetime.now() - timedelta(days=30)
    time_points = []
    current_time = start_time
    
    for i in range(8640):  # 30天 * 24小时 * 12个点/小时
        time_points.append(current_time)
        current_time += timedelta(minutes=5)
    
    # 生成基础数据
    data = []
    
    for i, timestamp in enumerate(time_points):
        # 时间特征
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # 基础趋势 (工作时间vs非工作时间)
        if 9 <= hour <= 17 and day_of_week < 5:  # 工作时间
            base_traffic = 0.7 + 0.2 * np.sin(2 * np.pi * hour / 24)
            base_cpu = 0.6 + 0.3 * np.sin(2 * np.pi * hour / 24)
        else:  # 非工作时间
            base_traffic = 0.3 + 0.1 * np.sin(2 * np.pi * hour / 24)
            base_cpu = 0.2 + 0.1 * np.sin(2 * np.pi * hour / 24)
        
        # 添加周期性波动
        weekly_factor = 1 + 0.1 * np.sin(2 * np.pi * day_of_week / 7)
        
        # 生成各项指标 - 使用合理的数值范围
        traffic_mbps = max(0, min(10000, (base_traffic * weekly_factor + 
                              np.random.normal(0, 0.1)) * 800))  # 0-8000 Mbps normal range
        
        latency_ms = max(0.1, min(100, 1.5 + base_traffic * 3 + np.random.normal(0, 0.5)))
        
        packet_loss_rate = max(0, min(0.1, 0.001 + base_traffic * 0.002 + 
                                     np.random.normal(0, 0.0005)))  # Max 10% loss
        
        cpu_usage = max(0, min(1, base_cpu * weekly_factor + 
                              np.random.normal(0, 0.05)))
        
        memory_usage = max(0, min(1, 0.3 + cpu_usage * 0.4 + 
                                 np.random.normal(0, 0.03)))
        
        disk_io_mbps = max(0, min(500, 20 + cpu_usage * 100 + np.random.normal(0, 10)))
        
        network_errors = max(0, int(traffic_mbps * packet_loss_rate * 0.001 + 
                                   np.random.poisson(1)))
        
        temperature_c = max(20, min(80, 30 + cpu_usage * 25 + np.random.normal(0, 3)))
        
        # 异常标记 (5%的数据点为异常)
        is_anomaly = False
        if np.random.random() < 0.05:
            is_anomaly = True
            # 异常情况下的指标变化
            if np.random.random() < 0.3:  # CPU异常
                cpu_usage = min(1, cpu_usage + np.random.uniform(0.3, 0.5))
                memory_usage = min(1, memory_usage + np.random.uniform(0.2, 0.4))
                temperature_c += np.random.uniform(10, 25)
            elif np.random.random() < 0.3:  # 网络异常
                latency_ms += np.random.uniform(10, 50)
                packet_loss_rate = min(1, packet_loss_rate + np.random.uniform(0.05, 0.2))
                network_errors += np.random.poisson(20)
            else:  # 流量异常
                traffic_mbps *= np.random.uniform(2, 5)
                disk_io_mbps *= np.random.uniform(3, 8)
        
        # 计算异常评分
        anomaly_score = 0
        if cpu_usage > 0.8: anomaly_score += 0.3
        if memory_usage > 0.85: anomaly_score += 0.2
        if latency_ms > 10: anomaly_score += 0.2
        if packet_loss_rate > 0.01: anomaly_score += 0.2
        if temperature_c > 60: anomaly_score += 0.1
        
        data.append({
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'traffic_mbps': round(traffic_mbps, 2),
            'latency_ms': round(latency_ms, 2),
            'packet_loss_rate': round(packet_loss_rate, 4),
            'cpu_usage': round(cpu_usage, 3),
            'memory_usage': round(memory_usage, 3),
            'disk_io_mbps': round(disk_io_mbps, 1),
            'network_errors': network_errors,
            'temperature_c': round(temperature_c, 1),
            'is_anomaly': is_anomaly,
            'anomaly_score': round(anomaly_score, 3)
        })
    
    return pd.DataFrame(data)

def generate_multi_node_data():
    """生成多节点监控数据"""
    
    np.random.seed(123)
    node_types = ['router', 'switch', 'server', 'client']
    nodes = [f'node_{i:02d}' for i in range(12)]
    
    # 为每个节点生成24小时的监控数据
    start_time = datetime.now() - timedelta(hours=24)
    timestamps = [start_time + timedelta(minutes=10*i) for i in range(144)]
    
    all_data = []
    
    for node_id in nodes:
        node_type = np.random.choice(node_types)
        
        for timestamp in timestamps:
            hour = timestamp.hour
            
            # 不同类型节点的基础负载不同
            if node_type == 'server':
                base_cpu = 0.4 + 0.3 * np.sin(2 * np.pi * hour / 24)
                base_memory = 0.6
            elif node_type == 'router':
                base_cpu = 0.2 + 0.2 * np.sin(2 * np.pi * hour / 24)
                base_memory = 0.3
            elif node_type == 'switch':
                base_cpu = 0.15 + 0.1 * np.sin(2 * np.pi * hour / 24)
                base_memory = 0.25
            else:  # client
                base_cpu = 0.1 + 0.4 * (1 if 9 <= hour <= 17 else 0.2)
                base_memory = 0.4
            
            # 添加随机波动
            cpu = max(0, min(1, base_cpu + np.random.normal(0, 0.1)))
            memory = max(0, min(1, base_memory + np.random.normal(0, 0.08)))
            
            # 网络指标
            throughput = max(0, 100 * (0.5 + 0.5 * np.sin(2 * np.pi * hour / 24)) + 
                            np.random.normal(0, 20))
            
            latency = max(0.1, 2 + cpu * 3 + np.random.normal(0, 0.5))
            
            # 异常检测
            is_anomaly = np.random.random() < 0.03
            if is_anomaly:
                cpu = min(1, cpu + np.random.uniform(0.2, 0.6))
                latency += np.random.uniform(5, 20)
            
            all_data.append({
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'node_id': node_id,
                'node_type': node_type,
                'cpu_usage': round(cpu, 3),
                'memory_usage': round(memory, 3),
                'throughput_mbps': round(throughput, 1),
                'latency_ms': round(latency, 2),
                'is_anomaly': is_anomaly
            })
    
    return pd.DataFrame(all_data)

def main():
    """生成所有测试数据"""
    
    print(">> 生成监控数据文件...")
    
    # 生成单节点详细监控数据
    print(">> 生成详细监控数据 (30天)...")
    detailed_data = generate_monitoring_data()
    detailed_data.to_csv('TestData/monitoring/detailed_monitoring.csv', index=False)
    print(f"OK 已生成: TestData/monitoring/detailed_monitoring.csv ({len(detailed_data)} 条记录)")
    
    # 生成多节点监控数据
    print(">> 生成多节点监控数据 (24小时)...")
    multi_node_data = generate_multi_node_data()
    multi_node_data.to_csv('TestData/monitoring/multi_node_monitoring.csv', index=False)
    print(f"OK 已生成: TestData/monitoring/multi_node_monitoring.csv ({len(multi_node_data)} 条记录)")
    
    # 生成JSON格式数据
    print(">> 生成JSON格式监控数据...")
    sample_data = detailed_data.tail(100).to_dict('records')
    with open('TestData/monitoring/sample_monitoring.json', 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'description': '网络监控数据样本',
                'total_records': len(sample_data),
                'time_range': '最近100个数据点',
                'features': list(detailed_data.columns)
            },
            'data': sample_data
        }, f, indent=2, ensure_ascii=False)
    print("OK 已生成: TestData/monitoring/sample_monitoring.json")
    
    # 生成Parquet格式数据
    try:
        detailed_data.to_parquet('TestData/monitoring/detailed_monitoring.parquet')
        print("OK 已生成: TestData/monitoring/detailed_monitoring.parquet")
    except ImportError:
        print("WARNING 跳过Parquet格式 (需要 pyarrow 或 fastparquet)")
    except Exception as e:
        print(f"WARNING 跳过Parquet格式: {e}")
    
    print("\n>> 数据统计:")
    print(f"  总记录数: {len(detailed_data):,}")
    print(f"  异常记录: {detailed_data['is_anomaly'].sum():,} ({detailed_data['is_anomaly'].mean():.1%})")
    print(f"  时间范围: {detailed_data['timestamp'].min()} 到 {detailed_data['timestamp'].max()}")
    print(f"  特征数量: {len(detailed_data.columns) - 2}")  # 除去timestamp和is_anomaly
    
    print("\n>> 所有测试数据已生成完成!")

if __name__ == "__main__":
    main()