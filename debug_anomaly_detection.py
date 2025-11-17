"""
调试异常检测问题
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def debug_feature_extraction():
    """调试特征提取问题"""
    
    print("调试异常检测特征提取问题")
    print("=" * 50)
    
    # 加载数据
    data = pd.read_csv("TestData/monitoring/detailed_monitoring.csv")
    print(f"原始数据形状: {data.shape}")
    print(f"列名: {list(data.columns)}")
    
    # 导入异常检测器
    from src.models.anomaly_detection.detectors import NetworkAnomalyAnalyzer, AnomalyDetectionConfig
    
    config = AnomalyDetectionConfig()
    analyzer = NetworkAnomalyAnalyzer(config)
    
    # 尝试特征准备
    print("\n测试特征准备...")
    try:
        features = analyzer.prepare_features(data)
        print(f"准备的特征矩阵形状: {features.shape}")
        print(f"特征列: {analyzer.feature_columns}")
        
        if len(analyzer.feature_columns) == 0:
            print("ERROR: 没有识别出任何特征!")
            
            # 检查数据类型
            print("\n数据类型检查:")
            for col in data.columns:
                print(f"  {col}: {data[col].dtype}")
                
            # 检查数值列
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            print(f"\n数值列: {list(numeric_cols)}")
            
            # 检查排除逻辑
            excluded_cols = ['timestamp', 'is_anomaly', 'anomaly_score', 'node_id']
            remaining_cols = [col for col in numeric_cols if col not in excluded_cols]
            print(f"排除特定列后: {remaining_cols}")
            
    except Exception as e:
        print(f"特征准备失败: {e}")
        import traceback
        traceback.print_exc()

def debug_training():
    """调试训练过程"""
    
    print("\n" + "=" * 50)
    print("调试训练过程")
    print("=" * 50)
    
    data = pd.read_csv("TestData/monitoring/detailed_monitoring.csv")
    
    from src.models.anomaly_detection.detectors import NetworkAnomalyAnalyzer, AnomalyDetectionConfig
    
    config = AnomalyDetectionConfig()
    config.contamination = 0.05  # 设置污染比例为5%
    analyzer = NetworkAnomalyAnalyzer(config)
    
    # 手动设置特征列
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in ['is_anomaly', 'anomaly_score']]
    print(f"手动设置特征列: {feature_cols}")
    
    # 训练数据
    train_data = data[:6000].copy()
    
    try:
        # 手动准备特征
        X_train = train_data[feature_cols].fillna(0)
        print(f"训练特征矩阵: {X_train.shape}")
        print(f"特征范围:")
        for col in feature_cols:
            print(f"  {col}: {X_train[col].min():.3f} - {X_train[col].max():.3f}")
        
        # 训练模型
        analyzer.feature_columns = feature_cols
        stats = analyzer.train(train_data)
        print(f"训练完成: {stats}")
        
        # 测试预测
        test_data = data[6000:6100].copy()
        results = analyzer.detect_anomalies(test_data)
        print(f"预测结果: {results['summary']}")
        
    except Exception as e:
        print(f"训练失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    debug_feature_extraction()
    debug_training()

if __name__ == "__main__":
    main()