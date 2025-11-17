"""
Basic system test without external dependencies.
"""

import os
import sys
import yaml

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing module imports...")
    
    try:
        # Test data generators
        from src.data.generators.network_generator import NetworkConfig
        from src.data.generators.data_generator import DataConfig
        print("  OK Data generators imported successfully")
        
        # Test anomaly detection
        from src.models.anomaly_detection.detectors import AnomalyDetectionConfig
        print("  OK Anomaly detection modules imported successfully")
        
        # Test cascading failure
        from src.models.cascading.failure_analyzer import CascadingFailureConfig
        print("  OK Cascading failure modules imported successfully")
        
        # Test explainability
        from src.models.explainable.shap_explainer import ExplainabilityConfig
        print("  OK Explainability modules imported successfully")
        
        # Test alerts
        from src.alerts.alert_manager import AlertManager, AlertLevel
        print("  OK Alert system imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"  FAILED Import error: {e}")
        return False
    except Exception as e:
        print(f"  FAILED Unexpected error: {e}")
        return False

def test_config_loading():
    """Test configuration loading"""
    print("\nTesting configuration loading...")
    
    try:
        config_path = "config/config.yaml"
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            print("  OK Configuration file loaded successfully")
            print(f"    - Network nodes: {config.get('network', {}).get('node_count', 'N/A')}")
            print(f"    - Time steps: {config.get('data', {}).get('time_steps', 'N/A')}")
            print(f"    - Anomaly methods: {config.get('anomaly_detection', {}).get('methods', 'N/A')}")
            
            return True
        else:
            print(f"  FAILED Configuration file not found: {config_path}")
            return False
            
    except Exception as e:
        print(f"  FAILED Configuration loading error: {e}")
        return False

def test_directory_structure():
    """Test project directory structure"""
    print("\nTesting directory structure...")
    
    expected_dirs = [
        "src",
        "src/data",
        "src/data/generators",
        "src/models",
        "src/models/anomaly_detection",
        "src/models/cascading",
        "src/models/explainable",
        "src/alerts",
        "src/visualization",
        "config",
        "tests",
        "docs",
        "notebooks",
        "data",
        "data/raw",
        "data/processed",
        "data/models"
    ]
    
    missing_dirs = []
    existing_dirs = []
    
    for dir_path in expected_dirs:
        if os.path.exists(dir_path):
            existing_dirs.append(dir_path)
        else:
            missing_dirs.append(dir_path)
    
    print(f"  OK {len(existing_dirs)} directories exist")
    
    if missing_dirs:
        print(f"  WARNING {len(missing_dirs)} directories missing:")
        for missing in missing_dirs[:5]:  # Show first 5
            print(f"    - {missing}")
        if len(missing_dirs) > 5:
            print(f"    ... and {len(missing_dirs) - 5} more")
    
    return len(missing_dirs) == 0

def test_core_files():
    """Test existence of core files"""
    print("\nTesting core files...")
    
    core_files = [
        "main.py",
        "run_dashboard.py",
        "requirements.txt",
        "README.md",
        "config/config.yaml",
        "src/__init__.py",
        "src/data/__init__.py",
        "src/data/generators/__init__.py",
        "src/data/generators/network_generator.py",
        "src/data/generators/data_generator.py",
        "src/models/__init__.py",
        "src/models/anomaly_detection/__init__.py",
        "src/models/anomaly_detection/detectors.py",
        "src/models/cascading/__init__.py",
        "src/models/cascading/failure_analyzer.py",
        "src/models/explainable/__init__.py",
        "src/models/explainable/shap_explainer.py",
        "src/alerts/__init__.py",
        "src/alerts/alert_manager.py",
        "src/visualization/__init__.py",
        "src/visualization/dashboard.py"
    ]
    
    missing_files = []
    existing_files = []
    
    for file_path in core_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
        else:
            missing_files.append(file_path)
    
    print(f"  OK {len(existing_files)} core files exist")
    
    if missing_files:
        print(f"  FAILED {len(missing_files)} core files missing:")
        for missing in missing_files:
            print(f"    - {missing}")
        return False
    
    return True

def test_dependencies():
    """Test if key dependencies would be available"""
    print("\nTesting dependency availability...")
    
    dependencies = [
        "numpy",
        "pandas", 
        "sklearn",
        "networkx",
        "streamlit",
        "plotly",
        "shap",
        "yaml"
    ]
    
    available = []
    missing = []
    
    for dep in dependencies:
        try:
            __import__(dep)
            available.append(dep)
        except ImportError:
            missing.append(dep)
    
    if available:
        print(f"  OK Available dependencies: {', '.join(available)}")
    
    if missing:
        print(f"  WARNING Missing dependencies: {', '.join(missing)}")
        print("    Run: pip install -r requirements.txt")
    
    return len(missing) == 0

def main():
    """Run all tests"""
    print("Complex Network Anomaly Detection and Cascading Failure Analysis System - Basic Test")
    print("=" * 60)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Core Files", test_core_files),
        ("Configuration", test_config_loading),
        ("Module Imports", test_imports),
        ("Dependencies", test_dependencies)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\nFAILED {test_name} test failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name:20}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall Result: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\nAll basic tests passed! System structure is complete.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Launch dashboard: python run_dashboard.py")
        print("3. Or run full analysis: python main.py --mode analysis")
    else:
        print(f"\nWARNING {len(results) - passed} tests failed, please check system configuration.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)