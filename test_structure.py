"""
Basic structure test without external dependencies.
"""

import os
import sys

def test_directory_structure():
    """Test project directory structure"""
    print("Testing directory structure...")
    
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
            print(f"  OK {dir_path}")
        else:
            missing_dirs.append(dir_path)
            print(f"  MISSING {dir_path}")
    
    print(f"\n{len(existing_dirs)}/{len(expected_dirs)} directories exist")
    
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
            print(f"  OK {file_path}")
        else:
            missing_files.append(file_path)
            print(f"  MISSING {file_path}")
    
    print(f"\n{len(existing_files)}/{len(core_files)} core files exist")
    
    return len(missing_files) == 0

def count_lines_of_code():
    """Count total lines of code"""
    print("\nCounting lines of code...")
    
    total_lines = 0
    total_files = 0
    
    for root, dirs, files in os.walk("src"):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        lines = len(f.readlines())
                        total_lines += lines
                        total_files += 1
                        print(f"  {filepath}: {lines} lines")
                except:
                    pass
    
    print(f"\nTotal: {total_lines} lines across {total_files} Python files")
    return total_lines, total_files

def main():
    """Run all tests"""
    print("Complex Network Anomaly Detection System - Structure Test")
    print("=" * 70)
    
    # Test directory structure
    dirs_ok = test_directory_structure()
    
    # Test core files
    files_ok = test_core_files()
    
    # Count code
    total_lines, total_files = count_lines_of_code()
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Results Summary")
    print("=" * 70)
    
    print(f"Directory Structure: {'COMPLETE' if dirs_ok else 'INCOMPLETE'}")
    print(f"Core Files: {'COMPLETE' if files_ok else 'INCOMPLETE'}")
    print(f"Code Statistics: {total_lines} lines of code, {total_files} files")
    
    if dirs_ok and files_ok:
        print("\nSYSTEM STRUCTURE COMPLETE!")
        print("\nMain Components:")
        print("  - Network topology generator")
        print("  - Simulation data generator")
        print("  - Anomaly detection algorithms (Isolation Forest, One-Class SVM, LOF)")
        print("  - Cascading failure analysis")
        print("  - Explainable AI analysis (SHAP integration)")
        print("  - Intelligent alert system")
        print("  - Streamlit visualization interface")
        
        print("\nUsage Instructions:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Launch dashboard: python run_dashboard.py")
        print("3. Or run analysis: python main.py --mode analysis")
        
        return True
    else:
        print("\nSYSTEM STRUCTURE INCOMPLETE - Please check missing files and directories")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)