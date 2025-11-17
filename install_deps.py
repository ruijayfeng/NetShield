"""
Installation script for dependencies with Python 3.13 compatibility.
"""

import subprocess
import sys
import os

def install_package(package_name):
    """Try to install a package and return success status"""
    try:
        print(f"Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"OK {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAILED to install {package_name}: {e}")
        return False
    except Exception as e:
        print(f"ERROR installing {package_name}: {e}")
        return False

def main():
    """Install dependencies step by step"""
    print("Installing dependencies for Network Anomaly Detection System")
    print("Python version:", sys.version)
    print("=" * 60)
    
    # Core packages (essential)
    core_packages = [
        "numpy>=1.26.0",
        "pandas>=2.1.0", 
        "scikit-learn>=1.4.0",
        "networkx>=3.2",
        "scipy>=1.11.0",
        "joblib>=1.3.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0"
    ]
    
    # Visualization packages
    viz_packages = [
        "streamlit>=1.28.0",
        "plotly>=5.17.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0"
    ]
    
    # Explainability packages
    explain_packages = [
        "shap>=0.43.0",
        "lime>=0.2.0.1"
    ]
    
    # Optional packages (can fail)
    optional_packages = [
        "torch>=2.1.0",
        "torch-geometric>=2.4.0"
    ]
    
    # Track installation results
    results = {
        "core": [],
        "visualization": [],
        "explainability": [], 
        "optional": []
    }
    
    print("Installing core packages...")
    for package in core_packages:
        success = install_package(package)
        results["core"].append((package, success))
    
    print("\nInstalling visualization packages...")
    for package in viz_packages:
        success = install_package(package)
        results["visualization"].append((package, success))
    
    print("\nInstalling explainability packages...")
    for package in explain_packages:
        success = install_package(package)
        results["explainability"].append((package, success))
    
    print("\nInstalling optional packages...")
    for package in optional_packages:
        success = install_package(package)
        results["optional"].append((package, success))
    
    # Summary
    print("\n" + "=" * 60)
    print("Installation Summary")
    print("=" * 60)
    
    for category, packages in results.items():
        successful = sum(1 for _, success in packages if success)
        total = len(packages)
        print(f"{category.title()}: {successful}/{total} packages installed")
        
        for package, success in packages:
            status = "OK" if success else "FAILED"
            print(f"  {status} {package.split('>=')[0]}")
    
    # Check if minimum requirements are met
    core_success = all(success for _, success in results["core"])
    viz_success = any(success for _, success in results["visualization"])
    
    if core_success and viz_success:
        print("\nInstallation successful! You can now run the system.")
        print("\nNext steps:")
        print("1. Launch dashboard: python run_dashboard.py")
        print("2. Or run analysis: python main.py --mode analysis")
        return True
    else:
        print("\nSome essential packages failed to install.")
        print("You may need to:")
        print("1. Update pip: python -m pip install --upgrade pip")
        print("2. Try installing packages individually")
        print("3. Consider using a different Python version (3.11 or 3.12)")
        return False

if __name__ == "__main__":
    success = main()
    input("\nPress Enter to exit...")
    sys.exit(0 if success else 1)