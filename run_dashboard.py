"""
Streamlit dashboard runner script.
"""

import sys
import os
import subprocess

def run_dashboard():
    """Run the Streamlit dashboard"""
    
    # Add current directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    
    # Path to dashboard script
    dashboard_path = os.path.join(current_dir, "src", "visualization", "dashboard.py")
    
    print("🚀 启动复杂网络异常检测与级联失效分析系统...")
    print("📊 Streamlit仪表板正在启动...")
    print(f"📁 仪表板路径: {dashboard_path}")
    print("-" * 60)
    
    try:
        # Run streamlit with localhost address for better compatibility
        cmd = [
            sys.executable, "-m", "streamlit", "run", dashboard_path,
            "--server.address", "localhost",
            "--server.port", "8501",
            "--browser.serverAddress", "localhost"
        ]
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 启动失败: {e}")
        print("💡 请确保已安装所有依赖: pip install -r requirements.txt")
        
    except KeyboardInterrupt:
        print("\n👋 仪表板已关闭")
        
    except Exception as e:
        print(f"❌ 未知错误: {e}")

if __name__ == "__main__":
    run_dashboard()