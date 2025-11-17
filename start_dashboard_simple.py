"""
Simple Streamlit dashboard starter - 简化版仪表板启动器
"""

import sys
import os
import subprocess

def main():
    """简单启动方式"""
    
    # 确保路径正确
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dashboard_path = os.path.join(current_dir, "src", "visualization", "dashboard.py")
    
    print(">> 启动网络分析仪表板...")
    print(f">> 路径: {dashboard_path}")
    print(">> 启动后请访问: http://localhost:8501")
    print("-" * 50)
    
    # 检查文件是否存在
    if not os.path.exists(dashboard_path):
        print(f"ERROR: 找不到仪表板文件 {dashboard_path}")
        return False
    
    try:
        # 最简单的启动方式
        os.chdir(current_dir)
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "src/visualization/dashboard.py"
        ])
        
    except FileNotFoundError:
        print("ERROR: 找不到 streamlit")
        print("TIP: 请先安装: pip install streamlit")
        return False
        
    except KeyboardInterrupt:
        print("\nOK: 仪表板已关闭")
        return True
        
    except Exception as e:
        print(f"ERROR: 启动失败: {e}")
        return False

if __name__ == "__main__":
    main()