#!/usr/bin/env python3
"""
AegisVision Dashboard Runner
Usage: python run_dashboard.py
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'torch',
        'transformers',
        'opencv-python',
        'pandas'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def main():
    """Run the AegisVision dashboard"""
    print("AegisVision Dashboard Startup")
    print("=" * 40)
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    app_file = current_dir / "app.py"
    
    if not app_file.exists():
        print("app.py not found in current directory")
        print(f"   Current directory: {current_dir}")
        print("   Please run this script from the AegisVision project root")
        return 1
    
    # Check model path
    model_path = current_dir / "checkpoints" / "videomae_finetuned_proper"
    if not model_path.exists():
        print(f" Model not found at: {model_path}")
        print("   Dashboard will still run, but you'll need to specify the correct model path")
    else:
        print(f"VideoMAE model found: {model_path}")
    
    # Check dependencies
    print("\nChecking dependencies...")
    if not check_dependencies():
        return 1
    
    print("All dependencies are installed")
    
    # Create output directory
    output_dir = current_dir / "output"
    output_dir.mkdir(exist_ok=True)
    print(f" Output directory ready: {output_dir}")
    
    print("\n Starting dashboard...")
    print("   Dashboard will open in your browser at: http://localhost:8501")
    print("   Press Ctrl+C to stop the dashboard")
    print("=" * 40)
    
    # Run streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n\n Dashboard stopped")
        return 0
    except Exception as e:
        print(f"\n Error running dashboard: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())