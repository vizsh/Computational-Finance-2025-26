#!/usr/bin/env python3
"""
Installation Fix Script for META Stock Prediction System

This script helps resolve NumPy compatibility issues by installing
compatible versions of packages.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔧 {description}...")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error code {e.returncode}")
        if e.stderr:
            print(f"Error: {e.stderr.strip()}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 7:
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python 3.7+ is required")
        return False

def check_current_installations():
    """Check currently installed packages"""
    print("\n📦 Checking current package installations...")
    
    packages_to_check = ['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'seaborn', 'yfinance']
    
    for package in packages_to_check:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is not installed")
        except Exception as e:
            print(f"⚠️ {package} has import issues: {e}")

def install_compatible_packages():
    """Install NumPy 1.x compatible packages"""
    print("\n🚀 Installing NumPy 1.x compatible packages...")
    
    # First, uninstall problematic packages
    uninstall_commands = [
        "pip uninstall numpy pandas scipy scikit-learn matplotlib seaborn -y",
        "conda remove numpy pandas scipy scikit-learn matplotlib seaborn -y" if os.path.exists(os.path.expanduser("~/.conda")) else "echo 'Conda not found, skipping conda uninstall'"
    ]
    
    for command in uninstall_commands:
        run_command(command, "Uninstalling packages")
    
    # Install compatible versions
    install_commands = [
        "pip install 'numpy<2.0.0'",
        "pip install 'pandas<2.2.0'",
        "pip install 'scipy<1.12.0'",
        "pip install 'scikit-learn<1.4.0'",
        "pip install 'matplotlib<3.8.0'",
        "pip install 'seaborn<0.13.0'",
        "pip install 'yfinance>=0.2.18'"
    ]
    
    success_count = 0
    for command in install_commands:
        if run_command(command, f"Installing package"):
            success_count += 1
    
    return success_count == len(install_commands)

def test_imports():
    """Test if all packages can be imported successfully"""
    print("\n🧪 Testing package imports...")
    
    test_packages = [
        'numpy',
        'pandas', 
        'scipy',
        'sklearn',
        'matplotlib',
        'seaborn',
        'yfinance'
    ]
    
    failed_imports = []
    
    for package in test_packages:
        try:
            if package == 'sklearn':
                import sklearn
                print(f"✅ sklearn imported successfully")
            else:
                __import__(package)
                print(f"✅ {package} imported successfully")
        except ImportError as e:
            print(f"❌ {package} import failed: {e}")
            failed_imports.append(package)
        except Exception as e:
            print(f"⚠️ {package} has issues: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed imports: {', '.join(failed_imports)}")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        return True

def create_virtual_environment():
    """Create a virtual environment for isolation"""
    print("\n🏗️ Creating virtual environment...")
    
    venv_name = "meta_prediction_env"
    
    # Check if virtual environment exists
    if os.path.exists(venv_name):
        print(f"Virtual environment '{venv_name}' already exists")
        return True
    
    # Create virtual environment
    if run_command(f"python -m venv {venv_name}", "Creating virtual environment"):
        print(f"\n✅ Virtual environment '{venv_name}' created successfully!")
        print(f"\nTo activate the virtual environment:")
        if sys.platform == "win32":
            print(f"  {venv_name}\\Scripts\\activate")
        else:
            print(f"  source {venv_name}/bin/activate")
        print(f"\nThen install packages: pip install -r requirements.txt")
        return True
    else:
        return False

def main():
    """Main installation fix function"""
    print("🔧 META Stock Prediction System - Installation Fix")
    print("="*60)
    print("This script will help resolve NumPy compatibility issues")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Please upgrade to Python 3.7+ and try again")
        return
    
    # Check current installations
    check_current_installations()
    
    # Ask user preference
    print("\n" + "="*60)
    print("Choose installation method:")
    print("1. Fix current environment (recommended)")
    print("2. Create virtual environment (isolated)")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        print("\n🔧 Fixing current environment...")
        if install_compatible_packages():
            print("\n✅ Package installation completed!")
            if test_imports():
                print("\n🎉 All issues resolved! You can now run the stock prediction system.")
            else:
                print("\n⚠️ Some packages still have issues. Consider using option 2 (virtual environment).")
        else:
            print("\n❌ Package installation failed. Consider using option 2 (virtual environment).")
    
    elif choice == "2":
        print("\n🏗️ Creating virtual environment...")
        if create_virtual_environment():
            print("\n✅ Virtual environment created!")
            print("\nNext steps:")
            print("1. Activate the virtual environment")
            print("2. Run: pip install -r requirements.txt")
            print("3. Run: python stockprice_pred_LR.py")
        else:
            print("\n❌ Failed to create virtual environment")
    
    elif choice == "3":
        print("\n👋 Exiting installation fix...")
        return
    
    else:
        print("\n❌ Invalid choice. Please run the script again.")
        return
    
    print("\n" + "="*60)
    print("📚 Additional troubleshooting tips:")
    print("="*60)
    print("1. If you still have issues, try creating a fresh virtual environment")
    print("2. Ensure you have pip updated: python -m pip install --upgrade pip")
    print("3. Check your internet connection for package downloads")
    print("4. On macOS, you might need: xcode-select --install")
    print("5. On Windows, ensure you have Visual C++ build tools")
    print("\nFor more help, check the README.md file")

if __name__ == "__main__":
    main() 