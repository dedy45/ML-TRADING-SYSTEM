"""
Automatic pandas installation fixer
"""
import subprocess
import sys

def run_command(cmd):
    """Run command and return result"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def fix_pandas():
    """Comprehensive pandas fix"""
    print("🔧 Starting pandas installation fix...")
    
    # Step 1: Update pip
    print("📦 Updating pip...")
    success, stdout, stderr = run_command("python -m pip install --upgrade pip")
    if success:
        print("✅ Pip updated successfully")
    else:
        print(f"⚠️ Pip update warning: {stderr}")
    
    # Step 2: Uninstall old pandas
    print("🗑️ Removing old pandas...")
    run_command("pip uninstall pandas -y")
    
    # Step 3: Install fresh pandas
    print("📥 Installing fresh pandas...")
    success, stdout, stderr = run_command("pip install pandas")
    if success:
        print("✅ Pandas installed successfully")
    else:
        print(f"❌ Pandas installation failed: {stderr}")
        
        # Try alternative method
        print("🔄 Trying alternative installation...")
        success, stdout, stderr = run_command("pip install --user pandas")
        if success:
            print("✅ Pandas installed with --user flag")
        else:
            print(f"❌ Alternative installation failed: {stderr}")
            return False
    
    # Step 4: Test installation
    print("🧪 Testing pandas...")
    try:
        import pandas as pd
        print(f"✅ Pandas working! Version: {pd.__version__}")
        return True
    except Exception as e:
        print(f"❌ Pandas test failed: {e}")
        return False

if __name__ == "__main__":
    success = fix_pandas()
    if success:
        print("🎉 Pandas fix completed successfully!")
    else:
        print("❌ Pandas fix failed. Try manual installation or conda.")