#!/usr/bin/env python3
"""
Easy-to-use script for running trading ML experiments
Perfect for beginners - just run and follow the prompts!
"""

import sys
import os
from datetime import datetime
from advanced_ml_pipeline import AdvancedTradingPipeline

def print_banner():
    """Print a nice banner"""
    print("=" * 70)
    print("🎯 TRADING ML EXPERIMENT RUNNER")
    print("=" * 70)
    print("📊 Analyze your trading data with machine learning")
    print("🔬 Track experiments with MLflow")
    print("💡 Perfect for beginners!")
    print("=" * 70)

def get_user_choice():
    """Get user's experiment choice"""
    print("\n🚀 Choose your experiment:")
    print("1. Quick Test (3 files) - Fast results")
    print("2. Medium Test (10 files) - Good balance")
    print("3. Large Test (25 files) - More comprehensive")
    print("4. Full Test (50 files) - Most comprehensive")
    print("5. Custom - Choose your own number")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            if choice in ['1', '2', '3', '4', '5']:
                return choice
            else:
                print("❌ Please enter a number between 1 and 5")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)

def get_file_count(choice):
    """Get number of files based on choice"""
    if choice == '1':
        return 3
    elif choice == '2':
        return 10
    elif choice == '3':
        return 25
    elif choice == '4':
        return 50
    elif choice == '5':
        while True:
            try:
                count = int(input("Enter number of files to process (1-544): "))
                if 1 <= count <= 544:
                    return count
                else:
                    print("❌ Please enter a number between 1 and 544")
            except ValueError:
                print("❌ Please enter a valid number")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                sys.exit(0)

def run_experiment(num_files):
    """Run the experiment"""
    print(f"\n🔬 Starting experiment with {num_files} files...")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)
    
    try:
        # Create pipeline
        pipeline = AdvancedTradingPipeline()
        
        # Run the complete pipeline
        result = pipeline.run_complete_pipeline(
            num_files=num_files, 
            target='is_profitable'
        )
        
        print("\n" + "=" * 50)
        print("🎉 EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("📊 Check your results in MLflow UI:")
        print("   👉 http://127.0.0.1:5000")
        print("\n💡 What to do next:")
        print("   1. Open the MLflow URL above")
        print("   2. Click on your latest experiment")
        print("   3. Compare model performance")
        print("   4. Check feature importance")
        print("   5. Review trading signals")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during experiment: {str(e)}")
        print("💡 Try running with fewer files or check your data")
        return False

def check_mlflow_ui():
    """Check if MLflow UI is running"""
    print("\n🔍 Checking MLflow UI status...")
    try:
        import requests
        response = requests.get("http://127.0.0.1:5000", timeout=2)
        print("✅ MLflow UI is running at http://127.0.0.1:5000")
        return True
    except:
        print("⚠️  MLflow UI is not running")
        print("💡 To start it, run: python -m mlflow ui --port 5000")
        return False

def main():
    """Main function"""
    print_banner()
    
    # Check if we have data
    data_path = "dataBT"
    if not os.path.exists(data_path):
        print(f"❌ Data folder not found: {data_path}")
        print("💡 Make sure your CSV files are in the dataBT folder")
        return
    
    # Count CSV files
    csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    print(f"📁 Found {len(csv_files)} CSV files in {data_path}")
    
    if len(csv_files) == 0:
        print("❌ No CSV files found!")
        print("💡 Add your trading data CSV files to the dataBT folder")
        return
    
    # Check MLflow UI
    check_mlflow_ui()
    
    # Get user choice
    choice = get_user_choice()
    num_files = get_file_count(choice)
    
    # Confirm with user
    print(f"\n📋 Experiment Summary:")
    print(f"   📊 Files to process: {num_files}")
    print(f"   🎯 Target: Profitable trades")
    print(f"   🤖 Models: Random Forest, Gradient Boosting, Logistic Regression")
    print(f"   📈 Features: Technical indicators, time features, risk metrics")
    
    confirm = input(f"\n🚀 Ready to start? (y/n): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("👋 Experiment cancelled!")
        return
    
    # Run experiment
    success = run_experiment(num_files)
    
    if success:
        # Ask if user wants to run another experiment
        print(f"\n🔄 Want to run another experiment?")
        again = input("Enter 'y' to run again, or any other key to exit: ").strip().lower()
        if again in ['y', 'yes']:
            main()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Experiment stopped by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        print("💡 Please check your setup and try again")
