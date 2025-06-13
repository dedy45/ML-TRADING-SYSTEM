#!/usr/bin/env python3
"""
Quick Fibonacci Analysis Runner
Pilihan cepat untuk analisis Fibonacci dengan parameter yang bisa disesuaikan
"""

import sys
import time
from fast_fibonacci_analyzer import FastFibonacciAnalyzer

def print_menu():
    """Print menu pilihan"""
    print("🔢 FIBONACCI ANALYSIS MENU")
    print("=" * 40)
    print("1. Quick Analysis (10 files, 500 rows each) - 30 seconds")
    print("2. Medium Analysis (30 files, 1000 rows each) - 2 minutes") 
    print("3. Large Analysis (50 files, 2000 rows each) - 5 minutes")
    print("4. Custom Analysis (choose your parameters)")
    print("5. Exit")
    print("=" * 40)

def quick_analysis():
    """Analisis cepat"""
    print("\n🚀 Quick Analysis Starting...")
    analyzer = FastFibonacciAnalyzer()
    
    start_time = time.time()
    file_stats = analyzer.load_sample_data(max_files=10, max_rows_per_file=500)
    
    if file_stats:
        report = analyzer.generate_comprehensive_report()
        elapsed = time.time() - start_time
        print(f"\n✅ Quick analysis completed in {elapsed:.1f} seconds!")
        return True
    return False

def medium_analysis():
    """Analisis menengah"""
    print("\n📊 Medium Analysis Starting...")
    analyzer = FastFibonacciAnalyzer()
    
    start_time = time.time()
    file_stats = analyzer.load_sample_data(max_files=30, max_rows_per_file=1000)
    
    if file_stats:
        report = analyzer.generate_comprehensive_report()
        elapsed = time.time() - start_time
        print(f"\n✅ Medium analysis completed in {elapsed:.1f} seconds!")
        return True
    return False

def large_analysis():
    """Analisis besar"""
    print("\n🔍 Large Analysis Starting...")
    analyzer = FastFibonacciAnalyzer()
    
    start_time = time.time()
    file_stats = analyzer.load_sample_data(max_files=50, max_rows_per_file=2000)
    
    if file_stats:
        report = analyzer.generate_comprehensive_report()
        elapsed = time.time() - start_time
        print(f"\n✅ Large analysis completed in {elapsed:.1f} seconds!")
        return True
    return False

def custom_analysis():
    """Analisis custom dengan parameter yang bisa disesuaikan"""
    print("\n⚙️ Custom Analysis Setup")
    print("-" * 30)
    
    try:
        max_files = int(input("Enter number of files to process (1-544): "))
        if max_files < 1 or max_files > 544:
            print("❌ Invalid number of files!")
            return False
        
        max_rows = int(input("Enter max rows per file (100-5000): "))
        if max_rows < 100 or max_rows > 5000:
            print("❌ Invalid number of rows!")
            return False
        
        print(f"\n🔧 Custom Analysis: {max_files} files, {max_rows} rows each")
        estimated_time = (max_files * max_rows) / 10000  # Rough estimation
        print(f"⏱️ Estimated time: {estimated_time:.1f} minutes")
        
        confirm = input("Continue? (y/n): ").lower()
        if confirm != 'y':
            return False
        
        analyzer = FastFibonacciAnalyzer()
        start_time = time.time()
        
        file_stats = analyzer.load_sample_data(max_files=max_files, max_rows_per_file=max_rows)
        
        if file_stats:
            report = analyzer.generate_comprehensive_report()
            elapsed = time.time() - start_time
            print(f"\n✅ Custom analysis completed in {elapsed:.1f} seconds!")
            return True
        
    except ValueError:
        print("❌ Invalid input! Please enter numbers only.")
        return False
    except KeyboardInterrupt:
        print("\n⏹️ Analysis cancelled by user")
        return False
    
    return False

def main():
    """Main function"""
    print("🎯 FAST FIBONACCI ANALYZER")
    print("Optimized for quick insights from trading data")
    print("")
    
    while True:
        try:
            print_menu()
            choice = input("Choose option (1-5): ").strip()
            
            if choice == '1':
                success = quick_analysis()
            elif choice == '2':
                success = medium_analysis()
            elif choice == '3':
                success = large_analysis()
            elif choice == '4':
                success = custom_analysis()
            elif choice == '5':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice! Please select 1-5")
                continue
            
            if success:
                print("\n💡 What's next?")
                print("   📊 Check the reports/ folder for detailed JSON results")
                print("   🔍 Review the console output for key insights")
                print("   🚀 Try a larger analysis for more comprehensive results")
                
                another = input("\nRun another analysis? (y/n): ").lower()
                if another != 'y':
                    print("👋 Thank you for using Fast Fibonacci Analyzer!")
                    break
            else:
                print("❌ Analysis failed. Please try again.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Exiting Fast Fibonacci Analyzer. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            print("Please try again.")

if __name__ == "__main__":
    main()
