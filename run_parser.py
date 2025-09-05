#!/usr/bin/env python3
"""
Simple launcher for the bank statement parser.
This is what you should run when clicking "Run" in Cursor.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("🏦 Bank Statement Parser Launcher")
    print("=" * 40)
    
    # Check if we're in the right environment
    try:
        import pdfplumber
        import pytesseract
        import pdf2image
        print("✅ All dependencies are available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: conda activate bank-statement")
        return
    
    print("\nChoose how to run the parser:")
    print("1. 🖥️  Desktop GUI (tkinter)")
    print("2. 🌐 Web Interface (Streamlit)")
    print("3. 💻 Command Line Interface")
    print("4. 📁 Process files in current directory")
    
    try:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            print("\n🖥️  Starting Desktop GUI...")
            subprocess.run([sys.executable, "gui_parser.py"])
            
        elif choice == "2":
            print("\n🌐 Starting Web Interface...")
            print("The app will open in your browser at http://localhost:8501")
            subprocess.run([sys.executable, "-m", "streamlit", "run", "bank_statement_transaction_exporter.py"])
            
        elif choice == "3":
            print("\n💻 Command Line Interface")
            print("Usage examples:")
            print("  python bank_statement_transaction_exporter.py file1.pdf file2.pdf")
            print("  python bank_statement_transaction_exporter.py *.pdf -o results.csv")
            print("  python bank_statement_transaction_exporter.py --help")
            print()
            
            files_input = input("Enter PDF file paths (space-separated) or press Enter to select files: ").strip()
            
            if files_input:
                files = files_input.split()
                cmd = [sys.executable, "bank_statement_transaction_exporter.py"] + files
                subprocess.run(cmd)
            else:
                # Use the interactive file selection
                subprocess.run([sys.executable, "bank_statement_transaction_exporter.py"])
                
        elif choice == "4":
            print("\n📁 Processing all PDF files in current directory...")
            pdf_files = list(Path(".").glob("*.pdf"))
            
            if not pdf_files:
                print("No PDF files found in current directory.")
                return
            
            print(f"Found {len(pdf_files)} PDF file(s):")
            for f in pdf_files:
                print(f"  - {f.name}")
            
            confirm = input("\nProcess these files? (y/n): ").strip().lower()
            if confirm == 'y':
                cmd = [sys.executable, "bank_statement_transaction_exporter.py"] + [str(f) for f in pdf_files]
                subprocess.run(cmd)
            else:
                print("Cancelled.")
                
        else:
            print("Invalid choice. Exiting.")
            
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
