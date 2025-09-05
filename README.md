# Bank Statement Transaction Exporter

A Python tool for parsing bank statement PDFs and extracting transaction data into CSV format.

## Features

- Supports multiple Canadian banks (RBC, BMO, Scotiabank)
- OCR fallback for scanned PDFs
- Multiple interfaces: Web (Streamlit), Desktop GUI (tkinter), and Command Line
- CSV export functionality

## Quick Start

### Option 1: Enhanced Web Interface (Recommended for Cursor)
```bash
conda activate bank-statement
python enhanced_parser.py
```
**Perfect for Cursor's "Run" button!** Opens in browser with:
- 📁 File upload interface
- 🔄 Real-time processing with progress bars
- ✅ Automatic validation of transaction totals
- 📊 Summary statistics and processing results
- 💾 One-click CSV export

### Option 2: Easy Launcher
```bash
conda activate bank-statement
python run_parser.py
```
This will give you a menu to choose your preferred interface.

### Option 3: Command Line Interface
```bash
conda activate bank-statement
python bank_statement_transaction_exporter.py statement1.pdf statement2.pdf
```

### Option 4: Desktop GUI
```bash
conda activate bank-statement
python gui_parser.py
```

## Installation

1. Create and activate the conda environment:
```bash
conda create -n bank-statement python=3.13
conda activate bank-statement
pip install -r requirements.txt
conda install -c conda-forge poppler
```

2. Run the application using any of the methods above.

## Usage Examples

### Command Line
```bash
# Process single file
python bank_statement_transaction_exporter.py statement.pdf

# Process multiple files
python bank_statement_transaction_exporter.py *.pdf -o results.csv

# Force OCR for all pages
python bank_statement_transaction_exporter.py statement.pdf --force-ocr

# Show debug information
python bank_statement_transaction_exporter.py statement.pdf --debug
```

### In Cursor IDE
1. Click the "Run" button on `run_parser.py` for the interactive launcher
2. Or use the debug configurations in `.vscode/launch.json`
3. Or run directly in the integrated terminal

## Supported Banks
- RBC (Chequing and Mastercard)
- BMO
- Scotiabank

## File Structure
- `bank_statement_transaction_exporter.py` - Main parser with CLI and Streamlit interfaces
- `gui_parser.py` - Desktop GUI version
- `run_parser.py` - Interactive launcher
- `requirements.txt` - Python dependencies 
