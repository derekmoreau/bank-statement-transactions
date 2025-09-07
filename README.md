# Bank Statement Transaction Exporter

Parse Canadian bank statement PDFs and export clean CSVs via a simple web UI, desktop GUI, or CLI. Includes robust, layout-aware parsers for RBC and Scotiabank with validation against statement totals.

## Highlights

- **Web UI (Streamlit)**: Upload multiple PDFs, see parsed tables, validate totals, download combined CSV
- **RBC Chequing (layout + chars)**: Accurate table reconstruction, multi-line descriptions, date carry-forward
- **RBC Savings (layout + bands)**: Numeric bands from header midpoints; chequing-style description assembly
- **RBC Mastercard (chars + spillover)**: Handles description spillover; posting date extraction
- **Scotiabank (balance-delta)**: Signs computed solely from balance deltas; improved description extraction
- **OCR fallback**: When the text layer is broken (e.g., collapsed spacing), selectively OCR pages
- Interfaces: Web (Streamlit), Desktop GUI (tkinter), Command Line

Supported banks today: RBC (Chequing, Savings, Mastercard), Scotiabank (chequing/savings e-statements), BMO

## Quick Start

### Option 1: Enhanced Web Interface (Recommended)
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
This immediately launches the Streamlit app in your browser.

### Option 3: Command Line Interface
```bash
conda activate bank-statement
python bank_statement_transaction_exporter.py statement1.pdf statement2.pdf
```
Tips:
- Filenames ending in `YYYY-MM-DD.pdf` improve year inference for RBC Savings/Mastercard.
- Add `--force-ocr` if text is garbled or spacing is collapsed.

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

## Parsers & Validation

- **RBC Chequing**
  - Character-aware description assembly (from per-line chars), strict column ranges
  - Date carry-forward when the date cell is blank
  - Validates deposits/withdrawals totals (from statement summary)

- **RBC Savings**
  - Numeric column bands derived from header midpoints prevent bleed between columns
  - Chequing-style description assembly

- **RBC Mastercard**
  - Two-date lines (transaction/posting); posting date used in CSV
  - Description = spillover after dates + middle slice; robust to wrapping
  - Validates against “Calculating Your Balance” totals

- **Scotiabank**
  - Balance-delta method for sign (no keyword heuristics)
  - Improved description extraction across the full row span
  - Validates deposits/withdrawals totals and closing balance

## File Structure
- `bank_statement_transaction_exporter.py` — Main parser (Streamlit + CLI) and all bank parsers
- `enhanced_parser.py` — Rich Streamlit UI used by the launcher
- `gui_parser.py` — Desktop GUI (tkinter)
- `run_parser.py` — Launcher that starts the Streamlit app
- `requirements.txt` — Dependencies

## Notes

- OCR requires Tesseract available on PATH. If needed, set `TESSERACT_CMD` to the full path.
- For macOS, `poppler` is required for `pdf2image` (installed via conda as shown above).
