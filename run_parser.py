#!/usr/bin/env python3
"""
Simple launcher for the bank statement parser.
Clicking "Run" in Cursor will open the web app in your browser.
"""

import os
import sys
import subprocess
from pathlib import Path


def main():
    print("🏦 Bank Statement Parser Launcher")
    print("=" * 40)

    # Resolve the path to the Streamlit app (enhanced_parser.py)
    repo_dir = Path(__file__).resolve().parent
    app_path = str(repo_dir / "enhanced_parser.py")

    # Quick dependency check (optional, non-fatal)
    missing = []
    try:
        import pdfplumber  # noqa: F401
    except Exception:
        missing.append("pdfplumber")
    try:
        import pytesseract  # noqa: F401
    except Exception:
        missing.append("pytesseract")
    try:
        import pdf2image  # noqa: F401
    except Exception:
        missing.append("pdf2image")
    try:
        import streamlit  # noqa: F401
    except Exception:
        missing.append("streamlit")

    if missing:
        print(f"⚠️ Missing dependencies detected: {', '.join(missing)}")
        print("Ensure you've installed requirements.txt before running.")

    # Launch Streamlit app; it will open a browser tab automatically
    print("\n🌐 Starting Web Interface (Streamlit)...")
    print("The app will open in your browser at http://localhost:8501")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_path,
        "--server.headless=false",
    ]
    # Forward current environment (PATH, TESSDATA_PREFIX, etc.)
    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\nError launching Streamlit: {e}")


if __name__ == "__main__":
    main()
