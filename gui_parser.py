#!/usr/bin/env python3
"""
Simple GUI version of the bank statement parser using tkinter.
This provides a desktop interface without requiring Streamlit.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import pandas as pd
from bank_statement_transaction_exporter import (
    extract_pages_text_with_ocr_fallback, 
    parse_any_statement_from_text,
    convert_from_bytes,
    pytesseract
)
import re

class BankStatementParserGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Bank Statement Parser")
        self.root.geometry("800x600")
        
        # Variables
        self.pdf_files = []
        self.results = pd.DataFrame()
        
        self.setup_ui()
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Bank Statement Parser", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # File selection
        ttk.Label(main_frame, text="PDF Files:").grid(row=1, column=0, sticky=tk.W, pady=5)
        
        file_frame = ttk.Frame(main_frame)
        file_frame.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        file_frame.columnconfigure(0, weight=1)
        
        self.file_listbox = tk.Listbox(file_frame, height=3)
        self.file_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        scrollbar = ttk.Scrollbar(file_frame, orient=tk.VERTICAL, command=self.file_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.file_listbox.configure(yscrollcommand=scrollbar.set)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=3, pady=10)
        
        ttk.Button(button_frame, text="Select PDF Files", 
                  command=self.select_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear Files", 
                  command=self.clear_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Process Files", 
                  command=self.process_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Export CSV", 
                  command=self.export_csv).pack(side=tk.LEFT, padx=5)
        
        # Options frame
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding="5")
        options_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        self.force_ocr_var = tk.BooleanVar()
        ttk.Checkbutton(options_frame, text="Force OCR for all pages", 
                       variable=self.force_ocr_var).pack(side=tk.LEFT, padx=5)
        
        self.debug_var = tk.BooleanVar()
        ttk.Checkbutton(options_frame, text="Show debug info", 
                       variable=self.debug_var).pack(side=tk.LEFT, padx=5)
        
        # Results area
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="5")
        results_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=15, width=80)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def select_files(self):
        files = filedialog.askopenfilenames(
            title="Select PDF Statement Files",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        
        for file in files:
            if file not in self.pdf_files:
                self.pdf_files.append(file)
                self.file_listbox.insert(tk.END, os.path.basename(file))
        
        self.status_var.set(f"Selected {len(self.pdf_files)} file(s)")
    
    def clear_files(self):
        self.pdf_files.clear()
        self.file_listbox.delete(0, tk.END)
        self.results_text.delete(1.0, tk.END)
        self.results = pd.DataFrame()
        self.status_var.set("Files cleared")
    
    def process_files(self):
        if not self.pdf_files:
            messagebox.showwarning("No Files", "Please select PDF files first.")
            return
        
        self.results_text.delete(1.0, tk.END)
        self.results = pd.DataFrame()
        
        self.status_var.set("Processing files...")
        self.root.update()
        
        for i, pdf_file in enumerate(self.pdf_files):
            try:
                self.results_text.insert(tk.END, f"Processing: {os.path.basename(pdf_file)}\n")
                self.root.update()
                
                with open(pdf_file, 'rb') as f:
                    data = f.read()
                
                if self.force_ocr_var.get():
                    # OCR every page unconditionally
                    images = convert_from_bytes(data, dpi=300)
                    pages_text = [pytesseract.image_to_string(img, lang="eng") for img in images]
                    full_text = re.sub(r'\s+', ' ', " ".join(pages_text)).strip()
                else:
                    pages_text, full_text = extract_pages_text_with_ocr_fallback(data)
                
                if self.debug_var.get():
                    self.results_text.insert(tk.END, f"Debug - First 200 chars: {full_text[:200]}...\n")
                
                first_page_text = pages_text[0] if pages_text else ""
                df, summary = parse_any_statement_from_text(first_page_text, full_text, data, os.path.basename(pdf_file))
                
                if df.empty:
                    self.results_text.insert(tk.END, f"  ⚠️  No transactions found\n")
                    continue
                
                self.results_text.insert(tk.END, f"  ✅ Found {len(df)} transactions\n")
                
                if summary:
                    pos_sum = round(df.loc[df['Amount'] > 0, 'Amount'].sum(), 2)
                    neg_sum = round(df.loc[df['Amount'] < 0, 'Amount'].sum(), 2)
                    pos_ok = abs(pos_sum - round(summary['pos_value'], 2)) < 0.01
                    neg_ok = abs(neg_sum - round(summary['neg_value'], 2)) < 0.01
                    
                    self.results_text.insert(tk.END, 
                        f"  📊 {summary['pos_label']}: {pos_sum:.2f} (expected: {summary['pos_value']:.2f}) {'✅' if pos_ok else '❌'}\n")
                    self.results_text.insert(tk.END, 
                        f"  📊 {summary['neg_label']}: {neg_sum:.2f} (expected: {summary['neg_value']:.2f}) {'✅' if neg_ok else '❌'}\n")
                
                df['Source File'] = os.path.basename(pdf_file)
                self.results = pd.concat([self.results, df], ignore_index=True)
                
            except Exception as e:
                self.results_text.insert(tk.END, f"  ❌ Error: {str(e)}\n")
                continue
        
        if not self.results.empty:
            self.results_text.insert(tk.END, f"\n🎉 Success! Found {len(self.results)} total transactions\n")
            self.status_var.set(f"Processed {len(self.pdf_files)} file(s) - {len(self.results)} transactions found")
        else:
            self.results_text.insert(tk.END, "\n❌ No transactions found in any files\n")
            self.status_var.set("No transactions found")
    
    def export_csv(self):
        if self.results.empty:
            messagebox.showwarning("No Data", "No data to export. Please process files first.")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save CSV File",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.results.to_csv(filename, index=False)
                messagebox.showinfo("Success", f"Data exported to {filename}")
                self.status_var.set(f"Exported to {os.path.basename(filename)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {str(e)}")

def main():
    root = tk.Tk()
    app = BankStatementParserGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
