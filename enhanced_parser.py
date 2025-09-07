#!/usr/bin/env python3
"""
Enhanced Bank Statement Parser with improved Streamlit interface.
This is the main file to run when clicking "Run" in Cursor.
"""

import io
import re
import pdfplumber
import pytesseract
import pandas as pd
import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes
from datetime import datetime

# Import all the parsing functions from the main file
from bank_statement_transaction_exporter import (
    extract_pages_text_with_ocr_fallback,
    parse_any_statement_from_text,
    convert_from_bytes,
    detect_statement_type
)

def main():
    st.set_page_config(
        page_title="Bank Statement Parser",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏦 Bank Statement Parser")
    st.markdown("**Extract transactions from Canadian bank statements and export to CSV**")
    
    # Sidebar for options
    with st.sidebar:
        st.header("⚙️ Options")
        force_ocr = st.checkbox("Force OCR for all pages", value=False, 
                               help="Slower but more robust for scanned PDFs")
        show_debug = st.checkbox("Show debug information", value=False,
                                help="Display extracted text for troubleshooting")
        auto_download = st.checkbox("Auto-download CSV", value=True,
                                   help="Automatically download CSV when processing is complete")
    
    # Main content area
    st.markdown("""
    ### 📋 How to use:
    1. **Upload** your PDF bank statements below
    2. **Review** the extracted transactions
    3. **Validate** that totals match the statement
    4. **Download** the combined CSV file
    """)
    
    # File upload
    files = st.file_uploader(
        "📁 Upload PDF Bank Statements", 
        type="pdf", 
        accept_multiple_files=True,
        help="Supports BMO, RBC (Chequing & Mastercard), and Scotiabank statements"
    )

    if not files:
        st.info("👆 Please upload one or more PDF files to get started")
        return

    # Processing section
    st.header("🔄 Processing Results")
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    combined = pd.DataFrame()
    processing_results = []
    
    for i, f in enumerate(files):
        status_text.text(f"Processing {i+1}/{len(files)}: {f.name}")
        progress_bar.progress((i * 4 + 1) / (len(files) * 4))  # 25% for start
        
        # Step 1: File reading
        status_text.text(f"Reading file {i+1}/{len(files)}: {f.name}")
        progress_bar.progress((i * 4 + 1) / (len(files) * 4))
        
        # Step 2: Text extraction/OCR
        status_text.text(f"Extracting text {i+1}/{len(files)}: {f.name}")
        progress_bar.progress((i * 4 + 2) / (len(files) * 4))
        
        # Step 3: Parsing transactions
        status_text.text(f"Parsing transactions {i+1}/{len(files)}: {f.name}")
        progress_bar.progress((i * 4 + 3) / (len(files) * 4))
        
        # Step 4: Processing complete
        status_text.text(f"Processing complete {i+1}/{len(files)}: {f.name}")
        progress_bar.progress((i * 4 + 4) / (len(files) * 4))
        
        with st.expander(f"📄 {f.name}", expanded=False):
            try:
                data = f.getvalue() if hasattr(f, "getvalue") else f.read()
                
                # For RBC statements, always use OCR as the text extraction loses spaces
                # Check filename first to detect RBC statements
                is_rbc = ('rbc' in f.name.lower() or 'royal' in f.name.lower() or 
                         'chequing' in f.name.lower() or 'banking' in f.name.lower())
                
                if force_ocr or is_rbc:
                    # OCR every page unconditionally with better settings
                    st.info("🔍 Using OCR for better text extraction...")
                    images = convert_from_bytes(data, dpi=300)
                    # Try different OCR configurations
                    pages_text = []
                    for img in images:
                        # Try with different OCR settings
                        try:
                            # First try with default settings
                            text = pytesseract.image_to_string(img, lang="eng")
                            if not text.strip() or len(text.strip()) < 50:
                                # Try with different page segmentation mode
                                text = pytesseract.image_to_string(img, lang="eng", config='--psm 6')
                            pages_text.append(text)
                        except Exception as e:
                            st.warning(f"OCR error: {e}")
                            pages_text.append("")
                    
                    full_text = re.sub(r'\s+', ' ', " ".join(pages_text)).strip()
                    first_page_text = pages_text[0] if pages_text else ""
                else:
                    # Try regular text extraction first
                    pages_text, full_text = extract_pages_text_with_ocr_fallback(data)
                    first_page_text = pages_text[0] if pages_text else ""
                    
                    # If text is concatenated (no spaces), fall back to OCR
                    if len(full_text) > 100 and ' ' not in full_text[:200]:
                        st.info("🔍 Text appears concatenated, switching to OCR...")
                        images = convert_from_bytes(data, dpi=300)
                        pages_text = [pytesseract.image_to_string(img, lang="eng") for img in images]
                        full_text = re.sub(r'\s+', ' ', " ".join(pages_text)).strip()
                        first_page_text = pages_text[0] if pages_text else ""

                if show_debug:
                    st.code((full_text[:800] + "…") if len(full_text) > 800 else full_text)
                
                # Always show some debug info for RBC statements
                if 'rbc' in first_page_text.lower() or 'royal bank' in first_page_text.lower():
                    st.subheader("🔍 RBC Debug Information")
                    
                    # Create debug output
                    debug_output = f"""🔍 RBC Debug Information
Statement type detected: {detect_statement_type(first_page_text)}

Full text length: {len(full_text)} characters

Processed text (full text):
{processed_text if 'processed_text' in locals() else 'Not processed yet'}

Table section detection:
"""
                    
                    # Check if we can find the transaction table
                    table_found = False
                    for pattern in ['Details of your account activity', 'Account Activity', 'Transaction Details', 'Opening Balance']:
                        if re.search(pattern, processed_text if 'processed_text' in locals() else full_text, re.IGNORECASE):
                            debug_output += f"✅ Found table section: '{pattern}'\n"
                            table_found = True
                            break
                    
                    if not table_found:
                        debug_output += "❌ Could not find transaction table section\n"
                        debug_output += "Looking for these patterns:\n"
                        for pattern in ['Details of your account activity', 'Account Activity', 'Transaction Details', 'Opening Balance']:
                            debug_output += f"- {pattern}\n"
                    
                    # Check for date patterns
                    date_matches = re.findall(r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*', processed_text if 'processed_text' in locals() else full_text, re.IGNORECASE)
                    if date_matches:
                        debug_output += f"✅ Found {len(date_matches)} date patterns: {date_matches[:5]}\n"
                    else:
                        debug_output += "❌ No date patterns found\n"
                    
                    # Check for amount patterns
                    amount_matches = re.findall(r'\$?[\d,]+\.\d{2}', processed_text if 'processed_text' in locals() else full_text)
                    if amount_matches:
                        debug_output += f"✅ Found {len(amount_matches)} amount patterns: {amount_matches[:5]}\n"
                    else:
                        debug_output += "❌ No amount patterns found\n"
                    
                    # Display debug info
                    st.write(f"**Statement type detected:** {detect_statement_type(first_page_text)}")
                    st.write(f"**Full text length:** {len(full_text)} characters")
                    
                    # Show processed text
                    processed_text = full_text
                    processed_text = re.sub(r'(\$?[\d,]+\.\d{2})', r' \1 ', processed_text)
                    processed_text = re.sub(r'(\d{1,2}(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)', r' \1 ', processed_text, flags=re.IGNORECASE)
                    processed_text = re.sub(r'\s+', ' ', processed_text)
                    
                    st.write(f"**Processed text (full text):**")
                    st.code(processed_text)
                    
                    # Update debug output with processed text
                    debug_output = debug_output.replace("Not processed yet", processed_text)
                    
                    # Display the debug info
                    st.write("**Table section detection:**")
                    if table_found:
                        st.success("✅ Found table section")
                    else:
                        st.warning("❌ Could not find transaction table section")
                        st.write("**Looking for these patterns:**")
                        for pattern in ['Details of your account activity', 'Account Activity', 'Transaction Details', 'Opening Balance']:
                            st.write(f"- {pattern}")
                    
                    st.write("**Date patterns:**")
                    if date_matches:
                        st.success(f"✅ Found {len(date_matches)} date patterns: {date_matches[:5]}")
                    else:
                        st.warning("❌ No date patterns found")
                    
                    st.write("**Amount patterns:**")
                    if amount_matches:
                        st.success(f"✅ Found {len(amount_matches)} amount patterns: {amount_matches[:5]}")
                    else:
                        st.warning("❌ No amount patterns found")
                    
                    
                    # Copy button for full debug output
                    if st.button("📋 Copy Full Debug Output", key=f"copy_debug_output_{i}"):
                        st.text_area("Debug Output (copy this):", debug_output, height=300)
                        st.write("Debug output displayed above - copy the text from the text area!")

                df, summary = parse_any_statement_from_text(first_page_text, full_text, data, getattr(f, 'name', None))
                
                # Add transactions info to debug output for RBC statements
                if 'rbc' in first_page_text.lower() or 'royal bank' in first_page_text.lower():
                    if not df.empty:
                        debug_output += f"\n✅ Found {len(df)} transactions:\n"
                        for idx, row in df.iterrows():
                            debug_output += f"  {idx+1}. {row['Date']} - {row['Description']} - ${row['Amount']:.2f}\n"
                        
                        # Display transactions in the debug section
                        st.write("**Transactions found:**")
                        st.success(f"✅ Found {len(df)} transactions:")
                        for idx, row in df.iterrows():
                            st.write(f"  {idx+1}. {row['Date']} - {row['Description']} - ${row['Amount']:.2f}")
                    else:
                        debug_output += f"\n❌ No transactions found\n"
                        st.write("**Transactions found:**")
                        st.warning("❌ No transactions found")
                
                if df.empty:
                    st.warning("⚠️ No transactions found — PDF may be low-quality scan or needs pattern tweaks")
                    processing_results.append({
                        'file': f.name,
                        'status': 'No transactions found',
                        'transactions': 0,
                        'validation': 'N/A'
                    })
                    continue

                # Display transactions
                st.write(f"✅ **Found {len(df)} transactions**")
                st.dataframe(df.reset_index(drop=True), use_container_width=True)

                # Validation section
                validation_passed = True
                if summary:
                    pos_sum = round(df.loc[df['Amount'] > 0, 'Amount'].sum(), 2)
                    neg_sum = round(df.loc[df['Amount'] < 0, 'Amount'].sum(), 2)
                    pos_ok = abs(pos_sum - round(summary['pos_value'], 2)) < 0.01
                    neg_ok = abs(neg_sum - round(summary['neg_value'], 2)) < 0.01
                    
                    st.subheader("🔍 Validation Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if pos_ok:
                            st.success(f"✅ **{summary['pos_label']}**: {pos_sum:.2f} (expected: {summary['pos_value']:.2f})")
                        else:
                            st.error(f"❌ **{summary['pos_label']}**: {pos_sum:.2f} (expected: {summary['pos_value']:.2f})")
                            validation_passed = False
                    
                    with col2:
                        if neg_ok:
                            st.success(f"✅ **{summary['neg_label']}**: {neg_sum:.2f} (expected: {summary['neg_value']:.2f})")
                        else:
                            st.error(f"❌ **{summary['neg_label']}**: {neg_sum:.2f} (expected: {summary['neg_value']:.2f})")
                            validation_passed = False
                else:
                    st.info("ℹ️ No summary block detected for validation")
                    validation_passed = "N/A"
                
                processing_results.append({
                    'file': f.name,
                    'status': 'Success' if validation_passed else 'Validation failed',
                    'transactions': len(df),
                    'validation': 'Passed' if validation_passed is True else 'Failed' if validation_passed is False else 'N/A'
                })

                df['Source File'] = f.name
                combined = pd.concat([combined, df], ignore_index=True)
                
            except Exception as e:
                st.error(f"❌ Error processing {f.name}: {str(e)}")
                processing_results.append({
                    'file': f.name,
                    'status': f'Error: {str(e)}',
                    'transactions': 0,
                    'validation': 'N/A'
                })

    # Summary section
    if not combined.empty:
        st.header("📊 Summary")
        
        # Processing summary table
        summary_df = pd.DataFrame(processing_results)
        st.subheader("Processing Summary")
        st.dataframe(summary_df, use_container_width=True)
        
        # Combined results
        st.subheader("📋 All Combined Transactions")
        st.write(f"**Total transactions:** {len(combined)}")
        st.write(f"**Total files processed:** {len(files)}")
        
        # Display combined data
        st.dataframe(combined, use_container_width=True)
        
        # Download section
        st.subheader("💾 Export to CSV")
        csv_data = combined.to_csv(index=False)
        
        # Auto-download if enabled
        if auto_download and len(combined) > 0:
            st.success("🎉 Processing complete! CSV is ready for download.")
        
        st.download_button(
            "📥 Download Combined CSV",
            csv_data,
            file_name="bank_statements_export.csv",
            mime="text/csv",
            help="Download all transactions as a CSV file"
        )
        
        # Additional statistics
        if len(combined) > 0:
            st.subheader("📈 Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_positive = combined[combined['Amount'] > 0]['Amount'].sum()
                st.metric("Total Deposits", f"${total_positive:.2f}")
            
            with col2:
                total_negative = combined[combined['Amount'] < 0]['Amount'].sum()
                st.metric("Total Withdrawals", f"${total_negative:.2f}")
            
            with col3:
                net_amount = combined['Amount'].sum()
                st.metric("Net Amount", f"${net_amount:.2f}")
            
            with col4:
                avg_transaction = combined['Amount'].mean()
                st.metric("Avg Transaction", f"${avg_transaction:.2f}")
    else:
        st.error("❌ No transactions were extracted from any of the uploaded files.")
        st.info("💡 Try enabling 'Force OCR for all pages' in the sidebar for scanned PDFs.")

if __name__ == "__main__":
    main()
