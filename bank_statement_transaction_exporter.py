import io
import re
import pdfplumber
import pytesseract
import pandas as pd
import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes
from datetime import datetime

# =========================
# Date helpers
# =========================

MONTHS_MAP = {
    'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
    'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
    'aug': 8, 'august': 8, 'sep': 9, 'sept': 9, 'september': 9,
    'oct': 10, 'october': 10, 'nov': 11, 'november': 11, 'dec': 12, 'december': 12
}

def parse_statement_period(full_text: str):
    """
    Supports:
      - From December 17, 2020 to January 15, 2021 (RBC chequing)
      - STATEMENT FROM DEC 30, 2020 TO JAN 25, 2021 (RBC MC)
      - April 1 to April 30, 2021 (Scotiabank)
      - Statement period  Dec. 5, 2024 - Jan. 4, 2025 (BMO)
    """
    patterns = [
        r'(?i)\bfrom\s+([A-Za-z\.]+\.?\s+\d{1,2},?\s+\d{4})\s+to\s+([A-Za-z\.]+\.?\s+\d{1,2},?\s+\d{4})',
        r'(?i)\bstatement\s+from\s+([A-Z]{3,}\s+\d{1,2},?\s+\d{4})\s+to\s+([A-Z]{3,}\s+\d{1,2},?\s+\d{4})',
        r'(?i)\bstatement\s*period\s+([A-Za-z\.]+\.?\s+\d{1,2},?\s+\d{4})\s*[-–]\s*([A-Za-z\.]+\.?\s+\d{1,2},?\s+\d{4})',
        r'(?i)\b([A-Za-z\.]+\.?\s+\d{1,2})\s+to\s+([A-Za-z\.]+\.?\s+\d{1,2}),?\s+(\d{4})',
    ]

    def norm_date(ds: str):
        ds = ds.replace('.', '').strip()
        for fmt in ["%b %d, %Y", "%B %d, %Y", "%b %d %Y", "%B %d %Y", "%b %d,%Y", "%B %d,%Y"]:
            try:
                return datetime.strptime(ds, fmt)
            except ValueError:
                continue
        return None

    for pat in patterns[:3]:
        m = re.search(pat, full_text)
        if m:
            d1 = norm_date(m.group(1))
            d2 = norm_date(m.group(2))
            return d1, d2

    # "April 1 to April 30, 2021"
    m = re.search(patterns[3], full_text)
    if m:
        d1s, d2s, year = m.group(1), m.group(2), int(m.group(3))
        def md_to_dt(md):
            parts = md.replace('.', '').split()
            if len(parts) != 2: return None
            mon = MONTHS_MAP.get(parts[0].lower())
            try: day = int(parts[1])
            except: return None
            return datetime(year, mon, day) if mon else None
        return md_to_dt(d1s), md_to_dt(d2s)

    return None, None

def interpret_transaction_date(raw: str, start_dt, end_dt):
    """
    Convert "Dec. 4", "DEC 28", "21 Dec", "Apr 30" -> "DD/MM/YYYY" using the statement period for the year.
    """
    s = raw.replace('.', '').strip()
    parts = s.split()
    mon = day = None

    if len(parts) >= 2:
        a, b = parts[0].lower(), parts[1].lower()
        if a in MONTHS_MAP:
            mon = MONTHS_MAP[a]
            try: day = int(parts[1])
            except: day = None
        elif b in MONTHS_MAP:
            mon = MONTHS_MAP[b]
            try: day = int(parts[0])
            except: day = None

    if not mon or not day:
        return raw

    if not start_dt or not end_dt:
        year = datetime.now().year
    else:
        if mon == start_dt.month:
            year = start_dt.year
        elif mon == end_dt.month:
            year = end_dt.year
        else:
            year = start_dt.year

    try:
        return datetime(year, mon, day).strftime("%d/%m/%Y")
    except ValueError:
        return raw

# =========================
# OCR-aware text extraction
# =========================

def _space_ratio(s: str) -> float:
    if not s: return 0.0
    spaces = sum(1 for ch in s if ch.isspace())
    return spaces / max(1, len(s))

def extract_pages_text_with_ocr_fallback(data_bytes: bytes, ocr_threshold_chars=40, ocr_min_space_ratio=0.03, dpi=300):
    """
    Returns (pages_text:list[str], full_text:str).

    Strategy:
      1) Extract text per page with pdfplumber.
      2) For each page, if too few characters OR space ratio < threshold, OCR that page.
      3) Join OCR-corrected pages; normalize whitespace to single spaces.
    """
    pages_text = []
    # pass 1: raw text
    with pdfplumber.open(io.BytesIO(data_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            pages_text.append(t)

    # pass 2: OCR pages with broken spacing or too little text
    for idx, t in enumerate(pages_text):
        txt = t or ""
        if (len(txt.strip()) < ocr_threshold_chars) or (_space_ratio(txt) < ocr_min_space_ratio):
            # OCR just this page
            images = convert_from_bytes(data_bytes, first_page=idx+1, last_page=idx+1, dpi=dpi)
            if images:
                ocr_text = pytesseract.image_to_string(images[0], lang="eng")
                pages_text[idx] = ocr_text or ""

    # Build normalized full text
    full = " ".join(pages_text)
    full = re.sub(r'\s+', ' ', full).strip()
    return pages_text, full

# =========================
# Summary detection (optional validation)
# =========================

def parse_summary_generic(full_text: str):
    """Return a harmonized summary dict or None."""
    def f2(x): return float(x.replace('$', '').replace(',', ''))

    # RBC chequing
    m_dep = re.search(r'(?i)Total\s+deposits\s+into\s+your\s+account.*?([+\-]?\$?[\d,]+\.\d{2})', full_text)
    m_wdr = re.search(r'(?i)Total\s+withdrawals\s+from\s+your\s+account.*?([+\-]?\$?[\d,]+\.\d{2})', full_text)
    if m_dep and m_wdr:
        return {'pos_label': 'Total deposits', 'pos_value': +f2(m_dep.group(1)),
                'neg_label': 'Total withdrawals', 'neg_value': -abs(f2(m_wdr.group(1)))}
    
    # RBC chequing (alternative format)
    m_dep_alt = re.search(r'(?i)Total\s+deposits.*?([+\-]?\$?[\d,]+\.\d{2})', full_text)
    m_wdr_alt = re.search(r'(?i)Total\s+withdrawals.*?([+\-]?\$?[\d,]+\.\d{2})', full_text)
    if m_dep_alt and m_wdr_alt:
        return {'pos_label': 'Total deposits', 'pos_value': +f2(m_dep_alt.group(1)),
                'neg_label': 'Total withdrawals', 'neg_value': -abs(f2(m_wdr_alt.group(1)))}

    # RBC Mastercard
    m_pay = re.search(r'(?i)Payments\s*&\s*credits\s+(-?\$?[\d,]+\.\d{2}|\([\d,]+\.\d{2}\))', full_text)
    m_pur = re.search(r'(?i)Purchases\s*&\s*debits\s+(\$?[\d,]+\.\d{2})', full_text)
    if m_pay and m_pur:
        pay_val = m_pay.group(1).strip()
        if pay_val.startswith('(') and pay_val.endswith(')'):
            pay_num = -float(pay_val.strip('()').replace(',', ''))
        else:
            pay_num = float(pay_val.replace('$', '').replace(',', ''))
        return {'pos_label': 'Purchases & debits', 'pos_value': +f2(m_pur.group(1)),
                'neg_label': 'Payments & credits', 'neg_value': -abs(pay_num)}

    # BMO
    m_bmo_pay = re.search(r'(?i)Payments\s+and\s+credits\s+(-?\$?[\d,]+\.\d{2})', full_text)
    m_bmo_pur = re.search(r'(?i)Purchases\s+and\s+other\s+charges\s+([+\-]?\$?[\d,]+\.\d{2})', full_text)
    if m_bmo_pay and m_bmo_pur:
        return {'pos_label': 'Purchases and other charges', 'pos_value': +f2(m_bmo_pur.group(1)),
                'neg_label': 'Payments and credits', 'neg_value': -abs(f2(m_bmo_pay.group(1)))}

    # Scotiabank
    m_sco_w = re.search(r'(?i)Minus\s+total\s+withdrawals\s+\$?([\d,]+\.\d{2})', full_text)
    m_sco_d = re.search(r'(?i)Plus\s+total\s+deposits\s+\$?([\d,]+\.\d{2})', full_text)
    if m_sco_w and m_sco_d:
        return {'pos_label': 'Total deposits', 'pos_value': +f2(m_sco_d.group(1)),
                'neg_label': 'Total withdrawals', 'neg_value': -abs(f2(m_sco_w.group(1)))}

    return None

# =========================
# Statement type detection
# =========================

def detect_statement_type(first_page_text: str):
    t = (first_page_text or "").lower()
    if 'rbc' in t or 'royal bank of canada' in t:
        if 'mastercard' in t or 'credit card payment' in t:
            return 'RBC_MC'
        return 'RBC_CHEQUING'
    if 'scotiabank' in t or 'preferred package' in t:
        return 'SCOTIA_CHEQUING'
    if ('bmo' in t) or ('statement period' in t and 'total interest charges' in t):
        return 'BMO'
    return 'UNKNOWN'

# =========================
# Amount normalization
# =========================

AMT = r'(-?\$?\(?\d{1,3}(?:,\d{3})*\.\d{2}\)?(?:\s*CR)?)'

def parse_amount(s: str) -> float:
    """
    Handle: $123.45, 123.45, (123.45), 123.45 CR
    """
    s = s.strip()
    is_cr = s.endswith('CR')
    s = s.replace('CR', '').strip()
    neg_paren = s.startswith('(') and s.endswith(')')
    s = s.strip('()').replace('$', '').replace(',', '')
    try:
        val = float(s)
    except:
        val = 0.0
    if is_cr or neg_paren:
        val = -val
    return val

# =========================
# Bank-specific parsers (using normalized full_text)
# =========================

def parse_bmo_from_text(full_text, start_dt, end_dt) -> pd.DataFrame:
    """
    BMO style:  Mon dd  Mon dd  <desc>  <amount or (amount) or amount CR>
    """
    pat = re.compile(r'([A-Za-z]{3}\.?\s*\d{1,2})\s+([A-Za-z]{3}\.?\s*\d{1,2})\s+(.*?)(?=(?:[A-Za-z]{3}\.?\s*\d{1,2}\s+[A-Za-z]{3}\.?\s*\d{1,2})|$)', re.IGNORECASE)
    amt_re = re.compile(AMT)
    rows = []
    for m in pat.finditer(full_text):
        td_raw, pd_raw, body = m.group(1), m.group(2), m.group(3).strip()
        am_match = None
        for am in amt_re.finditer(body):
            am_match = am
        if not am_match:
            continue
        amt_str = am_match.group(1).strip()
        desc = body[:am_match.start()].strip()
        rows.append({
            'Date': interpret_transaction_date(td_raw, start_dt, end_dt),
            'Description': desc,
            'Amount': parse_amount(amt_str)
        })
    return pd.DataFrame(rows)

def parse_rbc_mc_from_text(full_text, start_dt, end_dt) -> pd.DataFrame:
    """
    RBC Mastercard:  MON dd  MON dd  <desc>  <amount>
    """
    pat = re.compile(r'\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+(\d{1,2})\s+(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+(\d{1,2})\s+(.*?)(?=(?:\b(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+\d{1,2}\b)|$)', re.IGNORECASE)
    amt_re = re.compile(AMT)
    rows = []
    for m in pat.finditer(full_text):
        td_raw = f"{m.group(1)} {m.group(2)}"
        pd_raw = f"{m.group(3)} {m.group(4)}"
        body = m.group(5).strip()
        am_match = None
        for am in amt_re.finditer(body):
            am_match = am
        if not am_match:
            continue
        amt_str = am_match.group(1).strip()
        desc = body[:am_match.start()].strip()
        rows.append({
            'Date': interpret_transaction_date(td_raw, start_dt, end_dt),
            'Description': desc,
            'Amount': parse_amount(amt_str)
        })
    return pd.DataFrame(rows)

def parse_rbc_chequing_from_text(full_text, start_dt, end_dt) -> pd.DataFrame:
    """
    RBC chequing: handles the table format with Date, Description, Withdrawals ($), Deposits ($), Balance ($)
    Date format is "DD Mon" (e.g., "21 Jul", "29 Jul")
    """
    # The text is completely concatenated, so we need to be more aggressive with preprocessing
    processed_text = full_text
    
    # Add spaces around amounts (more aggressive)
    processed_text = re.sub(r'(\$?[\d,]+\.\d{2})', r' \1 ', processed_text)
    
    # Add spaces around dates (more aggressive)
    processed_text = re.sub(r'(\d{1,2}(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)', r' \1 ', processed_text, flags=re.IGNORECASE)
    
    # Add spaces around common transaction keywords
    transaction_keywords = [
        'Contactless Interac purchase', 'Interac purchase', 'Payroll Deposit', 'e-Transfer sent',
        'Online Transfer', 'Health/Dental Claim', 'Monthly fee', 'Opening Balance', 'Closing Balance'
    ]
    
    for keyword in transaction_keywords:
        processed_text = re.sub(r'(' + re.escape(keyword) + r')', r' \1 ', processed_text, flags=re.IGNORECASE)
    
    # Add spaces around common patterns
    processed_text = re.sub(r'(Details of your account activity)', r' \1 ', processed_text, flags=re.IGNORECASE)
    processed_text = re.sub(r'(Summary of your account for this period)', r' \1 ', processed_text, flags=re.IGNORECASE)
    
    # Normalize whitespace
    processed_text = re.sub(r'\s+', ' ', processed_text)
    
    # Look for the transaction table section with more flexible patterns
    table_start_patterns = [
        r'Details of your account activity',
        r'Details of your account activity - continued',
        r'Account Activity',
        r'Transaction Details',
        r'Opening Balance',  # Sometimes transactions start after opening balance
        r'Your opening balance'  # Alternative opening balance text
    ]
    
    table_start = 0
    for pattern in table_start_patterns:
        match = re.search(pattern, processed_text, re.IGNORECASE)
        if match:
            table_start = match.end()
            break
    
    # Extract the table content
    table_text = processed_text[table_start:]
    
    # Remove the summary section at the end
    summary_patterns = [
        r'Summary of your account for this period',
        r'Important information about your account',
        r'Please check this Account Statement',
        r'Monthly fee',
        r'Closing Balance'
    ]
    
    for pattern in summary_patterns:
        match = re.search(pattern, table_text, re.IGNORECASE)
        if match:
            table_text = table_text[:match.start()]
            break
    
    # If we still can't find the table section, try a different approach
    # Look for patterns that indicate transactions directly in the concatenated text
    if not table_text.strip():
        # Try to find transaction patterns in the original text
        # Look for date patterns followed by amounts
        transaction_pattern = re.compile(
            r'(\d{1,2}(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)'  # Date
            r'([^$]*?)'  # Description (everything until we hit a dollar amount)
            r'(\$?[\d,]+\.\d{2})'  # Amount
            r'(?=\d{1,2}(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*|$)',  # Lookahead for next date or end
            re.IGNORECASE | re.MULTILINE
        )
        table_text = full_text
    
    # Now look for transaction patterns in the processed text
    # Pattern: Date (DD Mon) followed by description and amounts
    transaction_pattern = re.compile(
        r'(\d{1,2}(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)\s+'  # Date
        r'([^$]+?)\s*'  # Description (everything until we hit a dollar amount)
        r'(\$?[\d,]+\.\d{2})?\s*'  # Optional withdrawal amount
        r'(\$?[\d,]+\.\d{2})?\s*'  # Optional deposit amount  
        r'(\$?[\d,]+\.\d{2})?\s*'  # Optional balance amount
        r'(?=\d{1,2}(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*|$)',  # Lookahead for next date or end
        re.IGNORECASE | re.MULTILINE
    )
    
    rows = []
    
    for match in transaction_pattern.finditer(table_text):
        date_str = match.group(1).strip()
        description = match.group(2).strip()
        withdrawal_str = match.group(3)
        deposit_str = match.group(4)
        balance_str = match.group(5)
        
        # Skip empty or invalid rows
        if not description or description.lower().strip() in ['', 'opening balance', 'closing balance']:
            continue
            
        # Determine amount and sign
        amount = 0.0
        if withdrawal_str and withdrawal_str.strip():
            amount = -parse_amount(withdrawal_str)
        elif deposit_str and deposit_str.strip():
            amount = parse_amount(deposit_str)
        else:
            # Try to determine from description
            desc_lower = description.lower()
            if any(k in desc_lower for k in ['deposit', 'payroll', 'refund', 'credit', 'received']):
                # This should be a deposit, but we don't have the amount
                continue
            elif any(k in desc_lower for k in ['withdrawal', 'purchase', 'sent', 'interac', 'payment', 'transfer', 'fee']):
                # This should be a withdrawal, but we don't have the amount
                continue
            else:
                continue  # Skip if we can't determine amount
        
        rows.append({
            'Date': interpret_transaction_date(date_str, start_dt, end_dt),
            'Description': description,
            'Amount': amount
        })
    
    return pd.DataFrame(rows)

def parse_scotia_chequing_from_text(full_text, start_dt, end_dt) -> pd.DataFrame:
    """
    Scotiabank chequing:  Mon dd  <desc>  <amount>  <balance>
    """
    pat = re.compile(
        rf'\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{{1,2}})\s+(.*?)\s+{AMT}\s+(\d{{1,3}}(?:,\d{{3}})*\.\d{{2}})',
        re.IGNORECASE
    )
    dep_kw = ('deposit', 'payroll', 'dep.', 'dépôt')
    wdr_kw = ('withdrawal', 'cheque', 'bill payment', 'ts bill payment', 'service charge', 'fee', 'insurance', 'mb-transfer', 'misc. payment', 'retrait', 'paiement', 'frais')

    rows = []
    for mon, day, desc, amt_str, _bal in pat.findall(full_text):
        dl = desc.lower()
        amt = parse_amount(amt_str)
        if any(k in dl for k in dep_kw):
            signed = +amt
        elif any(k in dl for k in wdr_kw):
            signed = -amt
        else:
            signed = +amt if ('deposit' in dl or 'dépôt' in dl) else -amt
        rows.append({
            'Date': interpret_transaction_date(f"{mon} {day}", start_dt, end_dt),
            'Description': desc.strip(),
            'Amount': signed
        })
    return pd.DataFrame(rows)

# =========================
# Master dispatcher
# =========================

def parse_any_statement_from_text(first_page_text: str, full_text: str):
    stype = detect_statement_type(first_page_text)
    start_dt, end_dt = parse_statement_period(full_text)
    summary = parse_summary_generic(full_text)

    if stype == 'BMO':
        df = parse_bmo_from_text(full_text, start_dt, end_dt)
    elif stype == 'RBC_MC':
        df = parse_rbc_mc_from_text(full_text, start_dt, end_dt)
    elif stype == 'RBC_CHEQUING':
        df = parse_rbc_chequing_from_text(full_text, start_dt, end_dt)
    elif stype == 'SCOTIA_CHEQUING':
        df = parse_scotia_chequing_from_text(full_text, start_dt, end_dt)
    else:
        # Try all if we couldn't detect type
        for fn in (parse_bmo_from_text, parse_rbc_mc_from_text, parse_rbc_chequing_from_text, parse_scotia_chequing_from_text):
            df = fn(full_text, start_dt, end_dt)
            if not df.empty:
                break

    if df.empty:
        return pd.DataFrame(), summary
    df._summary = summary
    return df, summary

# =========================
# Streamlit app
# =========================

def main():
    st.title("Multi-Bank Statement Parser (OCR-aware + RBC spacing fix)")
    st.write("""
    • Upload one or more PDF statements (BMO, RBC chequing, RBC Mastercard, Scotiabank).  
    • If a page's text has almost **no spaces**, we auto-OCR just that page.  
    • Dates normalized to **DD/MM/YYYY** using the statement period.  
    • Review per-file results and download a combined CSV.
    """)

    files = st.file_uploader("Upload PDF statements", type="pdf", accept_multiple_files=True)
    force_ocr = st.checkbox("Force OCR for all pages (slow, but robust)", value=False)
    show_debug = st.checkbox("Show debug text (first 800 chars per file)", value=False)

    if not files:
        return

    combined = pd.DataFrame()

    for f in files:
        st.markdown(f"**File:** {f.name}")
        data = f.getvalue() if hasattr(f, "getvalue") else f.read()

        if force_ocr:
            # OCR every page unconditionally
            images = convert_from_bytes(data, dpi=300)
            pages_text = [pytesseract.image_to_string(img, lang="eng") for img in images]
            full_text = re.sub(r'\s+', ' ', " ".join(pages_text)).strip()
        else:
            pages_text, full_text = extract_pages_text_with_ocr_fallback(data)

        if show_debug:
            st.code((full_text[:800] + "…") if len(full_text) > 800 else full_text)

        first_page_text = pages_text[0] if pages_text else ""

        df, summary = parse_any_statement_from_text(first_page_text, full_text)
        if df.empty:
            st.warning("No transactions parsed — PDF may be low-quality scan or needs minor pattern tweaks.")
            continue

        st.dataframe(df.reset_index(drop=True))

        if summary:
            pos_sum = round(df.loc[df['Amount'] > 0, 'Amount'].sum(), 2)
            neg_sum = round(df.loc[df['Amount'] < 0, 'Amount'].sum(), 2)
            pos_ok = abs(pos_sum - round(summary['pos_value'], 2)) < 0.01
            neg_ok = abs(neg_sum - round(summary['neg_value'], 2)) < 0.01
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**{summary['pos_label']}** expected: {summary['pos_value']:.2f}")
                st.success(f"Matches! ({pos_sum:.2f})") if pos_ok else st.error(f"Found {pos_sum:.2f}")
            with c2:
                st.write(f"**{summary['neg_label']}** expected: {summary['neg_value']:.2f}")
                st.success(f"Matches! ({neg_sum:.2f})") if neg_ok else st.error(f"Found {neg_sum:.2f}")
        else:
            st.info("No summary block detected (or parse didn’t find it).")

        df['Source File'] = f.name
        combined = pd.concat([combined, df], ignore_index=True)

    if not combined.empty:
        st.subheader("All Combined Transactions")
        st.dataframe(combined)
        st.download_button(
            "Download CSV",
            combined.to_csv(index=False),
            file_name="combined_statements.csv",
            mime="text/csv"
        )

def cli_main():
    """Command line interface for the bank statement parser."""
    import argparse
    import os
    import sys
    
    parser = argparse.ArgumentParser(description="Parse bank statement PDFs and export to CSV")
    parser.add_argument("pdf_files", nargs="+", help="Path(s) to PDF statement file(s)")
    parser.add_argument("-o", "--output", default="bank_statements.csv", help="Output CSV file name")
    parser.add_argument("--force-ocr", action="store_true", help="Force OCR for all pages")
    parser.add_argument("--debug", action="store_true", help="Show debug information")
    
    args = parser.parse_args()
    
    # Check if files exist
    for pdf_file in args.pdf_files:
        if not os.path.exists(pdf_file):
            print(f"Error: File '{pdf_file}' not found")
            sys.exit(1)
    
    combined = pd.DataFrame()
    
    for pdf_file in args.pdf_files:
        print(f"Processing: {pdf_file}")
        
        try:
            with open(pdf_file, 'rb') as f:
                data = f.read()
            
            if args.force_ocr:
                # OCR every page unconditionally
                images = convert_from_bytes(data, dpi=300)
                pages_text = [pytesseract.image_to_string(img, lang="eng") for img in images]
                full_text = re.sub(r'\s+', ' ', " ".join(pages_text)).strip()
            else:
                pages_text, full_text = extract_pages_text_with_ocr_fallback(data)
            
            if args.debug:
                print(f"Debug - First 200 chars: {full_text[:200]}...")
            
            first_page_text = pages_text[0] if pages_text else ""
            df, summary = parse_any_statement_from_text(first_page_text, full_text)
            
            if df.empty:
                print(f"  ⚠️  No transactions found in {pdf_file}")
                continue
            
            print(f"  ✅ Found {len(df)} transactions")
            
            if summary:
                pos_sum = round(df.loc[df['Amount'] > 0, 'Amount'].sum(), 2)
                neg_sum = round(df.loc[df['Amount'] < 0, 'Amount'].sum(), 2)
                pos_ok = abs(pos_sum - round(summary['pos_value'], 2)) < 0.01
                neg_ok = abs(neg_sum - round(summary['neg_value'], 2)) < 0.01
                
                print(f"  📊 {summary['pos_label']}: {pos_sum:.2f} (expected: {summary['pos_value']:.2f}) {'✅' if pos_ok else '❌'}")
                print(f"  📊 {summary['neg_label']}: {neg_sum:.2f} (expected: {summary['neg_value']:.2f}) {'✅' if neg_ok else '❌'}")
            
            df['Source File'] = os.path.basename(pdf_file)
            combined = pd.concat([combined, df], ignore_index=True)
            
        except Exception as e:
            print(f"  ❌ Error processing {pdf_file}: {str(e)}")
            continue
    
    if not combined.empty:
        combined.to_csv(args.output, index=False)
        print(f"\n🎉 Success! Exported {len(combined)} transactions to '{args.output}'")
        print(f"📁 Files processed: {len(args.pdf_files)}")
    else:
        print("\n❌ No transactions found in any of the provided files")
        sys.exit(1)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        cli_main()
    elif len(sys.argv) > 1:
        cli_main()
    else:
        # Default to Streamlit web interface when clicking "Run" in Cursor
        main()
