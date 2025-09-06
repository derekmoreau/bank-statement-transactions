import io
import re
import pdfplumber
import pytesseract
import pandas as pd
import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes
from datetime import datetime
from typing import Any, Optional, cast

# Optional backends for improved text extraction (lazy imports)
_PDFIUM: Any = None   # pypdfium2
_PYPDF2: Any = None   # PyPDF2

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
# Alternative text extraction preferring pdfium/PyPDF2
# =========================

def _try_import_pdfium():
    global _PDFIUM
    if _PDFIUM is not None:
        return True if _PDFIUM else False
    try:
        import pypdfium2 as pdfium  # type: ignore
        _PDFIUM = pdfium
        return True
    except Exception:
        _PDFIUM = False
        return False

def _try_import_pypdf2():
    global _PYPDF2
    if _PYPDF2 is not None:
        return True if _PYPDF2 else False
    try:
        import PyPDF2  # type: ignore
        _PYPDF2 = PyPDF2
        return True
    except Exception:
        _PYPDF2 = False
        return False

def extract_text_pages_prefer_pdfium(data_bytes: bytes):
    """Return (pages_text, full_text) using pdfium or PyPDF2; None if both unavailable."""
    # Try pdfium
    if _try_import_pdfium():
        try:
            pdf = _PDFIUM.PdfDocument(io.BytesIO(data_bytes))  # type: ignore
            try:
                pages_text = []
                for i in range(len(pdf)):
                    page = pdf.get_page(i)
                    try:
                        textpage = page.get_textpage()
                        txt = textpage.get_text_range() or ""
                        textpage.close()
                    finally:
                        page.close()
                    pages_text.append(txt)
                full = re.sub(r'\s+', ' ', ' '.join(pages_text)).strip()
                return pages_text, full
            finally:
                pdf.close()
        except Exception:
            pass
    # Fallback to PyPDF2
    if _try_import_pypdf2():
        try:
            reader = _PYPDF2.PdfReader(io.BytesIO(data_bytes))  # type: ignore
            pages_text = []
            for p in reader.pages:
                try:
                    txt = p.extract_text() or ""
                except Exception:
                    txt = ""
                pages_text.append(txt)
            full = re.sub(r'\s+', ' ', ' '.join(pages_text)).strip()
            return pages_text, full
        except Exception:
            pass
    return None, None

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

# =========================
# RBC chequing via pdfium/PyPDF2 text blocks (no layout dependency)
# =========================

def parse_rbc_chequing_from_text_blocks(full_text: str, start_dt, end_dt) -> pd.DataFrame:
    """
    Robust text-only parser:
      - Anchor by date tokens ("DD Mon" or "Mon DD")
      - Within each date block, capture repeated (description, amount [, balance])
      - Assign positive sign to deposits/credits keywords; negative otherwise
    """
    date_anchor = re.compile(
        r'((?:\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?))|'
        r'(?:(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}))\b',
        re.IGNORECASE,
    )
    amt_bal_re = re.compile(rf'(.*?){AMT}(?:\s+(\d{{1,3}}(?:,\d{{3}})*\.\d{{2}}))?(?:\s|$)')

    matches = list(date_anchor.finditer(full_text))
    rows = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        block = full_text[start:end].strip()
        head = m.group(1)
        rest = block[len(head):].strip()

        for mm in amt_bal_re.finditer(rest):
            desc = (mm.group(1) or '').strip()
            if not desc:
                continue
            amt = parse_amount(mm.group(2))
            dl = desc.lower()
            if any(k in dl for k in ['deposit', 'received', 'payroll', 'refund', 'credit', 'dépôt', 'remboursement']):
                signed = +amt
            elif any(k in dl for k in ['fee', 'purchase', 'withdrawal', 'sent', 'interac', 'payment', 'transfer', 'retrait', 'paiement', 'frais']):
                signed = -amt
            else:
                signed = -amt
            rows.append({
                'Date': interpret_transaction_date(head, start_dt, end_dt),
                'Description': desc,
                'Amount': signed,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        expected_cols = ['Date', 'Description', 'Amount']
        cols = [c for c in expected_cols if c in df.columns]
        if cols:
            df = df.loc[:, cols]
    return cast(pd.DataFrame, df)

def parse_rbc_chequing_pdfium_pipeline(data_bytes: bytes, start_dt, end_dt) -> pd.DataFrame:
    """Use pdfium/PyPDF2 text if available; fallback to OCR-aware extractor; then parse blocks."""
    pages_text, full_text = extract_text_pages_prefer_pdfium(data_bytes)
    if not full_text:
        _pages, full_text = extract_pages_text_with_ocr_fallback(data_bytes)
    # Normalize some joins
    full_text = re.sub(r'\s+', ' ', full_text).strip()
    if not full_text:
        return pd.DataFrame()
    return parse_rbc_chequing_from_text_blocks(full_text, start_dt, end_dt)

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

def parse_any_statement_from_text(first_page_text: str, full_text: str, data_bytes: Optional[bytes] = None, source_name: Optional[str] = None):
    """
    Master dispatcher that selects the appropriate parser.
    If `data_bytes` is provided and the statement is RBC chequing, a
    structure-aware PDF parser will be used for better accuracy.
    """
    stype = detect_statement_type(first_page_text)
    start_dt, end_dt = parse_statement_period(full_text)
    summary = parse_summary_generic(full_text)

    if stype == 'BMO':
        df = parse_bmo_from_text(full_text, start_dt, end_dt)
    elif stype == 'RBC_MC':
        df = parse_rbc_mc_from_text(full_text, start_dt, end_dt)
    elif stype == 'RBC_CHEQUING':
        # Prefer the new char-based pdfplumber logic using strict column ranges and date carry-forward
        df = pd.DataFrame()
        if data_bytes is not None:
            # Determine closing year/month from filename or statement period
            closing_year = None
            closing_month = None
            if source_name:
                m = re.search(r"(\d{4})-(\d{2})-(\d{2})\.pdf$", source_name, re.IGNORECASE)
                if m:
                    closing_year, closing_month = int(m.group(1)), int(m.group(2))
            if (closing_year is None or closing_month is None) and end_dt:
                closing_year, closing_month = end_dt.year, end_dt.month
            try:
                if closing_year is not None and closing_month is not None:
                    df = parse_rbc_chequing_from_pdf_chars(data_bytes, closing_year, closing_month)
            except Exception:
                df = pd.DataFrame()
        if df.empty and data_bytes is not None:
            # Fallback to pdfium/PyPDF2 text pipeline
            try:
                df = parse_rbc_chequing_pdfium_pipeline(data_bytes, start_dt, end_dt)
            except Exception:
                df = pd.DataFrame()
        if df.empty:
            # Last resort: old text heuristics
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

def _group_words_by_lines(words, y_tol=3.0):
    """Group pdfplumber words into line buckets by their y (top) coordinate."""
    lines = []
    for w in sorted(words, key=lambda x: (x.get('top', 0), x.get('x0', 0))):
        if not lines:
            lines.append([w])
            continue
        last_line = lines[-1]
        last_top = sum(x.get('top', 0) for x in last_line) / max(1, len(last_line))
        if abs(w.get('top', 0) - last_top) <= y_tol:
            last_line.append(w)
        else:
            lines.append([w])
    return lines

def parse_rbc_chequing_from_pdf(data_bytes: bytes, start_dt, end_dt) -> pd.DataFrame:
    """
    Layout-aware parser for RBC chequing statements following the requested flow:
      1) Find the title "Details of your account activity" (or continued).
      2) Set a strict left margin from that section and crop words to the table area.
      3) Detect and skip column headers and the first page "Opening Balance" line.
      4) Iterate transaction bands separated by dotted horizontal rules. Join 1–2 line descriptions,
         carry-forward dates, capture withdrawal/deposit amounts, stop at "Closing balance".
    """
    rows = []
    amount_regex = re.compile(AMT)

    def normalize_amount_text(s: str) -> str:
        m = amount_regex.search(s or '')
        return m.group(1) if m else ''

    with pdfplumber.open(io.BytesIO(data_bytes)) as pdf:
        for page_index, page in enumerate(pdf.pages):
            words = page.extract_words(
                use_text_flow=True,
                keep_blank_chars=False,
                x_tolerance=2,
                y_tolerance=2,
            )
            if not words:
                continue

            # 1) Find the title
            title_words = [
                w for w in words
                if 'details' in w.get('text', '').lower() and 'account' in w.get('text', '').lower()
            ]
            if not title_words:
                # No section on this page
                continue
            header_y = min(w['top'] for w in title_words)

            # 2) Set left margin based on the minimum x0 at the header line or the Date column
            # Try to use the Date header to align columns precisely
            def closest_word(target):
                c = [w for w in words if target.lower() in w.get('text', '').lower()]
                if not c:
                    return None
                c.sort(key=lambda w: abs(w['top'] - header_y))
                return c[0]

            date_hdr = closest_word('Date')
            desc_hdr = closest_word('Description')
            wdr_hdr = closest_word('Withdrawals')
            dep_hdr = closest_word('Deposits')
            bal_hdr = closest_word('Balance')

            if date_hdr and desc_hdr and wdr_hdr and dep_hdr and bal_hdr:
                date_x0 = date_hdr['x0']
                desc_x0 = desc_hdr['x0']
                wdr_x0 = wdr_hdr['x0']
                dep_x0 = dep_hdr['x0']
                bal_x0 = bal_hdr['x0']
            else:
                # Fallback approximate columns if headers not detected
                date_x0 = page.width * 0.08
                desc_x0 = page.width * 0.18
                wdr_x0 = page.width * 0.58
                dep_x0 = page.width * 0.72
                bal_x0 = page.width * 0.86

            left_margin = max(0.0, date_x0 - 4)

            col_bounds = {
                'date': (date_x0 - 2, desc_x0 - 2),
                'desc': (desc_x0 - 2, wdr_x0 - 6),
                'wdr': (wdr_x0 - 6, dep_x0 - 6),
                'dep': (dep_x0 - 6, bal_x0 - 6),
                'bal': (bal_x0 - 6, page.width),
            }

            # Keep only words in the table area (right of left_margin and below the title)
            body_words = [w for w in words if (w['top'] > header_y + 12 and w['x0'] >= left_margin)]
            if not body_words:
                continue

            # 3) Identify column header line to skip
            # We do this by removing any line that contains both Date and Description
            header_line_y = None
            for line in _group_words_by_lines(body_words, y_tol=3.0):
                t = ' '.join(w['text'] for w in line).lower()
                if 'date' in t and 'description' in t and ('withdrawals' in t or 'deposits' in t):
                    header_line_y = sum(w['top'] for w in line) / len(line)
                    break

            filtered_words = [w for w in body_words if (header_line_y is None or w['top'] > header_line_y + 2)]
            if not filtered_words:
                continue

            # Collect horizontal separators (dotted rules) using lines and many tiny rects
            sep_ys = []
            for ln in getattr(page, 'lines', []) or []:
                if ln.get('x1') is None or ln.get('x0') is None or ln.get('y0') is None or ln.get('y1') is None:
                    continue
                if min(ln['x0'], ln['x1']) < left_margin:
                    continue
                if abs(ln['y0'] - ln['y1']) <= 0.7 and (ln['x1'] - ln['x0']) > (page.width * 0.4):
                    sep_ys.append((ln['y0'] + ln['y1']) / 2.0)
            # Dotted rules often appear as many tiny rects; cluster their y centers
            tiny_rect_ys = []
            for r in getattr(page, 'rects', []) or []:
                if r.get('x0') is None or r.get('x1') is None or r.get('y0') is None or r.get('y1') is None:
                    continue
                if r['x0'] < left_margin:
                    continue
                h = abs(r['y1'] - r['y0'])
                wth = abs(r['x1'] - r['x0'])
                if h <= 1.2 and wth <= 3.5:  # small dots
                    tiny_rect_ys.append((r['y0'] + r['y1']) / 2.0)
            # Cluster tiny rect y's
            tiny_rect_ys.sort()
            clustered = []
            for y in tiny_rect_ys:
                if not clustered or abs(clustered[-1] - y) > 1.5:
                    clustered.append(y)
            sep_ys.extend(clustered)

            # Deduplicate and sort
            sep_ys = sorted(set(round(y, 1) for y in sep_ys if y > (header_line_y or header_y) + 8))
            if not sep_ys:
                # Fallback: synthesize separators by line gaps
                line_groups = _group_words_by_lines(filtered_words, y_tol=2.0)
                ys = [sum(w['top'] for w in g) / len(g) for g in line_groups]
                ys.sort()
                # Create separators halfway between successive lines
                sep_ys = [ys[0] - 2] + [ (ys[i] + ys[i+1]) / 2.0 for i in range(len(ys)-1) ] + [ys[-1] + 6]
            else:
                # Add top/bottom sentinels
                sep_ys = [min(sep_ys) - 2.0] + sep_ys + [page.height - 4.0]

            # Assign words to bands between separators
            band_words = [[] for _ in range(len(sep_ys) - 1)]
            for w in filtered_words:
                y_mid = w['top']
                # find band index via binary search
                lo, hi = 0, len(sep_ys) - 1
                idx = None
                while lo < hi:
                    mid = (lo + hi) // 2
                    if y_mid <= sep_ys[mid]:
                        hi = mid
                    else:
                        lo = mid + 1
                idx = max(0, lo - 1)
                if idx < len(band_words):
                    band_words[idx].append(w)

            current_date = None
            saw_closing = False

            for bi, group in enumerate(band_words):
                if not group:
                    continue
                # Map words to columns
                col_text = {'date': [], 'desc': [], 'wdr': [], 'dep': [], 'bal': []}
                for w in sorted(group, key=lambda x: x['x0']):
                    x = w['x0']
                    t = w['text']
                    for col, (x0, x1) in col_bounds.items():
                        if x >= x0 and x < x1:
                            col_text[col].append(t)
                            break

                date_text = ' '.join(col_text['date']).strip()
                desc_text = ' '.join(col_text['desc']).strip()
                wdr_text = normalize_amount_text(' '.join(col_text['wdr']))
                dep_text = normalize_amount_text(' '.join(col_text['dep']))

                # Skip any remaining header-ish groups
                lower = (desc_text + ' ' + date_text).lower()
                if ('date' in lower and 'description' in lower) or lower.strip() == '':
                    continue

                # 3) Skip Opening Balance on first page
                if page_index == 0 and desc_text.lower() == 'opening balance':
                    if date_text:
                        current_date = date_text
                    continue

                # Stop at Closing balance
                if desc_text.lower().startswith('closing balance'):
                    saw_closing = True
                    break

                # 4a) Record or carry forward date
                if date_text:
                    current_date = date_text

                # 4b-4d) Create transaction if any amount present
                if (wdr_text or dep_text) and desc_text:
                    rows.append({
                        'Date': interpret_transaction_date(current_date or '', start_dt, end_dt) if current_date else '',
                        'Description': desc_text,
                        'Amount': -parse_amount(wdr_text) if wdr_text else parse_amount(dep_text),
                    })

            if saw_closing:
                break

    return pd.DataFrame(rows)

def parse_rbc_chequing_from_pdf_chars(data_bytes: bytes, closing_year: int, closing_month: int) -> pd.DataFrame:
    """
    Char-aware extractor adapted from the provided reference logic.
    - Uses pdfplumber words to locate columns and numbers
    - Uses pdfplumber chars on each y-line to rebuild description text with correct spacing
    - Carries forward dates; computes positive deposits and negative withdrawals
    - Returns columns: Date, Description, Amount
    """
    try:
        import pdfplumber  # type: ignore
    except Exception as e:
        # pdfplumber required for this path; bubble up to caller for fallback
        raise e

    MONTHS = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
    num_re = re.compile(r"^\d{1,3}(?:,\d{3})*(?:\.\d{2})$|^\d+\.\d{2}$")

    def parse_iso_date(day_mon: str) -> str:
        m = re.match(r"^(\d{1,2})\s*([A-Za-z]{3})$", day_mon.strip())
        if not m: return day_mon
        day = int(m.group(1)); mon = MONTHS.get(m.group(2), None)
        if mon is None: return day_mon
        year = closing_year - 1 if mon > closing_month else closing_year
        return f"{year:04d}-{mon:02d}-{day:02d}"

    def find_header_columns_relaxed(words):
        name_to_variants = {
            "Date": ["Date"],
            "Description": ["Description"],
            "Withdrawals": ["Withdrawals", "Withdrawals($)"],
            "Deposits": ["Deposits", "Deposits($)"],
            "Balance": ["Balance", "Balance($)"],
        }
        found = {}
        for canonical, variants in name_to_variants.items():
            x0 = None
            for w in words:
                if w.get("text") in variants:
                    x0 = float(w["x0"]); break
            if x0 is None:
                return None
            found[canonical] = x0
        return found

    def assemble_line_text_from_chars(line_chars, gap_ratio: float = 0.33, min_abs_gap: float = 1.2) -> str:
        if not line_chars: return ""
        xs = [float(c["x0"]) for c in line_chars]
        x1s = [float(c["x1"]) for c in line_chars]
        widths = [x1s[i]-xs[i] for i in range(len(xs))]
        widths_sorted = sorted(widths)
        med_w = widths_sorted[len(widths_sorted)//2] if widths_sorted else 4.0
        pieces = [line_chars[0]["text"]]
        for i in range(len(line_chars)-1):
            gap = xs[i+1]-x1s[i]
            if gap > max(gap_ratio*med_w, min_abs_gap):
                pieces.append(" ")
            pieces.append(line_chars[i+1]["text"])
        return "".join(pieces)

    rows = []
    with pdfplumber.open(io.BytesIO(data_bytes)) as pdf:
        current_date = None
        desc_buffer = ""
        for page in pdf.pages:
            words = page.extract_words(x_tolerance=2, y_tolerance=3, keep_blank_chars=False, use_text_flow=True)
            header = find_header_columns_relaxed(words)
            if not header:
                continue

            x_desc = header["Description"]
            x_wdr = header["Withdrawals"]
            x_dep = header["Deposits"]
            x_bal = header["Balance"]

            # Group words by rounded y
            lines = {}
            for w in words:
                y = round(float(w["top"]), 1)
                lines.setdefault(y, []).append(w)
            line_ys = sorted(lines.keys())

            # Build char lines indexed by the same y
            char_lines = {}
            for c in page.chars:
                y = round(float(c["top"]), 1)
                char_lines.setdefault(y, []).append(c)
            for y in char_lines:
                char_lines[y] = sorted(char_lines[y], key=lambda c: float(c["x0"]))

            seen_header_line = False

            for y in line_ys:
                line_words = sorted(lines[y], key=lambda w: w["x0"])
                line_text = " ".join(w["text"] for w in line_words)

                # Skip header and boilerplate
                if not seen_header_line and ("Date" in line_text and "Description" in line_text and ("Withdrawals" in line_text or "Withdrawals($)" in line_text)):
                    seen_header_line = True
                    continue
                if not seen_header_line:
                    continue
                if "Detailsofyouraccountactivity" in line_text or "Detailsofyouraccountactivity-continued" in line_text:
                    continue
                if "PleasecheckthisAccountStatement" in line_text:
                    continue
                if "RoyalBankofCanadaGSTRegistrationNumber:" in line_text or "RoyalBankofCanadaGSTRegistrationNumber" in line_text:
                    continue
                if "ClosingBalance" in line_text:
                    continue

                # Date from tokens left of Description
                date_tokens = [w for w in line_words if float(w["x0"]) < x_desc - 2]
                date_str = None
                if date_tokens:
                    dtxt = "".join(w["text"] for w in date_tokens).strip()
                    m = re.match(r"^(\d{1,2})([A-Za-z]{3})$", dtxt)
                    if m:
                        date_str = parse_iso_date(f"{m.group(1)} {m.group(2)}")
                    elif "OpeningBalance" in dtxt.replace(" ", "") or "OpeningBalance" in dtxt:
                        desc_buffer = ""
                        continue

                # Description using chars between Description and Withdrawals
                cline = char_lines.get(y, [])
                if cline:
                    desc_chars = [c for c in cline if x_desc-1 <= float(c["x0"]) < x_wdr-1]
                    desc_text = assemble_line_text_from_chars(desc_chars).strip()
                else:
                    desc_tokens = [w for w in line_words if x_desc - 1 <= float(w["x0"]) < x_wdr - 1]
                    desc_text = " ".join(w["text"] for w in desc_tokens).strip()

                if date_str is not None:
                    desc_buffer = desc_text if desc_text else ""
                    current_date = date_str
                else:
                    if desc_text:
                        desc_buffer = (desc_buffer + " " + desc_text).strip() if desc_buffer else desc_text

                # Numeric tokens with strict column ranges
                def number_in_range(x_left: float, x_right: float):
                    out = []
                    for w in line_words:
                        x0 = float(w["x0"])
                        if x_left-0.5 <= x0 < x_right-0.5:
                            txt = w["text"].replace(",", "")
                            if num_re.match(txt):
                                out.append((float(txt), x0))
                    if out:
                        out.sort(key=lambda t: t[1])
                        return out[-1][0]
                    return None

                wdr_val = number_in_range(x_wdr, x_dep)
                dep_val = number_in_range(x_dep, x_bal)

                if (wdr_val is not None or dep_val is not None) and current_date is not None:
                    amount = (dep_val if dep_val is not None else 0.0) - (wdr_val if wdr_val is not None else 0.0)
                    rows.append({
                        'Date': current_date.replace('-', '/')[8:10] + '/' + current_date.replace('-', '/')[5:7] + '/' + current_date[:4] if re.match(r"^\d{4}-\d{2}-\d{2}$", current_date) else current_date,
                        'Description': desc_buffer,
                        'Amount': round(amount, 2),
                    })
                    desc_buffer = ""

    return pd.DataFrame(rows)

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

        df, summary = parse_any_statement_from_text(first_page_text, full_text, data)
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
            df, summary = parse_any_statement_from_text(first_page_text, full_text, data, os.path.basename(pdf_file))
            
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
    import os
    import subprocess
    # If arguments are provided, use CLI path
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        cli_main()
    elif len(sys.argv) > 1:
        cli_main()
    else:
        # When invoked directly without args, ensure we run via Streamlit to avoid bare-mode warnings
        if os.environ.get("BSI_LAUNCHED_VIA_STREAMLIT") == "1":
            main()
        else:
            env = os.environ.copy()
            env["BSI_LAUNCHED_VIA_STREAMLIT"] = "1"
            try:
                subprocess.run([sys.executable, "-m", "streamlit", "run", __file__], check=False, env=env)
            except KeyboardInterrupt:
                pass
