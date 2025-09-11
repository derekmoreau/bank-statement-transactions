import io
import re
import pdfplumber
import pytesseract
import pandas as pd
import streamlit as st
from PIL import Image
from pdf2image import convert_from_bytes
from datetime import datetime
from typing import Any, Optional, Dict, List, Tuple, cast

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
        # Optional balances from the Calculating Your Balance box
        m_prev = re.search(r'(?i)Previous\s*Account\s*Balance\s*\$?\s*([\d,]+\.\d{2})', full_text)
        m_new  = re.search(r'(?i)(?:Total\s*Account\s*Balance|NEW\s*BALANCE)\s*\$?\s*([\d,]+\.\d{2})', full_text)
        out = {'pos_label': 'Purchases & debits', 'pos_value': +f2(m_pur.group(1)),
               'neg_label': 'Payments & credits', 'neg_value': -abs(pay_num)}
        if m_prev:
            try:
                out['previous_balance'] = f2(m_prev.group(1))
            except Exception:
                pass
        if m_new:
            try:
                out['new_balance'] = f2(m_new.group(1))
            except Exception:
                pass
        return out

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
    # Savings must be detected before generic RBC chequing
    if ('savings account statement' in t) or ('high interest esavings' in t) or ('esavings' in t):
        return 'RBC_SAVINGS'
    if 'rbc' in t or 'royal bank of canada' in t or 'royal bank' in t:
        # Strengthen RBC Mastercard cues present on page 1
        mc_cues = (
            ('mastercard' in t) or
            ('purchases & debits' in t) or
            ('payments & credits' in t) or
            ('calculating your balance' in t) or
            ('activity description' in t) or
            ('amount ($)' in t)
        )
        if mc_cues:
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
    df = pd.DataFrame(rows)
    if not df.empty and 'Amount' in df.columns:
        df['Type'] = df['Amount'].apply(lambda x: 'payment_or_credit' if float(x) < 0 else 'purchase_or_debit')
    return df

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
# Scotiabank balance-delta extractor (coordinate aware)
# =========================

def _scotia_plumber_text_lines_from_bytes(data_bytes: bytes) -> list:
    lines = []
    with pdfplumber.open(io.BytesIO(data_bytes)) as pdf:
        for p in pdf.pages:
            txt = p.extract_text(x_tolerance=2, y_tolerance=2) or ""
            for ln in txt.splitlines():
                ln = (ln or "").strip()
                if ln:
                    lines.append(ln)
    return lines

def _scotia_parse_year_from_bytes(data_bytes: bytes, fallback_name: str = "") -> int:
    lines = _scotia_plumber_text_lines_from_bytes(data_bytes)
    for ln in lines:
        m = re.search(r"(?:Opening|Closing)\s+Balance\s+on\s+[A-Za-z]+\s+\d{1,2},\s*(\d{4})", ln)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    m = re.search(r"(20\d{2})", fallback_name or "")
    return int(m.group(1)) if m else datetime.now().year

def parse_scotia_from_pdf_balance_delta(data_bytes: bytes, source_name: str = "") -> pd.DataFrame:
    """
    Reconstruct Scotiabank rows using pdfplumber words and compute signed Amount from balance deltas only.
    Output columns: Date (DD/MM/YYYY), Description, Amount
    """
    MONTHS = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
    }
    MONTHS_ORDER = list(MONTHS.keys())
    MD_GLUED_RE = re.compile(r"^\d{4}(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(\d{1,2})$")
    MONEY_PAT = r"\d{1,3}(?:,\d{3})*\.\d{2}"

    year = _scotia_parse_year_from_bytes(data_bytes, source_name)

    rows = []
    with pdfplumber.open(io.BytesIO(data_bytes)) as pdf:
        for page in pdf.pages:
            words = page.extract_words(x_tolerance=2, y_tolerance=2) or []
            if not words:
                continue
            words.sort(key=lambda w: (round(w.get("top", 0.0), 1), w.get("x0", 0.0)))
            seq = [(w.get("text", ""), float(w.get("x0", 0.0)), float(w.get("top", 0.0))) for w in words]

            starts = []
            for idx in range(len(seq)):
                t0 = seq[idx][0]
                t1 = seq[idx+1][0] if idx+1 < len(seq) else ""
                if t0 in MONTHS_ORDER and re.fullmatch(r"\d{1,2}", t1 or ""):
                    starts.append((idx, t0, t1))
                else:
                    g = MD_GLUED_RE.match(t0 or "")
                    if g:
                        mon, day = g.group(1), g.group(2)
                        starts.append((idx, mon, day))

            indices = [i for (i, _, _) in starts]
            for si, (idx, mon, day) in enumerate(starts):
                end = indices[si+1] if si+1 < len(indices) else len(seq)
                chunk = seq[idx:end]
                tokens = [t for (t, _x, _y) in chunk[2:]]  # after date tokens

                money_pos = [i for i, t in enumerate(tokens) if re.fullmatch(MONEY_PAT, t or "")]
                balance_after = float(tokens[money_pos[-1]].replace(",", "")) if money_pos else None
                # Updated description logic: include all non-money tokens across the chunk
                desc_tokens = [tok for tok in tokens if (tok is not None) and (not re.fullmatch(MONEY_PAT, tok)) and tok not in {"-", "I", "R", "H", "|", "$"}]
                desc = " ".join(desc_tokens).strip()

                date_iso = f"{year:04d}-{MONTHS.get(mon, 1):02d}-{int(day):02d}"
                rows.append({"Date": date_iso, "Description": desc, "BalanceAfter": balance_after})

    out = []
    prev_bal = None
    for r in rows:
        desc = r.get("Description") or ""
        bal = r.get("BalanceAfter")
        if ("Opening Balance" in desc) and bal is not None and prev_bal is None:
            prev_bal = bal
            continue
        if "Closing Balance" in desc:
            prev_bal = bal
            continue
        if prev_bal is None or bal is None:
            continue
        delta = round(bal - prev_bal, 2)
        d = r["Date"]
        if re.match(r"^\d{4}-\d{2}-\d{2}$", d):
            d = f"{d[8:10]}/{d[5:7]}/{d[:4]}"
        out.append({
            'Date': d,
            'Description': desc,
            'Amount': float(delta),
        })
        prev_bal = bal

    return pd.DataFrame(out)

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
        # New preferred: word-based left-firstline extractor from provided spec
        df = pd.DataFrame()
        if data_bytes is not None:
            try:
                df = extract_rbc_mc_transactions_left_firstline_from_bytes(data_bytes, source_name, end_dt)
            except Exception:
                df = pd.DataFrame()
        if df.empty:
            # last resort fallback
            df = parse_rbc_mc_from_text(full_text, start_dt, end_dt)
    elif stype == 'RBC_SAVINGS':
        df = pd.DataFrame()
        if data_bytes is not None:
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
                    df = parse_rbc_savings_from_pdf_chars(data_bytes, closing_year, closing_month)
            except Exception:
                df = pd.DataFrame()
        if df.empty:
            # Fallback: try text blocks pipeline
            try:
                df = parse_rbc_chequing_pdfium_pipeline(data_bytes, start_dt, end_dt) if data_bytes is not None else pd.DataFrame()
            except Exception:
                df = pd.DataFrame()
    elif stype == 'RBC_CHEQUING':
        # Prefer v3 char-based extractor (exact descriptions/dates)
        df = pd.DataFrame()
        if data_bytes is not None:
            try:
                df = parse_rbc_chequing_from_pdf_chars_v3(data_bytes, start_dt, end_dt)
            except Exception:
                df = pd.DataFrame()
        if df.empty and data_bytes is not None:
            try:
                df = parse_rbc_chequing_pdfium_pipeline(data_bytes, start_dt, end_dt)
            except Exception:
                df = pd.DataFrame()
        if df.empty:
            df = parse_rbc_chequing_from_text(full_text, start_dt, end_dt)
    elif stype == 'SCOTIA_CHEQUING':
        # Prefer balance-delta coordinate parser when bytes are available
        if data_bytes is not None:
            try:
                df = parse_scotia_from_pdf_balance_delta(data_bytes, source_name or "")
            except Exception:
                df = parse_scotia_chequing_from_text(full_text, start_dt, end_dt)
        else:
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
            # Match clean extractor tolerances for better line grouping
            words = page.extract_words(x_tolerance=1.5, y_tolerance=1.5, keep_blank_chars=False, use_text_flow=True)
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
# RBC Chequing v3 (char-based, header-aware, multi-line descriptions)
# =========================

def _rbc_cq_v3_group_by_line_words(words, y_tol=1.5):
    lines = []
    for w in sorted(words, key=lambda x: x.get("top", 0.0)):
        if not lines:
            lines.append({"y": float(w.get("top", 0.0)), "words": [w]})
            continue
        attached = False
        for ln in lines:
            if abs(ln["y"] - float(w.get("top", 0.0))) <= y_tol:
                ln["words"].append(w)
                ln["y"] = (ln["y"] + float(w.get("top", 0.0))) / 2.0
                attached = True
                break
        if not attached:
            lines.append({"y": float(w.get("top", 0.0)), "words": [w]})
    for ln in lines:
        ln["words"] = sorted(ln["words"], key=lambda x: x.get("x0", 0.0))
    return sorted(lines, key=lambda l: l["y"])

def _rbc_cq_v3_group_chars_by_line(chars, y_tol=2.2):
    lines = []
    for c in sorted(chars, key=lambda x: (float(x.get("top", 0.0)) + float(x.get("bottom", 0.0))) / 2.0):
        y_mid = (float(c.get("top", 0.0)) + float(c.get("bottom", 0.0))) / 2.0
        attached = False
        for ln in lines:
            if abs(ln["y"] - y_mid) <= y_tol:
                ln["chars"].append(c)
                ln["y"] = (ln["y"] + y_mid) / 2.0
                attached = True
                break
        if not attached:
            lines.append({"y": y_mid, "chars": [c]})
    for ln in lines:
        ln["chars"] = sorted(ln["chars"], key=lambda x: x.get("x0", 0.0))
    return sorted(lines, key=lambda l: l["y"])

def _rbc_cq_v3_find_table_headers(page) -> Optional[Dict[str, float]]:
    words = page.extract_words(use_text_flow=True, keep_blank_chars=False) or []
    lines = _rbc_cq_v3_group_by_line_words(words)
    for ln in lines:
        texts = [str(w.get("text", "")) for w in ln["words"]]
        if ("Date" in texts) and ("Description" in texts) and any(str(t).startswith("Withdrawals") for t in texts) and any(str(t).startswith("Deposits") for t in texts):
            pos: Dict[str, float] = {"y": float(ln["y"]) }
            for w in ln["words"]:
                t = str(w.get("text", ""))
                x0 = float(w.get("x0", 0.0))
                if t == "Date": pos["Date"] = x0
                elif t == "Description": pos["Description"] = x0
                elif t.startswith("Withdrawals"): pos["Withdrawals"] = x0
                elif t.startswith("Deposits"): pos["Deposits"] = x0
                elif t.startswith("Balance"): pos["Balance"] = x0
            if "Balance" not in pos:
                pos["Balance"] = max(float(w.get("x0", 0.0)) for w in ln["words"]) + 100.0
            return pos
    # Fallback by chars
    chars = getattr(page, 'chars', []) or []
    for ln in _rbc_cq_v3_group_chars_by_line(chars, y_tol=2.5):
        s = "".join(str(c.get("text", "")) for c in ln["chars"])
        if ("Date" in s) and ("Description" in s) and ("Withdrawals" in s) and ("Deposits" in s):
            def find_x(word: str) -> Optional[float]:
                idx = s.find(word)
                if idx == -1:
                    return None
                running = ""
                for ch in ln["chars"]:
                    running += str(ch.get("text", ""))
                    if len(running) >= idx + 1:
                        return float(ch.get("x0", 0.0))
                return float(ln["chars"][0].get("x0", 0.0))
            return {
                "y": float(ln["y"]),
                "Date": find_x("Date") or 40.0,
                "Description": find_x("Description") or 100.0,
                "Withdrawals": find_x("Withdrawals") or 360.0,
                "Deposits": find_x("Deposits") or 450.0,
                "Balance": find_x("Balance") or (float(ln["chars"][-1].get("x1", 0.0)) + 100.0),
            }
    return None

def _rbc_cq_v3_reconstruct_from_chars(page, y_center: float, x0: float, x1: float, y_tol=2.2, gap_thresh=0.6) -> str:
    chars = [c for c in getattr(page, 'chars', []) or [] if (x0 - 2.0) <= float(c.get("x0", 0.0)) <= (x1 + 2.0) and abs(((float(c.get("top", 0.0)) + float(c.get("bottom", 0.0))) / 2.0) - y_center) <= y_tol]
    chars = sorted(chars, key=lambda c: float(c.get("x0", 0.0)))
    if not chars:
        return ""
    parts = [str(chars[0].get("text", ""))]
    for prev, cur in zip(chars, chars[1:]):
        gap = float(cur.get("x0", 0.0)) - float(prev.get("x1", 0.0))
        if gap > gap_thresh:
            parts.append(" ")
        parts.append(str(cur.get("text", "")))
    return "".join(parts).strip()

def _rbc_cq_v3_pick_amount_from_chars(line_chars: List[dict], x0: float, x1: float) -> Optional[float]:
    cells = [c for c in line_chars if (x0 - 2.0) <= float(c.get("x0", 0.0)) <= (x1 + 2.0)]
    cells = sorted(cells, key=lambda c: float(c.get("x0", 0.0)))
    if not cells:
        return None
    s = "".join(str(c.get("text", "")) for c in cells)
    m = re.findall(r"([0-9][0-9,]*\.\d{2})", s)
    return float(m[-1].replace(",", "")) if m else None

def _rbc_cq_v3_normalize_date(raw: str, start_dt: datetime, end_dt: datetime) -> Optional[str]:
    m = re.match(r"^(\d{1,2})\s*([A-Za-z]{3})$", raw.strip())
    if not m:
        return None
    day = int(m.group(1))
    mon_map = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
    mon = mon_map.get(m.group(2).title())
    year = start_dt.year
    if start_dt.month == 12 and mon == 1:
        year = end_dt.year
    try:
        return f"{year:04d}-{mon:02d}-{day:02d}"
    except Exception:
        return None

def _rbc_cq_v3_build_transactions_from_page(page, start_dt: datetime, end_dt: datetime, current_date: Optional[str] = None):
    header = _rbc_cq_v3_find_table_headers(page)
    if not header:
        return [], current_date
    x_date = float(header["Date"]) ; x_desc = float(header["Description"]) ; x_withd = float(header["Withdrawals"]) ; x_depos = float(header["Deposits"]) ; x_bal = float(header["Balance"])
    header_y = float(header["y"])
    chars = [c for c in getattr(page, 'chars', []) or [] if (((float(c.get("top", 0.0)) + float(c.get("bottom", 0.0))) / 2.0) > header_y - 0.5) and float(c.get("x0", 0.0)) < x_bal - 2.0]
    lines = _rbc_cq_v3_group_chars_by_line(chars, y_tol=2.2)

    txns = [] ; pending_desc: List[str] = [] ; pending_w: Optional[float] = None ; pending_d: Optional[float] = None

    def flush():
        nonlocal pending_desc, pending_w, pending_d, current_date, txns
        if pending_desc and (pending_w is not None or pending_d is not None):
            desc = " ".join(pending_desc).strip()
            txns.append({
                "date": current_date,
                "description": re.sub(r"\s+", " ", desc),
                "withdrawal": pending_w,
                "deposit": pending_d,
            })
        pending_desc = [] ; pending_w = None ; pending_d = None

    for ln in lines:
        line_chars = ln["chars"]
        full_line = "".join(str(c.get("text", "")) for c in line_chars)
        if full_line.startswith("ClosingBalance"):
            flush(); break

        left_text = _rbc_cq_v3_reconstruct_from_chars(page, float(ln["y"]), 0.0, x_desc - 5.0, y_tol=2.2, gap_thresh=0.6)
        m = re.search(r"(\d{1,2})\s*([A-Za-z]{3})", left_text)
        if m:
            nd = _rbc_cq_v3_normalize_date(f"{m.group(1)} {m.group(2)}", start_dt, end_dt)
            if nd:
                current_date = nd

        desc = _rbc_cq_v3_reconstruct_from_chars(page, float(ln["y"]), x_desc - 2.0, x_withd - 2.0, y_tol=2.2, gap_thresh=0.6)

        if re.match(r"^\s*Opening\s*Balance\b", desc, flags=re.IGNORECASE):
            continue
        if desc.startswith("Description") or ("Detailsof" in desc):
            continue

        w_amt = _rbc_cq_v3_pick_amount_from_chars(line_chars, x_withd, x_depos)
        d_amt = _rbc_cq_v3_pick_amount_from_chars(line_chars, x_depos, x_bal)

        if not desc and w_amt is None and d_amt is None:
            continue

        if desc:
            pending_desc.append(desc)
        if (w_amt is not None) or (d_amt is not None):
            pending_w = w_amt if w_amt is not None else None
            pending_d = d_amt if d_amt is not None else None
            flush()

    return txns, current_date

def parse_rbc_chequing_from_pdf_chars_v3(data_bytes: bytes, start_dt: Optional[datetime], end_dt: Optional[datetime]) -> pd.DataFrame:
    """Implement the provided v3 char-based chequing extractor and map to Date/Description/Amount."""
    if start_dt is None or end_dt is None:
        # Fallback to a reasonable default window if period missing
        start_dt = datetime.now().replace(month=1, day=1)
        end_dt = datetime.now()
    rows: List[Dict[str, Any]] = []
    with pdfplumber.open(io.BytesIO(data_bytes)) as pdf:
        current_date: Optional[str] = None
        for page in pdf.pages:
            txns, current_date = _rbc_cq_v3_build_transactions_from_page(page, start_dt, end_dt, current_date)
            for t in txns:
                iso = t.get("date") or ""
                date_out = iso
                if re.match(r"^\d{4}-\d{2}-\d{2}$", iso):
                    date_out = f"{iso[8:10]}/{iso[5:7]}/{iso[:4]}"
                withdrawal = t.get("withdrawal") if t.get("withdrawal") is not None else 0.0
                deposit = t.get("deposit") if t.get("deposit") is not None else 0.0
                amount = float(deposit) - float(withdrawal)
                rows.append({
                    "Date": date_out,
                    "Description": t.get("description", ""),
                    "Amount": round(amount, 2),
                })
    return pd.DataFrame(rows)
def parse_rbc_savings_from_pdf_chars(data_bytes: bytes, closing_year: int, closing_month: int) -> pd.DataFrame:
    """
    RBC Savings extractor using the same description reconstruction as chequing.
    - Column bands computed from header midpoints to classify numbers
    - Description is rebuilt from per-line chars between Description and Withdrawals
    - Date carry-forward; signs from column position (deposit positive, withdrawal negative)
    """
    import pdfplumber  # rely on existing dep

    def find_headers(words):
        variants = {
            "Date": ["Date"],
            "Description": ["Description"],
            "Withdrawals": ["Withdrawals($)", "Withdrawals"],
            "Deposits": ["Deposits($)", "Deposits"],
            "Balance": ["Balance($)", "Balance"],
        }
        got = {}
        for canonical, opts in variants.items():
            hit = next((w for w in words if w.get("text") in opts), None)
            if not hit:
                return None
            if canonical in ("Withdrawals", "Deposits", "Balance"):
                got[canonical] = (float(hit["x0"]) + float(hit["x1"])) / 2.0
            else:
                got[canonical] = float(hit["x0"])
        return got

    def column_bands_from_headers(h):
        xW, xD, xB = h["Withdrawals"], h["Deposits"], h["Balance"]
        mWD = (xW + xD) / 2.0
        mDB = (xD + xB) / 2.0
        return {
            "withdrawals": (xW - (mWD - xW), mWD),
            "deposits": (mWD, mDB),
            "balance": (mDB, mDB + (xB - xD)),
        }

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

    num_re = re.compile(r"^\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})$|^\$?\d+\.\d{2}$")

    rows = []
    with pdfplumber.open(io.BytesIO(data_bytes)) as pdf:
        for page in pdf.pages:
            words = page.extract_words(x_tolerance=2, y_tolerance=3, keep_blank_chars=False, use_text_flow=True)
            if not words:
                continue
            headers = find_headers(words)
            if not headers:
                continue
            bands = column_bands_from_headers(headers)

            # Build lines and char-lines
            lines = {}
            for w in words:
                y = round(float(w["top"]), 1)
                lines.setdefault(y, []).append(w)
            char_lines = {}
            for c in page.chars:
                y = round(float(c["top"]), 1)
                char_lines.setdefault(y, []).append(c)
            for y in char_lines:
                char_lines[y] = sorted(char_lines[y], key=lambda c: float(c["x0"]))

            seen_header = False
            current_date = None
            desc_buf = ""

            for y in sorted(lines.keys()):
                line_words = sorted(lines[y], key=lambda w: w["x0"])
                joined = "".join(w["text"] for w in line_words)

                if not seen_header and ("Date" in joined and "Description" in joined and "Withdrawals" in joined):
                    seen_header = True
                    continue
                if not seen_header:
                    continue
                if "Detailsofyouraccountactivity" in joined or "Detailsofyouraccountactivity-continued" in joined:
                    continue
                if "PleasecheckthisAccountStatement" in joined:
                    continue
                if "RoyalBankofCanadaGSTRegistrationNumber" in joined:
                    continue
                if "ClosingBalance" in joined:
                    continue

                # date to left of Description
                date_tokens = [w for w in line_words if float(w["x0"]) < headers["Description"] - 1]
                if date_tokens:
                    dtxt = "".join(w["text"] for w in date_tokens)
                    if "OpeningBalance" in dtxt:
                        desc_buf = ""
                        continue
                    m = re.match(r"^(\d{1,2})([A-Za-z]{3})$", dtxt.replace(" ", ""))
                    if m:
                        day = int(m.group(1))
                        mon = m.group(2)
                        # infer year
                        mon_num = {
                            "Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
                            "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12
                        }.get(mon, None)
                        if mon_num:
                            year = closing_year - 1 if mon_num > closing_month else closing_year
                            current_date = f"{year:04d}-{mon_num:02d}-{day:02d}"
                            desc_buf = ""

                # description via chars between Description and withdrawals band
                cline = char_lines.get(y, [])
                if cline:
                    desc_chars = [c for c in cline if headers["Description"]-1 <= float(c["x0"]) < bands["withdrawals"][0]-0.5]
                    desc_text = assemble_line_text_from_chars(desc_chars).strip()
                else:
                    desc_tokens = [w for w in line_words if headers["Description"]-1 <= float(w["x0"]) < bands["withdrawals"][0]-0.5]
                    desc_text = " ".join(w["text"] for w in desc_tokens).strip()

                if desc_text:
                    desc_buf = (f"{desc_buf} {desc_text}".strip() if desc_buf else desc_text)

                def nums_in_band(left: float, right: float):
                    out = []
                    for w in line_words:
                        x0, x1 = float(w["x0"]), float(w["x1"])
                        xc = (x0 + x1) / 2.0
                        if left <= xc < right:
                            txt = w["text"].replace(",", "").replace("$", "")
                            if num_re.match(txt):
                                out.append((float(txt), w))
                    out.sort(key=lambda t: t[1]["x0"])  # rightmost wins
                    return out

                wdrs = nums_in_band(*bands["withdrawals"])
                deps = nums_in_band(*bands["deposits"])
                bals = nums_in_band(*bands["balance"])

                if (wdrs or deps) and current_date:
                    amount = (deps[-1][0] if deps else 0.0) - (wdrs[-1][0] if wdrs else 0.0)
                    rows.append({
                        'Date': f"{current_date[8:10]}/{current_date[5:7]}/{current_date[:4]}",
                        'Description': (desc_buf or '').strip(),
                        'Amount': round(amount, 2),
                    })
                    desc_buf = ""

    return pd.DataFrame(rows)

def parse_rbc_mastercard_from_pdf_chars(data_bytes: bytes, closing_year: int, closing_month: int) -> pd.DataFrame:
    """
    Character-aware RBC Mastercard extractor.
    - Finds Description and Amount columns
    - Rebuilds description from chars with spillover handling
    - Uses two dates on the left (transaction & posting); outputs posting date
    - Amount sign comes from the text (negative for payments/credits)
    Returns DataFrame with columns: Date (DD/MM/YYYY), Description, Amount
    """
    import pdfplumber

    AMOUNT_RE = re.compile(r"-?\$?\d[\d,]*\.\d{2}")

    def to_iso_from_mon_day(mon_str: str, day_str: str) -> str:
        mon_map = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
        mon = mon_map.get(mon_str[:3].title())
        if mon is None:
            return f"{mon_str} {day_str}"
        year = closing_year - 1 if mon > closing_month else closing_year
        return f"{year:04d}-{mon:02d}-{int(day_str):02d}"

    def find_header_x_positions(words):
        x_desc = None
        x_amt = None
        for w in words:
            t = w.get("text", "").strip().upper().replace(" ", "")
            if t in ("ACTIVITY","ACTIVITYDESCRIPTION","DESCRIPTION") and x_desc is None:
                x_desc = float(w["x0"])
            if t in ("AMOUNT($)","AMOUNT","AMOUNT($)DATE","AMOUNT($)POSTING") and x_amt is None:
                x_amt = float(w["x0"])
        if x_desc is None or x_amt is None:
            xs = sorted(float(w.get("x0", 0.0)) for w in words)
            if x_desc is None and xs:
                x_desc = xs[len(xs)//3]
            if x_amt is None and xs:
                x_amt = xs[-1] - 100
        return x_desc, x_amt

    def group_chars_by_y(chars):
        lines = {}
        for c in chars:
            y = round(float(c.get("top", 0.0)), 1)
            lines.setdefault(y, []).append(c)
        for y in list(lines.keys()):
            lines[y] = sorted(lines[y], key=lambda cc: float(cc.get("x0", 0.0)))
        return dict(sorted(lines.items()))

    def assemble_from_chars(line_chars, gap_ratio: float = 0.33, min_abs_gap: float = 1.2) -> str:
        if not line_chars: return ""
        xs  = [float(c["x0"]) for c in line_chars]
        x1s = [float(c["x1"]) for c in line_chars]
        widths = [x1s[i]-xs[i] for i in range(len(xs))]
        med_w = sorted(widths)[len(widths)//2] if widths else 4.0
        out = [line_chars[0]["text"]]
        for i in range(len(line_chars)-1):
            gap = xs[i+1] - x1s[i]
            if gap > max(gap_ratio*med_w, min_abs_gap):
                out.append(" ")
            out.append(line_chars[i+1]["text"])
        return "".join(out)

    def strip_trailing_amount_and_headers(s: str) -> str:
        s = re.sub(r"\s*-?\$?\d{1,3}(?:,\d{3})*\.\d{2}\s*$", "", s).strip()
        s = re.sub(r"\s*(TOTAL ACCOUNT BALANCE|NEW BALANCE)\s*$", "", s, flags=re.I).strip()
        return s

    rows = []
    with pdfplumber.open(io.BytesIO(data_bytes)) as pdf:
        for page in pdf.pages:
            words = page.extract_words(x_tolerance=2, y_tolerance=3, keep_blank_chars=False, use_text_flow=True)
            if not words:
                continue
            x_desc, x_amt = find_header_x_positions(words)
            # Build word-based lines for robust multi-line description stitching
            words_sorted = sorted(words, key=lambda w: (round(float(w.get("top", 0.0)), 2), float(w.get("x0", 0.0))))
            word_lines = []
            cur_y = None
            buf = []
            for w in words_sorted:
                y = round(float(w.get("top", 0.0)), 2)
                if cur_y is None or abs(y - cur_y) <= 1.0:
                    buf.append(w)
                    cur_y = y if cur_y is None else (cur_y + y) / 2.0
                else:
                    if buf:
                        word_lines.append(buf)
                    buf = [w]
                    cur_y = y
            if buf:
                word_lines.append(buf)

            # Char lines for amount detection (preserve existing amount logic)
            char_lines = group_chars_by_y(page.chars)

            # Empirical column bands for RBC MC (match clean extractor)
            x_left_dates = 120.0
            x_desc_left, x_desc_right = 120.0, 460.0

            headerish_prefixes = (
                "TRANSACTION", "DATE", "POSTING", "CALCULATING", "PREVIOUS",
                "NEW BALANCE", "INTEREST RATE", "CASH BACK", "TOTAL ACCOUNT"
            )

            def nearest_char_line_y(avg_y: float) -> Optional[float]:
                if not char_lines:
                    return None
                keys = list(char_lines.keys())
                best = min(keys, key=lambda yy: abs(yy - avg_y))
                return best

            i = 0
            while i < len(word_lines):
                L = word_lines[i]
                avg_y = sum(float(w.get("top", 0.0)) for w in L) / max(1, len(L))
                # Detect two month/day tokens on the left (inclusive threshold)
                left_tokens = [str(w.get("text", "")) for w in L if float(w.get("x0", 0.0)) <= x_left_dates]
                left_str = " ".join(left_tokens)
                mdate = re.match(r"^\s*([A-Za-z]{3})\s+(\d{1,2})\s+([A-Za-z]{3})\s+(\d{1,2})", left_str)
                if not mdate:
                    i += 1
                    continue

                # Amount = rightmost number-like token anywhere on the word line
                amount = None
                for tok in [str(w.get("text", "")) for w in reversed(L)]:
                    t = tok.replace(",", "").replace("$", "").strip()
                    if re.fullmatch(r"-?\d+\.\d{2}", t):
                        try:
                            amount = float(t)
                            break
                        except Exception:
                            pass
                if amount is None:
                    i += 1
                    continue

                # Description on main line within band
                desc_tokens = [str(w.get("text", "")) for w in L if x_desc_left <= float(w.get("x0", 0.0)) <= x_desc_right]
                desc = " ".join(desc_tokens).strip()

                # Stitch continuation lines
                j = i + 1
                while j < len(word_lines):
                    L2 = word_lines[j]
                    left2 = " ".join([str(w.get("text", "")) for w in L2 if float(w.get("x0", 0.0)) <= x_left_dates])
                    if re.match(r"^\s*([A-Za-z]{3})\s+\d{1,2}\s+([A-Za-z]{3})\s+\d{1,2}", left2):
                        break
                    x_left_min = min((float(w.get("x0", 0.0)) for w in L2), default=1000.0)
                    if (x_left_min >= x_desc_left - 5.0) and (x_left_min < x_desc_left + 60.0):
                        cont = " ".join([str(w.get("text", "")) for w in L2 if x_desc_left <= float(w.get("x0", 0.0)) <= x_desc_right]).strip()
                        if cont and not any(cont.upper().startswith(h) for h in headerish_prefixes):
                            desc = (desc + " " + cont).strip()
                            j += 1
                            continue
                    cont_all = " ".join([str(w.get("text", "")) for w in L2]).strip()
                    if re.fullmatch(r"[0-9\s\-]+", cont_all or ""):
                        desc = (desc + " " + cont_all).strip()
                        j += 1
                        continue
                    break

                desc_text = strip_trailing_amount_and_headers(re.sub(r"\s+", " ", desc))
                if not desc_text:
                    i = j
                    continue

                # Dates
                tx_mon, tx_day, post_mon, post_day = mdate.groups()
                tx_iso = to_iso_from_mon_day(tx_mon.title(), tx_day)
                post_iso = to_iso_from_mon_day(post_mon.title(), post_day)

                # Amount value from word line
                val = float(amount)

                # Skip boilerplate rows
                if re.search(r"(TOTAL ACCOUNT BALANCE|NEW BALANCE|PREVIOUS ACCOUNT BALANCE|CASH BACK SUMMARY|INTEREST RATE)", desc_text, re.I):
                    i = j
                    continue

                # Output posting date as Date in DD/MM/YYYY
                date_out = post_iso
                if re.match(r"^\d{4}-\d{2}-\d{2}$", date_out):
                    date_out = f"{date_out[8:10]}/{date_out[5:7]}/{date_out[:4]}"

                # TransactionDate and PostingDate
                tx_date_out = tx_iso
                if re.match(r"^\d{4}-\d{2}-\d{2}$", tx_date_out):
                    tx_date_out = f"{tx_date_out[8:10]}/{tx_date_out[5:7]}/{tx_date_out[:4]}"
                post_date_out = date_out

                rows.append({
                    'Date': date_out,
                    'TransactionDate': tx_date_out,
                    'PostingDate': post_date_out,
                    'Description': desc_text,
                    'Amount': round(val, 2),
                })

                i = j

    return pd.DataFrame(rows)

# =========================
# RBC Mastercard (word-based left-firstline + right-summary)
# =========================

def _rbc_mc_parse_closing_from_name(name: str) -> Optional[tuple]:
    m = re.search(r"(20\d{2})-(\d{2})-(\d{2})", (name or ""))
    if not m:
        return None
    try:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    except Exception:
        return None

def _rbc_mc_group_lines(words: list) -> list:
    lines = []
    cur = []
    cur_y = None
    for w in sorted(words, key=lambda w: (round(float(w.get("top", 0.0)), 2), float(w.get("x0", 0.0)))):
        y = round(float(w.get("top", 0.0)), 2)
        if cur_y is None or abs(y - cur_y) <= 1.0:
            cur.append(w)
            cur_y = y if cur_y is None else (cur_y + y) / 2.0
        else:
            if cur:
                lines.append(cur)
            cur = [w]
            cur_y = y
    if cur:
        lines.append(cur)
    return lines

SUMMARY_KEYS = (
    "ANNUAL INTEREST RATES", "CALCULATING YOUR BALANCE", "PREVIOUS ACCOUNT BALANCE",
    "NEW BALANCE", "TOTAL ACCOUNT BALANCE", "INTEREST RATE", "CASH BACK", "FOREIGN CURRENCY"
)

def _rbc_mc_detect_left_right_split(page) -> tuple:
    words = page.extract_words(x_tolerance=1.6, y_tolerance=1.6, keep_blank_chars=False, use_text_flow=True) or []
    y_header = None
    for w in words:
        t = str(w.get("text", "")).upper()
        if "TRANSACTION" in t and "DATE" in t:
            y_header = float(w.get("top", 0.0)); break
    if y_header is None:
        y_header = 0.0

    x_right_keywords = []
    for w in words:
        t = str(w.get("text", "")).upper()
        if any(k in t for k in SUMMARY_KEYS) and float(w.get("top", 0.0)) > y_header + 10.0:
            x_right_keywords.append(float(w.get("x0", 0.0)))

    x_rect = None
    try:
        rects = getattr(page, 'rects', []) or []
        cands = []
        for r in rects:
            x0,y0,x1,y1 = float(r.get("x0", 0.0)), float(r.get("y0", 0.0)), float(r.get("x1", 0.0)), float(r.get("y1", 0.0))
            width = abs(x1 - x0); height = abs(y1 - y0); top = min(y0, y1)
            if width > page.width * 0.35 and height >= 8.0 and top > (y_header + 10.0):
                cands.append(x0)
        if cands:
            x_rect = min(cands)
    except Exception:
        pass

    xs = [x for x in [x_rect] + x_right_keywords if x is not None]
    x_split = min(xs) if xs else page.width * 0.62

    y_summary = None
    for w in words:
        t = str(w.get("text", "")).upper()
        if any(k in t for k in SUMMARY_KEYS) and float(w.get("top", 0.0)) > y_header + 10.0:
            y_summary = float(w.get("top", 0.0)); break
    y_top = (y_header or 0.0) + 8.0
    y_bottom = (y_summary - 8.0) if y_summary else page.height

    return (0.0, y_top, max(10.0, x_split - 6.0), y_bottom, x_split)

_AMT_TOKEN_RBC_MC = re.compile(r"^-?\$?\d{1,3}(?:,\d{3})*\.\d{2}$")

def extract_rbc_mc_transactions_left_firstline_from_bytes(data_bytes: bytes, source_name: Optional[str], fallback_end_dt) -> pd.DataFrame:
    closing_year = None
    closing_month = None
    if source_name:
        parsed = _rbc_mc_parse_closing_from_name(source_name)
        if parsed:
            closing_year, closing_month, _ = parsed
    if closing_year is None or closing_month is None:
        if fallback_end_dt:
            closing_year, closing_month = fallback_end_dt.year, fallback_end_dt.month
        else:
            now = datetime.now(); closing_year, closing_month = now.year, now.month

    rows = []
    with pdfplumber.open(io.BytesIO(data_bytes)) as pdf:
        for page in pdf.pages:
            x0, y0, x1, y1, _x_split = _rbc_mc_detect_left_right_split(page)
            pg = page.crop((x0, y0, x1, y1))
            words = pg.extract_words(x_tolerance=1.4, y_tolerance=1.4, keep_blank_chars=False, use_text_flow=True) or []
            lines = _rbc_mc_group_lines(words)
            x_left_dates = 120.0
            x_desc_left, x_desc_right = 120.0, 460.0
            for L in lines:
                left_tokens = [str(w.get("text", "")) for w in L if float(w.get("x0", 0.0)) <= x_left_dates]
                left_str = " ".join(left_tokens)
                m = re.match(r"^\s*([A-Za-z]{3})\s+(\d{1,2})\s+([A-Za-z]{3})\s+(\d{1,2})\b", left_str)
                if not m:
                    continue
                mon1, d1, mon2, d2 = m.groups()
                amount = None
                for tok in [str(w.get("text", "")) for w in sorted(L, key=lambda ww: float(ww.get("x0", 0.0)), reverse=True)]:
                    t = tok.replace(",", "").replace("$", "").strip()
                    if re.fullmatch(r"-?\d+\.\d{2}", t):
                        try:
                            amount = float(t); break
                        except Exception:
                            pass
                desc_tokens = [str(w.get("text", "")) for w in L if x_desc_left <= float(w.get("x0", 0.0)) <= x_desc_right]
                desc_tokens = [t for t in desc_tokens if not _AMT_TOKEN_RBC_MC.match(t)]
                desc = " ".join(desc_tokens).strip()
                if amount is None or not desc:
                    continue
                rows.append({
                    "transaction_date": to_iso_from_mon_day(mon1, d1, closing_year, closing_month),
                    "posting_date":     to_iso_from_mon_day(mon2, d2, closing_year, closing_month),
                    "description": " ".join(desc.split()),
                    "amount": round(float(amount), 2),
                })
    df = pd.DataFrame(rows, columns=["transaction_date","posting_date","description","amount"]) if rows else pd.DataFrame(columns=["transaction_date","posting_date","description","amount"])
    if not df.empty:
        df["type"] = df["amount"].apply(lambda x: "payment_or_credit" if float(x) < 0 else "purchase_or_debit")
    return df

def parse_rbc_mc_summary_right_from_bytes(data_bytes: bytes) -> Dict[str, Optional[float]]:
    LABELS = {
        "block_start":       [r"CALCULATING\s+YOUR\s+BALANCE", r"CALCUL.*VOTRE\s+SOLDE"],
        "block_end":         [r"NEW\s+BALANCE", r"TOTAL\s+ACCOUNT\s+BALANCE", r"NOUVEAU\s+SOLDE", r"TOTAL\s+DU\s+SOLDE"],
        "payments_credits":  [r"PAYMENTS\s*&\s*CREDITS?", r"PAYMENTS\s*&\s*CR(É|E)DITS?"],
        "purchases_debits":  [r"PURCHASES\s*&\s*DEBITS?",  r"ACHATS\s*&\s*D(É|E)BITS?"],
        "cash_advances":     [r"CASH\s*ADVANCES?",         r"AVANCES\s*d['’]ESP(È|E)CES?"],
        "interest":          [r"INTEREST",                 r"INT(É|E)R(Ê|E)T(?:S)?"],
        "fees":              [r"FEES",                     r"FRAIS"],
    }

    def detect_x_split(page):
        words = page.extract_words(x_tolerance=1.6, y_tolerance=1.6, keep_blank_chars=False, use_text_flow=True) or []
        y_header = None
        for w in words:
            if "TRANSACTION" in str(w.get("text", "")).upper() and "DATE" in str(w.get("text", "")).upper():
                y_header = float(w.get("top", 0.0)); break
        if y_header is None:
            y_header = 0.0
        x_right_keywords = []
        for w in words:
            t = str(w.get("text", "")).upper()
            if any(k in t for k in SUMMARY_KEYS) and float(w.get("top", 0.0)) > y_header + 10.0:
                x_right_keywords.append(float(w.get("x0", 0.0)))
        x_rect = None
        try:
            rects = getattr(page, 'rects', []) or []
            cands = []
            for r in rects:
                x0,y0,x1,y1 = float(r.get("x0", 0.0)), float(r.get("y0", 0.0)), float(r.get("x1", 0.0)), float(r.get("y1", 0.0))
                width = abs(x1 - x0); height = abs(y1 - y0); top = min(y0, y1)
                if width > page.width * 0.35 and height >= 8.0 and top > (y_header + 10.0):
                    cands.append(x0)
            if cands:
                x_rect = min(cands)
        except Exception:
            pass
        xs = [x for x in [x_rect] + x_right_keywords if x is not None]
        return (min(xs) if xs else page.width * 0.62)

    def group_lines(words):
        lines=[]; cur=[]; cur_y=None
        for w in sorted(words, key=lambda w: (round(float(w.get("top",0.0)),2), float(w.get("x0",0.0)))):
            y=round(float(w.get("top",0.0)),2)
            if cur_y is None or abs(y-cur_y)<=1.0:
                cur.append(w); cur_y=y if cur_y is None else (cur_y+y)/2.0
            else:
                lines.append(cur); cur=[w]; cur_y=y
        if cur: lines.append(cur)
        return lines

    def parse_amount_from_line(s: str) -> Optional[float]:
        s2 = s.replace(",", "")
        m = re.findall(r"(-?\$?\s*\d{1,3}(?:\d{3})*\.\d{2})(?!\s*%)", s2)
        if not m:
            m = re.findall(r"(-?\$?\s*\d+\.\d{2})(?!\s*%)", s2)
        if not m:
            return None
        raw = m[-1].replace("$","" ).replace(" ","")
        try:
            return abs(float(raw))
        except Exception:
            return None

    vals: Dict[str, Optional[float]] = {"payments_credits": None, "purchases_debits": None, "cash_advances": None, "interest": None, "fees": None}
    with pdfplumber.open(io.BytesIO(data_bytes)) as pdf:
        for page in pdf.pages:
            x_split = detect_x_split(page)
            rcrop = page.crop((x_split+2.0, 0.0, page.width, page.height))
            words = rcrop.extract_words(x_tolerance=1.6, y_tolerance=1.6, keep_blank_chars=False, use_text_flow=True) or []
            lines = group_lines(words)
            texts = [" ".join(str(w.get("text","")) for w in line) for line in lines]
            ups   = [t.upper() for t in texts]

            start_i = None
            for i, u in enumerate(ups):
                if re.search(r"CALCULATING\s+YOUR\s+BALANCE|CALCUL.*VOTRE\s+SOLDE", u):
                    start_i = i; break
            if start_i is None:
                continue
            end_i = len(ups) - 1
            for i in range(start_i+1, len(ups)):
                if re.search(r"NEW\s+BALANCE|TOTAL\s+ACCOUNT\s+BALANCE|NOUVEAU\s+SOLDE|TOTAL\s+DU\s+SOLDE", ups[i]):
                    end_i = i; break

            purchases_i = None
            for i in range(start_i, end_i+1):
                if re.search(r"PURCHASES\s*&\s*DEBITS?|ACHATS\s*&\s*D(É|E)BITS?", ups[i]):
                    amt = parse_amount_from_line(texts[i])
                    if amt is not None:
                        vals["purchases_debits"] = amt
                        purchases_i = i
                    break

            if purchases_i is None:
                for i in range(start_i, end_i+1):
                    if re.search(r"CASH\s*ADVANCES?|AVANCES\s*d['’]ESP(È|E)CES?", ups[i]):
                        amt = parse_amount_from_line(texts[i])
                        if amt is not None:
                            vals["cash_advances"] = amt
                            break
            else:
                for i in range(purchases_i+1, end_i+1):
                    if re.search(r"CASH\s*ADVANCES?|AVANCES\s*d['’]ESP(È|E)CES?", ups[i]):
                        amt = parse_amount_from_line(texts[i])
                        if amt is not None:
                            vals["cash_advances"] = amt
                            break

            for i in range(start_i, end_i+1):
                u = ups[i]
                if vals["payments_credits"] is None and re.search(r"PAYMENTS\s*&\s*CREDITS?|PAYMENTS\s*&\s*CR(É|E)DITS?", u):
                    amt = parse_amount_from_line(texts[i])
                    if amt is not None: vals["payments_credits"] = amt
                if vals["interest"] is None and re.search(r"INTEREST|INT(É|E)R(Ê|E)T(?:S)?", u):
                    amt = parse_amount_from_line(texts[i])
                    if amt is not None: vals["interest"] = amt
                if vals["fees"] is None and re.search(r"FEES|FRAIS", u):
                    amt = parse_amount_from_line(texts[i])
                    if amt is not None: vals["fees"] = amt

            if vals["purchases_debits"] is not None:
                break
    return vals

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
