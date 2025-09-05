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
    m_dep = re.search(r'(?i)Total\s+deposits.*?([+\-]?\$?[\d,]+\.\d{2})', full_text)
    m_wdr = re.search(r'(?i)Total\s+withdrawals.*?([+\-]?\$?[\d,]+\.\d{2})', full_text)
    if m_dep and m_wdr:
        return {'pos_label': 'Total deposits', 'pos_value': +f2(m_dep.group(1)),
                'neg_label': 'Total withdrawals', 'neg_value': -abs(f2(m_wdr.group(1)))}

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
            'Transaction Date': interpret_transaction_date(td_raw, start_dt, end_dt),
            'Posting Date': interpret_transaction_date(pd_raw, start_dt, end_dt),
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
            'Transaction Date': interpret_transaction_date(td_raw, start_dt, end_dt),
            'Posting Date': interpret_transaction_date(pd_raw, start_dt, end_dt),
            'Description': desc,
            'Amount': parse_amount(amt_str)
        })
    return pd.DataFrame(rows)

def parse_rbc_chequing_from_text(full_text, start_dt, end_dt) -> pd.DataFrame:
    """
    RBC chequing: supports BOTH "21 Dec" and "Dec 21" anchors.
    Each date block may contain multiple entries; we grab <desc> <amount> [balance].
    """
    # Find both "dd Mon" and "Mon dd"
    date_anchor = re.compile(
        r'((?:\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?))|'
        r'(?:(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}))\b',
        re.IGNORECASE
    )
    # In each block, capture: <desc> <amount> [balance]
    amt_bal_re = re.compile(rf'(.*?){AMT}(?:\s+(\d{{1,3}}(?:,\d{{3}})*\.\d{{2}}))?(?:\s|$)')

    matches = list(date_anchor.finditer(full_text))
    rows = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        block = full_text[start:end].strip()
        head = m.group(1)  # e.g., "21 Dec" OR "Dec 21"
        rest = block[len(head):].strip()

        for mm in amt_bal_re.finditer(rest):
            desc = (mm.group(1) or "").strip()
            if not desc:
                continue
            amount = parse_amount(mm.group(2))
            dl = desc.lower()
            # EN + FR sign heuristics
            if any(k in dl for k in ['deposit', 'received', 'payroll', 'refund', 'credit', 'dépôt', 'remboursement']):
                sign_amt = +amount
            elif any(k in dl for k in ['fee', 'purchase', 'withdrawal', 'sent', 'interac', 'payment', 'transfer', 'retrait', 'paiement', 'frais']):
                sign_amt = -amount
            else:
                sign_amt = -amount  # default
            rows.append({
                'Transaction Date': interpret_transaction_date(head, start_dt, end_dt),
                'Posting Date': None,
                'Description': desc,
                'Amount': sign_amt
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
            'Transaction Date': interpret_transaction_date(f"{mon} {day}", start_dt, end_dt),
            'Posting Date': None,
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

if __name__ == "__main__":
    main()
