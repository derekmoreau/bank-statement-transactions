#!/usr/bin/env python3
"""
RBC Mastercard PDF → CSV extractor (pdfplumber)
- Character-aware parsing with spillover description handling
- Per-statement validation against "Calculating Your Balance" box
Usage:
  python rbc_mastercard_extract.py MasterCard\ Statement-6810\ 2021-01-25.pdf ... --outdir mc_outputs
"""
import re
import io
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pdfplumber
import pandas as pd


# ---------------- Constants & Regex ----------------
MONTHS = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
AMOUNT_RE = re.compile(r"-?\$?\d[\d,]*\.\d{2}")


# ---------------- Helpers ----------------
def parse_closing_from_filename(path: Path) -> Tuple[int,int,int]:
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})\.pdf$", path.name)
    if not m:
        raise ValueError(f"Filename must end with YYYY-MM-DD.pdf: {path}")
    y, mo, d = map(int, m.groups())
    return y, mo, d


def to_iso_from_mon_day(mon_str: str, day_str: str, closing_year: int, closing_month: int) -> str:
    mon = MONTHS.get(mon_str[:3].title())
    if mon is None:
        return f"{mon_str} {day_str}"
    year = closing_year - 1 if mon > closing_month else closing_year
    return f"{year:04d}-{mon:02d}-{int(day_str):02d}"


def find_header_x_positions(words: List[dict]) -> Tuple[float, float]:
    """
    Locate Description and Amount column x positions from header tokens.
    Fallback heuristics if not found.
    Returns: (x_desc, x_amt)
    """
    x_desc: Optional[float] = None
    x_amt: Optional[float]  = None
    # Primary: look for explicit header tokens
    for w in words:
        t = str(w.get("text", "")).strip().upper().replace(" ", "")
        if t in ("ACTIVITY","ACTIVITYDESCRIPTION","DESCRIPTION"):
            if x_desc is None:
                x_desc = float(w["x0"])
        if t in ("AMOUNT($)","AMOUNT","AMOUNT($)DATE","AMOUNT($)POSTING"):
            if x_amt is None:
                x_amt = float(w["x0"])
    # Fallbacks
    if x_desc is None or x_amt is None:
        xs = sorted(float(w.get("x0", 0.0)) for w in words)
        if x_desc is None and xs:
            x_desc = xs[len(xs)//3]  # a rough third-in from the left
        if x_amt is None and xs:
            x_amt = xs[-1] - 100     # near the right edge
    return float(x_desc or 200.0), float(x_amt or 500.0)


def group_chars_by_y(chars: List[dict], tol: float = 0.1) -> Dict[float, List[dict]]:
    lines: Dict[float, List[dict]] = {}
    for c in chars:
        y = round(float(c.get("top", 0.0)), 1)
        lines.setdefault(y, []).append(c)
    for y in list(lines.keys()):
        lines[y] = sorted(lines[y], key=lambda cc: float(cc.get("x0", 0.0)))
    return dict(sorted(lines.items()))


def assemble_from_chars(line_chars: List[dict], gap_ratio: float = 0.33, min_abs_gap: float = 1.2) -> str:
    if not line_chars:
        return ""
    xs  = [float(c["x0"]) for c in line_chars]
    x1s = [float(c["x1"]) for c in line_chars]
    widths = [x1s[i]-xs[i] for i in range(len(xs))]
    med_w = sorted(widths)[len(widths)//2] if widths else 4.0
    out = [str(line_chars[0]["text"])]
    for i in range(len(line_chars)-1):
        gap = xs[i+1] - x1s[i]
        if gap > max(gap_ratio*med_w, min_abs_gap):
            out.append(" ")
        out.append(str(line_chars[i+1]["text"]))
    return "".join(out)


# ---------------- Summary truth ----------------
def parse_summary_truth(pdf_path: str) -> Dict[str, Optional[float]]:
    with pdfplumber.open(pdf_path) as pdf:
        text = "".join((p.extract_text() or "") for p in pdf.pages)
    t = text.replace(",", "")
    def grab(pat: str) -> Optional[float]:
        m = re.search(pat, t, flags=re.IGNORECASE)
        try:
            return float(m.group(1)) if m else None
        except Exception:
            return None
    return {
        "payments_credits": grab(r"Payments\s*&\s*credits\s*-\s*\$?\s*([0-9]+\.[0-9]{2})"),
        "purchases_debits": grab(r"Purchases\s*&\s*debits\s*\$?\s*([0-9]+\.[0-9]{2})"),
        "previous_balance": grab(r"Previous\s*Account\s*Balance\s*\$?\s*([0-9]+\.[0-9]{2})"),
        "new_balance":      grab(r"(?:Total\s*Account\s*Balance|NEW\s*BALANCE)\s*\$?\s*([0-9]+\.[0-9]{2})"),
    }


# ---------------- Core extractor (with spillover fix) ----------------
def extract_mc_transactions(pdf_path: str) -> pd.DataFrame:
    closing_year, closing_month, _ = parse_closing_from_filename(Path(pdf_path))
    rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            words = page.extract_words(x_tolerance=2, y_tolerance=3, keep_blank_chars=False, use_text_flow=True)
            if not words:
                continue
            x_desc, x_amt = find_header_x_positions(words)
            char_lines = group_chars_by_y(page.chars, tol=0.1)

            for _, cline in char_lines.items():
                # Build three slices
                left_text_narrow = assemble_from_chars([c for c in cline if float(c["x0"]) < (x_desc - 2)]).strip()
                # Widen the left capture window to pull any description spillover starting right after dates
                left_text_wide   = assemble_from_chars([c for c in cline if float(c["x0"]) < (x_desc + 40)])
                mid_text         = assemble_from_chars([c for c in cline if x_desc - 1 <= float(c["x0"]) < x_amt - 1]).strip()
                # Amount zone widened a bit to catch minus signs
                right_text       = assemble_from_chars([c for c in cline if float(c["x0"]) >= x_amt - 12]).strip()

                # Detect two dates on the left (Transaction / Posting)
                mdate = re.search(r"([A-Za-z]{3})\s*([0-9]{1,2})\s+([A-Za-z]{3})\s*([0-9]{1,2})", left_text_narrow)
                if not mdate:
                    continue

                # Extract the amount (must be on the same physical line)
                m_amt = list(AMOUNT_RE.finditer(right_text))
                if not m_amt:
                    continue

                # Spillover part after the two dates, from the widened-left slice
                m_spill = re.match(r"^\s*[A-Za-z]{3}\s*\d{1,2}\s+[A-Za-z]{3}\s*\d{1,2}\s*(.*)$", left_text_wide)
                spill = (m_spill.group(1) if m_spill else "").strip()

                # Final description = spill + mid
                desc_parts = [p for p in (spill, mid_text) if p]
                desc_text = re.sub(r"\s+", " ", " ".join(desc_parts)).strip()
                if not desc_text:
                    continue  # must have some description text

                # Dates → ISO
                tx_mon, tx_day, post_mon, post_day = mdate.groups()
                tx_date   = to_iso_from_mon_day(tx_mon.title(), tx_day, closing_year, closing_month)
                post_date = to_iso_from_mon_day(post_mon.title(), post_day, closing_year, closing_month)

                # Amount value
                val = float(m_amt[-1].group(0).replace(",", "").replace("$", ""))

                # Skip obvious non-rows if any (belt-and-suspenders)
                if re.search(r"(TOTAL ACCOUNT BALANCE|NEW BALANCE|PREVIOUS ACCOUNT BALANCE|CASH BACK SUMMARY|INTEREST RATE)", desc_text, re.I):
                    continue

                rows.append({
                    "transaction_date": tx_date,
                    "posting_date": post_date,
                    "description": desc_text,
                    "amount": round(val, 2),
                })

    df = pd.DataFrame(rows, columns=["transaction_date","posting_date","description","amount"])
    df["type"] = df["amount"].apply(lambda x: "payment_or_credit" if x < 0 else "purchase_or_debit")
    return df


# ---------------- CLI ----------------
def main():
    import argparse
    ap = argparse.ArgumentParser(description="RBC Mastercard PDF → CSV extractor with validation.")
    ap.add_argument("pdfs", nargs="+", help="One or more RBC Mastercard PDFs (filenames ending with YYYY-MM-DD.pdf).")
    ap.add_argument("--outdir", default="mc_outputs", help="Output directory for CSVs and validation report.")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    combined = []
    report = {}

    for p in args.pdfs:
        df = extract_mc_transactions(p)
        s  = parse_summary_truth(p)

        # Save CSV
        Path(outdir / (Path(p).stem + ".csv")).write_text(df.to_csv(index=False))

        # Validation
        pos = round(df.loc[df["amount"] > 0, "amount"].sum(), 2)
        neg = round(-df.loc[df["amount"] < 0, "amount"].sum(), 2)
        report[Path(p).name] = {
            "calc_purchases_debits": pos,
            "calc_payments_credits": neg,
            "source_purchases_debits": s.get("purchases_debits"),
            "source_payments_credits": s.get("payments_credits"),
            "match_purchases_debits": bool(s.get("purchases_debits") is not None and abs(pos - s["purchases_debits"]) < 0.01),
            "match_payments_credits": bool(s.get("payments_credits") is not None and abs(neg - s["payments_credits"]) < 0.01),
            "previous_balance": s.get("previous_balance"),
            "new_balance": s.get("new_balance"),
        }

        combined.append(df.assign(statement=Path(p).name))

    # Write combined outputs
    pd.concat(combined, ignore_index=True).to_csv(outdir / "combined_transactions.csv", index=False)
    (outdir / "validation_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


