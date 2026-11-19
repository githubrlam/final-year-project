"""Conservative structural admission for extracted tables."""
import re

_UNIT = re.compile(r"\b(thousand|million|billion)s?\b", re.IGNORECASE)
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


def table_quality(rows, page_text=None):
    """Return admission and reason codes without changing the raw extraction.

    Without page text, passing means only a header and labeled data are present.
    With page text, explicit units and multi-year headers in its first 20
    nonempty lines must also occur in the table. This cannot establish complete
    extraction or correct column alignment. Rejections fall back to page text.
    """
    clean = [["" if cell is None else str(cell).strip() for cell in row]
             for row in rows]
    reasons = []
    if len(clean) < 2:
        reasons.append("insufficient_rows")
    header = clean[0] if clean else []
    if not any(header):
        reasons.append("empty_header")
    elif not any(any(char.isalpha() for char in cell)
                 or re.fullmatch(r"(?:19|20)\d{2}", cell) for cell in header):
        reasons.append("missing_header_labels")
    if not any(sum(bool(cell) for cell in row) >= 2
               and any(char.isalpha() for cell in row for char in cell)
               for row in clean[1:]):
        reasons.append("no_labeled_data_rows")
    if page_text is not None:
        lines = [line.strip() for line in page_text.splitlines() if line.strip()][:20]
        table_text = "\n".join(cell for row in clean for cell in row)
        source_units = {unit.lower() for unit in _UNIT.findall("\n".join(lines))}
        table_units = {unit.lower() for unit in _UNIT.findall(table_text)}
        if source_units - table_units:
            reasons.append("missing_page_unit_context")
        source_years = set()
        for line in lines:
            years = _YEAR.findall(line)
            if len(set(years)) >= 2 and (line.split() == years or _UNIT.search(line)):
                source_years.update(years)
        if source_years - set(_YEAR.findall(table_text)):
            reasons.append("missing_page_year_context")
    return {"accepted": not reasons, "reasons": reasons}
