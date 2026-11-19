"""Token-bounded source slices that keep financial literals intact."""
import re


def split_text(text, tokenizer, size=350, overlap=50):
    """Prefer full lines, falling back to whole words for oversized lines.

    Overlap is a suffix of complete units, at most ``overlap`` tokens. Drop
    overlap when necessary to fit the next unit. An indivisible word over
    budget raises instead of returning a clipped number or oversized chunk.
    """
    if not 0 <= overlap < size:
        raise ValueError("Require 0 <= overlap < size.")
    if not text.strip():
        return []

    def count(value):
        return len(tokenizer.encode(value, add_special_tokens=False, truncation=False))

    units = []
    for line in text.splitlines(keepends=True):
        if count(line) <= size:
            units.append(line)
        else:
            words = re.findall(r"\s*\S+\s*|\s+", line)
            if any(count(word) > size for word in words):
                raise ValueError("Indivisible word exceeds chunk token budget; inspect source.")
            units.extend(words)

    chunks, start = [], 0
    while start < len(units):
        end = start + 1
        while end < len(units) and count("".join(units[start:end + 1])) <= size:
            end += 1
        chunks.append("".join(units[start:end]))
        if end == len(units):
            break
        next_start = end
        while overlap and next_start > start + 1 and count("".join(units[next_start - 1:end])) <= overlap:
            next_start -= 1
        while next_start < end and count("".join(units[next_start:end + 1])) > size:
            next_start += 1
        start = next_start
    return chunks
