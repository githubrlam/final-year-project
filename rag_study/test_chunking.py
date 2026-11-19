"""Source-preserving chunk boundaries; cached BGE regression needs no network."""
import json
import re
import unittest
from pathlib import Path

from chunking import split_text
from test_study import CharacterTokenizer

ROOT = Path(__file__).resolve().parent


class ChunkingChecks(unittest.TestCase):
    def test_complete_source_rows_and_whitespace_survive(self):
        rows = ["  Assets   120\n", "  Debt     -80\n", "  Equity    40\n"]
        parts = split_text("".join(rows), CharacterTokenizer(), size=25, overlap=0)
        self.assertEqual("".join(parts), "".join(rows))
        self.assertTrue(all(line in rows for part in parts for line in part.splitlines(keepends=True)))

    def test_long_line_keeps_signed_parenthesized_and_decimal_values_whole(self):
        text = "Amount -1,234.50 (2,345.67) +9,876.54 ending"
        parts = split_text(text, CharacterTokenizer(), size=18, overlap=3)
        for part in parts:
            start = text.index(part)
            end = start + len(part)
            for value in re.finditer(r"[-+]?\(?\d[\d,.]*\)?", text):
                self.assertFalse(value.start() < start < value.end(), part)
                self.assertFalse(value.start() < end < value.end(), part)
        for value in ["-1,234.50", "(2,345.67)", "+9,876.54"]:
            self.assertTrue(any(value in part for part in parts))

    def test_overlap_preserves_complete_rows_and_makes_progress(self):
        text = "a1\nb2\nc3\nd4\ne5\n"
        parts = split_text(text, CharacterTokenizer(), size=9, overlap=3)
        self.assertEqual(parts, ["a1\nb2\nc3\n", "c3\nd4\ne5\n"])

    def test_overlap_is_dropped_when_needed_for_next_complete_row(self):
        text = "a\nb\n1234567\n"
        self.assertEqual(split_text(text, CharacterTokenizer(), size=8, overlap=4),
                         ["a\nb\n", "1234567\n"])

    def test_oversized_indivisible_word_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "word.*budget"):
            split_text("123456789", CharacterTokenizer(), size=8, overlap=2)

    def test_empty_and_whitespace_only_input(self):
        for text in ["", "  \n\t"]:
            self.assertEqual(split_text(text, CharacterTokenizer(), size=8, overlap=2), [])

    def test_invalid_budgets_are_rejected_even_for_empty_input(self):
        for size, overlap in [(0, 0), (-1, 0), (8, -1), (8, 8)]:
            with self.subTest(size=size, overlap=overlap):
                with self.assertRaises(ValueError):
                    split_text("", CharacterTokenizer(), size=size, overlap=overlap)


class CachedBGERegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        try:
            cls.tokenizer = AutoTokenizer.from_pretrained(
                str(ROOT / "hf_cache/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"),
                local_files_only=True,
            )
        except OSError as exc:
            raise unittest.SkipTest(f"Cached BGE tokenizer unavailable: {exc}")

    def test_zero_overlap_does_not_duplicate_token_free_blank_lines(self):
        text = "One two.\n    \nThree four.\n"
        parts = split_text(text, self.tokenizer, size=4, overlap=0)
        self.assertEqual("".join(parts), text)

    def test_pfizer_balance_sheet_does_not_clip_25920(self):
        page = json.loads((ROOT / "outputs/data_audit/phase2/development_pages/PFIZER_2021_10K_p59.json").read_text())
        text = page["text"]
        parts = split_text(text, self.tokenizer)
        amount = text.index("25,920")
        for part in parts:
            start = text.index(part)
            end = start + len(part)
            self.assertFalse(amount < start < amount + len("25,920"), part[:100])
            self.assertFalse(amount < end < amount + len("25,920"), part[-100:])
        self.assertTrue(any("25,920" in part for part in parts))
        self.assertTrue(all(len(self.tokenizer.encode(part, add_special_tokens=False)) <= 350 for part in parts))


if __name__ == "__main__":
    unittest.main()
