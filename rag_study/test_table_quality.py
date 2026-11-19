"""Admission checks reject partial tables without repairing their contents."""
import copy
import unittest

from table_quality import table_quality


class TableQualityChecks(unittest.TestCase):
    def test_numeric_only_financial_fragment_is_rejected(self):
        rows = [["", None], ["$", "1,944"], ["29,125", None],
                ["14,882", None], ["181,476", None]]
        result = table_quality(rows)
        self.assertFalse(result["accepted"])
        self.assertIn("empty_header", result["reasons"])
        self.assertIn("no_labeled_data_rows", result["reasons"])

    def test_unit_year_header_and_labeled_values_are_accepted(self):
        rows = [["$ millions", "2022", "2021"], ["Revenue", "", ""],
                ["Operating", "-110", "-100"], ["Investing", "20", "30"]]
        self.assertEqual(table_quality(rows), {"accepted": True, "reasons": []})

    def test_simple_non_year_table_is_accepted(self):
        rows = [["Region", "Value"], ["North", "25"], ["South", "30"]]
        self.assertTrue(table_quality(rows)["accepted"])

    def test_numeric_zero_counts_as_a_value_and_input_is_unchanged(self):
        rows = [["Item", "Count"], ["Absent", 0]]
        original = copy.deepcopy(rows)
        self.assertTrue(table_quality(rows)["accepted"])
        self.assertEqual(rows, original)

    def test_header_without_labeled_body_is_rejected(self):
        rows = [["$ millions", "2022", "2021"], ["Revenue", "", ""],
                ["", "100", "90"], [None, "80", "70"]]
        result = table_quality(rows)
        self.assertFalse(result["accepted"])
        self.assertIn("no_labeled_data_rows", result["reasons"])

    def test_one_row_fragment_and_empty_table_are_rejected(self):
        for rows in ([], [["Cash", "1,000", "900"]]):
            with self.subTest(rows=rows):
                result = table_quality(rows)
                self.assertFalse(result["accepted"])
                self.assertIn("insufficient_rows", result["reasons"])

    def test_blank_label_cell_with_year_columns_is_accepted(self):
        rows = [["", "2022", "2021"], ["Revenue", "100", "90"]]
        self.assertTrue(table_quality(rows)["accepted"])

    def test_categorical_table_does_not_require_financial_numbers(self):
        rows = [["Name", "Role"], ["Ada", "Analyst"]]
        self.assertTrue(table_quality(rows)["accepted"])

    def test_explicit_source_units_and_years_retained_are_accepted(self):
        rows = [["Amounts in million", "2022", "2021"], ["Revenue", "100", "90"]]
        page = "Consolidated statements\n(In MILLIONS)\nYear ended December 31\n2022 2021"
        self.assertTrue(table_quality(rows, page)["accepted"])

    def test_missing_explicit_source_unit_or_year_is_rejected(self):
        page = "Consolidated statements\n(In millions)\n2022 2021"
        cases = [
            ([["Metric", "2022", "2021"], ["Revenue", "100", "90"]],
             ["missing_page_unit_context"]),
            ([["$ millions", "Current", "Prior"], ["Revenue", "100", "90"]],
             ["missing_page_year_context"]),
            ([["Metric", "Current", "Prior"], ["Revenue", "100", "90"]],
             ["missing_page_unit_context", "missing_page_year_context"]),
        ]
        for rows, reasons in cases:
            with self.subTest(reasons=reasons):
                original = copy.deepcopy(rows)
                self.assertTrue(table_quality(rows)["accepted"])
                result = table_quality(rows, page)
                self.assertFalse(result["accepted"])
                self.assertEqual(result["reasons"], reasons)
                self.assertEqual(rows, original)

    def test_simple_nonfinancial_page_does_not_require_units_or_years(self):
        rows = [["Region", "Value"], ["North", "25"], ["South", "30"]]
        page = "Regional breakdown\nRegion Value\nNorth 25\nSouth 30"
        self.assertTrue(table_quality(rows, page)["accepted"])

    def test_years_on_unit_header_line_are_checked(self):
        rows = [["$ billions", "Current", "Prior"], ["Revenue", "100", "90"]]
        result = table_quality(rows, "(BILLION, except per share data) 2022 2021")
        self.assertEqual(result["reasons"], ["missing_page_year_context"])

    def test_prose_year_mentions_are_not_treated_as_column_headers(self):
        rows = [["Region", "Value"], ["North", "25"]]
        page = "Comparison of changes in 2022 and 2021\nRegion Value\nNorth 25"
        self.assertTrue(table_quality(rows, page)["accepted"])

    def test_source_context_beyond_first_twenty_nonempty_lines_is_ignored(self):
        rows = [["Region", "Value"], ["North", "25"]]
        page = "\n\n".join(["Heading"] * 20 + ["(In thousands)", "2022 2021"])
        self.assertTrue(table_quality(rows, page)["accepted"])


if __name__ == "__main__":
    unittest.main()
