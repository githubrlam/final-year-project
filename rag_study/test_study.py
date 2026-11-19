"""Focused checks for the experiment's failure-prone boundaries; no model download."""
import json
import unittest
from unittest.mock import patch
import study

class CharacterTokenizer:
    def encode(self, text, **kwargs): return list(text)
    def __call__(self, text, **kwargs): return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}

CHUNK = {"id": "DEMO_2022_10K:p3:c0", "doc_name": "DEMO_2022_10K", "page0": 2, "pdf_page": 3,
         "text": "Revenue ($ millions): 2022 = -110; 2021 = -100.", "retrieval_score": .03}

class FakeRetriever:
    def __init__(self): self.queries = []; self.docs = []
    def retrieve(self, query, doc_name=None, previous=None):
        self.queries.append(query)
        self.docs.append(doc_name)
        return [CHUNK]

def fake_calls(objects):
    outputs = iter(objects)
    return lambda messages, tokenizer, schema=None: {"message": {"content": json.dumps(next(outputs))},
                                       "prompt_eval_count": 10, "eval_count": 5, "done_reason": "stop"}

FINAL = {"action": "answer", "answer": "Revenue was -$110 million", "citations": [CHUNK["id"]]}

class ProtocolChecks(unittest.TestCase):
    def test_generation_disables_thinking_in_request_and_token_count(self):
        class ChatTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.kwargs = kwargs
                return [1, 2, 3]
        tok = ChatTokenizer()
        with patch("requests.post") as post:
            post.return_value.json.return_value = {"message": {"content": "{}"}}
            result = study.ollama_call([{"role": "user", "content": "Revenue?"}], tok)
        self.assertEqual(result["message"]["content"], "{}")
        self.assertIs(post.call_args.kwargs["json"].get("think"), False)
        self.assertIs(tok.kwargs.get("enable_thinking"), False)

    def test_dictionary_tokenizer_cannot_bypass_prompt_budget(self):
        class DictionaryDefaultTokenizer:
            def apply_chat_template(self, messages, return_dict=True, **kwargs):
                ids = [1] * 4000
                return {"input_ids": ids, "attention_mask": ids} if return_dict else ids
        with patch("requests.post") as post:
            with self.assertRaisesRegex(ValueError, "Prompt exceeds"):
                study.ollama_call([{"role": "user", "content": "Long question"}], DictionaryDefaultTokenizer())
            post.assert_not_called()

    def test_context_budget_includes_separators(self):
        chunks = [{**CHUNK, "id": f"item{n}", "text": "Revenue: 1"} for n in range(2)]
        section = lambda c: f'[{c["id"]}] {c["doc_name"]}, PDF page {c["pdf_page"]}\n{c["text"]}'
        separate_length = sum(len(section(c)) for c in chunks)
        with patch.dict(study.CONFIG, {"evidence_tokens": separate_length, "top_k": 4}):
            text, chosen, used = study.pack_context(chunks, CharacterTokenizer())
        self.assertEqual(len(chosen), 1)
        self.assertEqual(used, len(text))
        self.assertLessEqual(used, separate_length)

    def test_context_reports_exact_joined_token_count(self):
        chunks = [{**CHUNK, "id": f"item{n}", "text": "Revenue: 1"} for n in range(2)]
        text, chosen, used = study.pack_context(chunks, CharacterTokenizer())
        self.assertEqual(len(chosen), 2)
        self.assertEqual(used, len(text))

    def test_pdf_page_is_one_based(self):
        self.assertEqual(CHUNK["pdf_page"], CHUNK["page0"] + 1)
        _, context, _ = study.pack_context([CHUNK], CharacterTokenizer())
        self.assertEqual(context[0]["page0"], 2)

    def test_arithmetic_units_and_negative_percentage(self):
        self.assertEqual(study.calculate({"operation": "percentage_change", "operands": [-110, -100]}), 10)
        self.assertAlmostEqual(study.calculate({"operation": "divide", "operands": [10, 40]}), .25)
        self.assertEqual(study.calculate({"operation": "subtract", "operands": [5, 8]}), -3)

    def test_arbitrary_code_and_invalid_operands_rejected(self):
        for request in [{"operation": "__import__", "operands": [1]}, {"operation": "add", "operands": [float("nan")]},
                        {"operation": "add", "operands": [True]}, {"operation": "add", "operands": ["1"]}]:
            with self.assertRaises(ValueError): study.calculate(request)
        with self.assertRaises(ZeroDivisionError): study.calculate({"operation": "divide", "operands": [1, 0]})

    def test_original_text_and_table_headers_preserved(self):
        tok = CharacterTokenizer()
        parts = study.split_text("$ -100\n   units", tok, size=8, overlap=2)
        self.assertIn("$ -100", parts[0])
        parts = study.table_chunks([["$ millions", "2022", "2021"], ["Revenue", "", ""],
                                  ["Operating", "-110", "-100"], ["Investing", "20", "30"]], tok, size=60)
        self.assertTrue(all("$ millions" in p and "2022" in p and "2021" in p for p in parts))
        self.assertTrue(any("-110" in p for p in parts))

    def test_single_pass_cannot_retrieve_again(self):
        retriever = FakeRetriever()
        row = study.run_question("Revenue?", CHUNK["doc_name"], "B0", retriever, CharacterTokenizer(), fake_calls([FINAL]))
        self.assertFalse(row["invalid"])
        self.assertEqual(len(retriever.queries), 1)

    def test_fixed_two_pass_and_adaptive_stop(self):
        retriever = FakeRetriever()
        row = study.run_question("Revenue?", CHUNK["doc_name"], "B1", retriever, CharacterTokenizer(), fake_calls([{"query": "Revenue 2022"}, FINAL]))
        self.assertEqual(row["retrieval_rounds"], 2)
        retriever = FakeRetriever()
        row = study.run_question("Revenue?", CHUNK["doc_name"], "B2", retriever, CharacterTokenizer(),
                                 fake_calls([{"sufficient": True, "missing_evidence": "", "query": ""}, FINAL]))
        self.assertEqual(row["retrieval_rounds"], 1)

    def test_adaptive_retry_calculator_and_termination(self):
        retriever = FakeRetriever()
        calc = {"action": "calculate", "operation": "subtract", "operands": [5, 8], "citations": [CHUNK["id"]], "units": "millions"}
        row = study.run_question("Difference?", CHUNK["doc_name"], "B2", retriever, CharacterTokenizer(), fake_calls([
            {"sufficient": False, "missing_evidence": "previous year", "query": "previous revenue"}, calc, calc, calc]))
        self.assertTrue(row["invalid"])
        self.assertEqual(row["model_calls"], 4)
        self.assertEqual(row["retrieval_rounds"], 2)
        self.assertLessEqual(row["model_calls"], study.CONFIG["max_model_calls"])

    def test_unknown_evidence_and_invalid_citations_visible(self):
        absent = {"action": "abstain", "reason": "Insufficient evidence"}
        row = study.run_question("Unknown?", CHUNK["doc_name"], "B0", FakeRetriever(), CharacterTokenizer(), fake_calls([absent]))
        self.assertTrue(row["abstain"])
        self.assertFalse(row["invalid"])
        row = study.run_question("Revenue?", CHUNK["doc_name"], "B0", FakeRetriever(), CharacterTokenizer(), fake_calls([{**FINAL, "citations": ["fake"]}]))
        self.assertTrue(row["invalid"])

    def test_model_call_limit_and_malformed_json(self):
        with patch.dict(study.CONFIG, {"max_model_calls": 1}):
            row = study.run_question("Revenue?", CHUNK["doc_name"], "B2", FakeRetriever(), CharacterTokenizer(),
                                     fake_calls([{"sufficient": True, "missing_evidence": "", "query": ""}]))
            self.assertTrue(row["invalid"])
            self.assertEqual(row["model_calls"], 1)
        row = study.run_question("Revenue?", CHUNK["doc_name"], "B0", FakeRetriever(), CharacterTokenizer(),
                                 lambda *args: {"message": {"content": "{bad"}})
        self.assertTrue(row["invalid"])

    def test_gold_record_cannot_cross_generation_boundary(self):
        with self.assertRaises(ValueError):
            study.run_question({"question": "Revenue?", "answer": "gold"}, CHUNK["doc_name"], "B0", FakeRetriever(), CharacterTokenizer())

if __name__ == "__main__": unittest.main()
