import unittest
import tempfile
import json
from pathlib import Path
import chat_semantic_assessment as assessment


class AssessmentChecks(unittest.TestCase):
    def test_payload_contains_justification_and_only_candidate_citations(self):
        row = {'answer': 'Yes', 'citations': ['A:p1:c0'], 'final_context': [
            {'id': 'A:p1:c0', 'text': 'cited', 'page0': 0},
            {'id': 'A:p2:c0', 'text': 'uncited', 'page0': 1}]}
        question = {'question': 'Did it grow?', 'answer': 'Yes',
                    'justification': '2 > 1', 'evidence': [{'evidence_text': 'gold'}]}
        payload = assessment.payload(row, question)
        self.assertEqual(payload['reference_justification'], '2 > 1')
        self.assertEqual(len(payload['candidate_cited_passages']), 1)
        self.assertNotIn('variant', payload)
        self.assertNotIn('reference_match', payload)

    def test_missing_duplicate_and_unknown_ratings_are_rejected(self):
        cases = [{'blind_id': 'E001'}]
        for ratings in [[], [{'blind_id': 'X'}], [{'blind_id': 'E001'}] * 2]:
            with self.subTest(ratings=ratings), self.assertRaises(ValueError):
                assessment.validate(cases, ratings)

    def test_unresolved_cases_remain_in_denominator(self):
        self.assertEqual(assessment.interval_counts(['pass', 'fail', 'not_assessable']),
                         {'n': 3, 'pass': 1, 'fail': 1, 'unresolved': 1,
                          'lower_bound': 1/3, 'upper_bound': 2/3, 'accuracy': None})

    def test_valid_abstention_is_not_invalid(self):
        rating = assessment.terminal_rating({'blind_id': 'E001', 'payload': {
            'candidate_abstain': True, 'candidate_invalid': False}})
        self.assertEqual(rating['answer_correctness'], 'fail')
        self.assertIn('valid abstention', rating['reason_correctness'])
        self.assertEqual(rating['citation_support'], 'not_assessable')

    def test_terminal_state_cannot_be_promoted_by_a_judge(self):
        case = {'blind_id': 'E001', 'payload': {'candidate_abstain': True}}
        rating = assessment.terminal_rating(case)
        rating['answer_correctness'] = 'pass'
        with self.assertRaises(ValueError):
            assessment.validate([case], [rating])

    def test_citation_nonanswers_are_not_uncertain_possible_passes(self):
        counts = assessment.interval_counts(['pass', 'fail', 'not_assessable'], not_applicable=1)
        self.assertEqual(counts['not_applicable'], 1)
        self.assertEqual(counts['unresolved'], 0)
        self.assertEqual(counts['upper_bound'], 1/3)

    def test_mapping_rejects_changed_payload_and_duplicate_ids(self):
        cases = [{'blind_id': 'E001', 'payload': {'candidate_answer': 'Yes'}}]
        mapping = [{'blind_id': 'E001', 'payload_sha256': 'wrong'}]
        for rows in [mapping, mapping * 2]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                assessment.validate_mapping(cases, rows)

    def test_finalized_rating_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            record = folder / 'evaluation_ratings.jsonl'
            record.write_text('original')
            (folder / 'final_assessment_manifest.json').write_text(json.dumps({
                'files': {record.name: assessment.digest(record)}}))
            assessment.verify_final_assessment(folder)
            record.write_text('changed')
            with self.assertRaises(ValueError):
                assessment.verify_final_assessment(folder)


if __name__ == '__main__':
    unittest.main()
