"""The automatic metric is a matching proxy, not a semantic or citation judge."""
import unittest
import evaluation

class MatchingChecks(unittest.TestCase):
    def score(self,answer,reference,question='Answer in USD millions.',**flags):
        return evaluation.score_answer({'answer':answer,**flags},{'answer':reference,'question':question})

    def test_source_precision_and_equivalent_scales(self):
        for answer in ['5,466.312 million USD','$5.466312 billion','5,466,312 thousand USD','5466.312']:
            with self.subTest(answer=answer):self.assertEqual(self.score(answer,'$5466.00')['answer_correct'],1)
        self.assertEqual(self.score('1,615.9 million USD','$1616.00')['answer_correct'],1)

    def test_question_precision_overrides_trailing_zeros(self):
        self.assertEqual(self.score('1.004','1.00','Round to two decimal places.')['answer_correct'],1)
        self.assertEqual(self.score('1.006','1.00','Round to two decimal places.')['answer_correct'],0)
        self.assertEqual(self.score('2.84%','2.8%','Round to one decimal place; answer in percents.')['answer_correct'],1)
        self.assertEqual(self.score('1.015','101.5%','Answer in percents.')['answer_correct'],0)

    def test_signs_currency_ratio_and_extra_numbers(self):
        self.assertEqual(self.score('(1,200)','-1200')['answer_correct'],1)
        self.assertEqual(self.score('1200','-1200')['answer_correct'],0)
        self.assertEqual(self.score('EUR 100','$100')['answer_correct'],0)
        self.assertEqual(self.score('$3.46','3.46','What is the inventory turnover ratio?')['answer_correct'],0)
        self.assertEqual(self.score('2020: 1616; 2019: 1864','$1616.00')['answer_correct'],0)
        self.assertEqual(self.score('Not 1616 but 100','$1616.00')['answer_correct'],0)

    def test_yes_no_and_normalized_text_are_explicit_proxies(self):
        self.assertEqual(self.score('Yes, it grew.','Yes, the change was positive','Did PP&E grow?')['answer_correct'],1)
        self.assertEqual(self.score('No, it did not.','Yes, the change was positive','Did PP&E grow?')['answer_correct'],0)
        self.assertEqual(self.score(' ARRAY, Trillium!','Array Trillium','Name the companies.')['answer_correct'],1)
        self.assertEqual(self.score('An increase occurred.','It grew.','Describe the change.')['answer_correct'],0)

    def test_failures_and_declines_cannot_pass(self):
        for flags in [{'invalid':True},{'abstain':True}]:self.assertEqual(self.score('1616','$1616.00',**flags)['answer_correct'],0)

if __name__=='__main__':unittest.main()

class AutomaticProtocolChecks(unittest.TestCase):
    def test_numeric_negation_is_not_a_match(self):
        self.assertEqual(evaluation.score_answer({'answer':'It is not 1616.'},{'answer':'1616','question':'Answer in USD millions.'})['answer_correct'],0)

    def test_metrics_need_no_rating_sheet_and_keep_report_pairs(self):
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        import study
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data=root/'data';out=root/'outputs';data.mkdir();out.mkdir()
            qs=[{'financebench_id':f'q{i}','doc_name':doc,'question':'Answer in USD millions.','answer':'10',
                 'evidence':[{'doc_name':doc,'evidence_page_num':0}]} for i,doc in enumerate(['A','A','B'])]
            rows=[{'financebench_id':q['financebench_id'],'doc_name':q['doc_name'],'variant':v,'answer':'10',
                   'abstain':False,'invalid':False,'seconds':1,'model_calls':1,'retrieval_rounds':1,
                   'input_tokens':10,'output_tokens':5,'final_context':[]} for q in qs for v in ['B0','B1','B2']]
            study.write_jsonl(data/'dev.jsonl',qs);study.write_jsonl(out/'dev_predictions.jsonl',rows)
            with patch.object(study,'DATA',data),patch.object(study,'OUT',out),patch.dict(study.CONFIG,{'bootstrap_replicates':20}):
                table=study.metrics('dev')
            self.assertTrue(table.answer_correct.eq(1).all())
            self.assertNotIn('supported_correct',table.columns)
            self.assertTrue((out/'dev_automatic_scores.csv').exists())
            self.assertFalse((out/'dev_blind_scoring.csv').exists())
            import pandas as pd
            intervals=pd.read_csv(out/'dev_paired_intervals.csv')
            self.assertTrue(intervals[['difference','lower_95','upper_95']].eq(0).all().all())

class NumericBoundaryChecks(unittest.TestCase):
    def test_sentence_stop_and_negation(self):
        q={'answer':'1616','question':'Answer in USD millions.'}
        self.assertEqual(evaluation.score_answer({'answer':'The value is 1616.'},q)['answer_correct'],1)
        self.assertEqual(evaluation.score_answer({'answer':'It is not 1616.'},q)['answer_correct'],0)
        self.assertEqual(evaluation.score_answer({'answer':'1,61,6'},q)['answer_correct'],0)

class EvaluationBindingChecks(unittest.TestCase):
    def test_editing_score_file_invalidates_automatic_readiness(self):
        import json,tempfile
        from pathlib import Path
        from unittest.mock import patch
        import phase5,study
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);out=root/'outputs';data=root/'data';out.mkdir();data.mkdir()
            study.write_jsonl(data/'dev.jsonl',[]);study.write_jsonl(out/'dev_predictions.jsonl',[])
            score=out/'dev_automatic_scores.csv';score.write_text('answer_correct\n1\n')
            with patch.object(study,'ROOT',root),patch.object(study,'DATA',data),patch.object(study,'OUT',out):
                phase5.directory().mkdir(parents=True)
                (phase5.directory()/'dev_evaluation_manifest.json').write_text(json.dumps({'inputs_sha256':{'outputs/dev_automatic_scores.csv':study.digest(score)}}))
                self.assertTrue(phase5.readiness()['checks']['automatic_evaluation_current'])
                score.write_text('answer_correct\n0\n')
                self.assertFalse(phase5.readiness()['checks']['automatic_evaluation_current'])

class FinancialSignChecks(unittest.TestCase):
    def test_currency_prefixes_and_unicode_minus_preserve_sign(self):
        for answer in ['-$100','−$100','- USD 100','($100)']:
            with self.subTest(answer=answer):
                q={'answer':'-100','question':'Answer in USD millions.'}
                self.assertEqual(evaluation.score_answer({'answer':answer},q)['answer_correct'],1)
                q['answer']='100'
                self.assertEqual(evaluation.score_answer({'answer':answer},q)['answer_correct'],0)
