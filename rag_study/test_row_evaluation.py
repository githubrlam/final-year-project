import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

class FrozenRowEvaluationChecks(unittest.TestCase):
    def test_freeze_is_immutable_and_detects_source_data_tampering(self):
        from row_evaluation import create_protocol,verify_protocol
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'source.py').write_text('x=1\n');(root/'test.jsonl').write_text('data')
            files={'source.py':root/'source.py'};data={'data/test.jsonl':root/'test.jsonl'}
            p=root/'protocol.json'
            create_protocol(p,{'generator':'pinned'},dict(SYSTEM='s'),files,data,['q'],['D'],'devhash')
            verify_protocol(json.loads(p.read_text()),files,data)
            with self.assertRaises(ValueError):create_protocol(p,{}, {},files,data,['q'],['D'],'devhash')
            (root/'source.py').write_text('x=2\n')
            with self.assertRaises(ValueError):verify_protocol(json.loads(p.read_text()),files,data)
            (root/'source.py').write_text('x=1\n');(root/'test.jsonl').write_text('changed')
            with self.assertRaises(ValueError):verify_protocol(json.loads(p.read_text()),files,data)

    def test_question_matrix_requires_exact_frozen_ids_and_documents(self):
        from row_evaluation import validate_questions
        questions=[dict(financebench_id=str(i),doc_name='D'+str(i%18)) for i in range(52)]
        protocol={'evaluation_question_ids':sorted(str(i) for i in range(52)),'evaluation_documents':sorted('D'+str(i) for i in range(18))}
        validate_questions(questions,protocol)
        with self.assertRaises(ValueError):validate_questions(questions[:-1],protocol)
        with self.assertRaises(ValueError):validate_questions(questions[:-1]+[questions[0]],protocol)
        with self.assertRaises(ValueError):validate_questions([dict(q,doc_name='OTHER') for q in questions],protocol)

    def test_generate_passes_only_question_doc_and_uses_all_policies(self):
        from row_evaluation import generate
        study=SimpleNamespace(run_question=Mock(return_value={'answer':'saved'}))
        original=study.run_question
        q=[dict(financebench_id=str(i),question='question'+str(i),doc_name='DOC',answer='SECRET_GOLD',evidence=['SECRET_EVIDENCE']) for i in range(52)]
        seen=[]
        def matrix(study,questions,matrix,folder,retriever_for,tokenizer,scorer,scope):
            self.assertEqual(matrix,[('row_verified',v) for v in ('B0','B1','B2')])
            for m,v in matrix:
                for item in questions:study.run_question(item['question'],item['doc_name'],v,retriever_for(m),tokenizer)
        adapter=Mock(return_value=SimpleNamespace(events=[]))
        from unittest.mock import patch
        with patch('row_evaluation.VerifiedRowRetriever',adapter):
            generate(study,q,Path('/unused'),object(),object(),matrix,object(),{})
        self.assertIs(study.run_question,original)
        self.assertEqual(original.call_count,156)
        for call in original.call_args_list:
            self.assertEqual(len(call.args),5)
            self.assertNotIn('SECRET',str(call))
            self.assertEqual(call.args[1],'DOC')

    def test_runtime_scope_rejects_prompt_or_setting_drift_before_generation(self):
        from row_evaluation import generate
        from final_row_tuning import COMMON
        from contextlib import nullcontext
        config=dict(COMMON,input_ablation_method='baseline',final_row_method='row_verified')
        prompts=dict(SYSTEM='s',ANSWER='a',REWRITE='r',ASSESS='c')
        study=SimpleNamespace(CONFIG=dict(config),PROMPTS={'v5':prompts},run_question=Mock())
        def matrix(study,questions,methods,folder,retriever_for,tokenizer,scorer,scope):
            with scope(study,'row_verified'):pass
        protocol={'config':dict(config),'prompts':dict(prompts)}
        generate(study,[],Path('/unused'),object(),object(),matrix,object(),protocol)
        protocol['config']['seed']=99
        with self.assertRaises(ValueError):generate(study,[],Path('/unused'),object(),object(),matrix,object(),protocol)
        self.assertEqual(study.run_question.call_count,0)
        protocol['config']['seed']=42;protocol['prompts']['ANSWER']='changed'
        with self.assertRaises(ValueError):generate(study,[],Path('/unused'),object(),object(),matrix,object(),protocol)

if __name__=='__main__':unittest.main()
