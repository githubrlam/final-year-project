# Final-year project · Budget-limited financial RAG

This project studies whether a small language model should search an annual report again when the first search misses a fact. It compares three policies on the same 52 FinanceBench questions from 18 reports. All policies use the same Qwen3.5-4B generator, retrieval pipeline and evidence cap.

| Policy | Search rule | Mean annotated-page recall | AI-assessed supported passes |
| --- | --- | ---: | ---: |
| B0 | Search once | 25.0% | 6/52 |
| B1 | Always search twice | 34.6% | 10/52 |
| B2 | Search again when the first passages appear to lack a required fact | 48.1% | 14/52 |

B2 made 12 fewer searches than B1 and used 5.6% more model tokens. Its four-pass advantage had a paired 95% interval of −1.9 to +17.0 percentage points. The scores come from a retrospective AI assessment on a previously inspected public benchmark subset. Agreement with human graders was not measured.

## Start here

- [Findings and limitations](results/FINDINGS.md) explains the results, the development comparisons and one financial calculation error.
- [Saved-results notebook](notebooks/00_saved_results.ipynb) recalculates the headline metrics from the 156 saved outcomes. View the executed output on GitHub or open the notebook from this checkout. Running it requires a Python 3 Jupyter kernel and the files in this repository.
- [Experiment notebooks](notebooks/) include data verification, PDF extraction, retrieval and the Kaggle workflows. The original workflow notebooks are templates for their recorded environments. They require the corresponding inputs and, for Kaggle runs, access to the original asset bundle or equivalent reconstructed inputs.
- [Source and frozen settings](rag_study/) contain the hash-verified generation snapshot, supporting code and selected tests. Look at [study.py](rag_study/study.py) for retrieval and answer generation, [row_evaluation.py](rag_study/row_evaluation.py) for the evaluation design and [the frozen protocol](rag_study/frozen_row_protocol.json) for settings and input hashes.

## How it works

The retriever searches the named filing with keyword and vector rankings, fuses the rankings and reranks the candidate passages. The generator sees at most four whole passages and 2,000 evidence tokens. A run can use up to two searches and four model calls per question. Benchmark references and annotated evidence are kept for scoring.

The saved-results notebook reproduces the published aggregates without model inference. To obtain the source benchmark and issuer reports for further work, install the Python dependencies and run the acquisition script from the repository root.

```sh
python -m pip install -r rag_study/requirements.txt
python rag_study/acquire.py
```

Exact regeneration of the frozen experiment also requires the pinned model and tokenizer, the historical inputs identified by hash, and a compatible Ollama and retrieval environment. See [the protocol](rag_study/frozen_row_protocol.json), [Kaggle runtime record](results/runtime.json) and [dependency snapshot](rag_study/requirements.kaggle.txt).

The benchmark is [FinanceBench](https://github.com/patronus-ai/financebench). See the [pinned source and licence information](results/financebench_provenance.json). The repository ships derived evaluation tables and notebook source. The acquisition script fetches issuer reports from their recorded sources.
