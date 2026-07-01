# Reproducibility: Retrieval Benchmark v1

## Setup

Use Python 3.10+ from the repository root.

```bash
pip install -r requirements.txt
```

The benchmark uses the existing three-paper sample corpus listed in `data/sample/sample_corpus_manifest.json`.

## Validate tests

```bash
pytest
```

The default pytest config disables pytest cache writes to avoid Windows cache permission issues.

## Build chunks and draft labels

```bash
python -m src.benchmarking.corpus
python evaluation/build_relevance_labels.py
```

`evaluation/build_relevance_labels.py` is only a helper for drafting labels. The benchmark scorer reads finalized `evaluation/relevance_labels.json` and does not use keyword matching.


## Recommended Conda GPU run

On this machine, the base Python Torch install can fail with a Windows DLL initialization error. Use the existing `torch_gpu` Conda environment for the real SentenceTransformers + Chroma benchmark:

```powershell
$env:PAPERQA_USE_SENTENCE_TRANSFORMERS = "1"
$env:HF_HUB_OFFLINE = "1"
conda --no-plugins run -n torch_gpu python evaluation\run_retrieval_benchmark.py
```

`HF_HUB_OFFLINE=1` avoids slow Hugging Face adapter-check retries when the model is already cached locally. `conda --no-plugins` avoids Conda's CUDA virtual-package plugin permission issue in this sandbox.
## Run benchmark

```bash
python evaluation/run_retrieval_benchmark.py
```

Generated outputs:

- `data/processed/benchmark_chunks_simple.json`
- `data/processed/benchmark_chunks_section_aware_v2.json`
- `evaluation/results/retrieval_benchmark_v1.json`
- `evaluation/results/retrieval_benchmark_v1.csv`
- `docs/retrieval_benchmark_v1.md`

By default, dense retrieval uses a lightweight in-memory lexical-vector fallback so the benchmark runs even when local Torch DLLs are unavailable. To opt into SentenceTransformers + Chroma, set `PAPERQA_USE_SENTENCE_TRANSFORMERS=1`; stores are written under `chroma_benchmark_simple` and `chroma_benchmark_section_aware_v2`.


