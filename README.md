# Text-Into-Meaning-RAG

A lightweight, efficient Retrieval-Augmented Generation (RAG) framework baseline, currently configured for culinary domain question answering.

## Key Features

- **Dynamic Hardware Detection**: Zero-configuration required. The system automatically detects and routes execution to **NVIDIA GPU (CUDA)**, **Apple Silicon (MPS)**, or gracefully falls back to **CPU**, ensuring seamless cross-platform compatibility.
- **Local Embedding & Vector Search**: Utilizes `sentence-transformers` and `FAISS` for fast, offline similarity search.
- **Open-Source LLM Integration**: Powered by `Qwen/Qwen2.5-0.5B-Instruct` for precise answer generation.
- **End-to-End Pipeline**: Includes complete scripts for indexing, inference, and exact-match/F1 token evaluation.

## Directory Structure

```text
rag_baseline/
├── data/
│   ├── corpus.jsonl               # Background knowledge corpus
│   ├── benchmark.json             # Labelled benchmark for evaluation
│   └── test_queries.json          # Inference input questions
├── index_store/                   # Generated FAISS vector index (auto-created)
├── outputs/                       # Generated answers and eval metrics (auto-created)
├── config.py                      # Centralized configuration & hardware routing
├── utils.py                       # Helper functions and metric calculations
├── install.py                     # Environment setup and dependency installation script
├── build_index.py                 # Script to chunk and embed the corpus
├── inference.py                   # Script to run RAG generation
├── evaluation.py                  # Script to score model outputs
├── requirements.txt               # Project dependencies
└── README.md
```

## Environment Setup

You only need to run this once. The script will automatically detect your hardware (CUDA/MPS/CPU) and install the correct dependencies.

```bash
python install.py
```

## Quick Start

Before running the pipeline, ensure your background text is in `data/corpus.jsonl` and your test questions are in `data/test_queries.json` and `data/benchmark.json`.

### 1. Build the Vector Index
This will parse the corpus, chunk the text, generate embeddings, and save the FAISS index to `index_store/`.
```bash
python build_index.py
```

### 2. Run Inference
The system will automatically utilize available hardware resources to retrieve context and generate answers.
```bash
python inference.py
```
Outputs will be saved to `outputs/inference_output.json`.

### 3. Evaluate Performance
Compare generated answers against the gold standard to calculate Exact Match and Token F1 scores.
```bash
python evaluation.py
```
Metrics will be saved to `outputs/evaluation_results.json`.

## Configuration

All major parameters (chunk size, model selection, generation temperature, etc.) can be easily adjusted in `config.py`.