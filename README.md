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
├── data/                          # Processed data for RAG (auto-generated)
│   ├── corpus.jsonl               # Background knowledge corpus
│   ├── benchmark.json             # Labelled benchmark for evaluation
│   └── test_queries.json          # Inference input questions
├── index_store/                   # Generated FAISS vector index (auto-created)
├── outputs/                       # Generated answers and eval metrics (auto-created)
├── Blog_EastAsian_Cuisines.txt    # Raw culinary corpus 1
├── East_Asian_Corpus_Massive.txt  # Raw culinary corpus 2
├── Wikibooks_EastAsian_Recipes.txt# Raw culinary corpus 3
├── east_asia_benchmark_test.json  # Raw Q&A pairs
├── config.py                      # Centralized configuration & hardware routing
├── utils.py                       # Helper functions and metric calculations
├── install.py                     # Environment setup and dependency installation script
├── prepare_data.py                # Script to convert raw files to data/ folder
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

### 1. Prepare Data
Convert raw text and JSON files into the formatted dataset required by the system.
```bash
python prepare_data.py
```

### 2. Build the Vector Index
This will parse the corpus, chunk the text, generate embeddings, and save the FAISS index to `index_store/`.
```bash
python build_index.py
```

### 3. Run Inference
The system will automatically utilize available hardware resources to retrieve context and generate answers.
```bash
python inference.py
```
Outputs will be saved to `outputs/inference_output.json`.

### 4. Evaluate Performance
Compare generated answers against the gold standard to calculate Exact Match and Token F1 scores.
```bash
python evaluation.py
```
Metrics will be saved to `outputs/evaluation_results.json`.

## Configuration

All major parameters (chunk size, model selection, generation temperature, etc.) can be easily adjusted in `config.py`.