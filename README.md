# Text-Into-Meaning-RAG

The repo is built for COMP64702 Transforming Text Into Meaning Coursework.

This repo mainly contains:

- **Inference and Evaluation Pipelines:** `./inference_eval_pipeline.ipynb`

- **Experiment Source Code:** `./chunking`, `./embedding`,  `./retrieval`, `./generation`, and  `./evaluation`
- **Corpus JSONL, and Web Crawler:** `./corpus/*.jsonl`, `corpus/east-asian-corpus-builder`
- **Benchmark for Inference Demo:** `./benchmark/`
- **Chunked Files and FAISS Vector Base:** `./artifacts/`

## Highlights

- **Embedding Similarity Chunking**: Split or merge text chunks by language embedding models.
- **Hybrid retrieval**: BM25 + dense embeddings to reduce missed evidence.
- **Reciprocal Rank Fusion (RRF)**: Stabilize candidate quality across retrieval modes.
- **Cross-encoder reranking**: Higher precision in final context selection.
- **Stepwise Prompting**: Enhance generation quality by reducing hallucination.

# 1. Pipeline Codebase

## 1.1 Environment Set-up

- Python 3.11 recommended.
- Optional GPU for faster embedding and generation.

```bash
conda create -n <env_name> python=3.11 	# Init a conda environment with python 3.11
conda activate <env_name> 			   	# Activate the created environment
cd <REPO_ROOT>							# Change directory to the root of this repo
pip install -r requirements.txt			# Install required packages with Pip
```

**Notes:**

- If you have CUDA, install a CUDA-enabled `torch` build from the official PyTorch index.
- First run may download HuggingFace models.

## 1.2 Inference and Evaluation Pipelines

The Inference and Evaluation Pipelines are in the notebook:`./inference_pipeline.ipynb`

Run it in order from top to bottom.

## 1.3 Optional: Rebuild Artifacts from Scratch

This step can also be implemented by running the `Chunking` and `Chunk Embedding` sections in the notebook. 

If you want to regenerate chunks and FAISS artifacts instead of using existing files: 

### 1) Chunk the Corpus

```bash
python chunking/chunker.py
```

Default input/output:

- Input: `corpus/corpus.jsonl`
- Output: `artifacts/chunks/corpus_fixed.jsonl`

### 2) Build Embeddings + FAISS

```bash
python embedding/embed_chunks.py --input artifacts/chunks/corpus_fixed.jsonl --model Qwen/Qwen3-Embedding-0.6B
```

Generated files are written under `artifacts/faiss/`:

- `*.index`
- `*.meta.jsonl`
- `*.manifest.json`

After this, update the notebook config paths if file names differ.

# 2. Benchmark

This repository includes a small factoid-style benchmark for retrieval and generation experiments. The benchmark is stored as a JSON file with a top-level `queries` list. Each entry contains a natural-language question, a short reference answer, source metadata, and the gold chunk id used for retrieval evaluation.

### 2.1 Structure

Each benchmark item follows this structure:

```json
{
  "query": "What is gochujang?",
  "answer": "Gochujang is a fermented bean paste made with red pepper powder.",
  "source": "Wikipedia",
  "title": "Korean cuisine",
  "section": "Gochujang",
  "chunk_ids": {
    "embedding": 1150
  },
  "query_id": 78
}
```

Field descriptions:

- `query`: the user-style question used at inference time
- `answer`: the gold short answer
- `source`: the corpus source file (`80Cuisine`, `Wikibooks`, or `Wikipedia`)
- `title`: the document title in the source corpus
- `section`: the section heading associated with the evidence
- `chunk_ids.embedding`: the gold chunk id for chunk-level retrieval evaluation
- `query_id`: unique benchmark item id

### 2.2 Question Types

The benchmark currently contains **50** questions. It is designed as a compact **factoid-style benchmark** focused on short-answer retrieval. Most questions ask for a single factual span or a concise explanatory phrase rather than long-form synthesis.

Typical question patterns include:

- **Definition / description**  
  Examples: *What is a baozi?*, *What is daigakuimo?*, *What is xiǎochī in Taiwanese cuisine?*
- **Ingredients / composition**  
  Examples: *What is dashi traditionally made with?*, *What is inside a takoyaki?*, *What type of vinegar is used in Kung Pao Chicken sauce?*
- **Technique / preparation**  
  Examples: *What cooking technique is used to make tempura?*, *What cooking method characterises Wuxi-style cuisine?*
- **Origin / location**  
  Examples: *Where did takoyaki originate?*, *Where is the Taiwanese oyster omelette thought to have originated?*
- **Reason / cause**  
  Examples: *Why is hot pot considered a healthy meal?*, *Why should rice not be overwashed when making congee?*
- **Time / temperature / quantity-style factual queries**  
  Examples: *How long should wonton wrapper dough be refrigerated?*, *What temperature is used to fry Cantonese crispy fried chicken?*

In the current version, the benchmark is dominated by short factual prompts:
- `definition/description`: **21**
- `ingredients/composition`: **11**
- `technique/preparation`: **3**
- `origin/location`: **2**
- `reason/cause`: **3**
- `time/temperature`: **4**
- `other factual`: **6**

This distribution makes the benchmark especially suitable for evaluating:
- first-stage retrieval recall
- reranking quality
- evidence grounding for concise answer generation

### 2.3 Source Distribution

The benchmark draws evidence from three corpus sources:

- **Wikibooks**: **22** questions
- **Wikipedia**: **15** questions
- **80Cuisine**: **13** questions

This gives the benchmark a mix of:
- **recipe-style procedural content** from Wikibooks
- **encyclopedic cuisine background** from Wikipedia
- **course-style regional cuisine summaries** from 80Cuisine

As a result, retrieval models are tested on both:
- short recipe and ingredient descriptions
- broader cultural or historical cuisine passages
- section-specific evidence linked to named dishes, ingredients, and preparation methods

### 2.4 Intended Use

This benchmark is mainly intended for:
1. comparing chunking strategies
2. evaluating dense, sparse, and hybrid retrieval
3. testing reranking performance
4. validating whether the retrieved evidence supports short-form grounded answers

Because the benchmark is relatively small, it is best used for **rapid iteration and method comparison**, rather than as a large-scale final evaluation set.



# 3. Corpus

The corpus is stored as a JSONL file (`./corpus/corpus.jsonl`). Each line is a semi-structured record representing one section-level text unit collected from one of three sources: **Wikipedia**, **Wikibooks**, and **80Cuisine**. The corpus is designed to support both retrieval experiments and chunking experiments, so it combines encyclopedic cuisine descriptions, recipe-style procedural text, and shorter blog-style regional cuisine summaries.

## 3.1 Content Overview

The current corpus contains **1003 records** in total, spanning **147 document titles** across the three sources.

It mixes three complementary content types:

- **Wikipedia**: cuisine overviews, sub-cuisine pages, ingredient descriptions, and background knowledge
- **Wikibooks**: recipe pages with ingredient lists, preparation steps, notes, and variations
- **80Cuisine**: shorter cuisine blog posts with regional introductions, named dishes, and informal descriptive passages

This mixture is useful for RAG because it provides both:
- **factoid-style evidence**, such as ingredient names, cooking methods, or origins
- **longer descriptive evidence**, such as cuisine history, regional characteristics, and dish explanations

## 3.2 Record Structure

Most corpus entries share the same core fields:

- `source`: source collection (`Wikipedia`, `Wikibooks`, or `80Cuisine`)
- `page_type`: coarse page category, such as `sub_cuisine`, `overview`, `recipe`, `ingredient`, or `blog`
- `title`: document title
- `section`: section heading associated with the text
- `url`: original source URL
- `text`: raw cleaned text content for that section

Some entries also contain:

- `doc_id`: unique document-level identifier
- `cuisine`: cuisine label or regional cuisine tag

A typical entry looks like this:

```json
{
  "doc_id": "wikibooks_cookbook:broccoli_stir_fry_1",
  "source": "Wikibooks",
  "page_type": "recipe",
  "title": "Cookbook:Broccoli Stir Fry",
  "section": "Ingredients",
  "url": "https://en.wikibooks.org/wiki/Cookbook:Broccoli_Stir_Fry",
  "cuisine": "East Asian",
  "text": "Chicken or other meat\n\nBroccoli\n\nPowdered ginger (optional)\n\nSoy sauce\n\nOil to fry with, possibly including sesame oil\n\nSesame seeds"
}
```

This section-level structure makes the corpus easy to:
- convert into fixed-size or semantic chunks
- preserve source metadata during retrieval
- trace benchmark evidence back to an original title and section

## 3.3 Source Distribution

The corpus is distributed across three sources:

- **Wikipedia**: **605** records
- **Wikibooks**: **370** records
- **80Cuisine**: **28** records

In terms of page type, the corpus is dominated by section-based reference and recipe material:

```text
- sub_cuisine: 437
- recipe: 338
- overview: 168
- ingredient: 32
- blog: 28
```

This means the corpus is not purely recipe-oriented or purely encyclopedic. Instead, it combines:
- **broad cuisine background**
- **dish- and ingredient-level descriptions**
- **stepwise procedural cooking content**

## 3.4 Why This Structure Is Useful for RAG

The corpus was organised at the section level before chunking so that:
1. semantically meaningful units such as `Ingredients`, `Procedure`, or `History` can be preserved
2. chunking methods can be compared under the same original document structure
3. retrieval results remain interpretable through `source`, `title`, and `section` metadata

Because the corpus contains both structured recipe sections and noisier descriptive web text, it is also useful for testing how different chunking strategies behave on heterogeneous source material.
