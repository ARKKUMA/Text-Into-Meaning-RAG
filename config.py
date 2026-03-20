from pathlib import Path
import torch

# ===== Paths =====
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "index_store"
OUTPUT_DIR = BASE_DIR / "outputs"

CORPUS_PATH = DATA_DIR / "corpus.jsonl"
BENCHMARK_PATH = DATA_DIR / "benchmark.json"
TEST_QUERIES_PATH = DATA_DIR / "test_queries.json"

INDEX_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== Models =====
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# ===== Baseline settings =====
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5
RETRIEVAL_CANDIDATE_K = 12
RETRIEVAL_ALPHA = 0.75
MAX_NEW_TOKENS = 128

# ===== Runtime =====
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available(): 
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"Initialize System: Running on {DEVICE.upper()} mode")

TEMPERATURE = 0.0

# ===== Files =====
FAISS_INDEX_PATH = str(INDEX_DIR / "faiss_index")
INFERENCE_OUTPUT_PATH = OUTPUT_DIR / "inference_output.json"
EVAL_OUTPUT_PATH = OUTPUT_DIR / "evaluation_results.json"