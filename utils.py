import json
import re
from pathlib import Path
from typing import List, Dict, Any


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def exact_match(pred: str, gold: str) -> int:
    return int(normalize_text(pred) == normalize_text(gold))


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counts = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1

    gold_counts = {}
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1

    common = 0
    for token, count in pred_counts.items():
        if token in gold_counts:
            common += min(count, gold_counts[token])

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def lexical_overlap_score(query: str, text: str) -> float:
    query_tokens = set(normalize_text(query).split())
    if not query_tokens:
        return 0.0
    text_tokens = set(normalize_text(text).split())
    if not text_tokens:
        return 0.0
    return len(query_tokens.intersection(text_tokens)) / len(query_tokens)


def reciprocal_rank(retrieved_doc_ids: List[str], gold_doc_ids: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in gold_doc_ids:
            return 1.0 / rank
    return 0.0


def build_prompt(query: str, retrieved_chunks: List[str]) -> str:
    context = "\n\n".join(
        [f"[Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(retrieved_chunks)]
    )
    return f"""You are a culinary question-answering assistant specialised in cuisine.
Use only the provided context to answer.
If the answer is not in the context, output exactly: "I cannot determine this from the provided context."
Keep the answer concise (1-2 sentences), factual, and avoid adding assumptions.

Context:
{context}

Question:
{query}

Final Answer:"""