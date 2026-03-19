from config import BENCHMARK_PATH, INFERENCE_OUTPUT_PATH, EVAL_OUTPUT_PATH
from utils import load_json, save_json, exact_match, token_f1


def main() -> None:
    benchmark = load_json(BENCHMARK_PATH)
    predictions = load_json(INFERENCE_OUTPUT_PATH)

    pred_map = {x["id"]: x for x in predictions}

    results = []
    retrieval_hits = []
    em_scores = []
    f1_scores = []

    for ex in benchmark:
        qid = ex["id"]
        gold_answer = ex["gold_answer"]
        gold_doc_ids = set(ex["gold_evidence_doc_ids"])

        if qid not in pred_map:
            continue

        pred = pred_map[qid]
        pred_answer = pred["answer"]

        retrieved_doc_ids = {
            chunk["metadata"]["doc_id"] for chunk in pred["retrieved_chunks"]
        }

        hit_at_5 = int(len(gold_doc_ids.intersection(retrieved_doc_ids)) > 0)
        em = exact_match(pred_answer, gold_answer)
        f1 = token_f1(pred_answer, gold_answer)

        retrieval_hits.append(hit_at_5)
        em_scores.append(em)
        f1_scores.append(f1)

        results.append(
            {
                "id": qid,
                "query": ex["query"],
                "gold_answer": gold_answer,
                "pred_answer": pred_answer,
                "hit@5": hit_at_5,
                "exact_match": em,
                "token_f1": round(f1, 4),
            }
        )

    summary = {
        "num_examples": len(results),
        "retrieval_hit@5": round(sum(retrieval_hits) / len(retrieval_hits), 4) if retrieval_hits else 0.0,
        "generation_exact_match": round(sum(em_scores) / len(em_scores), 4) if em_scores else 0.0,
        "generation_token_f1": round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0,
        "details": results,
    }

    save_json(summary, EVAL_OUTPUT_PATH)
    print("Evaluation summary:")
    print(summary)
    print(f"Saved evaluation results to {EVAL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()