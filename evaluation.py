from config import BENCHMARK_PATH, INFERENCE_OUTPUT_PATH, EVAL_OUTPUT_PATH
from utils import load_json, save_json, exact_match, token_f1, reciprocal_rank


def main() -> None:
    benchmark = load_json(BENCHMARK_PATH)
    predictions = load_json(INFERENCE_OUTPUT_PATH)

    pred_map = {x["id"]: x for x in predictions}

    results = []
    retrieval_hits = []
    retrieval_recalls = []
    retrieval_precisions = []
    retrieval_mrr = []
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

        retrieved_doc_ids_list = [chunk["metadata"]["doc_id"] for chunk in pred["retrieved_chunks"]]
        retrieved_doc_ids = set(retrieved_doc_ids_list)

        hit_at_5 = int(len(gold_doc_ids.intersection(retrieved_doc_ids)) > 0)
        recall_at_5 = len(gold_doc_ids.intersection(retrieved_doc_ids)) / len(gold_doc_ids) if gold_doc_ids else 0.0
        precision_at_5 = len(gold_doc_ids.intersection(retrieved_doc_ids)) / len(retrieved_doc_ids) if retrieved_doc_ids else 0.0
        mrr_at_5 = reciprocal_rank(retrieved_doc_ids_list, gold_doc_ids)
        em = exact_match(pred_answer, gold_answer)
        f1 = token_f1(pred_answer, gold_answer)

        retrieval_hits.append(hit_at_5)
        retrieval_recalls.append(recall_at_5)
        retrieval_precisions.append(precision_at_5)
        retrieval_mrr.append(mrr_at_5)
        em_scores.append(em)
        f1_scores.append(f1)

        results.append(
            {
                "id": qid,
                "query": ex["query"],
                "gold_answer": gold_answer,
                "pred_answer": pred_answer,
                "hit@5": hit_at_5,
                "recall@5": round(recall_at_5, 4),
                "precision@5": round(precision_at_5, 4),
                "mrr@5": round(mrr_at_5, 4),
                "exact_match": em,
                "token_f1": round(f1, 4),
            }
        )

    summary = {
        "num_examples": len(results),
        "retrieval_hit@5": round(sum(retrieval_hits) / len(retrieval_hits), 4) if retrieval_hits else 0.0,
        "retrieval_recall@5": round(sum(retrieval_recalls) / len(retrieval_recalls), 4) if retrieval_recalls else 0.0,
        "retrieval_precision@5": round(sum(retrieval_precisions) / len(retrieval_precisions), 4) if retrieval_precisions else 0.0,
        "retrieval_mrr@5": round(sum(retrieval_mrr) / len(retrieval_mrr), 4) if retrieval_mrr else 0.0,
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