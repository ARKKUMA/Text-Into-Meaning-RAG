"""
End-to-end retrieval pipeline: BM25 + Dense → RRF → Cross-encoder rerank.

Input / output conform to the payload schema:
  Input:  {"queries": [{"query_id": "0", "query": "..."}]}
  Output: {"results": [{"query_id": "0", "query": "...",
                         "response": "",
                         "retrieved_context": [{"doc_id": <chunk_id>,
                                                "vector_id": <vector_id>,
                                                "text": "..."}]}]}
"""
from pathlib import Path
from typing import Any, Dict, List, Optional

from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.fusion import rrf_fuse
from retrieval.reranker import CrossEncoderReranker


class RetrievalPipeline:
    """
    Parameters
    ----------
    meta_jsonl_path:      Path to FAISS meta JSONL (shared corpus for BM25 & Dense).
    faiss_manifest_path:  Path to FAISS manifest JSON (locates index + model).
    cross_encoder_model:  HuggingFace ID or local path for the cross-encoder.
    bm25_top_k:           Candidates retrieved per query by BM25.
    dense_top_k:          Candidates retrieved per query by Dense.
    rrf_k:                RRF smoothing constant (default 60).
    rrf_weights:          [bm25_weight, dense_weight] (default [1.0, 1.0]).
    rerank_input_k:       How many fused candidates to feed the cross-encoder.
    final_top_k:          Final number of results returned per query.
    """

    def __init__(
        self,
        meta_jsonl_path: Path,
        faiss_manifest_path: Path,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        bm25_top_k: int = 100,
        dense_top_k: int = 100,
        rrf_k: int = 60,
        rrf_weights: Optional[List[float]] = None,
        rerank_input_k: int = 30,
        final_top_k: int = 5,
    ) -> None:
        self.bm25_top_k = bm25_top_k
        self.dense_top_k = dense_top_k
        self.rrf_k = rrf_k
        self.rrf_weights = rrf_weights or [1.0, 1.0]
        self.rerank_input_k = rerank_input_k
        self.final_top_k = final_top_k

        self.bm25 = BM25Retriever(meta_jsonl_path)
        self.dense = DenseRetriever(faiss_manifest_path)
        self.reranker = CrossEncoderReranker(cross_encoder_model)

    # ── single-query retrieval ────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run full pipeline for one query.

        Args:
            query: Raw query string.
            top_k: Number of results to return.  Defaults to ``self.final_top_k``.
                   Pass a larger value (e.g. ``max(eval_ks)``) during evaluation
                   so that Recall@k and NDCG@k are computed over enough candidates.

        Returns a list of dicts with keys:
          doc_id, vector_id, chunk_id, text, title, sections, source, score
        """
        n_results = top_k if top_k is not None else self.final_top_k

        # Stage 1: sparse + dense retrieval
        bm25_hits = self.bm25.search(query, top_k=self.bm25_top_k)
        dense_hits = self.dense.search(query, top_k=self.dense_top_k)

        # Stage 2: RRF fusion
        fused = rrf_fuse(
            [bm25_hits, dense_hits],
            k=self.rrf_k,
            weights=self.rrf_weights,
        )

        # Stage 3: cross-encoder rerank — feed at least rerank_input_k candidates
        rerank_pool = max(self.rerank_input_k, n_results)
        reranked = self.reranker.rerank(
            query,
            fused[:rerank_pool],
            top_k=n_results,
        )

        results = []
        for rank, (vec_id, score, meta) in enumerate(reranked):
            results.append(
                {
                    "doc_id": str(rank).zfill(3),
                    "vector_id": vec_id,
                    "chunk_id": meta.get("chunk_id"),
                    "text": meta.get("text", ""),
                    "title": meta.get("title", ""),
                    "sections": meta.get("sections", []),
                    "source": meta.get("source", ""),
                    "score": round(score, 6),
                }
            )
        return results

    # ── batch payload ─────────────────────────────────────────────────────────

    def run_batch(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a full input payload and return an output payload.

        Input:  {"queries": [{"query_id": "0", "query": "..."}]}
        Output: {"results": [{"query_id": "0", "query": "...",
                               "response": "",
                               "retrieved_context": [{"doc_id": <chunk_id>,
                                                       "vector_id": <vector_id>,
                                                       "text": "..."}]}]}
        """
        results = []
        for item in payload["queries"]:
            hits = self.retrieve(item["query"])
            context = [
                {
                    "doc_id": h["chunk_id"],
                    "vector_id": h["vector_id"],
                    "text": h["text"],
                }
                for h in hits
            ]
            results.append(
                {
                    "query_id": item["query_id"],
                    "query": item["query"],
                    "response": "",
                    "retrieved_context": context,
                }
            )
        return {"results": results}
