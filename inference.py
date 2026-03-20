import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import (
    TEST_QUERIES_PATH,
    EMBEDDING_MODEL_NAME,
    GENERATION_MODEL_NAME,
    FAISS_INDEX_PATH,
    TOP_K,
    RETRIEVAL_CANDIDATE_K,
    RETRIEVAL_ALPHA,
    MAX_NEW_TOKENS,
    DEVICE,
    TEMPERATURE,
    INFERENCE_OUTPUT_PATH,
)
from utils import load_json, save_json, build_prompt, lexical_overlap_score


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def load_generator():
    tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        GENERATION_MODEL_NAME,
        torch_dtype=torch.float32 if DEVICE == "cpu" else torch.float16,
    )
    if DEVICE == "cuda":
        model = model.to("cuda")
    model.eval()
    return tokenizer, model


def generate_answer(tokenizer, model, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    if DEVICE == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False if TEMPERATURE == 0.0 else True,
            temperature=TEMPERATURE,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if "Final Answer:" in decoded:
        return decoded.split("Final Answer:", 1)[-1].strip()
    if "Answer:" in decoded:
        return decoded.split("Answer:", 1)[-1].strip()
    return decoded.strip().split("\n")[0]


def hybrid_retrieve(vectorstore, query: str):
    # 1) semantic recall
    docs_with_scores = vectorstore.similarity_search_with_score(
        query, k=RETRIEVAL_CANDIDATE_K
    )
    if not docs_with_scores:
        return []

    # 2) score normalization and lexical re-ranking
    max_dist = max(score for _, score in docs_with_scores)
    min_dist = min(score for _, score in docs_with_scores)

    reranked = []
    for doc, dist in docs_with_scores:
        if max_dist == min_dist:
            semantic = 1.0
        else:
            semantic = 1.0 - ((dist - min_dist) / (max_dist - min_dist))
        lexical = lexical_overlap_score(query, doc.page_content)
        hybrid = RETRIEVAL_ALPHA * semantic + (1.0 - RETRIEVAL_ALPHA) * lexical
        reranked.append((doc, hybrid, semantic, lexical))

    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked[:TOP_K]


def main() -> None:
    queries = load_json(TEST_QUERIES_PATH)
    vectorstore = load_vectorstore()
    tokenizer, model = load_generator()

    outputs = []

    for item in queries:
        qid = item["id"]
        query = item["query"]

        reranked = hybrid_retrieve(vectorstore, query)
        docs = [x[0] for x in reranked]
        retrieved_chunks = [d.page_content for d in docs]

        prompt = build_prompt(query, retrieved_chunks)
        answer = generate_answer(tokenizer, model, prompt)

        outputs.append(
            {
                "id": qid,
                "query": query,
                "answer": answer,
                "retrieved_chunks": [
                    {
                        "text": d.page_content,
                        "metadata": d.metadata,
                        "hybrid_score": round(reranked[i][1], 4),
                        "semantic_score": round(reranked[i][2], 4),
                        "lexical_score": round(reranked[i][3], 4),
                    }
                    for i, d in enumerate(docs)
                ],
            }
        )

        print(f"[{qid}] {query}")
        print(f"Answer: {answer}")
        print("-" * 60)

    save_json(outputs, INFERENCE_OUTPUT_PATH)
    print(f"Saved inference outputs to {INFERENCE_OUTPUT_PATH}")


if __name__ == "__main__":
    main()