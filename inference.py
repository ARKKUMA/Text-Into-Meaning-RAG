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
    MAX_NEW_TOKENS,
    DEVICE,
    TEMPERATURE,
    INFERENCE_OUTPUT_PATH,
)
from utils import load_json, save_json, build_prompt


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

    if "Answer:" in decoded:
        return decoded.split("Answer:", 1)[-1].strip()
    return decoded.strip()


def main() -> None:
    queries = load_json(TEST_QUERIES_PATH)
    vectorstore = load_vectorstore()
    tokenizer, model = load_generator()

    outputs = []

    for item in queries:
        qid = item["id"]
        query = item["query"]

        docs = vectorstore.similarity_search(query, k=TOP_K)

        retrieved_chunks = [d.page_content for d in docs]
        retrieved_metadata = [d.metadata for d in docs]

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
                    }
                    for d in docs
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