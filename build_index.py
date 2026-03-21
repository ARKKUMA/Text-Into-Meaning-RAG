from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import (
    CORPUS_PATH,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
)
from utils import load_jsonl


def main() -> None:
    corpus = load_jsonl(CORPUS_PATH)

    documents = []
    for row in corpus:
        chunk_id = row["doc_id"]
        title = row.get("title", "")
        source = row.get("source", "")
        document_type = row.get("document_type", "")
        text = row["text"]

        enriched_chunk = f"Title: {title}\nSource: {source}\nType: {document_type}\n\n{text}"
        documents.append(
            Document(
                page_content=enriched_chunk,
                metadata={
                    "doc_id": chunk_id, 
                    "title": title,
                    "source": source,
                    "document_type": document_type,
                    "chunk_id": chunk_id,
                },
            )
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)

    print(f"Built FAISS index with {len(documents)} chunks.")
    print(f"Saved to: {FAISS_INDEX_PATH}")


if __name__ == "__main__":
    main()