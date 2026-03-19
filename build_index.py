from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import (
    CORPUS_PATH,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    FAISS_INDEX_PATH,
)
from utils import load_jsonl


def main() -> None:
    corpus = load_jsonl(CORPUS_PATH)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    documents = []
    for row in corpus:
        doc_id = row["doc_id"]
        title = row.get("title", "")
        source = row.get("source", "")
        document_type = row.get("document_type", "")
        text = row["text"]

        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "doc_id": doc_id,
                        "title": title,
                        "source": source,
                        "document_type": document_type,
                        "chunk_id": f"{doc_id}_chunk_{i}",
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