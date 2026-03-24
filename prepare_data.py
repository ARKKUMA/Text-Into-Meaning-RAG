import json
import os
import re
from utils import get_text_splitter

# file config
txt_files = [
    "Blog_EastAsian_Cuisines.md",
    "East_Asian_Corpus_Massive.md",
    "Wikibooks_EastAsian_Recipes_Clean.md"
]
qa_file = "east_asia_benchmark_test.json"

os.makedirs("data", exist_ok=True)

splitter = get_text_splitter()

def get_keywords(text):
    # simple keyword extraction
    words = re.findall(r'\w+', text.lower())
    return set(w for w in words if len(w) >= 3)

# process corpus
docs_cache = {}
with open("data/corpus.jsonl", "w", encoding="utf-8") as out_f:
    for idx, file_name in enumerate(txt_files):
        if os.path.exists(file_name):
            with open(file_name, "r", encoding="utf-8") as in_f:
                content = in_f.read()
            
            doc_id_base = f"doc_{idx+1}"
            chunks = splitter.split_text(content)

            if "Blog" in file_name:
                doc_type = "Blog"
                actual_source = "East Asian Culinary Blogs"
            elif "Wiki" in file_name:
                doc_type = "Wiki"
                actual_source = "Wikibooks Recipes"
            else:
                doc_type = "Corpus"
                actual_source = "Massive Text Corpus"
            
            for i, chunk_text in enumerate(chunks):
                chunk_id = f"{doc_id_base}_chunk_{i}"
                doc_data = {
                    "doc_id": chunk_id,
                    "title": file_name,
                    "text": chunk_text,
                    "source": actual_source,
                    "document_type": doc_type
                }
                out_f.write(json.dumps(doc_data, ensure_ascii=False) + "\n")
                # cache text and keywords for matching
                docs_cache[chunk_id] = {
                    "text": chunk_text,
                    "keywords": get_keywords(chunk_text)
                }

# process test data
if os.path.exists(qa_file):
    with open(qa_file, "r", encoding="utf-8") as f:
        raw_qa = json.load(f)

    if isinstance(raw_qa, list):
        # Handle format: [{doc_id, queries: [{query, answer}, ...]}, ...]
        qa_items = []
        for item in raw_qa:
            if isinstance(item, dict):
                if "queries" in item and isinstance(item.get("queries"), list):
                    qa_items.extend(item.get("queries", []))
                elif "query" in item:
                    # Direct Q&A format
                    qa_items.append(item)
    elif isinstance(raw_qa, dict):
        if "sources" in raw_qa and isinstance(raw_qa.get("sources"), list):
            qa_items = []
            for source_block in raw_qa.get("sources", []):
                if isinstance(source_block, dict):
                    qa_items.extend(source_block.get("questions", []))
        else:
            qa_items = raw_qa.get("questions", [])
    else:
        qa_items = []
        
    benchmark_data = []
    test_queries_data = []
    
    global_q_count = 0
    for item in qa_items:
        if not isinstance(item, dict):
            continue

        global_q_count += 1

        q_id = item.get("id", str(global_q_count))
        query = item.get("query", item.get("question", ""))
        answer = item.get("answer", item.get("gold_answer", ""))

        answer_keywords = get_keywords(answer)
        best_chunk_id = None
        max_overlap = 0

        for c_id, c_info in docs_cache.items():
            overlap = len(answer_keywords.intersection(c_info["keywords"]))
            if overlap > max_overlap:
                max_overlap = overlap
                best_chunk_id = c_id

        evidence = [best_chunk_id] if best_chunk_id and max_overlap > 0 else ["doc_1_chunk_0"]

        benchmark_data.append({
            "id": str(q_id),
            "query": query,
            "gold_answer": answer,
            "gold_evidence_doc_ids": evidence
        })

        test_queries_data.append({
            "id": str(q_id),
            "query": query
        })
        
    with open("data/benchmark.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, ensure_ascii=False, indent=2)
        
    with open("data/test_queries.json", "w", encoding="utf-8") as f:
        json.dump(test_queries_data, f, ensure_ascii=False, indent=2)

print("Data ready: corpus.jsonl is physically chunked and benchmark.json uses keyword matching!")