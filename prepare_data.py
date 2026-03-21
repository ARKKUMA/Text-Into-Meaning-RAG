import json
import os
import re
from utils import get_text_splitter

# file config
txt_files = [
    "Blog_EastAsian_Cuisines.txt",
    "East_Asian_Corpus_Massive.txt",
    "Wikibooks_EastAsian_Recipes_Clean.txt"
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
            
            for i, chunk_text in enumerate(chunks):
                chunk_id = f"{doc_id_base}_chunk_{i}"
                doc_data = {
                    "doc_id": chunk_id,
                    "title": file_name,
                    "text": chunk_text
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
        
    benchmark_data = []
    test_queries_data = []
    
    for i, item in enumerate(raw_qa):
        q_id = item.get("id", str(i+1))
        query = item.get("query", item.get("question", ""))
        answer = item.get("answer", item.get("gold_answer", ""))
        
        # smart matching via keyword overlap
        answer_keywords = get_keywords(answer)
        best_chunk_id = None
        max_overlap = 0
        
        for c_id, c_info in docs_cache.items():
            overlap = len(answer_keywords.intersection(c_info["keywords"]))
            if overlap > max_overlap:
                max_overlap = overlap
                best_chunk_id = c_id
        
        # fallback to first chunk if no overlap found
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