import json
import os

# file config
txt_files = [
    "Blog_EastAsian_Cuisines.txt",
    "East_Asian_Corpus_Massive.txt",
    "Wikibooks_EastAsian_Recipes_Clean.txt"
]
qa_file = "east_asia_benchmark_test.json"

os.makedirs("data", exist_ok=True)

# process corpus
with open("data/corpus.jsonl", "w", encoding="utf-8") as out_f:
    for idx, file_name in enumerate(txt_files):
        if os.path.exists(file_name):
            with open(file_name, "r", encoding="utf-8") as in_f:
                content = in_f.read()
            
            doc_data = {
                "doc_id": f"doc_{idx+1}",
                "title": file_name,
                "text": content
            }
            out_f.write(json.dumps(doc_data, ensure_ascii=False) + "\n")

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
        
        evidence = item.get("doc_id", item.get("evidence", []))
        if isinstance(evidence, str):
            evidence = [evidence]
        if not evidence:
            evidence = ["doc_1", "doc_2", "doc_3"] 
            
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