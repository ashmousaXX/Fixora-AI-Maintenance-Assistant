from retrieval import semantic_search
results = semantic_search("screen on the patient monitor is black", device_id=None, top_k=10)
for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
    print(f"{dist:.4f}  {meta.get('device_id')}")