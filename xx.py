from retrieval import semantic_search

probe = semantic_search(query="Why does my monitor show nothing when I plug it in?", device_id=None, top_k=5)
for doc, meta, dist in zip(probe["documents"][0], probe["metadatas"][0], probe["distances"][0]):
    print(f"[{dist:.4f}] {meta.get('device')}: {doc[:150]}")