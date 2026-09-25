#from preprocessing import extract_pdf_text
#from pathlib import Path

#pages = extract_pdf_text(Path("Data/Maintencie/Copy of Service Manual.pdf"))
#print("--- PAGE 1 ---")
#print(pages[0]["text"])
#print("\n--- PAGE 2 ---")
#print(pages[1]["text"][:1500])

from retrieval import semantic_search

results = semantic_search(query="the screen is blank", device_id="sc6002xl", top_k=5)
for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
    print(f"[{dist:.4f}] Section: {meta.get('section')}")
    print(doc[:250])
    print("-" * 60)