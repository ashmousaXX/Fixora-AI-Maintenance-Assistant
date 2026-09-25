from retrieval import hybrid_search
results = hybrid_search("The ventilator has a power supply problem", device_id="servo_ventilator", top_k=10)
for doc, dist in zip(results["documents"][0], results["distances"][0]):
    print(f"{dist:.4f} | {doc[:90]}")