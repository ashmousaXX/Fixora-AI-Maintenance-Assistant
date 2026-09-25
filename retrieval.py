import json
import re
import chromadb

from sentence_transformers import SentenceTransformer
from config import (
    PROCESSED_DIR,
    VECTOR_DB_DIR,
)


# =========================================================
# Configuration
# =========================================================

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

COLLECTION_NAME = "maintai_manuals"

MAX_DISTANCE = 0.55


embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)


# =========================================================
# Get available devices
# =========================================================

def get_available_devices():

    client = chromadb.PersistentClient(
        path=str(VECTOR_DB_DIR)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    data = collection.get()

    devices = set()

    for metadata in data["metadatas"]:

        device_id = metadata.get(
            "device_id",
            ""
        )

        if device_id:
            devices.add(device_id)

    return sorted(
        list(devices)
    )


# =========================================================
# Load chunks
# =========================================================

def load_chunks():

    chunks_file = (
        PROCESSED_DIR
        /
        "maintai_chunks.json"
    )


    if not chunks_file.exists():

        raise FileNotFoundError(
            f"Chunks file not found: {chunks_file}"
        )


    with open(
        chunks_file,
        "r",
        encoding="utf-8",
    ) as file:

        chunks = json.load(file)


    return chunks



# =========================================================
# Create Vector Database
# =========================================================

def create_vector_database():

    print("=" * 70)
    print("CREATING VECTOR DATABASE")
    print("=" * 70)


    chunks = load_chunks()


    print(
        f"Loaded chunks: {len(chunks)}"
    )


    VECTOR_DB_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    client = chromadb.PersistentClient(
        path=str(VECTOR_DB_DIR)
    )


    try:

        client.delete_collection(
            COLLECTION_NAME
        )

        print(
            "Old collection deleted"
        )

    except Exception:

        pass



    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine"
        },
    )


    texts = []
    ids = []
    metadatas = []


    for chunk in chunks:


        text = chunk.get(
            "text",
            ""
        ).strip()


        if not text:
            continue


        chunk_id = chunk.get(
            "chunk_id"
        )


        if not chunk_id:
            continue



        metadata = {

            "device_id":
                str(
                    chunk.get(
                        "device_id",
                        ""
                    )
                ),


            "device":
                str(
                    chunk.get(
                        "device",
                        ""
                    )
                ),


            "manufacturer":
                str(
                    chunk.get(
                        "manufacturer",
                        ""
                    )
                ),


            "page":
                int(
                    chunk.get(
                        "page",
                        0
                    )
                ),


            "section":
                str(
                    chunk.get(
                        "section",
                        ""
                    )
                ),


            "chunk_type":
                str(
                    chunk.get(
                        "chunk_type",
                        "text"
                    )
                ),


            "error_code":
                (
                    str(
                        chunk["error_code"]
                    )
                    if chunk.get("error_code") is not None
                    else ""
                ),

        }


        texts.append(text)
        ids.append(chunk_id)
        metadatas.append(metadata)



    print(
        f"Documents prepared: {len(texts)}"
    )



    # =========================================
    # Create embeddings
    # =========================================


    embeddings = embedding_model.encode(

        texts,

        show_progress_bar=True,

        normalize_embeddings=True,

        batch_size=32,

    )



    # =========================================
    # Insert into ChromaDB in batches
    # =========================================


    CHROMA_BATCH_SIZE = 1000


    total = len(texts)


    for start in range(
        0,
        total,
        CHROMA_BATCH_SIZE
    ):


        end = min(
            start + CHROMA_BATCH_SIZE,
            total
        )


        print(
            f"Adding batch {start} - {end} / {total}"
        )



        collection.add(

            ids=
                ids[start:end],


            documents=
                texts[start:end],


            metadatas=
                metadatas[start:end],


            embeddings=
                embeddings[start:end].tolist(),

        )


    print()

    print(
        f"Stored vectors: {collection.count()}"
    )


    print("=" * 70)



# =========================================================
# Semantic Search
# =========================================================

def semantic_search(
    query,
    device_id=None,
    top_k=5,
):


    client = chromadb.PersistentClient(
        path=str(VECTOR_DB_DIR)
    )


    collection = client.get_collection(
        name=COLLECTION_NAME
    )


    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True,
    )


    search_arguments = {

        "query_embeddings":
            [
                query_embedding.tolist()
            ],

        "n_results":
            top_k,
    }



    if device_id:

        search_arguments["where"] = {

            "device_id":
                device_id
        }



    results = collection.query(
        **search_arguments
    )


    return results


# =========================================================
# Keyword overlap (lexical) scoring — no hardcoded synonyms,
# just literal terms from the query itself, so it stays
# general-purpose instead of tuned to one known test case.
# =========================================================
STOPWORDS = {
    "the", "a", "an", "is", "has", "have", "of", "to", "in",
    "on", "for", "with", "and", "or", "problem", "issue",
}
DOMAIN_SYNONYMS = {
    "power": {"power", "voltage", "electrical", "electric", "mains", "volt"},
    "screen": {"screen", "display", "monitor"},
    "leak": {"leak", "leakage"},
    "battery": {"battery", "voltage", "charge"},
}


def extract_keywords(query):
    words = re.findall(r"[a-zA-Z]+", query.lower())
    keywords = set()
    for w in words:
        if w in STOPWORDS or len(w) <= 2:
            continue
        keywords.add(w)
        keywords.update(DOMAIN_SYNONYMS.get(w, set()))
    return keywords


def keyword_overlap_score(text, keywords):
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


def lexical_search(device_id, keywords, top_k=10):
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)

    data = collection.get(where={"device_id": device_id})

    scored = []
    for doc, meta, doc_id in zip(data["documents"], data["metadatas"], data["ids"]):
        overlap = keyword_overlap_score(doc, keywords)
        if overlap > 0:
            scored.append((overlap, doc, meta, doc_id))

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:top_k]


def hybrid_search(
    query,
    device_id=None,
    top_k=5,
    semantic_pool=15,
    lexical_pool=15,
    rrf_k=60,
):
    semantic_raw = semantic_search(query=query, device_id=device_id, top_k=semantic_pool)

    sem_docs = semantic_raw["documents"][0]
    sem_dists = semantic_raw["distances"][0]
    sem_metas = semantic_raw["metadatas"][0]
    sem_ids = semantic_raw["ids"][0] if semantic_raw.get("ids") else [None] * len(sem_docs)

    raw_best_distance = min(sem_dists) if sem_dists else None

    keywords = extract_keywords(query)
    lexical_hits = lexical_search(device_id, keywords, top_k=lexical_pool) if device_id else []

    rrf_scores = {}
    doc_lookup = {}
    distance_lookup = {}

    for rank, (doc, dist, meta, doc_id) in enumerate(zip(sem_docs, sem_dists, sem_metas, sem_ids)):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (rrf_k + rank + 1)
        doc_lookup[doc_id] = (doc, meta)
        distance_lookup[doc_id] = dist

    for rank, (overlap, doc, meta, doc_id) in enumerate(lexical_hits):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (rrf_k + rank + 1)
        doc_lookup.setdefault(doc_id, (doc, meta))

    ranked_ids = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)[:top_k]

    missing_ids = [i for i in ranked_ids if i not in distance_lookup]
    if missing_ids:
        query_embedding = embedding_model.encode(query, normalize_embeddings=True)
        missing_texts = [doc_lookup[i][0] for i in missing_ids]
        missing_embeddings = embedding_model.encode(missing_texts, normalize_embeddings=True)
        for doc_id, emb in zip(missing_ids, missing_embeddings):
            distance_lookup[doc_id] = 1 - float(query_embedding @ emb)

    return {
        "documents": [[doc_lookup[i][0] for i in ranked_ids]],
        "distances": [[distance_lookup[i] for i in ranked_ids]],
        "metadatas": [[doc_lookup[i][1] for i in ranked_ids]],
        "ids": [ranked_ids],
        "raw_best_distance": raw_best_distance,
    }
# =========================================================
# Exact Error Search
# =========================================================

def exact_error_search(
    error_code,
    device_id=None,
):


    client = chromadb.PersistentClient(
        path=str(VECTOR_DB_DIR)
    )


    collection = client.get_collection(
        name=COLLECTION_NAME
    )


    filters = [

        {
            "error_code":
                str(error_code)
        }

    ]


    if device_id:

        filters.append(
            {
                "device_id":
                    device_id
            }
        )


    # Chroma rejects "$and" with fewer than 2 conditions,
    # so only wrap in "$and" when we actually have 2+ filters.
    if len(filters) == 1:
        where_clause = filters[0]
    else:
        where_clause = {"$and": filters}


    results = collection.get(

        where=where_clause

    )


    return results



# =========================================================
# Detect Error Code
# =========================================================

def detect_error_code(query):


    patterns = [

        r"\berror\s+code\s+(\d{1,5})\b",

        r"\berror\s+(\d{1,5})\b",

        r"\bcode\s+(\d{1,5})\b",

        r"\bE[-\s]?(\d{1,5})\b",

        r"\bERR[-\s]?(\d{1,5})\b",

    ]


    for pattern in patterns:

        match = re.search(
            pattern,
            query,
            re.IGNORECASE,
        )


        if match:

            return match.group(1)


    return None



# =========================================================
# Main Retrieval
# =========================================================

def retrieve(
    query,
    device_id=None,
    error_code=None,
    top_k=5,
):

    if error_code is None:
        error_code = detect_error_code(query)

    if error_code:
        exact_results = exact_error_search(error_code, device_id)

        if exact_results["ids"]:
            return {
                "retrieval_type": "exact_error",
                "detected_error_code": error_code,
                "detected_device": exact_results["metadatas"][0].get("device_id", ""),
                "results": exact_results,
            }

    search_device_id = device_id

    if search_device_id is None:
        probe_results = semantic_search(query=query, device_id=None, top_k=1)

        if not probe_results["documents"][0]:
            return {
                "retrieval_type": "not_found",
                "detected_error_code": error_code,
                "detected_device": None,
                "results": probe_results,
            }

        probe_distance = probe_results["distances"][0][0]

        if probe_distance > MAX_DISTANCE:
            return {
                "retrieval_type": "not_found",
                "detected_error_code": error_code,
                "detected_device": None,
                "results": probe_results,
            }

        search_device_id = probe_results["metadatas"][0][0].get("device_id", "")

    semantic_results = hybrid_search(
        query=query,
        device_id=search_device_id,
        top_k=top_k,
    )

    if not semantic_results["documents"][0]:
        return {
            "retrieval_type": "not_found",
            "detected_error_code": error_code,
            "detected_device": None,
            "results": semantic_results,
        }

    best_distance = semantic_results.get(
        "raw_best_distance",
        semantic_results["distances"][0][0],
    )

    print(f"Best semantic distance: {best_distance:.4f}")
    print(f"Maximum allowed distance: {MAX_DISTANCE:.4f}")

    if best_distance > MAX_DISTANCE:
        return {
            "retrieval_type": "not_found",
            "detected_error_code": error_code,
            "detected_device": None,
            "results": semantic_results,
        }

    detected_device = semantic_results["metadatas"][0][0].get("device_id", "")

    return {
        "retrieval_type": "semantic",
        "detected_error_code": error_code,
        "detected_device": detected_device,
        "results": semantic_results,
    }

# =========================================================
# Test
# =========================================================

def test_retrieve():


    query = (
        "The ventilator has a power supply problem"
    )


    result = retrieve(
        query=query,
        top_k=5,
    )


    print("="*70)

    print(
        "TYPE:",
        result["retrieval_type"]
    )


    print(
        "DEVICE:",
        result["detected_device"]
    )


    print(
        result["results"]["documents"][0]
    )



if __name__ == "__main__":

    test_retrieve()