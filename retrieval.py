import json
import os
import re
import chromadb
from dotenv import load_dotenv

from groq import Groq
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from config import (
    PROCESSED_DIR,
    VECTOR_DB_DIR,
)


load_dotenv(override=True)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "maintai_manuals"
MAX_DISTANCE = 0.5
BM25_TOP_K = 20

# Top-N BM25 hits are allowed to survive the distance filter, so a
# strong keyword match (e.g. "voltage supply") is not thrown away just
# because its embedding is far from the query.
BM25_RESCUE_RANK = 3
BM25_RESCUE_MAX_DISTANCE = 0.65
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

HYDE_MODEL_NAME = "openai/gpt-oss-20b"
_hyde_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# =========================================================
# Query expansion
# =========================================================

SYNONYM_EXPANSIONS = {
    "power supply": "power supply voltage supply internal supply voltage",
    "power problem": "voltage supply mains power",
    "blank screen": "blank display no display",
}


def expand_query(query, device_id=None):
    """
    Appends manual-style synonyms to the query. device_id is accepted
    (and currently unused) so callers can pass it without breaking.
    """
    q = query.lower()
    extra = [v for k, v in SYNONYM_EXPANSIONS.items() if k in q]
    return query + (" " + " ".join(extra) if extra else "")


# =========================================================
# Vector DB helpers
# =========================================================

def get_available_devices():
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)
    data = collection.get()
    devices = set()
    for metadata in data["metadatas"]:
        device_id = metadata.get("device_id", "")
        if device_id:
            devices.add(device_id)
    return sorted(list(devices))


def load_chunks():
    chunks_file = PROCESSED_DIR / "maintai_chunks.json"
    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_file}")

    with open(chunks_file, "r", encoding="utf-8") as file:
        chunks = json.load(file)
    return chunks


def create_vector_database():
    print("=" * 70)
    print("CREATING VECTOR DATABASE")
    print("=" * 70)
    chunks = load_chunks()
    print(f"Loaded chunks: {len(chunks)}")
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Old collection deleted")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    texts = []
    ids = []
    metadatas = []
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue
        chunk_id = chunk.get("chunk_id")
        if not chunk_id:
            continue

        metadata = {
            "device_id": str(chunk.get("device_id", "")),
            "device": str(chunk.get("device", "")),
            "manufacturer": str(chunk.get("manufacturer", "")),
            "page": int(chunk.get("page", 0)),
            "section": str(chunk.get("section", "")),
            "chunk_type": str(chunk.get("chunk_type", "text")),
            "error_code": (
                str(chunk["error_code"])
                if chunk.get("error_code") is not None
                else ""
            ),
        }
        texts.append(text)
        ids.append(chunk_id)
        metadatas.append(metadata)
    print(f"Documents prepared: {len(texts)}")

    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,
        batch_size=32,
    )

    CHROMA_BATCH_SIZE = 1000
    total = len(texts)
    for start in range(0, total, CHROMA_BATCH_SIZE):
        end = min(start + CHROMA_BATCH_SIZE, total)
        print(f"Adding batch {start} - {end} / {total}")
        collection.add(
            ids=ids[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end].tolist(),
        )
    print()
    print(f"Stored vectors: {collection.count()}")
    print("=" * 70)


# =========================================================
# Semantic search
# =========================================================

def semantic_search(
    query,
    device_id=None,
    top_k=8,
    expand=False,
):
    """
    expand=False by default: the device-detection probe and the HyDE
    leg must see the query exactly as written. Synonym expansion is
    only requested explicitly by hybrid_search for the raw-query leg.
    """
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)

    search_query = expand_query(query, device_id=device_id) if expand else query

    query_embedding = embedding_model.encode(
        search_query,
        normalize_embeddings=True,
    )
    search_arguments = {
        "query_embeddings": [query_embedding.tolist()],
        "n_results": top_k,
    }

    if device_id:
        search_arguments["where"] = {"device_id": device_id}

    results = collection.query(**search_arguments)
    return results


# =========================================================
# HyDE
# =========================================================

def generate_hypothetical_passage(query):
    try:
        response = _hyde_client.chat.completions.create(
            model=HYDE_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rewrite the user's symptom as a short technical "
                        "phrase in the style of a medical-equipment service "
                        "manual's troubleshooting table. One sentence. Do "
                        "not answer or solve anything, only rephrase."
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0,  # deterministic: same query -> same rewrite
            # gpt-oss models "think" first; a tiny limit could be spent
            # entirely on reasoning and leave an empty answer.
            max_completion_tokens=200,
            reasoning_effort="low",
        )
        rewritten = (response.choices[0].message.content or "").strip()
        return rewritten if rewritten else query
    except Exception as error:
        print(f"  HyDE rewrite failed, using original query: {error}")
        return query


# =========================================================
# BM25 keyword search
# =========================================================

_bm25_cache = {}


def _get_bm25_index(device_id):
    if device_id in _bm25_cache:
        return _bm25_cache[device_id]
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)
    data = collection.get(where={"device_id": device_id})

    ids = data["ids"]
    texts = data["documents"]
    metadatas = data["metadatas"]
    tokenized_corpus = [
        re.findall(r"[a-z0-9]+", text.lower())
        for text in texts
    ]

    bm25 = BM25Okapi(tokenized_corpus)
    entry = {
        "bm25": bm25,
        "ids": ids,
        "texts": texts,
        "metadatas": metadatas,
    }
    _bm25_cache[device_id] = entry
    return entry


def keyword_search(
    query,
    device_id,
    top_k=BM25_TOP_K,
):
    index = _get_bm25_index(device_id)
    # BM25 benefits the most from synonym expansion
    # (e.g. "power supply" -> "voltage supply").
    tokenized_query = re.findall(
        r"[a-z0-9]+",
        expand_query(query).lower(),
    )
    scores = index["bm25"].get_scores(tokenized_query)
    ranked_positions = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )

    ranked_positions = [i for i in ranked_positions if scores[i] > 0][:top_k]
    return {
        "ids": [index["ids"][i] for i in ranked_positions],
        "documents": [index["texts"][i] for i in ranked_positions],
        "metadatas": [index["metadatas"][i] for i in ranked_positions],
        "scores": [scores[i] for i in ranked_positions],
    }


# =========================================================
# Hybrid search
#   legs: raw query (expanded) + HyDE rewrite + BM25, fused with RRF
#   distance gate: min(distance to raw query, distance to HyDE query)
#   -> HyDE can only help a chunk pass, never make it fail.
# =========================================================

def hybrid_search(
    query,
    device_id,
    top_k=8,
    hyde_query=None,
):
    if hyde_query is None:
        hyde_query = generate_hypothetical_passage(query)

    use_hyde = bool(hyde_query) and hyde_query.strip() != query.strip()

    semantic_top_k = max(top_k * 3, 24)

    empty_result = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }

    candidates = {}

    def _get(doc_id, document, metadata):
        if doc_id not in candidates:
            candidates[doc_id] = {
                "id": doc_id,
                "document": document,
                "metadata": metadata,
                "query_rank": None,
                "hyde_rank": None,
                "bm25_rank": None,
                "dist_query": None,
                "dist_hyde": None,
            }
        return candidates[doc_id]

    # ---- Leg 1: raw query (with synonym expansion) ----
    query_results = semantic_search(
        query=query,
        device_id=device_id,
        top_k=semantic_top_k,
        expand=True,
    )
    for rank, (doc_id, doc, meta) in enumerate(
        zip(
            query_results["ids"][0],
            query_results["documents"][0],
            query_results["metadatas"][0],
        ),
        start=1,
    ):
        _get(doc_id, doc, meta)["query_rank"] = rank

    # ---- Leg 2: HyDE rewrite ----
    if use_hyde:
        hyde_results = semantic_search(
            query=hyde_query,
            device_id=device_id,
            top_k=semantic_top_k,
            expand=False,
        )
        for rank, (doc_id, doc, meta) in enumerate(
            zip(
                hyde_results["ids"][0],
                hyde_results["documents"][0],
                hyde_results["metadatas"][0],
            ),
            start=1,
        ):
            _get(doc_id, doc, meta)["hyde_rank"] = rank

    # ---- Leg 3: BM25 ----
    keyword_results = keyword_search(
        query=query,
        device_id=device_id,
        top_k=BM25_TOP_K,
    )
    for rank, (doc_id, doc, meta) in enumerate(
        zip(
            keyword_results["ids"],
            keyword_results["documents"],
            keyword_results["metadatas"],
        ),
        start=1,
    ):
        _get(doc_id, doc, meta)["bm25_rank"] = rank

    if not candidates:
        return empty_result

    # ---- Uniform distances for every candidate ----
    # Always measured against the ORIGINAL (un-expanded) query, and
    # against the HyDE rewrite when it exists.
    candidate_list = list(candidates.values())
    doc_embeddings = embedding_model.encode(
        [c["document"] for c in candidate_list],
        normalize_embeddings=True,
        batch_size=32,
    )
    query_embedding = embedding_model.encode(query, normalize_embeddings=True)
    hyde_embedding = (
        embedding_model.encode(hyde_query, normalize_embeddings=True)
        if use_hyde
        else None
    )

    for candidate, doc_embedding in zip(candidate_list, doc_embeddings):
        candidate["dist_query"] = float(1.0 - (doc_embedding @ query_embedding))
        if hyde_embedding is not None:
            candidate["dist_hyde"] = float(1.0 - (doc_embedding @ hyde_embedding))

        distances = [
            d for d in (candidate["dist_query"], candidate["dist_hyde"])
            if d is not None
        ]
        candidate["semantic_distance"] = min(distances)

    # ---- Distance filter (+ BM25 rescue) ----
    filtered_candidates = [
        c for c in candidate_list
        if c["semantic_distance"] <= MAX_DISTANCE
        or (
            c["bm25_rank"] is not None
            and c["bm25_rank"] <= BM25_RESCUE_RANK
            and c["semantic_distance"] <= BM25_RESCUE_MAX_DISTANCE
        )
    ]

    if not filtered_candidates:
        return empty_result

    # ---- Reciprocal Rank Fusion ----
    RRF_K = 60
    QUERY_WEIGHT = 1.5
    HYDE_WEIGHT = 1.0
    BM25_WEIGHT = 1.0

    for candidate in filtered_candidates:
        score = 0.0
        if candidate["query_rank"] is not None:
            score += QUERY_WEIGHT / (RRF_K + candidate["query_rank"])
        if candidate["hyde_rank"] is not None:
            score += HYDE_WEIGHT / (RRF_K + candidate["hyde_rank"])
        if candidate["bm25_rank"] is not None:
            score += BM25_WEIGHT / (RRF_K + candidate["bm25_rank"])
        candidate["rrf_score"] = score

    ranked_candidates = sorted(
        filtered_candidates,
        key=lambda item: (item["rrf_score"], -item["semantic_distance"]),
        reverse=True,
    )[:top_k]

    true_best_distance = min(
        c["semantic_distance"] for c in filtered_candidates
    )

    return {
        "ids": [[c["id"] for c in ranked_candidates]],
        "documents": [[c["document"] for c in ranked_candidates]],
        "metadatas": [[c["metadata"] for c in ranked_candidates]],
        "distances": [[c["semantic_distance"] for c in ranked_candidates]],
        "true_best_distance": true_best_distance,
    }


# =========================================================
# Exact error-code search
# =========================================================

MAX_EXACT_MATCHES = 5


def exact_error_search(
    error_code,
    device_id=None,
):
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)
    filters = [{"error_code": str(error_code)}]

    if device_id:
        filters.append({"device_id": device_id})
    if len(filters) == 1:
        where_clause = filters[0]
    else:
        where_clause = {"$and": filters}
    results = collection.get(where=where_clause)
    if len(results.get("ids", [])) > MAX_EXACT_MATCHES:
        for key in ("ids", "documents", "metadatas"):
            if key in results and results[key]:
                results[key] = results[key][:MAX_EXACT_MATCHES]
    return results


FALSE_POSITIVE_ERROR_CODES = {"382"}


def detect_error_code(query):
    patterns = [
        r"\berror\s+code\s+(\d{1,5})\b",
        r"\berror\s+(\d{1,5})\b",
        r"\bcode\s+(\d{1,5})\b",
        r"\bE[-\s]?(\d{1,5})\b",
        r"\bERR[-\s]?(\d{1,5})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            code = match.group(1)
            if code in FALSE_POSITIVE_ERROR_CODES:
                return None
            return code
    return None


# =========================================================
# Main retrieval entry point
# =========================================================

def retrieve(
    query,
    device_id=None,
    error_code=None,
    top_k=8,
):
    if error_code is None:
        error_code = detect_error_code(query)
    if error_code:
        exact_results = exact_error_search(error_code, device_id)
        if exact_results["ids"]:
            return {
                "retrieval_type": "exact_error",
                "detected_error_code": error_code,
                "detected_device": exact_results["metadatas"][0].get(
                    "device_id", ""
                ),
                "results": exact_results,
            }

    search_device_id = device_id

    # ---- Device detection probe: RAW query, no expansion, no HyDE ----
    # (expansion here used to drag e.g. Philips questions to the wrong
    # device because "voltage supply" matches other manuals better)
    if search_device_id is None:
        probe_results = semantic_search(
            query=query,
            device_id=None,
            top_k=1,
        )

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

        search_device_id = probe_results["metadatas"][0][0].get(
            "device_id", ""
        )

    # HyDE only after the device is known and the query is relevant
    # (saves one Groq call for irrelevant questions).
    hyde_query = generate_hypothetical_passage(query)
    print(f"HyDE rewrite: {hyde_query}")

    semantic_results = hybrid_search(
        query=query,
        device_id=search_device_id,
        top_k=top_k,
        hyde_query=hyde_query,
    )

    if not semantic_results["documents"][0]:
        return {
            "retrieval_type": "not_found",
            "detected_error_code": error_code,
            "detected_device": None,
            "results": semantic_results,
        }

    best_distance = semantic_results.get(
        "true_best_distance",
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


def test_retrieve():
    query = "The ventilator has a power supply problem"
    result = retrieve(
        query=query,
        top_k=8,
    )
    print("=" * 70)
    print("TYPE:", result["retrieval_type"])
    print("DEVICE:", result["detected_device"])
    for doc in result["results"]["documents"][0]:
        print("-" * 70)
        print(doc[:300])


if __name__ == "__main__":
    test_retrieve()