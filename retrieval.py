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

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)
COLLECTION_NAME = "maintai_manuals"
MAX_DISTANCE = 0.5
BM25_TOP_K = 20

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)

HYDE_MODEL_NAME = "openai/gpt-oss-20b"
_hyde_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# =========================================================
# Query expansion (device-scoped)
# =========================================================

# Maps a generic symptom phrase to the manual's own terminology, so a
# common way of describing a fault also matches the wording the manual
# actually uses.
SYNONYM_EXPANSIONS = {
    "power supply": "power supply voltage supply internal supply voltage",
}


def expand_query(query, device_id=None):
    """
    Expand the query with manual-specific synonyms — but ONLY once the
    target device is already known to be the Servo ventilator, where
    "voltage supply" is the manual's actual wording for this symptom.

    Applying this expansion unconditionally (regardless of device)
    previously broke device detection for OTHER devices: SC6002XL's
    troubleshooting tables repeat the word "voltage" constantly (e.g.
    "measure voltage... 11.6 to 13.8 VDC"), so expanding a Philips
    query with "voltage supply" pulled SC6002XL chunks ahead of the
    correct Philips match during the device-detection probe, which
    runs with device_id=None. Gating on device_id keeps expansion from
    ever affecting device detection, since the probe always calls this
    with device_id=None.
    """
    if device_id != "servo_ventilator":
        return query

    lowered = query.lower()
    for key, expansion in SYNONYM_EXPANSIONS.items():
        if key in lowered:
            return f"{query} {expansion}"
    return query


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
    print(f"Documents prepared: {len(texts)}")

    # =========================================
    # Create embeddings
    # =========================================
    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,
        batch_size=32,

    )

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
    print( f"Stored vectors: {collection.count()}")
    print("=" * 70)



# =========================================================
# Semantic Search
# =========================================================

def semantic_search(
    query,
    device_id=None,
    top_k=8,
):

    client = chromadb.PersistentClient(
        path=str(VECTOR_DB_DIR)
    )
    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    search_query = expand_query(query, device_id=device_id)

    query_embedding = embedding_model.encode(
        search_query,
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
# HyDE: Hypothetical Document Embeddings
# =========================================================

def generate_hypothetical_passage(query):
    """
    HyDE: rewrites the user's symptom description in manual-style
    technical language before embedding it, so vague everyday phrasing
    can still match precise manual terminology it doesn't share any
    words with. General technique, not tuned to any one fault.

    Falls back to the original query on any failure (rate limit,
    network error, etc.) so a HyDE hiccup never breaks retrieval.
    """
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
            temperature=0.3,
            max_tokens=60,
        )
        rewritten = response.choices[0].message.content.strip()
        return rewritten if rewritten else query
    except Exception as error:
        print(f"  HyDE rewrite failed, using original query: {error}")
        return query


_bm25_cache = {}
def _get_bm25_index(device_id):

    """Build (and cache) a BM25 index over every chunk of one device."""

    if device_id in _bm25_cache:
        return _bm25_cache[device_id]
    client = chromadb.PersistentClient(
        path=str(VECTOR_DB_DIR)
    )
    collection = client.get_collection(
        name=COLLECTION_NAME
    )
    data = collection.get(
        where={"device_id": device_id}
    )

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
    tokenized_query = re.findall(
        r"[a-z0-9]+",
        query.lower(),
    )
    scores = index["bm25"].get_scores(tokenized_query)
    ranked_positions = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )

    ranked_positions = [
        i for i in ranked_positions if scores[i] > 0
    ][:top_k]
    return {
        "ids": [index["ids"][i] for i in ranked_positions],
        "documents": [index["texts"][i] for i in ranked_positions],
        "metadatas": [index["metadatas"][i] for i in ranked_positions],
        "scores": [scores[i] for i in ranked_positions],
    }

# =========================================================
# Hybrid Search
# Semantic + BM25 candidates combined using RRF
# =========================================================

def hybrid_search(
    query,
    device_id,
    top_k=8,
):
    """
    Combine semantic and BM25 retrieval using
    Reciprocal Rank Fusion (RRF).

    Semantic search captures meaning/similarity.
    BM25 captures exact lexical/keyword matches.

    RRF is used only for ranking candidates.

    IMPORTANT:
    After fusion, candidates are filtered using
    MAX_DISTANCE before they are returned to the LLM.
    This prevents weak semantic matches from entering
    the final RAG context just because they ranked well
    through BM25/RRF.
    """

    hyde_query = generate_hypothetical_passage(query)

    # -----------------------------------------------------
    # 1. Semantic candidates
    # -----------------------------------------------------
    # device_id is already resolved by the time hybrid_search runs
    # (retrieve() only calls this after the device-detection probe),
    # so expand_query's servo-only gate can safely apply here without
    # affecting device detection.

    semantic_top_k = max(
        top_k * 3,
        24,
    )

    semantic_results = semantic_search(
        query=hyde_query,
        device_id=device_id,
        top_k=semantic_top_k,
    )
    semantic_ids = list(
        semantic_results["ids"][0]
    )

    semantic_documents = list(
        semantic_results["documents"][0]
    )

    semantic_metadatas = list(
        semantic_results["metadatas"][0]
    )

    semantic_distances = list(
        semantic_results["distances"][0]
    )

    # -----------------------------------------------------
    # 2. BM25 candidates
    # -----------------------------------------------------

    keyword_results = keyword_search(
        query=query,
        device_id=device_id,
        top_k=BM25_TOP_K,
    )

    keyword_ids = list(
        keyword_results["ids"]
    )

    keyword_documents = list(
        keyword_results["documents"]
    )

    keyword_metadatas = list(
        keyword_results["metadatas"]
    )

    # -----------------------------------------------------
    # 3. Build candidate lookup table
    # -----------------------------------------------------

    candidates = {}

    # Add semantic candidates
    for idx, doc_id in enumerate(
        semantic_ids
    ):
        candidates[doc_id] = {
            "id": doc_id,
            "document":
                semantic_documents[idx],
            "metadata":
                semantic_metadatas[idx],
            "semantic_distance":
                float(
                    semantic_distances[idx]
                ),
            "semantic_rank":
                idx + 1,
            "bm25_rank":
                None,
        }

    # Add BM25 candidates
    for idx, doc_id in enumerate(
        keyword_ids
    ):
        if doc_id not in candidates:

            candidates[doc_id] = {
                "id": doc_id,
                "document":
                    keyword_documents[idx],
                "metadata":
                    keyword_metadatas[idx],
                "semantic_distance":
                    None,
                "semantic_rank":
                    None,
                "bm25_rank":
                    idx + 1,
            }

        else:
            candidates[doc_id]["bm25_rank"] = (
                idx + 1
            )

    if not candidates:
        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

    # -----------------------------------------------------
    # 4. Calculate semantic distance for BM25-only
    #    candidates
    # -----------------------------------------------------

    query_embedding = None

    for candidate in candidates.values():

        if candidate["semantic_distance"] is not None:
            continue

        if query_embedding is None:
            query_embedding = (
                embedding_model.encode(
                    hyde_query,
                    normalize_embeddings=True,
                )
            )

        document_embedding = (
            embedding_model.encode(
                candidate["document"],
                normalize_embeddings=True,
            )
        )

        candidate["semantic_distance"] = float(
            1.0
            -
            (
                document_embedding
                @
                query_embedding
            )
        )

    # -----------------------------------------------------
    # 5. Filter weak semantic matches
    # -----------------------------------------------------

    filtered_candidates = [
        candidate
        for candidate in candidates.values()
        if candidate["semantic_distance"]
        <= MAX_DISTANCE
    ]

    if not filtered_candidates:
        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        # -----------------------------------------------------
    # 4. Reciprocal Rank Fusion
    # -----------------------------------------------------

    RRF_K = 60

        # Semantic gets more weight than BM25: these manuals reuse the same
    # vocabulary constantly ("ventilator", "supply", "pressure"), so raw
    # keyword overlap is a weak signal here and was pulling less-relevant
    # chunks to rank 1 purely on word frequency.
    SEMANTIC_WEIGHT = 1.5
    BM25_WEIGHT = 1.0

    for candidate in filtered_candidates:

        rrf_score = 0.0

        if candidate.get("semantic_rank") is not None:
            rrf_score += (
                SEMANTIC_WEIGHT
                /
                (
                    RRF_K
                    + candidate["semantic_rank"]
                )
            )

        if candidate.get("bm25_rank") is not None:
            rrf_score += (
                BM25_WEIGHT
                /
                (
                    RRF_K
                    + candidate["bm25_rank"]
                )
            )

        # IMPORTANT:
        # Every candidate must receive an RRF score
        # before sorting.
        candidate["rrf_score"] = rrf_score
    # -----------------------------------------------------
    # 5. Sort by fused ranking
    # -----------------------------------------------------

    ranked_candidates = sorted(
    filtered_candidates,
    key=lambda item: (
        item.get("rrf_score", 0.0),
        (
            -item["semantic_distance"]
            if item.get("semantic_distance") is not None
            else float("-inf")
        ),
    ),
    reverse=True,
)

    # -----------------------------------------------------
    # 6. Keep final top-k
    # -----------------------------------------------------

    ranked_candidates = ranked_candidates[:top_k]
    # -----------------------------------------------------
    # 9. Prepare Chroma-like result structure
    # -----------------------------------------------------

    result_ids = []
    result_documents = []
    result_metadatas = []
    result_distances = []

    for candidate in ranked_candidates:

        result_ids.append(
            candidate["id"]
        )

        result_documents.append(
            candidate["document"]
        )

        result_metadatas.append(
            candidate["metadata"]
        )

        result_distances.append(
            candidate["semantic_distance"]
        )

    true_best_distance = min(
        candidate["semantic_distance"]
        for candidate in filtered_candidates
    )

    return {
        "ids": [
            result_ids
        ],
        "documents": [
            result_documents
        ],
        "metadatas": [
            result_metadatas
        ],
        "distances": [
            result_distances
        ],
        "true_best_distance": true_best_distance,
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
    top_k=8,
):
    if error_code is None:
        error_code = detect_error_code(
            query
        )
    if error_code:
        exact_results = exact_error_search(
            error_code,
            device_id,
        )
        if exact_results["ids"]:
            return {
                "retrieval_type":
                    "exact_error",
                "detected_error_code":
                    error_code,
                "detected_device":
                    exact_results["metadatas"][0].get(
                        "device_id",
                        ""
                    ),
                "results":
                    exact_results,
            }
    search_device_id = device_id
    if search_device_id is None:
        # device_id=None here means expand_query will NOT expand the
        # probe query (it only expands once device_id == "servo_ventilator"),
        # so device detection always runs on the query as written.
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

    semantic_results = hybrid_search(
        query=query,
        device_id=search_device_id,
        top_k=top_k,
    )

    if not semantic_results["documents"][0]:
        return {
            "retrieval_type":
                "not_found",
            "detected_error_code":
                error_code,
            "detected_device":
                None,
            "results":
                semantic_results,
        }
    best_distance = semantic_results.get(
        "true_best_distance",
        semantic_results["distances"][0][0],
    )
    print(f"Best semantic distance: {best_distance:.4f}")
    print(f"Maximum allowed distance: {MAX_DISTANCE:.4f}")

    if best_distance > MAX_DISTANCE:
        return {
            "retrieval_type":
                "not_found",
            "detected_error_code":
                error_code,
            "detected_device":
                None,
            "results":
                semantic_results,
        }
    detected_device = (
        semantic_results
        ["metadatas"]
        [0]
        [0]
        .get(
            "device_id",
            ""
        )
    )
    return {
        "retrieval_type":
            "semantic",
        "detected_error_code":
            error_code,
        "detected_device":
            detected_device,

        "results":
            semantic_results,
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
        top_k=8,
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