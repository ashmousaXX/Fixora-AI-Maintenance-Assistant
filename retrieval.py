import json
import re
import chromadb

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
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

# How many candidates BM25 should look at per device before fusing
# with the semantic results. Keyword search is cheap, so we can
# afford to scan the whole device corpus instead of a small pool.
BM25_TOP_K = 20


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
    top_k=8,
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
# Keyword (BM25) Search
# =========================================================
#
# Dense embeddings alone can rank an exact-term match low if the
# query is phrased differently than the manual (e.g. query says
# "power supply problem" but the manual chunk says "voltage supply").
# BM25 catches these cases because it scores on literal token
# overlap, independent of embedding geometry.
#
# Scoped to a single device_id (found first via the semantic probe
# in retrieve()) so this stays cheap even on a large multi-manual
# corpus.

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
    )[:top_k]

    # Drop zero-score matches; they add noise, not signal.
    ranked_positions = [
        i for i in ranked_positions if scores[i] > 0
    ]

    return {
        "ids": [index["ids"][i] for i in ranked_positions],
        "documents": [index["texts"][i] for i in ranked_positions],
        "metadatas": [index["metadatas"][i] for i in ranked_positions],
        "scores": [scores[i] for i in ranked_positions],
    }



# =========================================================
# Hybrid Search (semantic order preserved, BM25 used as a RESCUE)
# =========================================================
#
# Earlier version used full Reciprocal Rank Fusion, giving BM25 equal
# weight to semantic similarity. That backfired: a chunk that merely
# repeats the query's literal words (e.g. a general "gas supply is
# regulated" description) would outrank the chunk that actually
# answers the question but uses different wording (e.g. "Leakage in
# ventilator"), because BM25 has no notion of "this is the actionable
# troubleshooting answer" vs "this is background description".
#
# So instead we treat BM25 as a targeted rescue, not a re-ranker:
#   - The semantic ranking is trusted and left untouched.
#   - We only ask BM25 one question: "is there an exact-term match
#     that semantic search missed entirely?"
#   - If yes, that single best keyword hit is added, replacing only
#     the weakest (highest-distance) semantic candidate — so a
#     correct exact-term chunk (like one mentioning "voltage supply")
#     can still surface even if semantic search buried it, without
#     disturbing chunks semantic search already got right.

def hybrid_search(
    query,
    device_id,
    top_k=8,
):

    semantic_results = semantic_search(
        query=query,
        device_id=device_id,
        top_k=top_k,
    )

    semantic_ids = list(semantic_results["ids"][0])
    documents = list(semantic_results["documents"][0])
    metadatas = list(semantic_results["metadatas"][0])
    distances = list(semantic_results["distances"][0])

    if not semantic_ids:
        return semantic_results

    keyword_results = keyword_search(
        query=query,
        device_id=device_id,
        top_k=BM25_TOP_K,
    )

    # Find the single best keyword hit that semantic search missed
    # entirely (not just ranked low — genuinely absent from the pool).
    rescue_index = None

    for idx, doc_id in enumerate(keyword_results["ids"]):
        if doc_id not in semantic_ids:
            rescue_index = idx
            break

    if rescue_index is not None:

        rescue_text = keyword_results["documents"][rescue_index]

        query_embedding = embedding_model.encode(
            query,
            normalize_embeddings=True,
        )

        rescue_embedding = embedding_model.encode(
            rescue_text,
            normalize_embeddings=True,
        )

        rescue_distance = float(
            1.0 - (rescue_embedding @ query_embedding)
        )

        # Swap out the weakest current slot (highest distance) for
        # the rescued chunk, so the total candidate count stays the
        # same and the LLM context doesn't balloon.
        weakest_position = distances.index(max(distances))

        semantic_ids[weakest_position] = keyword_results["ids"][rescue_index]
        documents[weakest_position] = rescue_text
        metadatas[weakest_position] = keyword_results["metadatas"][rescue_index]
        distances[weakest_position] = rescue_distance

    # Re-sort everything by distance so "Best semantic distance" and
    # the source ordering shown to the LLM stay consistent (closest
    # match first), regardless of whether a rescue happened.
    order = sorted(
        range(len(semantic_ids)),
        key=lambda i: distances[i],
    )

    return {
        "ids": [[semantic_ids[i] for i in order]],
        "documents": [[documents[i] for i in order]],
        "metadatas": [[metadatas[i] for i in order]],
        "distances": [[distances[i] for i in order]],
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
    top_k=8,
):


    if error_code is None:

        error_code = detect_error_code(
            query
        )



    # Exact error search

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



    # Hybrid search (semantic + BM25)
    #
    # When no device was pre-selected, do this in TWO stages instead of
    # one global search: first a cheap probe (top_k=1) across every
    # device just to find which single device is the best match, then a
    # second search restricted to ONLY that device for the real top_k.
    # Without this, the top-5 results can freely mix chunks from
    # unrelated devices (e.g. a ventilator query pulling in an
    # ultrasound machine's chunk just because it ranked 4th globally),
    # and the LLM ends up blending both into one answer.
    #
    # The second stage now fuses semantic + BM25 (see hybrid_search)
    # instead of relying on semantic distance alone, so an exact
    # domain term in the manual (e.g. "voltage supply") isn't missed
    # just because the query used different wording.

    search_device_id = device_id

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




    best_distance = semantic_results["distances"][0][0]



    print(
        f"Best semantic distance: {best_distance:.4f}"
    )


    print(
        f"Maximum allowed distance: {MAX_DISTANCE:.4f}"
    )



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