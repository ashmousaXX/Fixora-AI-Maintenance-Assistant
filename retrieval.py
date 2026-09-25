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

# Words that commonly precede a stray number in service-manual prose
# without that number being an actual device error/fault code (page
# refs, section numbers, figure numbers, model/part numbers, etc.).
# With 39 manuals in the corpus, "code 5" or "fault 12" shows up
# constantly as ordinary text -- this blocklist filters those out
# before a match is accepted.
ERROR_CODE_FALSE_POSITIVE_CONTEXT = {
    "page", "pg", "p", "chapter", "ch", "section", "sec",
    "table", "tbl", "figure", "fig", "step", "item", "note",
    "rev", "revision", "version", "ver", "model", "type",
    "part", "no", "number", "appendix", "volume", "vol",
    "manual", "chart", "diagram", "row", "column", "col",
}


def _is_false_positive_context(text, match_start):
    """
    Look at the word immediately before a candidate error-code match
    and reject the match if that word is a known false-positive
    trigger (a page/section/figure/table reference, etc.) rather than
    genuine error-code language.
    """
    preceding = text[:match_start].strip().split()
    if not preceding:
        return False
    context_word = preceding[-1].lower().strip(".,:;()[]-")
    return context_word in ERROR_CODE_FALSE_POSITIVE_CONTEXT


def detect_error_code(text):
    """
    Detect an error/fault code mentioned in text.

    Deliberately narrower than a naive "any number near the word
    code" match: across a multi-device, multi-manual corpus, loose
    patterns like a bare "code \\d+" false-positive constantly on
    page numbers, section numbers, figure numbers, etc., and those
    false positives get indexed as real error_code metadata -- which
    then lets exact_error_search() return the wrong device's chunk
    for an unrelated query. This function requires the match to be
    introduced by "error"/"fault" specifically (not the generic word
    "code" alone) and rejects matches immediately preceded by a
    reference-style word (see ERROR_CODE_FALSE_POSITIVE_CONTEXT).
    """
    patterns = [
        r"\berror\s+code\s*[:#]?\s*(\d{1,5})\b",
        r"\bfault\s+code\s*[:#]?\s*(\d{1,5})\b",
        r"\berror\s+(\d{1,5})\b",
        r"\bfault\s+(\d{1,5})\b",
        r"\bE[-\s]?(\d{1,5})\b",
        r"\bERR[-\s]?(\d{1,5})\b",
    ]
    for pattern in patterns:
        for match in re.finditer(
            pattern,
            text,
            re.IGNORECASE,
        ):
            if _is_false_positive_context(text, match.start()):
                continue
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
    # NOTE: default widened from 5 to 8. In a safety-relevant domain
    # (medical device troubleshooting), narrowing recall to save a
    # little context length is the wrong tradeoff -- a relevant
    # passage that exists in the manual but doesn't rank in the top 5
    # should still have a real chance to reach the LLM, rather than
    # silently being dropped and the assistant reporting "not found"
    # when the manual actually does have an answer.
    if error_code is None:
        error_code = detect_error_code(
            query
        )

    # ---------------------------------------------------
    # Two-stage device resolution for exact error-code
    # lookups.
    #
    # Error codes are NOT unique across devices/manuals, so
    # blindly running exact_error_search() with no device
    # filter (the old behavior) can return a completely
    # unrelated device's chunk just because it happens to
    # share the same code number. To avoid that, when the
    # caller hasn't already pinned a device_id, we first run
    # an unscoped semantic search purely to guess which
    # device the query is actually about, then scope the
    # exact search to that device. Only if the device-scoped
    # exact search comes up empty do we fall back to a
    # global (unscoped) exact search, since an unambiguous
    # code with no semantic device match is still better
    # than returning nothing.
    # ---------------------------------------------------
    inferred_device_id = device_id

    if error_code and inferred_device_id is None:
        device_guess = semantic_search(
            query=query,
            device_id=None,
            top_k=1,
        )
        guess_metadatas = device_guess.get("metadatas") or [[]]
        guess_distances = device_guess.get("distances") or [[]]
        if guess_metadatas[0] and guess_distances[0]:
            guess_distance = guess_distances[0][0]
            if guess_distance <= MAX_DISTANCE:
                inferred_device_id = (
                    guess_metadatas[0][0].get("device_id") or None
                )

    # Exact error search
    if error_code:
        exact_results = None
        scoped_to_device = None

        if inferred_device_id:
            exact_results = exact_error_search(
                error_code,
                inferred_device_id,
            )
            scoped_to_device = inferred_device_id

        if not exact_results or not exact_results["ids"]:
            # Fall back to a global lookup -- either we couldn't
            # confidently guess a device, or the guessed device
            # doesn't actually have this code (e.g. the query was
            # ambiguous enough that semantic search leaned the
            # wrong way).
            exact_results = exact_error_search(
                error_code,
                device_id,
            )
            scoped_to_device = device_id

        if exact_results["ids"]:
            devices_in_result = {
                metadata.get("device_id")
                for metadata in exact_results["metadatas"]
            }
            if scoped_to_device is None and len(devices_in_result) > 1:
                print(
                    f"WARNING: error code {error_code} matched "
                    f"multiple devices {sorted(d for d in devices_in_result if d)} "
                    f"with no device scoping available; returning the "
                    f"first match. Consider asking the user which "
                    f"device they mean."
                )
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

    # Semantic search
    semantic_results = semantic_search(
        query=query,
        device_id=device_id,
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