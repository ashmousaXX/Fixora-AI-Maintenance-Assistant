from retrieval import retrieve
from llm import generate_answer


# =========================================================
# Format exact error results
# =========================================================

def format_exact_results(results):

    if not results.get("ids"):
        return ""

    context_parts = []

    for index, (
        document,
        metadata,
    ) in enumerate(
        zip(
            results["documents"],
            results["metadatas"],
        ),
        start=1,
    ):

        source = (
            f"SOURCE {index}\n"
            f"Device: {metadata.get('device','')}\n"
            f"Page: {metadata.get('page','')}\n"
            f"Section: {metadata.get('section','')}\n"
            f"Error code: {metadata.get('error_code','')}\n"
            f"Manual evidence:\n"
            f"{document}"
        )

        context_parts.append(source)

    return "\n\n".join(context_parts)



# =========================================================
# Format semantic results
# =========================================================

def format_semantic_results(results):

    documents = results.get("documents")

    if not documents:
        return ""

    if not documents[0]:
        return ""

    documents = documents[0]
    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    distances = results.get(
        "distances",
        [[]]
    )[0]


    context_parts = []


    for index, (
        document,
        metadata,
        distance,
    ) in enumerate(
        zip(
            documents,
            metadatas,
            distances,
        ),
        start=1,
    ):

        source = (
            f"SOURCE {index}\n"
            f"Device: {metadata.get('device','')}\n"
            f"Page: {metadata.get('page','')}\n"
            f"Section: {metadata.get('section','')}\n"
            f"Distance: {distance:.4f}\n"
            f"Manual evidence:\n"
            f"{document}"
        )

        context_parts.append(source)


    return "\n\n".join(context_parts)



# =========================================================
# Build RAG Context
# =========================================================

def build_rag_context(
    query,
    device_id=None,
    top_k=5,
):

    retrieval_output = retrieve(
        query=query,
        device_id=device_id,
        top_k=top_k,
    )


    retrieval_type = retrieval_output[
        "retrieval_type"
    ]


    results = retrieval_output[
        "results"
    ]


    if retrieval_type == "exact_error":

        context = format_exact_results(
            results
        )

    elif retrieval_type == "not_found":
        context = ""

    else:
        context = format_semantic_results(
            results
        )


    return {

        "retrieval_type":
            retrieval_type,

        "detected_error_code":
            retrieval_output.get(
                "detected_error_code"
            ),

        # retrieval.py already computes this correctly for both the
        # exact_error (.get()-style flat metadata) and semantic
        # (.query()-style nested metadata) cases — reuse it directly
        # instead of re-deriving it here.
        "detected_device":
            retrieval_output.get(
                "detected_device"
            ),

        "context":
            context,

        "raw_results":
            results,

    }



# =========================================================
# Main Answer Function
# =========================================================

def answer_query(
    query,
    top_k=5,
    device_id=None,
):


    rag_result = build_rag_context(
        query=query,
        device_id=device_id,
        top_k=top_k,
    )


    context = rag_result["context"]



    if not context:

        return {

            "answer":
                "No relevant information was found in the service manuals.",

            "speech_answer":
                "No relevant information was found in the service manuals.",

            "detected_device":
                rag_result.get(
                    "detected_device"
                ),

            "retrieval_type":
                rag_result["retrieval_type"],

            "context":
                "",

        }



    generated = generate_answer(
    query=query,
    context=context,
    device=rag_result.get(
        "detected_device"
    ),
)



    return {
    "answer": generated.get(
        "display_answer",
        ""
    ),
    "speech_answer": generated.get(
        "speech_answer",
        ""
    ),
    "detected_device":
        rag_result.get(
            "detected_device"
        ),
    "retrieval_type":
        rag_result["retrieval_type"],
    "context": context,
}



# =========================================================
# Test
# =========================================================

def test_full_rag():

    query = (
        "The ventilator has a gas supply problem"
    )


    result = answer_query(
        query=query,
        top_k=5,
    )


    print("="*70)

    print(
        "DEVICE:",
        result["detected_device"]
    )


    print(
        "TYPE:",
        result["retrieval_type"]
    )


    print()

    print(
        result["answer"]
    )


    print("="*70)



if __name__ == "__main__":

    test_full_rag()