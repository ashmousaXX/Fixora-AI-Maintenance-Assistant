from retrieval import retrieve


# =========================================================
# Evaluation test cases
# =========================================================

TEST_CASES = [

    # -----------------------------------------------------
    # Siemens Servo Ventilator
    # -----------------------------------------------------

    {
        "name": "Servo - Error 35",
        "query": "The ventilator has error 35",
        "expected_device": "servo_ventilator",
        "expected_type": "exact_error",
        "expected_error_code": "35",
        "expected_answer_terms": [
            "GAS SUPPLY FAILURE",
            "35",
        ],
    },

    {
        "name": "Servo - Gas supply",
        "query": "The ventilator has a gas supply problem",
        "expected_device": "servo_ventilator",
        "expected_type": "semantic",
        "expected_answer_terms": [
            "gas supply",
            "pressure",
            "leakage",
        ],
    },

    {
        "name": "Servo - Power supply",
        "query": "The ventilator has a power supply problem",
        "expected_device": "servo_ventilator",
        "expected_type": "semantic",
        "expected_answer_terms": [
            "voltage supply",
            "power supply",
        ],
    },

    {
        "name": "Servo - Pressure problem",
        "query": "The ventilator has a pressure problem",
        "expected_device": "servo_ventilator",
        "expected_type": "semantic",
        "expected_answer_terms": [
            "pressure",
        ],
    },


    # -----------------------------------------------------
    # Philips / Agilent Monitor
    # -----------------------------------------------------

    {
        "name": "Philips - Blank screen",
        "query": "The patient monitor screen is blank",
        "expected_device":
            "philips_v24_v25_agilent_m1205_monitor_service_manual",
        "expected_type": "semantic",
        "expected_answer_terms": [
            "blank",
            "display",
            "power supply",
        ],
    },

    {
        "name": "Philips - Power problem",
        "query": "The monitor has a power supply problem",
        "expected_device":
            "philips_v24_v25_agilent_m1205_monitor_service_manual",
        "expected_type": "semantic",
        "expected_answer_terms": [
            "power supply",
            "fuse",
        ],
    },


    # -----------------------------------------------------
    # Siemens SC6002XL
    # -----------------------------------------------------

    {
        "name": "SC6002XL - Blank screen",
        "query":
            "The monitor turns on but the screen is blank",
        "expected_device": "sc6002xl",
        "expected_type": "semantic",
        "expected_answer_terms": [
            "screen",
            "blank",
        ],
    },


    # -----------------------------------------------------
    # Negative tests
    # -----------------------------------------------------

    {
        "name": "Negative - Unrelated problem",
        "query":
            "How do I repair a home coffee machine?",
        "expected_device": None,
        "expected_type": "not_found",
        "expected_answer_terms": [],
    },
]


# =========================================================
# Helpers
# =========================================================

def normalize(text):
    return str(text).lower().strip()


def contains_expected_terms(
    text,
    expected_terms,
):
    """
    Check whether important expected terms
    appear in the returned answer.
    """

    text = normalize(text)

    if not expected_terms:
        return True, []

    missing = []

    for term in expected_terms:

        if normalize(term) not in text:
            missing.append(term)

    return len(missing) == 0, missing


# =========================================================
# Retrieval evaluation
# =========================================================

def evaluate_retrieval():

    total = len(TEST_CASES)

    device_correct = 0
    type_correct = 0
    error_code_correct = 0

    results = []

    print()
    print("=" * 80)
    print("FIXORA RETRIEVAL EVALUATION")
    print("=" * 80)

    for index, case in enumerate(
        TEST_CASES,
        start=1,
    ):

        print()
        print(
            f"[{index}/{total}] "
            f"{case['name']}"
        )

        print(
            f"Query: {case['query']}"
        )

        result = retrieve(
            query=case["query"],
            top_k=5,
        )

        actual_type = result.get(
            "retrieval_type"
        )

        actual_device = result.get(
            "detected_device"
        )

        actual_error = result.get(
            "detected_error_code"
        )

        # ---------------------------------------------
        # Metrics
        # ---------------------------------------------

        type_ok = (
            actual_type
            == case["expected_type"]
        )

        device_ok = (
            actual_device
            == case["expected_device"]
        )

        error_ok = True

        if case.get("expected_error_code"):

            error_ok = (
                str(actual_error)
                ==
                str(
                    case[
                        "expected_error_code"
                    ]
                )
            )

        if type_ok:
            type_correct += 1

        if device_ok:
            device_correct += 1

        if error_ok and case.get(
            "expected_error_code"
        ):
            error_code_correct += 1

        print(
            f"Expected type : "
            f"{case['expected_type']}"
        )

        print(
            f"Actual type   : "
            f"{actual_type}"
        )

        print(
            f"Expected device: "
            f"{case['expected_device']}"
        )

        print(
            f"Actual device  : "
            f"{actual_device}"
        )

        if case.get("expected_error_code"):

            print(
                f"Expected error: "
                f"{case['expected_error_code']}"
            )

            print(
                f"Actual error  : "
                f"{actual_error}"
            )

        print(
            f"Type: "
            f"{'PASS' if type_ok else 'FAIL'}"
        )

        print(
            f"Device: "
            f"{'PASS' if device_ok else 'FAIL'}"
        )

        if case.get("expected_error_code"):

            print(
                f"Error code: "
                f"{'PASS' if error_ok else 'FAIL'}"
            )

        results.append(
            {
                "name": case["name"],
                "query": case["query"],
                "result": result,
            }
        )

    # =====================================================
    # Summary
    # =====================================================

    print()
    print("=" * 80)
    print("RETRIEVAL SUMMARY")
    print("=" * 80)

    print(
        f"Total tests: {total}"
    )

    print(
        f"Device accuracy: "
        f"{device_correct / total * 100:.1f}%"
    )

    print(
        f"Retrieval type accuracy: "
        f"{type_correct / total * 100:.1f}%"
    )

    error_tests = sum(
        1
        for case in TEST_CASES
        if case.get("expected_error_code")
    )

    if error_tests:

        print(
            f"Error-code accuracy: "
            f"{error_code_correct / error_tests * 100:.1f}%"
        )

    print("=" * 80)

    return results


# =========================================================
# Answer evaluation
# =========================================================

def evaluate_answers():

    print()
    print("=" * 80)
    print("FIXORA ANSWER EVALUATION")
    print("=" * 80)

    answer_correct = 0
    answer_total = 0

    for index, case in enumerate(
        TEST_CASES,
        start=1,
    ):

        print()
        print(
            f"[{index}/{len(TEST_CASES)}] "
            f"{case['name']}"
        )

        result = retrieve(
            query=case["query"],
            top_k=5,
        )

        # -------------------------------------------------
        # Only evaluate answer if retrieval succeeded
        # -------------------------------------------------

        if result.get(
            "retrieval_type"
        ) == "not_found":

            print(
                "Retrieval returned NOT_FOUND."
            )

            if case["expected_type"] == "not_found":

                print(
                    "Negative test: PASS"
                )

                answer_correct += 1
                answer_total += 1

            continue

        # -------------------------------------------------
        # Import RAG only when answer evaluation is needed
        # -------------------------------------------------

        from rag import answer_query

        rag_result = answer_query(
            query=case["query"],
            top_k=5,
        )

        answer = rag_result.get(
            "answer",
            "",
        )

        ok, missing = (
            contains_expected_terms(
                answer,
                case.get(
                    "expected_answer_terms",
                    [],
                ),
            )
        )

        answer_total += 1

        if ok:
            answer_correct += 1

        print(
            f"Answer check: "
            f"{'PASS' if ok else 'FAIL'}"
        )

        if missing:

            print(
                "Missing expected terms:",
                ", ".join(missing),
            )

        print()
        print("Answer:")
        print(answer[:1000])

    # =====================================================
    # Summary
    # =====================================================

    print()
    print("=" * 80)
    print("ANSWER SUMMARY")
    print("=" * 80)

    print(
        f"Evaluated answers: "
        f"{answer_total}"
    )

    if answer_total:

        print(
            f"Answer fact-match accuracy: "
            f"{answer_correct / answer_total * 100:.1f}%"
        )

    print("=" * 80)


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    retrieval_results = evaluate_retrieval()

    print()

    choice = input(
        "Run LLM answer evaluation? (y/n): "
    ).strip().lower()

    if choice == "y":
        evaluate_answers()
    else:
        print(
            "LLM answer evaluation skipped."
        )