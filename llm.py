import json
import os
import re
import time

from dotenv import load_dotenv
from groq import Groq, RateLimitError

from prompts import SYSTEM_PROMPT, build_user_prompt


# =========================================================
# Configuration
# =========================================================

load_dotenv()

MODEL_NAME = "openai/gpt-oss-20b"

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# =========================================================
# Clean speech text (safety net against symbols)
# =========================================================

def clean_speech_text(text):
    """
    Strip/replace symbols that Orpheus TTS would read aloud
    literally (e.g. "equals", "dash", "open parenthesis").
    """
    text = text.replace("=", " means ")
    text = re.sub(r"[–—-]", ", ", text)      # dashes -> natural pause
    text = re.sub(r"[()]", "", text)          # drop parentheses, keep content
    text = re.sub(r"[*_#|/\\]", "", text)     # strip leftover markdown symbols
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =========================================================
# Truncate speech text without cutting mid-sentence
# =========================================================

def truncate_speech_text(text, limit=180):
    """
    Cuts speech_answer down to `limit` characters WITHOUT ending
    mid-sentence. Prefers cutting at the last complete sentence;
    only falls back to a word boundary if no sentence fits at all.
    """
    if len(text) <= limit:
        return text

    truncated = text[:limit]

    last_period = truncated.rfind(".")
    if last_period > 40:
        return truncated[: last_period + 1]

    # No full sentence fits — fall back to last full word, add a period
    return truncated.rsplit(" ", 1)[0].rstrip(",") + "."


# =========================================================
# Call Groq with automatic retry
#
# Handles two distinct failure modes with one retry loop:
#   - RateLimitError from the Groq client (transient, free-tier TPM caps)
#   - A "successful" response with empty content (occasionally the model
#     returns finish_reason without usable text)
#
# NOTE: this replaces the previous version of llm.py, which defined
# call_groq_with_retry twice — once at module level (handling only
# RateLimitError) and again, shadowing it, inside generate_answer()
# (handling only empty content). Neither retried on both failure
# modes. This single version retries on either.
# =========================================================

def call_groq_with_retry(
    messages,
    max_retries=2,
    wait_seconds=1.5,
):
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.2,
                max_completion_tokens=1200,
                reasoning_effort="low",
            )
            finish_reason = response.choices[0].finish_reason
            content = response.choices[0].message.content

            # DEBUG:
            print(
                f"[DEBUG] attempt={attempt} finish_reason={finish_reason} "
                f"content_len={len(content) if content else 0}"
            )

            if content and content.strip():
                return content.strip()
            last_error = f"Empty content, finish_reason={finish_reason}"

        except RateLimitError:
            last_error = "Rate limit hit"
            print(
                f"[INFO] Groq rate limit hit — waiting {wait_seconds}s "
                f"before retry {attempt + 1}/{max_retries}..."
            )
        except Exception as error:
            last_error = str(error)
            print(f"[DEBUG] attempt={attempt} exception: {last_error}")

        if attempt < max_retries:
            time.sleep(wait_seconds)

    raise RuntimeError(f"Groq call failed after retries: {last_error}")


# =========================================================
# Generate answer
# =========================================================

def generate_answer(
    query,
    context,
    device=None,
):
    system_prompt = SYSTEM_PROMPT
    user_prompt = build_user_prompt(
        query=query,
        device=device,
        context=context,
    )

    try:
        raw = call_groq_with_retry(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
    except RuntimeError as error:
        # -------------------------------------------------
        # Guard against total call failure (all retries
        # exhausted), not just an empty-but-successful call.
        # -------------------------------------------------
        fallback_text = (
            "The assistant did not return a response for this query. "
            "Please try rephrasing the question or asking again."
        )
        print(f"[DEBUG] generate_answer giving up: {error}")
        return {
            "display_answer": fallback_text,
            "speech_answer": fallback_text,
        }

    # -----------------------------------------------------
    # Guard against a genuinely empty completion from the API
    # -----------------------------------------------------

    if not raw:
        fallback_text = (
            "The assistant did not return a response for this query. "
            "Please try rephrasing the question or asking again."
        )
        return {
            "display_answer": fallback_text,
            "speech_answer": fallback_text,
        }

    # -----------------------------------------------------
    # Remove accidental markdown code fences
    # -----------------------------------------------------

    if raw.startswith("```"):
        raw = raw.strip("`").strip()

        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    # -----------------------------------------------------
    # Parse JSON
    # -----------------------------------------------------

    try:
        parsed = json.loads(raw)

        display_answer = str(
            parsed.get(
                "display_answer",
                "",
            )
        ).strip()

        speech_answer = str(
            parsed.get(
                "speech_answer",
                "",
            )
        ).strip()

        # -------------------------------------------------
        # Fallback if fields are missing
        # -------------------------------------------------

        if not display_answer:
            display_answer = raw

        if not speech_answer:
            speech_answer = display_answer

        # -------------------------------------------------
        # Safety net: strip symbols, then truncate at a
        # sentence boundary (never mid-sentence)
        # -------------------------------------------------

        speech_answer = clean_speech_text(speech_answer)
        speech_answer = truncate_speech_text(speech_answer)

        return {
            "display_answer": display_answer,
            "speech_answer": speech_answer,
        }

    except json.JSONDecodeError:
        # -------------------------------------------------
        # Fallback if LLM did not return valid JSON
        # -------------------------------------------------

        speech_answer = clean_speech_text(raw)
        speech_answer = truncate_speech_text(speech_answer)

        return {
            "display_answer": raw,
            "speech_answer": speech_answer,
        }