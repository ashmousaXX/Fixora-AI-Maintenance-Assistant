import json
import os
import re
import time

from dotenv import load_dotenv
from groq import Groq, RateLimitError
from prompts import SYSTEM_PROMPT, build_user_prompt

load_dotenv(override=True)
MODEL_NAME = "openai/gpt-oss-120b"
MAX_COMPLETION_TOKENS = 2500

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

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

def truncate_speech_text(text, limit=180):
    """
    Cuts speech_answer down to `limit` characters WITHOUT ending
    mid-sentence. Prefers cutting at the last complete sentence; only
    falls back to a word boundary if no sentence fits at all — and
    when falling back, keeps dropping trailing "weak" words (articles,
    prepositions, conjunctions) so it never ends on a dangling word
    like "...consulting the manual for." which reads as broken when
    spoken aloud.
    """
    if len(text) <= limit:
        return text
    truncated = text[:limit]
    last_period = truncated.rfind(".")
    if last_period > 40:
        return truncated[: last_period + 1]
    weak_endings = {
        "a", "an", "the", "to", "for", "of", "in", "on", "at", "by",
        "with", "and", "or", "but", "is", "are", "as", "from", "that",
        "this", "it", "its", "if",
    }
    words = truncated.rstrip(" ,").split(" ")
    while words and words[-1].lower().strip(",.") in weak_endings:
        words.pop()
    if not words:
        return truncated.rsplit(" ", 1)[0].rstrip(",") + "."
    return " ".join(words).rstrip(",") + "."

def call_groq_with_retry(
    messages,
    max_retries=4,
    wait_seconds=8,
):
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.2,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                reasoning_effort="low",
            )
            finish_reason = response.choices[0].finish_reason
            content = response.choices[0].message.content
            print(f"[DEBUG] attempt={attempt} finish_reason={finish_reason} content_len={len(content) if content else 0}")

            if finish_reason == "length":
                last_error = f"Truncated, finish_reason=length"
            elif content and content.strip():
                return content.strip()
            else:
                last_error = f"Empty content, finish_reason={finish_reason}"

        except RateLimitError as e:
            wait = wait_seconds
            try:
                retry_after = e.response.headers.get("retry-after")
                if retry_after:
                    wait = float(retry_after)
            except Exception:
                pass

            if wait > 60:
                raise RuntimeError(
                    f"Groq daily/rate quota exhausted — retry after {wait:.0f}s"
                )

            last_error = "Rate limit hit"
            print(
                f"[INFO] Groq rate limit — waiting {wait}s before retry "
                f"{attempt + 1}/{max_retries}..."
            )
            time.sleep(wait)

            continue

        except Exception as error:
            last_error = str(error)
            print(f"[DEBUG] attempt={attempt} exception: {last_error}")

        if attempt < max_retries:
            time.sleep(wait_seconds)

    raise RuntimeError(f"Groq call failed after retries: {last_error}")

MAX_CONTEXT_CHARS = 8000
def _truncate_context(context, limit=MAX_CONTEXT_CHARS):
    if len(context) <= limit:
        return context
    truncated = context[:limit]

    last_source = truncated.rfind("\n\nSOURCE ")
    if last_source > 0:
        truncated = truncated[:last_source]
    return truncated + (
        "\n\n[Additional matching sources were omitted to stay within "
        "the model's request size limit.]"
    )

def generate_answer(
    query,
    context,
    device=None,
):
    context = _truncate_context(context)
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
        fallback_text = (
            "The assistant did not return a response for this query. "
            "Please try rephrasing the question or asking again."
        )
        print(f"[DEBUG] generate_answer giving up: {error}")
        return {
            "display_answer": fallback_text,
            "speech_answer": fallback_text,
        }

    if not raw:
        fallback_text = (
            "The assistant did not return a response for this query. "
            "Please try rephrasing the question or asking again."
        )
        return {
            "display_answer": fallback_text,
            "speech_answer": fallback_text,
        }

    if raw.startswith("```"):
        raw = raw.strip("`").strip()

        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

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

        if not display_answer:
            display_answer = raw

        if not speech_answer:
            speech_answer = display_answer

        speech_answer = clean_speech_text(speech_answer)
        speech_answer = truncate_speech_text(speech_answer)

        return {
            "display_answer": display_answer,
            "speech_answer": speech_answer,
        }

    except json.JSONDecodeError:

        speech_answer = clean_speech_text(raw)
        speech_answer = truncate_speech_text(speech_answer)
        return {
            "display_answer": raw,
            "speech_answer": speech_answer,
        }