import json
import os
import re
import time

from dotenv import load_dotenv
from groq import Groq, RateLimitError
from prompts import SYSTEM_PROMPT, build_user_prompt

load_dotenv(override=True)
MODEL_NAME = "qwen/qwen3.8-27b"
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
            retry_after = None
            try:
                retry_after = float(e.response.headers.get("retry-after"))
            except Exception:
                pass

            if retry_after and retry_after > 60:
                raise RuntimeError(
                    f"Groq daily/rate quota exhausted — retry after {retry_after:.0f}s"
                )

            # Exponential backoff (8s, 16s, 32s, 60s, 60s), never shorter
            # than what Groq asked for via retry-after.
            wait = min(
                max(wait_seconds * (2 ** attempt), (retry_after or 0) + 1),
                60,
            )

            last_error = "Rate limit hit"
            if attempt >= max_retries:
                break  # out of retries: don't sleep for nothing

            print(
                f"[INFO] Groq rate limit — waiting {wait:.0f}s before retry "
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
def _context_pages(context):
    """
    Every page number that genuinely appears anywhere in the
    retrieved context -- both as a SOURCE's own "Page: N" header
    AND as any "page N" mention inside a SOURCE's quoted manual
    text (e.g. an internal cross-reference the manual itself makes,
    like "refer to X on page 14"). A number only counts as grounded
    if it is actually present in the evidence, by either route.
    """
    header_pages = set(int(n) for n in re.findall(r"^Page:\s*(\d+)", context, re.MULTILINE))
    body_pages = set(int(n) for n in re.findall(r"[Pp]age\s*:?\s*(\d+)", context))
    return header_pages | body_pages

def _answer_pages(display_answer):
    """Every page number the model's answer claims to cite."""
    return set(int(n) for n in re.findall(r"[Pp]age\s*:?\s*(\d+)", display_answer))


def _find_unsupported_pages(display_answer, context):
    """
    Page numbers cited in the answer that do not appear anywhere in
    the retrieved context -- a strong signal the model pulled a
    detail from outside knowledge instead of the provided evidence,
    even when that detail happens to be factually accurate.
    """
    return _answer_pages(display_answer) - _context_pages(context)

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
        unsupported = _find_unsupported_pages(display_answer, context)
        if unsupported:
            print(f"[WARN] Answer cited page(s) not in context: {sorted(unsupported)} — retrying once")
            corrective_prompt = (
                user_prompt
                + "\n\nYour previous answer cited page number(s) "
                + ", ".join(str(p) for p in sorted(unsupported))
                + ", which do NOT appear in the SERVICE MANUAL EVIDENCE above. "
                + "This means you used information from outside the provided "
                + "evidence. Rewrite your answer using ONLY the pages that "
                + "actually appear above: "
                + ", ".join(str(p) for p in sorted(_context_pages(context)))
                + ". Remove any claim tied to a page not in that list."
            )
            try:
                retry_raw = call_groq_with_retry(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": corrective_prompt},
                    ]
                )
                if retry_raw.startswith("```"):
                    retry_raw = retry_raw.strip("`").strip()
                    if retry_raw.lower().startswith("json"):
                        retry_raw = retry_raw[4:].strip()
                retry_parsed = json.loads(retry_raw)
                retry_display = str(retry_parsed.get("display_answer", "")).strip()
                retry_speech = str(retry_parsed.get("speech_answer", "")).strip()

                if retry_display and not _find_unsupported_pages(retry_display, context):
                    display_answer = retry_display
                    speech_answer = retry_speech or retry_display
                else:
                    print("[WARN] Retry still cites unsupported page(s) — flagging answer")
                    display_answer += (
                        "\n\n*Note: part of this answer could not be fully "
                        "verified against the retrieved manual pages and may "
                        "need manual double-checking.*"
                    )
            except Exception as error:
                print(f"[WARN] Corrective retry failed: {error} — flagging original answer")
                display_answer += (
                    "\n\n*Note: part of this answer could not be fully "
                    "verified against the retrieved manual pages and may "
                    "need manual double-checking.*"
                )

        if not speech_answer:
            speech_answer = display_answer

        speech_answer = clean_speech_text(speech_answer)
        speech_answer = truncate_speech_text(speech_answer)

        return {
            "display_answer": display_answer,
            "speech_answer": speech_answer,
        }

    except json.JSONDecodeError:
        # The model occasionally emits a value without its opening
        # quote (seen with an unquoted "speech_answer"). Try to pull
        # display_answer and speech_answer out with a tolerant regex
        # before falling back to dumping the raw JSON text as-is.
        display_match = re.search(
            r'"display_answer"\s*:\s*"(.*?)"\s*,\s*"speech_answer"',
            raw, re.DOTALL,
        )
        speech_match = re.search(
            r'"speech_answer"\s*:\s*"?(.*?)"?\s*\}?\s*$',
            raw, re.DOTALL,
        )

        if display_match:
            display_answer = display_match.group(1).replace('\\"', '"').replace("\\n", "\n")
            speech_answer = (
                speech_match.group(1).strip().rstrip('"')
                if speech_match else display_answer
            )
            speech_answer = clean_speech_text(speech_answer)
            speech_answer = truncate_speech_text(speech_answer)
            return {
                "display_answer": display_answer,
                "speech_answer": speech_answer,
            }

        speech_answer = clean_speech_text(raw)
        speech_answer = truncate_speech_text(speech_answer)
        return {
            "display_answer": raw,
            "speech_answer": speech_answer,
        }