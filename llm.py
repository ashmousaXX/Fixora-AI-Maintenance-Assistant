import json
import os
import re

from dotenv import load_dotenv
from groq import Groq


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
# Generate answer
# =========================================================

def generate_answer(
    query,
    context,
    device=None,
):

    system_prompt = """
You are Fixora, a technical maintenance assistant.

Your job is to answer the user's question using only the
provided service-manual evidence.

Rules:

1. Use only the provided manual evidence.

2. Do not invent causes, procedures, measurements, or steps.

3. If the manual evidence is insufficient, clearly say that.

4. Preserve technical terminology from the manual.

5. Preserve the order and relationships found in the provided evidence.
Do not invent a troubleshooting priority or sequence unless the manual
explicitly provides one.

6. Mention the manual page and section when available.

7. Do not claim something is confirmed unless the evidence confirms it.

8. If the manual gives multiple possible causes, present them as possibilities.

9. Keep the answer practical and concise, but do not add procedural wording
that is not present in the evidence.

10. Do not add an "order of checks", priority, diagnosis, or recommendation
unless that ordering is explicitly supported by the provided evidence.

11. If multiple retrieved chunks describe different possible causes for the
same symptom, present them as separate possible causes without ranking them.

12. Do not tell the user to check causes "in turn", "first", "next", or in any
sequence unless the manual explicitly provides that sequence.

13. End the answer after presenting the supported causes, actions, and references.
Do not add a concluding instruction unless that instruction is explicitly present
in the manual evidence.

14. Return valid JSON only with exactly these two keys:

{
  "display_answer": "Full detailed answer for the screen. Markdown is allowed.",
  "speech_answer": "One or two short, COMPLETE spoken sentences, totaling under 170
  characters. Never start a sentence you cannot finish within that budget — if the
  full explanation does not fit, mention only the single most important cause and
  action. Plain words only — no markdown, no symbols such as =, -, (), /, :, or *.
  Write everything as natural words instead (e.g. 'means' instead of '=')."
}

The speech_answer must communicate the same supported conclusion as the
display_answer.

If the evidence includes DANGER, WARNING, or CAUTION, mention that first
in speech_answer.

Do not invent information that is not supported by the evidence.
"""

    user_prompt = f"""
USER QUESTION:
{query}

DEVICE INFORMATION:
{device}

SERVICE MANUAL EVIDENCE:
{context}

Answer the question using only the evidence above.

Return raw JSON only.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.2,
    )

    raw = (
        response
        .choices[0]
        .message
        .content
        .strip()
    )

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
            "display_answer":
                display_answer,

            "speech_answer":
                speech_answer,
        }

    except json.JSONDecodeError:

        # -------------------------------------------------
        # Fallback if LLM did not return valid JSON
        # -------------------------------------------------

        speech_answer = clean_speech_text(raw)
        speech_answer = truncate_speech_text(speech_answer)

        return {
            "display_answer":
                raw,

            "speech_answer":
                speech_answer,
        }