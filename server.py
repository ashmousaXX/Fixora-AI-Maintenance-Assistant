import os
import uuid
import tempfile

from flask import Flask, request, jsonify, send_from_directory, send_file
from dotenv import load_dotenv
from groq import Groq
import torch
from transformers import pipeline

from rag import answer_query


# =========================================================
# Configuration
# =========================================================

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STT_MODEL = "openai/whisper-large-v3-turbo"

TTS_MODEL = "canopylabs/orpheus-v1-english"
TTS_VOICE = "troy"

RECORDINGS_DIR = os.path.join(BASE_DIR, "voice_recordings")
os.makedirs(RECORDINGS_DIR, exist_ok=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in .env")

groq_client = Groq(api_key=GROQ_API_KEY)


# =========================================================
# Load STT model once at startup (not per-request)
# =========================================================

print(f"Loading STT model: {STT_MODEL}")

stt = pipeline(
    "automatic-speech-recognition",
    model=STT_MODEL,
    device=-1,
    dtype=torch.float32,
)

print("STT model loaded.")


# =========================================================
# Flask app
# =========================================================

app = Flask(__name__)


# ---------------------------------------------------------
# Serve the chat UI itself
# ---------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "fixora-ui.html")


# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------

@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


# ---------------------------------------------------------
# Ask a question (RAG)
# ---------------------------------------------------------

@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()

    if not query:
        return jsonify({"error": "query is required"}), 400

    result = answer_query(
        query=query,
        top_k=8,
    )

    return jsonify(
        {
            "answer": result.get("answer", ""),
            "speech_answer": result.get("speech_answer", ""),
            "device": result.get("detected_device"),
            "retrieval_type": result.get("retrieval_type"),
        }
    )


# ---------------------------------------------------------
# Text-to-speech
# ---------------------------------------------------------

@app.route("/api/speech", methods=["POST"])
def speech():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "text is required"}), 400

    output_path = os.path.join(
        RECORDINGS_DIR,
        f"reply_{uuid.uuid4().hex}.wav",
    )

    response = groq_client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=text,
        response_format="wav",
    )

    response.write_to_file(output_path)

    return send_file(
        output_path,
        mimetype="audio/wav",
    )


# ---------------------------------------------------------
# Speech-to-text
# ---------------------------------------------------------
#
# NOTE: browser microphone recordings (via the MediaRecorder API used
# in fixora-ui.html) typically arrive as WebM/Opus, not WAV. Decoding
# that requires ffmpeg to be installed and on PATH on this machine.
# If transcription fails with a decoding error, install ffmpeg
# (https://ffmpeg.org/download.html) and make sure `ffmpeg` runs from
# a plain terminal.
# ---------------------------------------------------------

@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "audio file is required"}), 400

    audio_file = request.files["audio"]

    suffix = os.path.splitext(audio_file.filename or "")[1] or ".webm"

    with tempfile.NamedTemporaryFile(
        suffix=suffix,
        delete=False,
    ) as temp_file:
        audio_file.save(temp_file.name)
        temp_path = temp_file.name

    try:
        result = stt(
            temp_path,
            generate_kwargs={
                "language": "english",
                "task": "transcribe",
            },
        )
        text = result["text"].strip()
    finally:
        os.remove(temp_path)

    return jsonify({"text": text})


# =========================================================
# Run
# =========================================================

if __name__ == "__main__":
    app.run(
        debug=True,
        port=5000,
    )
    