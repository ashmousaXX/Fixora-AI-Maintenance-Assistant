import os
import time
import uuid
import tempfile
from flask import Flask, request, jsonify, send_from_directory, send_file
from dotenv import load_dotenv
from groq import Groq
import torch
from transformers import pipeline
from rag import answer_query

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


print(f"Loading STT model: {STT_MODEL}")

stt = pipeline(
    "automatic-speech-recognition",
    model=STT_MODEL,
    device=-1,
    dtype=torch.float32,
)
print("STT model loaded.")
app = Flask(__name__)

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "fixora-ui.html")

@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()

    if not query:
        return jsonify({"error": "query is required"}), 400

    t0 = time.time()
    result = answer_query(
        query=query,
        top_k=8,
    )
    print(f"[timing] /api/ask (retrieval + LLM) took {time.time() - t0:.2f}s")

    return jsonify(
        {
            "answer": result.get("answer", ""),
            "speech_answer": result.get("speech_answer", ""),
            "device": result.get("detected_device"),
            "retrieval_type": result.get("retrieval_type"),
        }
    )


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

    t0 = time.time()
    response = groq_client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=text,
        response_format="wav",
    )
    response.write_to_file(output_path)
    print(f"[timing] /api/speech (TTS, {len(text)} chars) took {time.time() - t0:.2f}s")

    return send_file(
        output_path,
        mimetype="audio/wav",
    )


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
        t0 = time.time()
        result = stt(
            temp_path,
            generate_kwargs={
                "language": "english",
                "task": "transcribe",
            },
        )
        text = result["text"].strip()
        print(f"[timing] /api/transcribe (STT) took {time.time() - t0:.2f}s")
    finally:
        os.remove(temp_path)

    return jsonify({"text": text})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)