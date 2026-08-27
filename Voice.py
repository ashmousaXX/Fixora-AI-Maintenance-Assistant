import os

import sounddevice as sd
import soundfile as sf
import torch

from dotenv import load_dotenv
from groq import Groq
from transformers import pipeline

from rag import answer_query


# =========================================================
# Configuration
# =========================================================

STT_MODEL = "openai/whisper-large-v3-turbo"

TTS_MODEL = "canopylabs/orpheus-v1-english"
TTS_VOICE = "troy"

SAMPLE_RATE = 16000
RECORD_SECONDS = 5

RECORDINGS_DIR = "voice_recordings"

os.makedirs(
    RECORDINGS_DIR,
    exist_ok=True,
)


# =========================================================
# Groq
# =========================================================

load_dotenv()

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY not found in .env"
    )


groq_client = Groq(
    api_key=GROQ_API_KEY
)


# =========================================================
# Whisper STT
# =========================================================

print(
    f"Loading STT model: {STT_MODEL}"
)


stt = pipeline(
    "automatic-speech-recognition",
    model=STT_MODEL,
    device=-1,
    dtype=torch.float32,
)


print(
    "STT model loaded."
)


# =========================================================
# Record Audio
# =========================================================

def record_audio(
    duration=RECORD_SECONDS,
):

    print(
        f"Recording for {duration} seconds..."
    )


    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )


    sd.wait()


    path = os.path.join(
        RECORDINGS_DIR,
        "input.wav",
    )


    sf.write(
        path,
        audio,
        SAMPLE_RATE,
    )


    return path



# =========================================================
# STT
# =========================================================

def transcribe_audio(
    audio_path,
):

    result = stt(
        audio_path,
        generate_kwargs={
            "language": "english",
            "task": "transcribe",
        },
    )


    return result["text"].strip()



# =========================================================
# TTS
# =========================================================

def generate_speech(
    text,
):

    text = text.strip()


    if not text:
        return None


    output_path = os.path.join(
        RECORDINGS_DIR,
        "answer.wav",
    )


    response = groq_client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=text,
        response_format="wav",
    )


    response.write_to_file(
        output_path
    )


    print(
        f"Audio saved: {output_path}"
    )


    return output_path



# =========================================================
# Full Voice Pipeline
# =========================================================

def voice_query():

    # -------------------------
    # Record
    # -------------------------

    audio_path = record_audio()


    # -------------------------
    # STT
    # -------------------------

    print(
        "Transcribing..."
    )


    text = transcribe_audio(
        audio_path
    )


    print()
    print(
        "User:",
        text
    )


    if not text:

        print(
            "No speech detected."
        )

        return



    # -------------------------
    # RAG + LLM
    # -------------------------

    print()
    print(
        "Running RAG..."
    )


    result = answer_query(
        query=text,
        top_k=5,
    )



    answer = result.get(
        "answer",
        ""
    )


    speech_text = result.get(
        "speech_answer",
        ""
    )


    # NOTE: answer_query() returns this field as "detected_device",
    # not "device" — using the wrong key here silently returns the
    # "unknown" default every single time, no matter what was detected.
    detected_device = result.get(
        "detected_device",
        "unknown"
    )



    print()

    print(
        "Detected device:",
        detected_device
    )


    print()

    print(
        "Answer:"
    )


    print(
        answer
    )



    # -------------------------
    # TTS
    # -------------------------

    if speech_text:

        print()

        print(
            "Speech:"
        )


        print(
            speech_text
        )


        print()

        print(
            "Generating speech..."
        )


        generate_speech(
            speech_text
        )



# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    voice_query()