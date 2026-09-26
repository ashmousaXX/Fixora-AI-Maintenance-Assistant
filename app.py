import os
import tempfile
import uuid

import streamlit as st
import torch
from dotenv import load_dotenv
from groq import Groq
from transformers import pipeline

from rag import answer_query


# =========================================================
# Configuration
# =========================================================

load_dotenv()

STT_MODEL = "openai/whisper-large-v3-turbo"

TTS_MODEL = "canopylabs/orpheus-v1-english"
TTS_VOICE = "troy"

RECORDINGS_DIR = "voice_recordings"
os.makedirs(RECORDINGS_DIR, exist_ok=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

USER_AVATAR = "🧑‍🔧"
ASSISTANT_AVATAR = "🛠️"

EXAMPLE_PROMPTS = [
    "The ventilator has a power supply problem",
    "The patient monitor screen is blank",
    "The inspiratory flow transducer is defective",
    "Parts of the display are missing or discolored",
]


# =========================================================
# Page setup
# =========================================================

st.set_page_config(
    page_title="Fixora",
    page_icon="🛠️",
    layout="centered",
    initial_sidebar_state="expanded",
)


# =========================================================
# Custom CSS — ChatGPT-style dark theme
# =========================================================

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    /* ---------- App background ---------- */
    .stApp {
        background-color: #212121;
        color: #ececec;
    }

    /* Hide default Streamlit chrome for a cleaner, app-like look */
    #MainMenu, footer, header {
        visibility: hidden;
    }

    /* Center column, ChatGPT-width content */
    .block-container {
        max-width: 780px;
        padding-top: 1.5rem;
        padding-bottom: 8rem;
        margin: 0 auto;
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background-color: #171717;
        border-right: 1px solid #2f2f2f;
    }
    [data-testid="stSidebar"] * {
        color: #ececec !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-weight: 600;
    }

    /* ---------- Welcome / empty state ---------- */
    .welcome-wrap {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 3.5rem 1rem 2rem 1rem;
    }
    .welcome-wrap .icon {
        font-size: 2.6rem;
        margin-bottom: 0.6rem;
    }
    .welcome-wrap h2 {
        color: #ececec;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .welcome-wrap p {
        color: #9b9b9b;
        font-size: 0.95rem;
        max-width: 460px;
    }

    /* Suggestion chip buttons */
    div[data-testid="column"] .stButton button {
        background-color: #2a2a2a;
        color: #ececec;
        border: 1px solid #3a3a3a;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        text-align: left;
        font-size: 0.88rem;
        font-weight: 400;
        width: 100%;
        height: auto;
        white-space: normal;
        transition: background-color 0.15s ease, border-color 0.15s ease;
    }
    div[data-testid="column"] .stButton button:hover {
        background-color: #343434;
        border-color: #4d4d4d;
        color: #ececec;
    }

    /* ---------- Chat messages (row style, not bubbles) ---------- */
    [data-testid="stChatMessage"] {
        background-color: transparent;
        padding: 1.1rem 0.2rem;
        border-bottom: 1px solid #2a2a2a;
        gap: 0.9rem;
    }
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"] {
        background-color: #2f2f2f;
        border-radius: 50%;
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li {
        color: #ececec;
        font-size: 0.95rem;
        line-height: 1.65;
    }
    [data-testid="stChatMessage"] strong {
        color: #ffffff;
    }
    [data-testid="stChatMessage"] code {
        background-color: #2f2f2f;
        border-radius: 4px;
        padding: 0.1rem 0.35rem;
    }

    /* Device badge */
    .device-badge {
        display: inline-block;
        margin-top: 0.4rem;
        padding: 0.2rem 0.65rem;
        background-color: #2a3a2f;
        color: #7fd99a;
        border: 1px solid #375a41;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 500;
    }

    /* ---------- Chat input (pill, fixed at bottom) ---------- */
    [data-testid="stChatInput"] {
        background-color: #212121;
    }
    [data-testid="stChatInput"] textarea {
        background-color: #2f2f2f !important;
        color: #ececec !important;
        border: 1px solid #4d4d4d !important;
        border-radius: 26px !important;
        padding: 0.85rem 1.2rem !important;
        font-size: 0.95rem !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #8e8e8e !important;
    }
    [data-testid="stChatInput"] button {
        background-color: #ececec !important;
        border-radius: 50% !important;
    }

    /* ---------- Generic buttons (sidebar etc.) ---------- */
    .stButton button {
        background-color: #2a2a2a;
        color: #ececec;
        border: 1px solid #3a3a3a;
        border-radius: 10px;
        font-weight: 500;
    }
    .stButton button:hover {
        background-color: #3a3a3a;
        border-color: #4d4d4d;
        color: #ffffff;
    }

    /* Audio player container */
    audio {
        margin-top: 0.6rem;
        height: 34px;
        width: 100%;
        border-radius: 8px;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-thumb {
        background-color: #4d4d4d;
        border-radius: 8px;
    }
    ::-webkit-scrollbar-track {
        background-color: #212121;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Cached resources (loaded once per server process)
# =========================================================

@st.cache_resource(show_spinner="Loading speech recognition model (first run only)...")
def load_stt_pipeline():
    return pipeline(
        "automatic-speech-recognition",
        model=STT_MODEL,
        device=-1,
        dtype=torch.float32,
    )


@st.cache_resource(show_spinner=False)
def load_groq_client():
    if not GROQ_API_KEY:
        return None
    return Groq(api_key=GROQ_API_KEY)


groq_client = load_groq_client()

if groq_client is None:
    st.error(
        "GROQ_API_KEY was not found in .env. Add it before using Fixora."
    )
    st.stop()


# =========================================================
# Helpers
# =========================================================

def transcribe_audio_bytes(audio_bytes):
    stt = load_stt_pipeline()

    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False,
    ) as temp_file:
        temp_file.write(audio_bytes)
        temp_path = temp_file.name

    try:
        result = stt(
            temp_path,
            generate_kwargs={
                "language": "english",
                "task": "transcribe",
            },
        )
        return result["text"].strip()
    finally:
        os.remove(temp_path)


def synthesize_speech(text):
    text = text.strip()

    if not text:
        return None

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

    return output_path


def render_message(role, text, device=None, audio_path=None):
    avatar = USER_AVATAR if role == "user" else ASSISTANT_AVATAR

    with st.chat_message(role, avatar=avatar):
        st.markdown(text)

        if role == "assistant" and device:
            st.markdown(
                f'<span class="device-badge">🔧 {device}</span>',
                unsafe_allow_html=True,
            )

        if audio_path and os.path.exists(audio_path):
            st.audio(audio_path)


def handle_query(query_text):
    st.session_state.messages.append(
        {"role": "user", "text": query_text}
    )
    render_message("user", query_text)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.spinner("Checking the service manuals..."):
            result = answer_query(
                query=query_text,
                top_k=8,
            )

        answer = result.get("answer", "")
        speech_text = result.get("speech_answer", "")
        device = result.get("detected_device")

        st.markdown(answer)

        if device:
            st.markdown(
                f'<span class="device-badge">🔧 {device}</span>',
                unsafe_allow_html=True,
            )

        audio_path = None

        if speech_text:
            try:
                with st.spinner("Generating voice reply..."):
                    audio_path = synthesize_speech(speech_text)
                st.audio(audio_path)
            except Exception as error:
                st.warning(f"Voice reply unavailable: {error}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "text": answer,
            "device": device,
            "audio_path": audio_path,
        }
    )


# =========================================================
# Session state
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:
    st.markdown("### 🛠️ Fixora")

    if st.button("➕ New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.markdown("#### 🎤 Ask by voice")

    audio_value = st.audio_input("Record your question", label_visibility="collapsed")

    if audio_value is not None:
        if st.button("Send voice message", use_container_width=True):
            audio_bytes = audio_value.getvalue()

            with st.spinner("Transcribing..."):
                transcribed = transcribe_audio_bytes(audio_bytes)

            if transcribed:
                handle_query(transcribed)
                st.rerun()
            else:
                st.warning("No speech detected — try again.")

    st.divider()
    st.caption("Fixora — AI Maintenance Assistant")
    st.caption("For medical & diagnostic equipment service manuals.")


# =========================================================
# Main chat area
# =========================================================

if not st.session_state.messages:

    st.markdown(
        """
        <div class="welcome-wrap">
            <div class="icon">🛠️</div>
            <h2>How can I help you today?</h2>
            <p>Describe a fault or symptom on a ventilator, patient monitor,
            or other supported device, and Fixora will search the service
            manuals for you.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(2)

    for index, prompt in enumerate(EXAMPLE_PROMPTS):
        with cols[index % 2]:
            if st.button(prompt, key=f"example_{index}", use_container_width=True):
                handle_query(prompt)
                st.rerun()

else:
    for message in st.session_state.messages:
        render_message(
            message["role"],
            message["text"],
            device=message.get("device"),
            audio_path=message.get("audio_path"),
        )

text_query = st.chat_input("Describe the problem...")

if text_query:
    handle_query(text_query)
    st.rerun()