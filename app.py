import os
import tempfile

import streamlit as st

from rag import answer_query
from Voice import transcribe_audio, generate_speech


st.set_page_config(page_title="Fixora — AI Maintenance Assistant", page_icon="🔧")

st.title("🔧 Fixora — Voice-Enabled Maintenance Assistant")
st.write(
    "Record your question about a device malfunction. Fixora will search "
    "the service manuals, answer, and read the answer back to you."
)

audio_value = st.audio_input("Record your question")

if audio_value is not None:

    # Save the browser-recorded audio to a temp file so transcribe_audio()
    # (which expects a file path) can read it.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_value.read())
        input_audio_path = tmp_file.name

    try:
        with st.spinner("Transcribing..."):
            question_text = transcribe_audio(input_audio_path)

        st.markdown(f"**You asked:** {question_text}")

        if not question_text.strip():
            st.warning("No speech detected. Please try recording again.")
        else:
            with st.spinner("Searching manuals and generating answer..."):
                result = answer_query(query=question_text, top_k=5)

            detected_device = result.get("detected_device") or "unknown"
            st.caption(f"Detected device: {detected_device}")

            st.markdown("### Answer")
            st.markdown(result["answer"])

            speech_text = result.get("speech_answer", "")
            if speech_text:
                with st.spinner("Generating speech..."):
                    output_audio_path = generate_speech(speech_text)
                if output_audio_path:
                    st.audio(output_audio_path)
    finally:
        os.unlink(input_audio_path)