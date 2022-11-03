# Requirements yang dibutuhkan ------------------------------------------------------------

import streamlit as st
import requests
import json
import os
from pytube import YouTube
import librosa
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from pydub import AudioSegment

# Mengatur judul dan ikon pada situs Streamlit ------------------------------------------------------------

st.set_page_config(
    page_title="Audio dan Video Transcription App", page_icon="📝", layout="wide"
)

# Mengatur layout pada situs Streamlit -------------------------------------------------


def _max_width_():
    max_width_str = f"max-width: 1200px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )


_max_width_()

# Mengatur menu pada situs Streamlit -------------------------------------------------


def main():
    pages = {
        "🔊 Audio Transcriber": demo,
        # "📹 Video Transcription": vid,
    }

    if "page" not in st.session_state:
        st.session_state.update(
            {
                # Default page
                "page": "Home",
            }
        )

    with st.sidebar:
        page = st.radio("Select your mode", tuple(pages.keys()))

    pages[page]()


# Audio Transcriber -------------------------------------------------

def demo():

    c1, c2, c3 = st.columns([1, 4, 1])
    with c2:

        with st.form(key="my_form"):

            f = st.file_uploader("", type=[".wav"])

            st.info(
                f"""
                        👆 Upload a .wav file.
                        """
            )

            submit_button = st.form_submit_button(label="Transcribe")

    if f is not None:
        path_in = f.name
        # Get file size from buffer
        # Source: https://stackoverflow.com/a/19079887
        old_file_position = f.tell()
        f.seek(0, os.SEEK_END)
        getsize = f.tell()  # os.path.getsize(path_in)
        f.seek(old_file_position, os.SEEK_SET)
        getsize = round((getsize / 1000000), 1)

        if getsize < 2:  # File more than 2MB
            # To read file as bytes:
            bytes_data = f.getvalue()

            # Load your API key from an environment variable or secret management service
            api_token = st.secrets["api_token"]

            # endregion API key
            API_URL = "https://api-inference.huggingface.co/models/indonesian-nlp/wav2vec2-large-xlsr-indonesian"
            headers = {"Authorization": f"Bearer {api_token}"}

            def query(data):
                response = requests.request(
                    "POST", API_URL, headers=headers, data=data)
                return json.loads(response.content.decode("utf-8"))

            data = query(bytes_data)

            values_view = data.values()
            value_iterator = iter(values_view)
            text_value = next(value_iterator)
            text_value = text_value.lower()

            st.success(text_value)

            c0, c1 = st.columns([2, 2])

            with c0:
                st.download_button(
                    "Download the transcription",
                    text_value,
                    file_name=None,
                    mime=None,
                    key=None,
                    help=None,
                    on_click=None,
                    args=None,
                    kwargs=None,
                )

        else:
            st.warning(
                "🚨 The file you uploaded is more than 2MB!"
            )
            st.stop()

    else:
        path_in = None
        st.stop()

# Video Transcriber from Youtube -------------------------------------------------


# def vid():
#    c1, c2, c3 = st.columns([1, 4, 1])
#    with c2:

#        with st.form(key="my_form"):

#            link = st.text_input('YouTube URL')

#            st.info(
#                f"""
#                        👆 Insert YouTube URL.
#                        """
#            )

#            submit_button = st.form_submit_button(label="Transcribe")

#    if link is not None:
#        path_in = link.name
#        audio_path = 'C://'
#        try:
#            yt = YouTube(link)
#        except:
#            st.success('Connection Error!')
#        yt.streams.filter(file_extension='mp4')
#        stream = yt.streams.get_by_itag(139)
#        stream.download(audio_path, "output.mp4")
#        given_audio = AudioSegment.from_file('C://output.mp4', format="mp4")
#        given_audio.export("output.wav", format="wav")
#        sp, rate = sf.read("output.wav")
#        sp = librosa.resample(sp.T, rate, 16000)
#        sf.write("output.wav", sp.T, 16000, subtype='PCM_24')

#        tokenizer = Wav2Vec2Tokenizer.from_pretrained(
#            "indonesian-nlp/wav2vec2-large-xlsr-indonesian")
#        model = Wav2Vec2ForCTC.from_pretrained(
#            "indonesian-nlp/wav2vec2-large-xlsr-indonesian")

#        stream = librosa.stream(
#            "output.wav",
#            block_length=5,
#            frame_length=16000,
#            hop_length=16000
#        )
#        for speech in stream:
#            if len(speech.shape) > 1:
#                speech = speech[:, 0] + speech[:, 1]

#            input_values = tokenizer(speech, return_tensors="pt").input_values
#            logits = model(input_values).logits

#            predicted_ids = torch.argmax(logits, dim=-1)
#            transcription = tokenizer.decode(predicted_ids[0])
#            st.success(transcription)

#            c0, c1 = st.columns([2, 2])

#            with c0:
#                st.download_button(
#                    "Download the transcription",
#                    transcription,
#                    file_name=None,
#                    mime=None,
#                    key=None,
#                    help=None,
#                    on_click=None,
#                    args=None,
#                    kwargs=None,
#                )
#    else:
#        path_in = None
#        st.warning(
#            "🚨 Program Error"
#        )
#        st.stop()

    # Catatan tambahan -------------------------------------------------
with st.expander("ℹ️ - About this app", expanded=False):

    st.write(
        """     
Aplikasi ini hanyalah sebagai model AI yang akan dikembangkan selanjutnya. Diajukan untuk memenuhi salah satu syarat tugas mata kuliah Praktik Kerja Lapangan (PKL)
	    """
    )

    st.markdown("")

if __name__ == "__main__":
    main()
