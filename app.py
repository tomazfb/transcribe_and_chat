import streamlit as st
from transcricao import Transcricao
from chat_with_embeddings import ChatWithEmbeddings
import os
import pathlib
import uuid


def show():
    st.title('Transcribe And Chat')

    st.write('')

    uploaded_file = st.file_uploader('Upload a file',
                            help="Upload a file in .wav or .mp3 or a text file in .csv or .txt or even a .xlsx file",
                            type=['wav', 'mp3', 'txt', 'csv', 'xlsx'],
                            accept_multiple_files=False)

    if uploaded_file:

        data = uploaded_file.getvalue()
        
        # create an unique temporary file name based on a UUID
        destination_file_name = uuid.uuid4().hex + uploaded_file.name

        # create ./tmp/ directory if it doesn't exist
        pathlib.Path("./tmp/").mkdir(parents=True, exist_ok=True)

        destination_path = os.path.join("./tmp/", destination_file_name)

        with open(destination_path, "wb") as destination_file:
            destination_file.write(data)
            destination_file.close()

            # save the file with a new temporary unique name
            extension_lowercase = uploaded_file.name.split(".")[-1].lower()

            if extension_lowercase == 'wav' or extension_lowercase == "mp3":
                if st.button("Transcribe it"):
                    t = Transcricao(destination_path)
                    with st.spinner('Transcribing...'):
                        transcription = t.obter_transcricao_audio()
                    st.code(transcription)
                    pass
            elif extension_lowercase == "txt" or extension_lowercase == "csv" or extension_lowercase == "xlsx":
                if st.button("Chat it"):
                    pass

        # remove the temporary file
        os.remove(destination_path)

show()