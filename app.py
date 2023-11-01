import streamlit as st
from transcricao import Transcricao
from chat_with_embeddings import ChatWithEmbeddings
import os
import pathlib
import uuid
from streamlit_extras.stylable_container import stylable_container
from langchain.callbacks import get_openai_callback
from streamlit_chat import message
import dotenv
import pandas as pd
import openai

try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except:
    pass

def show():
    st.title('Transcribe And Chat')

    st.write('')

    dotenv.load_dotenv()

    key = os.getenv("OPENAI_API_KEY")
    if not os.getenv("OPENAI_API_KEY"):
        #show input to user set its own API_KEY
        key = st.text_input("Your OPENAI API KEY")

    if key and len(key) > 5:
        openai.api_key = key
    else:
        st.stop()

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

            transcription = ""

            if extension_lowercase == 'wav' or extension_lowercase == "mp3":
                btn_transcribe = st.button("Transcribe it", key="btn_transcribe")

                if btn_transcribe:
                    t = Transcricao(destination_path)
                    with st.spinner('Transcribing...'):
                        transcription = t.obter_transcricao_audio()
                    with stylable_container(
                        "codeblock",
                        """
                        code {
                            white-space: pre-wrap !important;
                        }
                        """,
                    ):
                        output_code = st.code(transcription, language="markdown")

                        st.download_button("Download", data=transcription, file_name=uploaded_file.name+".txt")

                        with st.expander("Custo:"):
                            st.write(f"US${t.last_transcription_cost:0.3f}")
                    
                    
            elif extension_lowercase == "txt" or extension_lowercase == "csv" or extension_lowercase == "xlsx":
                model = st.selectbox("Model", ChatWithEmbeddings.obter_modelos(), 0)

                if st.session_state.get("chatter") is None:
                    if extension_lowercase.endswith("txt"):
                        loader = ChatWithEmbeddings.create_text_loader(destination_path)
                    elif extension_lowercase.endswith("csv"):
                        df = pd.read_csv(destination_path)
                        # first 10 lines
                        df = df.head(10)
                        st.dataframe(df)
                        st.write("(first 10 lines)")
                        loader = ChatWithEmbeddings.create_csv_loader(destination_path)
                    else: # extension_lowercase.endswith("xlsx"):
                        df = pd.read_excel(destination_path)
                        # first 10 lines
                        df = df.head(10)
                        st.dataframe(df)
                        st.write("(first 10 lines)")
                        loader = ChatWithEmbeddings.create_unstructured_excel_loader(destination_path)
                    st.session_state["chatter"] = ChatWithEmbeddings(loader)

                input = st.chat_input()
                if input:
                    with get_openai_callback() as cb:
                        c = st.session_state["chatter"]
                        c.chat(input, model=model)

                        for msg in c.memory.buffer_as_messages:
                            message(msg.content, is_user=(msg.type=='human'), allow_html=True)
                        with st.expander("Custo:"):
                            st.write(cb)

show()