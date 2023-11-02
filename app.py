import streamlit as st
import os
from streamlit_extras.stylable_container import stylable_container
from langchain.callbacks import get_openai_callback
from streamlit_chat import message
import dotenv
import openai
from frontend_generator import FrontendGenerator

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

    if not os.getenv("OPENAI_API_KEY"):
        #show input to user set its own API_KEY
        key = st.text_input("Your OPENAI API KEY")
        if key and len(key) > 5:
            openai.api_key = key
        if not key:
            st.stop()

    linked_url, uploaded_file = None, None

    if not linked_url:
        uploaded_file = st.file_uploader('Upload a file',
                                help="Upload a file in .wav or .mp3 or a text file in .csv or .txt or even a .xlsx file",
                                type=['wav', 'mp3', 'txt', 'csv', 'xlsx'],
                                accept_multiple_files=False)

    if not linked_url and not uploaded_file:
        st.write('... or ...')

    if not uploaded_file:
        linked_url = st.text_input("Inform URL for a file or YouTube Video:")

    if uploaded_file or linked_url:
        generator = FrontendGenerator.create(uploaded_file, linked_url)
        generator.generate()

show()