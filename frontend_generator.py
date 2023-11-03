import uuid
import os
import streamlit as st
import pandas as pd
import pathlib
import urllib
from typing import Self
from streamlit.runtime.uploaded_file_manager import UploadedFile
from abc import ABC, abstractmethod
from genai.chat_with_embeddings import ChatWithEmbeddings
from genai.transcription import Transcription
from streamlit_extras.stylable_container import stylable_container
from streamlit.delta_generator import DeltaGenerator
from langchain.callbacks import get_openai_callback
from streamlit_chat import message
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
from io import BytesIO 
from pydub import AudioSegment
import mimetypes

# create ./tmp/ directory if it doesn't exist
pathlib.Path("./tmp/").mkdir(parents=True, exist_ok=True)

class FrontendGenerator(ABC):
    '''
    Base abstract class and Factory for frontend generators
    '''

    @abstractmethod
    def generate(self) -> None:
        pass

    @staticmethod
    def create(uploaded_file : UploadedFile = None, link : str = None) -> Self:
        if uploaded_file:
            return FileUploadFrontEndGenerator(uploaded_file)
        elif link:
            return UrlFrontEndGenerator(link)
        else:
            raise Exception("Either uploaded_file or link must be provided")

    def st_output_code(self, content : str) -> DeltaGenerator:
        with stylable_container(
            "codeblock",
            """
            code {
                white-space: pre-wrap !important;
            }
            """,
        ):
            output_code = st.code(content, language="markdown")
        
        return output_code

    def st_df(self, df : pd.DataFrame, lines : int = 10):
        # first n lines
        df2 = df.head(lines)
        st.dataframe(df2)
        st.write(f"(first {lines} lines)")

class LocalFileFrontEndGenerator(FrontendGenerator):
    '''
    Frontend generator for local files
    '''
    def __init__(self, path : str) -> None:
        super().__init__()
        self.path = path

    def generate(self) -> None:
        # save the file with a new temporary unique name
        path = self.path
        extension_lowercase = path.split(".")[-1].lower()

        transcription = ""

        if extension_lowercase == 'wav' or extension_lowercase == "mp3":
            btn_transcribe = st.button("Transcribe it", key="btn_transcribe")

            if btn_transcribe:
                t = Transcription(path)
                with st.spinner('Transcribing...'):
                    transcription = t.obter_transcricao_audio()
                
                self.st_output_code(transcription)

                # gets just filename portion from path
                filename = os.path.basename(path) + ".txt"

                st.download_button("Download", data=transcription, file_name=filename)

                with st.expander("Custo:"):
                    st.write(f"US${t.last_transcription_cost:0.3f}")
                
        elif extension_lowercase == "txt" or extension_lowercase == "csv" or extension_lowercase == "xlsx":
            model = st.selectbox("Model", ChatWithEmbeddings.obter_modelos(), 0)

            if st.session_state.get("chatter") is None:
                if extension_lowercase.endswith("txt"):
                    loader = ChatWithEmbeddings.create_text_loader(self.path)
                elif extension_lowercase.endswith("csv"):
                    df = pd.read_csv(self.path)
                    self.st_df(df, 10)
                    loader = ChatWithEmbeddings.create_csv_loader(self.path)
                else: # extension_lowercase.endswith("xlsx"):
                    df = pd.read_excel(self.path)
                    self.st_df(df, 10)
                    loader = ChatWithEmbeddings.create_unstructured_excel_loader(self.path)
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

class FileUploadFrontEndGenerator(LocalFileFrontEndGenerator):
    '''
    Frontend generator for uploaded files
    '''
    def __init__(self, uploaded_file : UploadedFile) -> None:
        self.__uploaded_file = uploaded_file

        # create an unique temporary file name based on a UUID
        destination_file_name = uuid.uuid4().hex + self.__uploaded_file.name
        self.path = os.path.join("./tmp/", destination_file_name)

        with open(self.path, "wb") as destination_file:
            destination_file.write(self.__uploaded_file.getvalue())
            destination_file.close()

class UrlFrontEndGenerator(FrontendGenerator):
    '''
    Frontend generator for a given URL
    '''
    def __init__(self, input : str) -> None:
        self.__input = input
        try:
            urllib.parse.urlparse(input)
        except:
            raise Exception("Invalid URL")
        
        self.__video_id = self.__get_video_id()
        self.__is_youtube = self.__video_id is not None
        if self.__is_youtube:
            self.__transcription_languages = YouTubeTranscriptApi.list_transcripts(self.__video_id)
            """
            # the Transcript object provides metadata properties
            transcript.video_id,
            transcript.language,
            transcript.language_code,
            # whether it has been manually created or generated by YouTube
            transcript.is_generated,
            # a list of languages the transcript can be translated to
            transcript.translation_languages,
            """
        self.__audio_buffer = None
        self.__download_mp3_name = None

    def __get_video_id(self):
        """    
        Examples of URLs:
        Valid:
            'http://youtu.be/_lOT2p_FCvA',
            'www.youtube.com/watch?v=_lOT2p_FCvA&feature=feedu',
            'http://www.youtube.com/embed/_lOT2p_FCvA',
            'http://www.youtube.com/v/_lOT2p_FCvA?version=3&amp;hl=en_US',
            'https://www.youtube.com/watch?v=rTHlyTphWP0&index=6&list=PLjeDyYvG6-40qawYNR4juzvSOg-ezZ2a6',
            'youtube.com/watch?v=_lOT2p_FCvA',
        
        Invalid:
            'youtu.be/watch?v=_lOT2p_FCvA',
        """
        if self.__input.startswith(('youtu', 'www')):
            self.__input = 'http://' + self.__input

        query = urllib.parse.urlparse(self.__input)

        if 'youtube' in query.hostname:
            if query.path == '/watch':
                return urllib.parse.parse_qs(query.query)['v'][0]
            elif query.path.startswith(('/embed/', '/v/', '/shorts/')):
                return query.path.split('/')[2]
        elif 'youtu.be' in query.hostname:
            return query.path[1:]
        # fail?
        return None

    def __get_audio(self) -> None:
        if not self.__audio_buffer:
            with st.spinner("In progress..."):
                yt = YouTube(self.__input)

                prefix = uuid.uuid4().hex + "_" + self.__video_id

                audio = yt.streams.filter(only_audio = True).first()
                mp4_file = audio.download("tmp/")

                self.__download_mp3_name = prefix + ".mp3"
                mp3_path = os.path.join("tmp/", self.__download_mp3_name)

                audio_segment = AudioSegment.from_file(mp4_file)
                audio_segment.export(mp3_path, format="mp3")

                # delete mp4_file
                os.remove(mp4_file)

                # create BytesIO
                self.__audio_buffer = BytesIO()

                # read self__download_mp3_name into __audio_buffer
                with open(mp3_path, "rb") as f:
                    self.__audio_buffer.write(f.read())
                    f.close()
                
                # delete mp3 file
                os.remove(mp3_path)
                

        self.__audio_buffer.seek(0)        

    def generate(self) -> None:
        if self.__is_youtube:
            st.write("It is an Youtube video")
            if self.__transcription_languages:
                tab1, tab2 = st.tabs(["Get Audio", "Get Transcription"])
            else:
                tab1 = st.tabs(["Get Audio"])
                tab2 = None

            with tab1:
                if st.button("Get content mp3 Audio (free)", disabled = self.__audio_buffer is not None):
                    self.__get_audio()
                        
                if self.__download_mp3_name:
                    #download button for file
                    st.download_button(label = f"Click here to Download {self.__video_id}.mp3",
                                    data = self.__audio_buffer,
                                    file_name = self.__download_mp3_name,
                                    mime = "audio/mpeg")
                        
            with tab2:
                if self.__transcription_languages:    
                    lang = st.selectbox(label = "",
                                        options =self.__transcription_languages,
                                        index = 0,
                                        format_func = lambda x: f"{x.language} ({x.language_code})")
                    if st.button("Get Transcription (free)", disabled=not lang):
                        transcription = YouTubeTranscriptApi.get_transcript(self.__video_id, languages=[lang.language_code])
                        self.st_output_code(transcription)
                        st.download_button("Download", data=str(transcription), file_name=self.__video_id+".txt")
        else:
            # create an unique temporary file name based on a UUID
            destination_file_name = uuid.uuid4().hex

            path = os.path.join("./tmp/", destination_file_name)

            # download file from url in local path
            path, readers = urllib.request.urlretrieve(self.__input, path)

            content_type = readers.get_content_type()

            new_path = path + mimetypes.guess_extension(content_type)

            #move path to new_path
            os.rename(path, new_path)

            inner_generator = LocalFileFrontEndGenerator(new_path)

            inner_generator.generate()