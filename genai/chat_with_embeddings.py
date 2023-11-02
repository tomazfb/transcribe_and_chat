from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders.base import BaseLoader
from langchain.text_splitter import BaseDocumentTransformer
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders.text import TextLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import DataFrameLoader
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import pandas as pd
from typing import List
import openai

class ChatWithEmbeddings:
    @staticmethod
    def create_text_loader(path: str) -> BaseLoader:
        return TextLoader(path)

    @staticmethod
    def create_unstructured_file_loader(path: str) -> BaseLoader:
        return UnstructuredFileLoader(path)

    @staticmethod
    def create_unstructured_excel_loader(path: str) -> BaseLoader:
        return UnstructuredExcelLoader(path)

    @staticmethod
    def create_csv_loader(path: str) -> BaseLoader:
        return CSVLoader(file_path=path)

    @staticmethod
    def create_excel_loader(path: str, page_content_column : str = None) -> BaseLoader:
        #read file with pandas
        df = pd.read_excel(path)

        #create loader
        return ChatWithEmbeddings.create_dataframe_loader(df, page_content_column=page_content_column)

    @staticmethod
    def create_dataframe_loader(df: pd.DataFrame, page_content_column : str = None) -> BaseLoader:
        if not page_content_column:
            page_content_column = df.columns[0]

        return DataFrameLoader(df, page_content_column=page_content_column)

    @staticmethod
    def create_recursive_character_text_splitter() -> BaseDocumentTransformer:
        return RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0
        )


    @staticmethod
    def obter_modelos() -> List[str]:
        return ["gpt-3.5-turbo", "gpt-4", "gpt-4-32k"]

    def __init__(self, document_loader: BaseLoader, document_transformer: BaseDocumentTransformer = None) -> None:
        self.__document_loader = document_loader
        if document_transformer:
            self.__document_transformer = document_transformer
        else:
            self.__document_transformer = ChatWithEmbeddings.create_recursive_character_text_splitter()
        self.memory = ConversationBufferMemory()
        self.__vectordb = None
        self.__retrievalQA = None

    def chat(self, prompt : str, model : str = "gpt-3.5-turbo") -> str :
        if not self.__retrievalQA:
            #load data
            data = self.__document_loader.load()

            # split
            splits = self.__document_transformer.transform_documents(data)

            # VectorDB
            embedding = OpenAIEmbeddings(openai_api_key=openai.api_key)
            self.__vectordb = Chroma.from_documents(documents=splits, embedding=embedding)

            llm = ChatOpenAI(model=model, openai_api_key=openai.api_key)

            query_retriever = MultiQueryRetriever.from_llm(
                retriever=self.__vectordb.as_retriever(), llm=llm
            )

            # RetrievalQA
            self.__retrievalQA = RetrievalQA.from_llm(llm=llm, retriever=query_retriever, memory=self.memory)
        
        return self.__retrievalQA(prompt)


