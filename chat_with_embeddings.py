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

class ChatWithEmbeddings:
    @staticmethod
    def create_text_loader(path: str) -> BaseLoader:
        return TextLoader(path)

    @staticmethod
    def create_unstructured_file_loader(path: str) -> BaseLoader:
        return UnstructuredFileLoader(path)

    @staticmethod
    def create_recursive_character_text_splitter() -> BaseDocumentTransformer:
        return RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0
        )

    def __init__(self, document_loader: BaseLoader, document_transformer: BaseDocumentTransformer = create_recursive_character_text_splitter()) -> None:
        self.document_loader = document_loader
        self.document_transformer = document_transformer
        self.memory = ConversationBufferMemory()

    def chat(self, prompt):
        #load data
        data = self.document_loader.load()

        # split
        splits = self.document_transformer.transform_documents(data)

        # VectorDB
        embedding = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(documents=splits, embedding=embedding)

        llm = ChatOpenAI(model="gpt-3.5-turbo")
        retriever = MultiQueryRetriever.from_llm(
            retriever=vectordb.as_retriever(), llm=llm
        )

        # RetrievalQA
        retrievalQA = RetrievalQA.from_llm(llm=llm, retriever=retriever, memory=self.memory)
        
        return retrievalQA(prompt)


