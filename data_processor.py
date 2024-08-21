from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

from tqdm.auto import tqdm
from langchain_experimental.text_splitter import SemanticChunker
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
import os
import mimetypes
from dotenv import load_dotenv
load_dotenv()
# to mitigate having to change the pattern of code secrets importing add this line below
userdata = os.environ


def document_extraction(path: str) -> List[Document]:
    """
    Handle the document Extraction.
    Args:
      path: str
    Returns:
      List[Document]
    """
    if not os.path.exists(path):
        raise ValueError(f"Path {path} does not exist")
    try:
        mime_type, _ = mimetypes.guess_type(path)
        if mime_type == "application/pdf":
            loader = PyPDFLoader(path)
            documents = loader.load()
            return documents
        else:
            raise ValueError(f"Unsupported file type: {mime_type}")
    except Exception as e:
        raise e


def post_processing(document: Document, semantic: bool = True, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> List[Document]:
    """
    Performs the chunking of the document. if the user prefers semantic chunking, it will be done.
    Args:
      document: Document
      semantic: bool
    Returns:
      List[Document]
    """
    if semantic:
        from langchain_huggingface import HuggingFaceEmbeddings
        text_splitter = SemanticChunker(
            embeddings=HuggingFaceEmbeddings(model_name=model_name),
            breakpoint_threshold_type="percentile"
        )
        documents = text_splitter.split_documents([document])
        return documents
    else:
        """Here because the user might not have access to HF, they may need to implement the use of chunking of the document of recursive chunking techinques of characters"""
        text_splitter = CharacterTextSplitter(
            chunk_size=3000, chunk_overlap=150)
        documents = text_splitter.split_documents([document])
        return documents


def create_embeddings(text: str,
                      local: bool = True,
                      model: str = "sentence-transformers/all-MiniLM-L6-v2") -> List[float]:
    """
    To make sure that everyone can generate embeddings and test their models.
    Args:
      text: str
    Returns:
      List[float]
    """
    if local:
        from langchain.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name=model)
        embedding = embeddings.embed_query(text)
        return embedding
    else:
        import requests
        url = userdata.get("EMBEDDING_SERVICE_URL")
        headers = {
            "Content-Type": "application/json"
        }
        data = {
            "text": text
        }
        response = requests.get(url, headers=headers, json=data)
        embedding = response.json()["embedding"]
        # check if embedding is instance of list of float
        if not isinstance(embedding, list):
            raise ValueError("Embedding is not a list of float")
        return embedding

