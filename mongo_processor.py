from pymongo import MongoClient
from typing import List
from langchain_core.documents import Document
from tqdm.auto import tqdm
from dotenv import load_dotenv
from data_processor import create_embeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings
import os
load_dotenv()

# to mitigate having to change the pattern of code secrets importing add this line below
userdata = os.environ


async def perform_document_upload(documents: List[Document],
                                  database_uri: str = userdata.get(
                                      "DATABASE_URI"),
                                  database_name: str = userdata.get(
                                      "DATABASE_NAME"),
                                  collection_name: str = userdata.get(
                                      "TARGET_COLLECTION")
                                  ) -> None:
    """
    Performs the document upload to the mongodb.
    Args:
      documents: List[Document]
    Returns:
      None
    """
    client = MongoClient(database_uri)
    db = client[database_name]
    collection = db[collection_name]
    document_clean = []
    for document in tqdm(documents, desc="Creating content embeddings"):
        document_clean.append({
            "text": document.page_content,
            "embedding":  create_embeddings(document.page_content),
        }
        )
    collection.insert_many(document_clean)
    client.close()

    return None

def get_mongodbAtlasEmbeddings(db_uri: str = userdata.get("DATABASE_URI"),
                               db_name: str = userdata.get("DATABASE_NAME"),
                               collection_name: str = userdata.get(
                                   "TARGET_COLLECTION"),
                               atlas_vector_search_index: str = "book_index",
                               embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    The Method get the stored vector embeddings from the mongodb atlas database

    Args:
      db_uri (str): The URI of the MongoDB Atlas database
      db_name (str): The name of the MongoDB Atlas database
      collection_name (str): The name of the MongoDB Atlas collection
      atlas_vector_search_index (str): The name of the MongoDB Atlas vector search index
      embedding_model (str, optional): The name of the Hugging Face embedding model. Defaults to "sentence-transformers/all-MiniLM-L6-v2".

    Returns:
      MongoDBAtlasVectorSearch: The vector search object that can be used for similarity search operations
    """
    vector_search = MongoDBAtlasVectorSearch.from_connection_string(
        db_uri,
        db_name + "." + collection_name,
        HuggingFaceEmbeddings(model_name=embedding_model),
        index_name="book_index")
 
    return vector_search


def perform_document_retrieval(query: str,
                                     database_uri: str = userdata.get(
                                         "DATABASE_URI"),
                                     database_name: str = userdata.get(
                                         "DATABASE_NAME"),
                                     collection_name: str = userdata.get(
                                         "TARGET_COLLECTION")
                                     ) -> List[Document]:
    """
    Performs the document retrieval from the mongodb.
    Args:
      query: str
    Returns:
      List[Document]
    """
    client = MongoClient(database_uri)
    db = client[database_name]
    collection = db[collection_name]
    query_embedding =  create_embeddings(query,local=False)
    """
    The aggregation pipeline below is used to find the most similar document to the query
    so your task is to implement the aggregation pipeline that will find the most similar document to the query    
    Ensure that the vector score is projected in the aggregation pipeline
    
    """
    #********************************************
    # Uncomment the code below to implement the aggregation pipeline
    #********************************************    
    # aggregation = [
    #     # code here to implement the aggregation pipeline
    #     # make sure you makes use of the query_embedding variable
    #     # add as many aggregation stages as you need
    # ]

    results = collection.aggregate(aggregation)
    client.close()
    return results
