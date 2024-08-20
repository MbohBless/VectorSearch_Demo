from langchain_groq import ChatGroq
# from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser
from mongo_processor import get_mongodbAtlasEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
import os
import mimetypes
from dotenv import load_dotenv
load_dotenv()

userdata = os.environ

llm = ChatGroq(
    model="gemma2-9b-it",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
vectorstore = get_mongodbAtlasEmbeddings()
# retriever = vectorstore.as_retriever()
def format_docs(docs):
    """Convert Documents to a single string.:"""
    formatted = [
        f"Article Snippet: {doc.page_content}"
        for doc in docs
    ]
    return "\n\n" + "\n\n".join(formatted)


# building the chatprompt


def create_chat_template_aug() -> ChatPromptTemplate:
    """
    The Method is to format the base user prompt and the base response to the chat prompt template

    Args:
        user_prompt (str): The user prompt to be formatted
        base_response (str): The base response to be formatted

    Returns:
        ChatPromptTemplate: The chat prompt template object
    """
    system_prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. \n"
        "Context of query:\n"
        "{context}"
    )

    chat_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}")
        ]
    )
    return chat_template


def generate_non_history_response(
    query: str,
    vectorstore,
    model: str = "default",
    temperature: float = 0.0
):
    GROQ_API_KEY = userdata.get("GROQ_API_KEY")

    model_map = {
        "mixtral": "mixtral-8x7b-32768",
        "gemma": "gemma2-9b-it",
        "llama3-70": "llama3-70b-8192",
        "default": "llama3-8b-8192"
    }

    model_name = model_map.get(model, model_map["default"])

    llm = ChatGroq(
        temperature=temperature,
        model_name=model_name,
        groq_api_key=GROQ_API_KEY
    )

    retriever = vectorstore.as_retriever(search_type="mmr")
    prompt = create_chat_template_aug()
    format = itemgetter("docs") | RunnableLambda(format_docs)

    answer = (
        prompt
        | llm
    )

    rag_chain_with_source = RunnableParallel(
        {"docs": retriever, "input": RunnablePassthrough()}
    ).assign(context=format).assign(answer=answer).pick(["answer", "docs"])

    return rag_chain_with_source.invoke(query)


def generate_response(query: str, model: str = "default", temperature: float = 0.0):
    """
    Generate a response to a query using the retrieval QA model.
    Args:
        query: The query to generate a response to.
        vectorstore: The vector store to use for retrieval.
        model: The model to use for generation.
        temperature: The temperature to use for generation.
    """
    response_chain = generate_non_history_response(
        query=query,
        vectorstore=vectorstore,
        model=model,
        temperature=temperature
    )
    # time.sleep(2)

    return response_chain['answer'].content

