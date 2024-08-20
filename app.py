import streamlit as st
import random
import time
from mongo_processor import perform_document_retrieval, perform_document_upload

def response_generator(query):
    response = """**I found the following Segments for you:**\n\n"""
    # for word in response.split():
    #     yield word + " "
    #     time.sleep(0.05)
    documents =  perform_document_retrieval(query)
    
    for document in documents:
        response += f"**Relevance Score**:"+str(document["score"])+ "\n\n" +"**Content**:"+ document['text'] +"\n\n"
    return response
st.title("Vector Retrival with MongoDB vector search")
# c = st.container()
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What do you need to find"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write(
            response_generator(prompt)
        )

    st.session_state.messages.append(
        {"role": "assistant", "content": response})
