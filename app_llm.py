
import streamlit as st
import time
import llm_processor
def response_stream_simulator(query):
    response = llm_processor.generate_response(query)
    for word in response.split():
        yield word + " "
        time.sleep(0.05)
        
st.title("Vector Retrival with MongoDB vector search")
if "messages" not in st.session_state:
    st.session_state.messages = []
    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
if prompt := st.chat_input("What do you need to find"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        response = st.write(
            response_stream_simulator(prompt)
        )
        
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
    
    