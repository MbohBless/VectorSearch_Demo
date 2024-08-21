#!/bin/bash

echo "Running the LLM script"
streamlit run app_llm.py &

# Wait for Streamlit to start
sleep 5

# Forward the port using gh command and make it public
gh codespace ports visibility 8502:public -c $CODESPACE_NAME