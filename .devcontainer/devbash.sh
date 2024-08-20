#!/bin/bash 

# This script is run when the container is started. It is used to set up the environment for the user.

touch .env

# Set up the environment
while [[ -z "$DATABASE_URI" ]]; do
    echo "Enter the database URI: "
    read DATABASE_URI
    if [[ -z "$DATABASE_URI" ]]; then
        echo "Database URI cannot be empty."
    fi
done

while [[ -z "$TARGET_COLLECTION" ]]; do
    echo "Enter the target collection: "
    read TARGET_COLLECTION
    if [[ -z "$TARGET_COLLECTION" ]]; then
        echo "Target collection cannot be empty."
    fi
done

while [[ -z "$DATABASE_NAME" ]]; do
    echo "Enter the database name: "
    read DATABASE_NAME
    if [[ -z "$DATABASE_NAME" ]]; then
        echo "Database name cannot be empty."
    fi
done

while [[ -z "$EMBEDDING_SERVICE_URL" ]]; do
    echo "Enter the embedding service URL: "
    read EMBEDDING_SERVICE_URL
    if [[ -z "$EMBEDDING_SERVICE_URL" ]]; then
        echo "Embedding service URL cannot be empty."
    fi
done

echo "Enter the value for the Huggingface token:"
read HF_TOKEN

echo "Enter the groq api key:"
read GROQ_API_KEY

echo "DATABASE_URI=$DATABASE_URI" >> .env
echo "TARGET_COLLECTION=$TARGET_COLLECTION" >> .env
echo "DATABASE_NAME=$DATABASE_NAME" >> .env
echo "EMBEDDING_SERVICE_URL=$EMBEDDING_SERVICE_URL" >> .env

if [[ -n "$HF_TOKEN" ]]; then
    echo "HF_TOKEN=$HF_TOKEN" >> .env
fi

if [[ -n "$GROQ_API_KEY" ]]; then
    echo "GROQ_API_KEY=$GROQ_API_KEY" >> .env
fi

echo "Environment set up successfully."