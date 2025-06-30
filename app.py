# app.py

# Import necessary libraries
from flask import Flask, request, jsonify
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline
import os

# --- CORRECTED IMPORT ---
# We must import CORS from the flask_cors library specifically.
from flask_cors import CORS

# Initialize the Flask application
app = Flask(__name__)

# This now correctly applies the CORS rules to your app,
# allowing it to handle preflight requests from the browser.
CORS(app)

# --- Global Variables ---
vector_store = None
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")


@app.route('/submit', methods=['POST'])
def submit_document():
    """
    API Endpoint for submitting a document.
    """
    global vector_store

    data = request.get_json()
    if 'text' not in data or not data['text'].strip():
        return jsonify({"error": "No text provided"}), 400
    
    document_text = data['text']

    try:
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(document_text)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)

        return jsonify({"message": "Document processed successfully!"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/ask', methods=['POST'])
def ask_question():
    """
    API Endpoint for asking a question.
    """
    global vector_store
    global qa_pipeline

    if vector_store is None:
        return jsonify({"error": "No document has been submitted yet. Please submit a document first."}), 400

    data = request.get_json()
    if 'question' not in data or not data['question'].strip():
        return jsonify({"error": "No question provided"}), 400
    
    question = data['question']

    try:
        docs = vector_store.similarity_search(question, k=1)
        if not docs:
            return jsonify({"answer": "Sorry, I couldn't find any relevant information in the document."})

        context = docs[0].page_content
        result = qa_pipeline(question=question, context=context)
        bot_response = result['answer']
        
        return jsonify({"answer": bot_response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)