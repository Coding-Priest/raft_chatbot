import streamlit as st
import PyPDF2
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Constants
MODEL_NAME = "Anirudh6778/t5_fineTuned_RAFT"  # Your Hugging Face model name
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"    # Embedding model
TOP_K = 5                                   # Number of top chunks to retrieve

@st.cache_resource
def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer

@st.cache_resource
def load_model():
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device

@st.cache_resource
def load_embedding_model():
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return model

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return text

def split_text_into_chunks(text, max_length=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_length):
        chunk = ' '.join(words[i:i + max_length])
        chunks.append(chunk)
    return chunks

def compute_embeddings(model, chunks):
    embeddings = model.encode(chunks, convert_to_tensor=True)
    return embeddings

def retrieve_top_k_chunks(query, embedding_model, chunks, chunk_embeddings, k=TOP_K):
    query_embedding = embedding_model.encode([query], convert_to_tensor=True)
    similarities = cosine_similarity(query_embedding.cpu().numpy(), chunk_embeddings.cpu().numpy())[0]
    top_k_indices = similarities.argsort()[-k:][::-1]
    top_k_chunks = [chunks[i] for i in top_k_indices]
    return top_k_chunks

def generate_response(model, tokenizer, device, query, context, max_length=150):
    input_text = f"question: {query} context: {context}"
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=2
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    st.set_page_config(page_title="RAG Chatbot", layout="wide")
    st.title("📄 PDF-based Retrieval-Augmented Generation (RAG) Chatbot")

    # Sidebar for instructions
    with st.sidebar:
        st.header("Instructions")
        st.markdown("""
        1. **Upload a PDF**: Use the file uploader to upload your PDF document.
        2. **Enter Your Query**: Type your question related to the PDF content.
        3. **Get Response**: Click the "Get Response" button to receive an answer.
        """)

    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        with st.spinner("Extracting text from PDF..."):
            text = extract_text_from_pdf(uploaded_file)
        st.success("PDF text extracted successfully!")

        if not text.strip():
            st.error("No text found in the uploaded PDF.")
            return

        # Split into chunks
        chunks = split_text_into_chunks(text)
        st.info(f"Total chunks created: {len(chunks)}")

        # Load embedding model
        embedding_model = load_embedding_model()

        with st.spinner("Computing embeddings for chunks..."):
            chunk_embeddings = compute_embeddings(embedding_model, chunks)
        st.success("Embeddings computed successfully!")

        # User query input
        query = st.text_input("Enter your question:", "")

        if st.button("Get Response") and query:
            with st.spinner("Retrieving relevant chunks..."):
                top_k_chunks = retrieve_top_k_chunks(query, embedding_model, chunks, chunk_embeddings, TOP_K)
                context = " ".join(top_k_chunks)
            st.success("Relevant chunks retrieved!")

            # Load tokenizer and model
            tokenizer = load_tokenizer()
            model, device = load_model()

            with st.spinner("Generating response..."):
                response = generate_response(model, tokenizer, device, query, context)
            st.success("Response generated!")

            st.markdown("### 🤖 Response")
            st.write(response)

if __name__ == "__main__":
    main()
