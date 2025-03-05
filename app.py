import streamlit as st
import uuid
import os
# Remove this since you won't need a hard-coded import anymore:
# from config import pinecone_key
from openai import OpenAI
from pinecone import Pinecone
import PyPDF2
import re
from sentence_transformers import SentenceTransformer

# We no longer define PINECONE_API_KEY = pinecone_key, since it will come from the user

@st.cache_resource
def init():
    print("Loading model and connecting to Pinecone only once...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Now we get Pinecone key from st.session_state
    pinecone_api_key = st.session_state["pinecone_api_key"]
    pc = Pinecone(api_key=pinecone_api_key)
    INDEX_NAME = "testingindex"  # Change to your actual index name
    index = pc.Index(INDEX_NAME)

    return model, index

def extract_text_from_pdf(file) -> str:
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=500):
    text = text.replace("\n", " ")
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end
    return chunks

def get_embedding(text: str, sentence_model) -> list[float]:
    return sentence_model.encode(text, convert_to_numpy=True).tolist()

def upsert_pdf_chunks_to_pinecone(user_id, pdf_file, index, model):
    full_text = extract_text_from_pdf(pdf_file)
    chunks = chunk_text(full_text, chunk_size=500)

    vectors_to_upsert = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk, model)
        vector_id = f"{user_id}_{i}"
        metadata = {"user_id": user_id, "chunk_index": i, "text": chunk}
        vectors_to_upsert.append((vector_id, embedding, metadata))
    
    index.upsert(vectors=vectors_to_upsert)

def query_pinecone(user_id, query_text, index, model, top_k=3):
    query_embedding = get_embedding(query_text, model)
    result = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter={"user_id": {"$eq": user_id}}
    )
    return result

def create_context(matches):
    if not matches:
        return ""
    context_texts = [m['metadata']['text'] for m in matches]
    return "\n\n".join(context_texts)

def ask_openai_with_context(user_query, context, chat_history, openai_api_key):
    system_prompt = f"""
    You are a helpful assistant.
    Use the following context to answer the user's question. If the answer is available in the context, provide it.
    Context : (
    {context})

    If the answer is not in the context, say:
    "I'm sorry, but I don't have that information."
    """

    client = OpenAI(api_key=openai_api_key)
    messages = [{"role": "system", "content": system_prompt}]
    for chat in chat_history:
        messages.append({"role": "user", "content": chat["query"]})
        messages.append({"role": "assistant", "content": chat["response"]})
    messages.append({"role": "user", "content": user_query})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7
    )
    return response.choices[0].message.content

def chat_init(user_query, user_id, index, model, openai_api_key):
    pinecone_results = query_pinecone(user_id, user_query, index, model, top_k=3)
    print("Pinecone result : ", pinecone_results)
    top_chunks = pinecone_results.get("matches", [])
    context = create_context(top_chunks)
    chat_history = st.session_state["chat_history"]
    print("Chat history : ",chat_history)
    answer = ask_openai_with_context(user_query, context, chat_history, openai_api_key)
    return answer

def main():
    st.title("PDF Chatbot with Pinecone & OpenAI")

    # 1) Prompt for OpenAI Key
    if "openai_api_key" not in st.session_state:
        st.subheader("Enter your OpenAI API key to begin")
        user_key = st.text_input("OpenAI API Key", type="password")
        
        if st.button("Submit OpenAI Key"):
            if user_key.strip():
                st.session_state["openai_api_key"] = user_key.strip()
                st.rerun()
            else:
                st.warning("Please provide a valid OpenAI API Key.")
        st.stop()

    # 2) Prompt for Pinecone Key
    if "pinecone_api_key" not in st.session_state:
        st.subheader("Enter your Pinecone API key to begin")
        pinecone_key_input = st.text_input("Pinecone API Key", type="password")

        if st.button("Submit Pinecone Key"):
            if pinecone_key_input.strip():
                st.session_state["pinecone_api_key"] = pinecone_key_input.strip()
                st.rerun()
            else:
                st.warning("Please provide a valid Pinecone API Key.")
        st.stop()

    # 3) Init Model & Index
    model, index = init()
    openai_api_key = st.session_state["openai_api_key"]

    # Assign or retrieve a unique user ID
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = str(uuid.uuid4())
    user_id = st.session_state["user_id"]

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    st.write(f"Your user ID is: **{user_id}**")

    # PDF Upload
    st.header("Upload Your PDF")
    uploaded_pdf = st.file_uploader("Choose a PDF file", type=["pdf"])
    if uploaded_pdf is not None:
        upsert_pdf_chunks_to_pinecone(user_id, uploaded_pdf, index, model)
        st.success("Your PDF has been indexed successfully!")

    # Display chat history
    st.header("Chat History")
    for chat in st.session_state["chat_history"]:
        st.markdown(f"**You:** {chat['query']}")
        st.markdown(f"**Bot:** {chat['response']}")
        st.markdown("---")

    # Chat Interface
    st.header("Ask a Question About Your PDF")
    user_query = st.text_input("Type your question here...")

    if st.button("Send"):
        if user_query.strip():
            answer = chat_init(user_query, user_id, index, model, openai_api_key)
            st.session_state["chat_history"].append(
                {"query": user_query, "response": answer}
            )
            st.rerun()
        else:
            st.warning("Please enter a question first.")

if __name__ == "__main__":
    main()
