import streamlit as st

from langchain_community.document_loaders import SeleniumURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""
model_name = "llama3"
embeddings_model_name = "nomic-embed-text"

# Initialize session state untuk chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    embeddings = OllamaEmbeddings(model=embeddings_model_name)
    st.session_state.vector_store = InMemoryVectorStore(embeddings)

if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

model = OllamaLLM(model=model_name)

def load_page(url):
    loader = SeleniumURLLoader(
        urls=[url]
    )
    documents = loader.load()
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    data = text_splitter.split_documents(documents)
    return data

def index_docs(documents):
    st.session_state.vector_store.add_documents(documents)

def retrieve_docs(query):
    return st.session_state.vector_store.similarity_search(query)

def answer_question(question, context):
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})

st.title("AI Crawler")

# URL input dan load documents
url = st.text_input("Enter URL:")

if url and not st.session_state.documents_loaded:
    with st.spinner("Loading and processing documents..."):
        try:
            documents = load_page(url)
            chunked_documents = split_text(documents)
            index_docs(chunked_documents)
            st.session_state.documents_loaded = True
            st.success("Documents loaded successfully!")
        except Exception as e:
            st.error(f"Error loading documents: {str(e)}")

# Tombol untuk clear chat history
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

with col2:
    if st.button("Load New URL"):
        st.session_state.documents_loaded = False
        st.session_state.messages = []
        # Reset vector store
        embeddings = OllamaEmbeddings(model=embeddings_model_name)
        st.session_state.vector_store = InMemoryVectorStore(embeddings)
        st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
question = st.chat_input("Ask a question about the loaded document...")

if question and st.session_state.documents_loaded:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})
    
    # Display user message
    with st.chat_message("user"):
        st.write(question)
    
    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                retrieve_documents = retrieve_docs(question)
                context = "\n\n".join([doc.page_content for doc in retrieve_documents])
                answer = answer_question(question, context)
                st.write(answer)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_msg = f"Error generating answer: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

elif question and not st.session_state.documents_loaded:
    st.warning("Please enter a URL and load documents first!")