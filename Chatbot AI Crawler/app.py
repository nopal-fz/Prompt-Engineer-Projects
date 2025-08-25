from flask import Flask, render_template, request, jsonify, session
import os
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

app = Flask(__name__)

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

model_name = "llama3"
embeddings_model_name = "nomic-embed-text"

# Global variables
vector_stores = {}
models = {}

def get_embeddings():
    return OllamaEmbeddings(model=embeddings_model_name)

def get_model():
    return OllamaLLM(model=model_name)

def load_page(url):
    loader = SeleniumURLLoader(urls=[url])
    documents = loader.load()
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

def index_docs(session_id, documents):
    if session_id not in vector_stores:
        embeddings = get_embeddings()
        vector_stores[session_id] = InMemoryVectorStore(embeddings)
    vector_stores[session_id].add_documents(documents)

def retrieve_docs(session_id, query):
    if session_id in vector_stores:
        return vector_stores[session_id].similarity_search(query)
    return []

def answer_question(question, context):
    model = get_model()
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load_url', methods=['POST'])
def load_url():
    try:
        data = request.get_json()
        url = data.get('url')
        
        if not url:
            return jsonify({'success': False, 'message': 'URL is required'})
        
        # Generate session ID if not exists
        if 'session_id' not in session:
            session['session_id'] = os.urandom(16).hex()
        
        session_id = session['session_id']
        
        # Load and process documents
        documents = load_page(url)
        chunked_documents = split_text(documents)
        index_docs(session_id, chunked_documents)
        
        # Initialize chat history
        session['messages'] = []
        session['documents_loaded'] = True
        session['current_url'] = url
        
        return jsonify({
            'success': True, 
            'message': f'Successfully loaded and processed {len(chunked_documents)} document chunks'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error loading URL: {str(e)}'})

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        question = data.get('question')
        
        if not question:
            return jsonify({'success': False, 'message': 'Question is required'})
        
        if 'session_id' not in session or not session.get('documents_loaded'):
            return jsonify({'success': False, 'message': 'Please load a URL first'})
        
        session_id = session['session_id']
        
        # Retrieve relevant documents
        retrieved_docs = retrieve_docs(session_id, question)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Generate answer
        answer = answer_question(question, context)
        
        # Update chat history
        if 'messages' not in session:
            session['messages'] = []
        
        session['messages'].append({'role': 'user', 'content': question})
        session['messages'].append({'role': 'assistant', 'content': answer})
        session.modified = True
        
        return jsonify({
            'success': True,
            'answer': answer,
            'messages': session['messages']
        })
    
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error generating answer: {str(e)}'})

@app.route('/get_messages')
def get_messages():
    messages = session.get('messages', [])
    return jsonify({'messages': messages})

@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    session['messages'] = []
    session.modified = True
    return jsonify({'success': True, 'message': 'Chat history cleared'})

@app.route('/reset_session', methods=['POST'])
def reset_session():
    session_id = session.get('session_id')
    if session_id and session_id in vector_stores:
        del vector_stores[session_id]
    
    session.clear()
    return jsonify({'success': True, 'message': 'Session reset successfully'})

if __name__ == '__main__':
    app.run(debug=True)