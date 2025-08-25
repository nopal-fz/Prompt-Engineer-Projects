from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import os
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from docx import Document
from werkzeug.utils import secure_filename
import tempfile

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize Ollama LLM with Llama3
def get_llm():
    """Initialize the Ollama LLM with Llama3"""
    return OllamaLLM(
        model="llama3:latest",
        base_url="http://localhost:11434",
        temperature=0.8,
        num_predict=1000
    )

def read_txt_file(file_path):
    """Read text file content"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def read_docx_file(file_path):
    """Read docx file content"""
    doc = Document(file_path)
    return '\n'.join([para.text for para in doc.paragraphs])

def generate_roasting_with_ollama(content):
    """Generate roasting using Ollama with LangChain"""
    llm = get_llm()
    
    roast_prompt = ChatPromptTemplate.from_template("""
    Instruksi: Berikan roast sarkastik terhadap cover letter berikut. Gunakan metafora dan perumpamaan yang tajam untuk mengkritik kekurangan dan kekurangan dalam surat lamaran. Buat dalam bentuk paragraf dan tambahkan beberapa saran untuk cover letter tersebut.

    Cover Letter:
    {content}
    
    Berikan roasting yang membangun dengan gaya sarkastik yang cerdas dan humoris.
    """)
    
    roast_chain = roast_prompt | llm
    
    try:
        response = roast_chain.invoke({"content": content})
        return response.strip()
    except Exception as e:
        raise Exception(f"Gagal menghasilkan roasting: {str(e)}")

def process_file(file_path, filename):
    """Process uploaded file and generate roasting"""
    file_type = filename.split('.')[-1].lower()
    
    try:
        if file_type == 'txt':
            content = read_txt_file(file_path)
        elif file_type == 'docx':
            content = read_docx_file(file_path)
        else:
            return None, "Unsupported file type. Please upload .txt or .docx files only."
        
        if not content.strip():
            return None, "File is empty. Please upload a file with content."
        
        roasting = generate_roasting_with_ollama(content)
        return roasting, None
        
    except Exception as e:
        return None, f"Error processing file: {str(e)}"

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        file_ext = filename.split('.')[-1].lower()
        
        if file_ext not in ['txt', 'docx']:
            return jsonify({'error': 'Only .txt and .docx files are allowed'}), 400
        
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}') as tmp_file:
            file.save(tmp_file.name)
            tmp_file_path = tmp_file.name
        
        try:
            roasting, error = process_file(tmp_file_path, filename)
            
            # Clean up temp file
            os.unlink(tmp_file_path)
            
            if error:
                return jsonify({'error': error}), 400
            
            return jsonify({
                'success': True,
                'roasting': roasting,
                'filename': filename
            })
            
        except Exception as e:
            # Clean up temp file
            os.unlink(tmp_file_path)
            return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        # Test Ollama connection
        llm = get_llm()
        test_response = llm.invoke("Test")
        return jsonify({'status': 'healthy', 'ollama': 'connected'})
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)