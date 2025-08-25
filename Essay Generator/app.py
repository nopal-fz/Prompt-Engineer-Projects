from flask import Flask, render_template, request, jsonify, send_file
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from io import BytesIO
from fpdf import FPDF
import os
import json

app = Flask(__name__)

# Inisialisasi model
llm = OllamaLLM(model='llama3', temperature=0.7)

# Setup prompt template
def create_chat_prompt(style, length):
    return ChatPromptTemplate.from_messages([
        ("system", 
         f"""You are an expert essay writer. Write a detailed essay in Bahasa Indonesia.
         The essay should have a {style.lower()} style and {length.lower()} length.
         Make sure to include an introduction, body paragraphs, and a conclusion.
         The essay should be clear, engaging, and suitable for academic or student purposes."""
        ),
        ("human", "{user}")
    ])

# Fungsi PDF Generator
def generate_pdf(text, title="Essay"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, title, ln=True, align="C")
    pdf.ln(10)
    
    # Content
    pdf.set_font("Arial", size=12)
    for line in text.split("\n"):
        if line.strip():
            pdf.multi_cell(0, 8, line.encode('latin-1', 'replace').decode('latin-1'))
            pdf.ln(2)

    # Output as bytes
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    buffer = BytesIO(pdf_bytes)
    buffer.seek(0)
    return buffer

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_essay():
    try:
        data = request.get_json()
        user_prompt = data.get('prompt', '').strip()
        essay_style = data.get('style', 'Akademik')
        essay_length = data.get('length', 'Menengah (~300 kata)')
        
        if not user_prompt:
            return jsonify({'error': 'Topik tidak boleh kosong'}), 400
        
        # Generate essay
        chat_prompt = create_chat_prompt(essay_style, essay_length)
        essay_chain = chat_prompt | llm
        response = essay_chain.invoke({"user": user_prompt})
        
        return jsonify({
            'success': True,
            'essay': response,
            'topic': user_prompt
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download-pdf', methods=['POST'])
def download_pdf():
    try:
        data = request.get_json()
        essay_text = data.get('essay', '')
        topic = data.get('topic', 'Essay')
        
        if not essay_text:
            return jsonify({'error': 'Tidak ada konten untuk diunduh'}), 400
        
        # Generate PDF
        pdf_buffer = generate_pdf(essay_text, topic)
        
        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=f'essay_{topic[:30]}.pdf',
            mimetype='application/pdf'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)