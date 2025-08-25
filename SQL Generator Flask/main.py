from flask import Flask, render_template, request, jsonify
import os
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Initialize Ollama with Llama3 model
model = OllamaLLM(model="llama3",
                  temperature=0.8
                  )

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_sql():
    try:
        data = request.get_json()
        text_input = data.get('query', '').strip()
        
        if not text_input:
            return jsonify({'error': 'Masukkan deskripsi Query terlebih dahulu!'}), 400
        
        # Template untuk generate SQL
        sql_template = ChatPromptTemplate.from_template("""
            Buatlah kueri SQL berdasarkan teks berikut. Jika teks tidak relevan, jawab dengan pesan:
            "Error: Input tidak relevan untuk membuat query SQL."

            Teks input: {text_input}

            Saya hanya ingin generate output nya kueri SQL saja, tanpa penjelasan tambahan!
        """)
        
        # Generate SQL query
        sql_chain = sql_template | model
        sql_response = sql_chain.invoke({"text_input": text_input})
        
        if "Error:" in sql_response:
            return jsonify({'error': sql_response}), 400
        
        # Template untuk explain SQL
        explain_template = ChatPromptTemplate.from_template("""
            Jelaskan kueri SQL dibawah ini dengan bahasa yang sederhana dan mudah dipahami:

            SQL Query: {sql_query}

            Berikan penjelasan yang singkat dan jelas tentang apa yang dilakukan query ini!
        """)
        
        # Generate explanation
        explain_chain = explain_template | model
        explanation_response = explain_chain.invoke({"sql_query": sql_response})
        
        return jsonify({
            'sql_query': sql_response,
            'explanation': explanation_response
        })
        
    except Exception as e:
        return jsonify({'error': f'Terjadi kesalahan: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)