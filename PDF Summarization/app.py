import streamlit as st
from transformers import BertTokenizer, EncoderDecoderModel, EncoderDecoderConfig
import PyPDF2

model_ckpt = 'ardavey/bert2gpt-indosum'
tokenizer = BertTokenizer.from_pretrained(model_ckpt)
tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token

config = EncoderDecoderConfig.from_pretrained(model_ckpt)
config.early_stopping = True

model = EncoderDecoderModel.from_pretrained(model_ckpt, config=config)

def extract_text_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def summarize_text(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    summary_ids = model.generate(
        inputs['input_ids'],
        max_length=150,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

st.title('Text Summarization Pdf')

option = st.selectbox('Select input type', ('PDF', 'Text'))

if option == 'PDF':
    uploaded_file = st.file_uploader('Upload PDF files', type='pdf')
    if uploaded_file is not None:
        pdf_text = extract_text_pdf(uploaded_file)
        st.header('Text extracted from PDF:')
        st.write(pdf_text)
        
        if st.button('Compact PDF'):
            summary = summarize_text(pdf_text)
            st.header('Summary Result:')
            st.write(summary)

elif option == 'Text':
    text_input = st.text_area('Enter the text you want to summary')
    if st.button('Concise text'):
        if len(text_input) > 0:
            summary = summarize_text(text_input)
            st.header('Summary Result:')
            st.write(summary)
        else:
            st.warning('Enter the text first!')
