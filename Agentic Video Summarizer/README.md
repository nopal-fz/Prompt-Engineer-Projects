# Agentic Video Summarizer 📽️🧠

Agentic Video Summarizer is a Streamlit-based application that allows users to upload videos and receive AI-generated summaries with the help of intelligent agents.

## 🚀 Features
- 📂 **Upload Video**: Supports `.mp4`, `.mov`, and `.avi` formats.
- 🤖 **AI Agent**: Uses **Gemini (Google AI)** to analyze and summarize video content.
- 🔎 **External Search**: Leverages **DuckDuckGo** to enrich insights with additional web searches.
- 📝 **Contextual Summarization**: Answers user questions based on uploaded video content.

## 🛠️ Installation
Make sure you have **Python 3.8+** and **pip** installed on your system.

```sh
# Clone repository
git clone https://github.com/username/agentic-video-summarizer.git
cd agentic-video-summarizer

# Create virtual environment
python -m venv env
source env/bin/activate # For macOS/Linux
env\Scripts\activate # For Windows

# Install dependencies
pip install -r requirements.txt
```

## 🎬 Usage
Run the application with the following command:

```sh
streamlit run app.py
```

Then, open a browser and access `http://localhost:8501/`.

## 📦 Dependencies
- `streamlit`
- `phi`
- `google-generativeai`
- `duckduckgo_search`
- `dotenv`

Make sure all dependencies are installed with:
```sh
pip install -r requirements.txt
```

## 🛠️ AI Agent Configuration
This app uses `phi.Agent` with the **Gemini** model from Google AI:
```python
multimodal_agent = Agent(
name="Video AI Summarizer",
model=Gemini(id="gemini-2.0-flash-exp"),
tools=[DuckDuckGo()],
markdown=True,
)
```
This agent analyzes videos and provides informative summaries, and complements insights with web searches.

## 🛠️ Project Structure
```
agentic-video-summarizer/
│── app.py # Main Streamlit app
│── requirements.txt # Dependencies
│── README.md # Documentation
```

## 📜 License
This project is licensed under the **MIT License**. You are free to use and modify this project as needed.

---
**Made with ❤️ by Naufal Faiz**
