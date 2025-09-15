import streamlit as st 
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file,get_file
import google.generativeai as genai
import time
from pathlib import Path
import tempfile
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY=os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

# page config
st.set_page_config(
    page_title="Agentic Video Summarizer",
    page_icon="🧠",
    layout="centered"
)

st.markdown(
    """
    <div style="text-align: center;">
        <h1>Agentic Video Summarizer 📽️</h1>
        <h2>Summarize any video with Agentic AI!</h2>
    </div>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def initialize_agent():
    return Agent(
        name="Video AI Summarizer",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True,
    )

# Initialize the agent
multimodal_Agent=initialize_agent()

# File uploader
video_file = st.file_uploader(
    "Upload a video file", type=['mp4', 'mov', 'avi'], help="Upload a video file to summarize"
)

if video_file:
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.write(video_file.read())
        video_path = temp.name

    st.video(video_path, format="video/mp4", start_time=0)

    user_query = st.text_area(
        "What insight are you seeking from this video?",
        placeholder="Enter a question or topic to summarize the video.",
        help="Provide specific questions or topics to generate a more accurate summary."
    )

    if st.button("🔍 Summarize Video"):
        if not user_query:
            st.warning("Please provide a question or topic to summarize the video.")
        else:
            try:
                with st.spinner("Processing video summary..."):
                    processed_video = upload_file(video_path, mime_type="video/mp4")
                    while processed_video.state.name == "PROCESSING":
                        time.sleep(1)
                        processed_video = get_file(processed_video.name)

                    analysis_prompt = (
                        f"""
                        Analyze the uploaded video for content and context.
                        Respond to the following query using video insights and supplementary web research:
                        "{user_query}"
                        
                        Provide a detailed, user-friendly, and informative summary of the video content.
                        """
                    )

                    response = multimodal_Agent.run(analysis_prompt, videos=[processed_video])

                st.subheader("Summary Results 📝")
                st.markdown(response.content)

            except Exception as error:
                st.error(f"An error occurred: {error}")
            finally:
                Path(video_path).unlink(missing_ok=True)
else:
    st.info("Upload a video file to get started!")