from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import streamlit as st
import os

# Load environment variables from .env
load_dotenv()

# Set up necessary keys
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

HF_TOKEN = os.getenv("HF_TOKEN")

# UI
st.title("📸 Caption Generator Bot")
st.markdown("Get a catchy caption for your next social media post!")

# User input
topic = st.text_input("Enter a topic for your post(e.g., beach, coffee, fashion):")

# Use ChatPromptTemplate
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "Generate only ONE short, catchy caption based on the given topic. Keep it fun and engaging.Do NOT include extra labels, tags, or additional words"),
    ("user", "Topic: {topic}")
])

# Hugging Face LLM via LangChain
llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
    huggingfacehub_api_token=HF_TOKEN
)

# Output parser
output_parser = StrOutputParser()

# Chain
chain = chat_prompt | llm | output_parser

# Run the chain if topic is entered
if topic:
    with st.spinner("Generating your caption..."):
        response = chain.invoke({"topic": topic})
        st.success("Here’s your caption:")
        st.write(response.strip())
