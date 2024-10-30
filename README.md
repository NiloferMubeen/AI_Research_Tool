# AI_Research_Tool

This project, AI Research Tool, is a streamlined application designed to help users extract insights from online resources. Users can input up to three URLs, which are processed, split into chunks, and stored in a vector database. The application allows for natural language queries, retrieving relevant answers with the help of large language models (LLMs).

## Features
* **Multi-URL Input:** Accepts up to three URLs to analyze.
* **Vector Storage with FAISS:** Efficient storage and retrieval of document embeddings using FAISS.
* **LLM-Based Querying:** Leverages the LLaMA model for accurate, contextual responses to queries.
* **Embedding Generation:** Uses Hugging Face embeddings to represent document content in vector format.
* **Interactive UI:** Powered by Streamlit for a simple, user-friendly interface

## Technologies Used
1. Streamlit
2. Python
3. LangChain
4. Groq
5. Hugging Face
6. FAISS (Facebook AI Similarity Search)

# Setup
### _Prerequisites_
* Python 3.10 
* Required Python packages: See `requirements.txt`

# Installation
### 1. Clone the Repository
```
git clone https://github.com/yourusername/AI-Research-Tool.git
cd AI-Research-Tool
```
### 2. Create a virtual environment and activate
```
.venv/Scripts/activate
```
### 3. Install Required Packages
```
pip install -r requirements.txt
```
### 4. Set API keys
* Sign up at Groq cloud to get an API key.
* Add your Groq API key to the `.env` file
```
GROQ_API_KEY = your_groq_api_key
```
### 5. Run the Application
```
streamlit run app.py
```
