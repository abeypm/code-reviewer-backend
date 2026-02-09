# rag_server.py
"""
This script runs a Flask web server that acts as the backend for a RAG-based code reviewer.
It exposes a single endpoint `/ask` which accepts a code file upload.

The server performs the following steps:
1. Receives a code file.
2. Searches a ChromaDB vector database (pre-indexed with a textbook) for relevant passages.
3. Constructs a detailed prompt containing the code and the retrieved passages.
4. **Saves the full prompt to a file named 'last_prompt.log' for debugging.**
5. Calls an LLM (either standard OpenAI or Azure OpenAI, based on .env configuration) to get a review.
6. Returns the AI-generated review as a JSON response.

To run:
1. Ensure all dependencies from requirements.txt are installed.
2. Ensure you have run index.py to create the 'db' folder.
3. Ensure your .env file is correctly configured for either standard OpenAI or Azure OpenAI.
4. Run `python rag_server.py` in your terminal.
"""
import os
import chromadb
import openai
from flask import Flask, request, jsonify
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# Load environment variables from .env file at the very start
load_dotenv()

# --- Configuration ---
COLLECTION_NAME = "clean_code_book"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LOG_FILE_NAME = "last_prompt.log" # The file where the prompt will be saved

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Global Objects - Loaded once at startup for efficiency ---
print("Loading models and setting up clients... This may take a moment.")
client = None
model_to_use = None

# --- Client Setup: Detects and configures either Azure or Standard OpenAI ---
api_type = os.getenv("OPENAI_API_TYPE")

if api_type == "azure":
    print("Configuring for Azure OpenAI...")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("OPENAI_API_KEY")
    api_version = os.getenv("OPENAI_API_VERSION")
    chat_deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

    if not all([azure_endpoint, api_key, api_version, chat_deployment_name]):
        raise EnvironmentError("Azure OpenAI config incomplete in .env file.")

    client = openai.AzureOpenAI(
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
    )
    model_to_use = chat_deployment_name
    print("Azure OpenAI client configured.")
else:
    print("Configuring for standard OpenAI...")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not found in .env file for standard OpenAI.")

    client = openai.OpenAI(api_key=api_key)
    model_to_use = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
    print("Standard OpenAI client configured.")

# --- Embedding Function and Vector DB Setup ---
try:
    print("Setting up embedding function and vector database...")
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
    db_client = chromadb.PersistentClient(path="db")
    collection = db_client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=sentence_transformer_ef
    )
    print("Embedding function and vector database loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load models or database. Have you run index.py? Error: {e}")

print("✅ Server is fully initialized and ready to accept requests.")


# --- Main API Endpoint ---
@app.route("/ask", methods=['POST'])
def ask_agent():
    """Handles the file upload and returns the AI-powered code review."""
    print("\nReceived a new request to /ask...")

    # 1. Handle File Upload
    if 'code_file' not in request.files:
        print("  - Error: No 'code_file' part in the request.")
        return jsonify({"error": "No 'code_file' part in the request"}), 400
    file = request.files['code_file']
    if file.filename == '':
        print("  - Error: No file selected.")
        return jsonify({"error": "No selected file"}), 400

    if file:
        try:
            code_content = file.read().decode('utf-8')
            print(f"  - Received file '{file.filename}' ({len(code_content)} chars).")

            # 2. Retrieve Relevant Passages (RAG)
            print("  - Searching knowledge base for relevant passages...")
            results = collection.query(
                query_texts=[code_content],
                n_results=3
            )
            retrieved_passages = "\n\n---\n\n".join(results['documents'][0])
            print("  - Found relevant passages.")

            # 3. Augment Prompt
            system_prompt = """
You are an world-class expert software engineer specializing in writing clean, maintainable code. 
You will be provided with passages from a software engineering textbook and a piece of code.
Your task is to act as a helpful code reviewer. Analyze the code and provide a review based ONLY on the principles and examples found in the provided textbook passages.
Do not use any outside knowledge. If the passages are not relevant, state that you cannot provide a review based on the given context.
Structure your feedback clearly. Start with a high-level summary, then use bullet points for specific suggestions.
"""
            
            user_prompt = f"""
**TEXTBOOK PASSAGES:**
<passages>
{retrieved_passages}
</passages>

**CODE TO REVIEW:**
<code>
{code_content}
</code>

Please provide your expert review based on the textbook passages.
"""
            
            # --- MODIFIED FOR DEBUGGING ---
            # Construct the full prompt content and save it to a log file
            prompt_content_to_log = f"""
{'='*80}
--- CONSTRUCTED LLM PROMPT (Model: {model_to_use}) ---
{'='*80}

[SYSTEM PROMPT]
{system_prompt}

{'-'*80}

[USER PROMPT]
{user_prompt}

{'='*80}
--- END OF PROMPT ---
"""
            try:
                with open(LOG_FILE_NAME, "w", encoding="utf-8") as f:
                    f.write(prompt_content_to_log)
                print(f"  - Full LLM prompt saved to '{LOG_FILE_NAME}' for debugging.")
            except Exception as log_e:
                print(f"  - Warning: Could not write prompt to log file. Error: {log_e}")
            # --- END OF DEBUGGING BLOCK ---

            # 4. Call LLM
            print(f"  - Calling LLM ('{model_to_use}')...")
            response = client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            llm_answer = response.choices[0].message.content
            print("  - Successfully received response from LLM.")
            return jsonify({"answer": llm_answer})

        except Exception as e:
            print(f"  - An error occurred during processing: {e}")
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "An unknown server error occurred"}), 500


# --- Run Flask App ---
if __name__ == '__main__':
    app.run(port=5000, debug=True)
