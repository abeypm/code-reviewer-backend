# rag_server.py
"""
This script runs a Flask web server for a RAG-based code reviewer.
This version has been converted to support streaming and includes a new endpoint
to serve an HTML page (`index.html`) to correctly render the stream.
"""
import os
import json
import chromadb
import openai
import datetime
# NEW: Import Response and send_from_directory
from flask import Flask, request, jsonify, Response, send_from_directory
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# Load environment variables from .env file at the very start
load_dotenv()

# --- Configuration ---
COLLECTION_NAME = "clean_code_book"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OUTPUT_DIR = "review_outputs"

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Global Objects - Loaded once at startup ---
print("Loading models and setting up clients... This may take a moment.")
client = None
model_to_use = None
api_type = os.getenv("OPENAI_API_TYPE")
if api_type == "azure":
    print("Configuring for Azure OpenAI...")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("OPENAI_API_KEY")
    api_version = os.getenv("OPENAI_API_VERSION")
    chat_deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
    if not all([azure_endpoint, api_key, api_version, chat_deployment_name]):
        raise EnvironmentError("Azure OpenAI config incomplete in .env file.")
    client = openai.AzureOpenAI(api_key=api_key, azure_endpoint=azure_endpoint, api_version=api_version)
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

try:
    print("Setting up embedding function and vector database...")
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
    db_client = chromadb.PersistentClient(path="db")
    collection = db_client.get_collection(name=COLLECTION_NAME, embedding_function=sentence_transformer_ef)
    print(f"  - Collection '{collection.name}' contains {collection.count()} documents.")
    print("Embedding function and vector database loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load models or database. Have you run index.py? Error: {e}")

print("✅ Server is fully initialized and ready to accept requests.")


# --- NEW: Route to serve the HTML front-end page ---
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


# --- Main API Endpoint (Converted to Streaming) ---
@app.route("/ask", methods=['POST'])
def ask_agent():
    """Handles the file upload and streams back a series of specialist reviews."""
    print("\nReceived a new request for a direct specialist stream...")

    if 'code_file' not in request.files:
        return jsonify({"error": "No 'code_file' part in the request"}), 400
    file = request.files['code_file']
    original_filename = file.filename
    if original_filename == '':
        return jsonify({"error": "No selected file"}), 400
    code_content = file.read().decode('utf-8')

    # The main logic is now inside a generator function
    def generate_reviews(code_to_review, filename):
        try:
            # Buffer flush to defeat proxy buffering
            yield (' ' * 4096).encode('utf-8')

            print(f"  - Starting review generator for file '{filename}' ({len(code_to_review)} chars).")
            with open('review_checklist.json', 'r', encoding="utf-8") as f:
                checklist = json.load(f)
            print(f"  - Loaded {len(checklist)} items from checklist.")

            os.makedirs(OUTPUT_DIR, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{timestamp}_review_for_{filename}.md"
            output_filepath = os.path.join(OUTPUT_DIR, output_filename)
            
            for item in checklist:
                print(f"    - Starting review for: '{item['focus']}'...")
                heading = f"\n\n---\n\n## Specialist Review for: {item['focus']}\n\n"
                yield heading.encode('utf-8')
                
                results = collection.query(query_texts=[item['query']], n_results=2)
                retrieved_passages = ""
                if results and results.get('documents') and len(results['documents']) > 0 and results['documents'][0]:
                    retrieved_passages = "\n\n---\n\n".join(results['documents'][0])
                else:
                    warning_message = f"_Note: Could not find specific textbook passages for '{item['focus']}'. Review is based on general knowledge._\n\n"
                    yield warning_message.encode('utf-8')
                    with open(output_filepath, "a", encoding="utf-8") as f:
                        f.write(heading + warning_message)
                
                system_prompt = f"""
You are a highly specialized code reviewer. Your ONLY focus is on reviewing code for one specific principle: **{item['focus']}**.
You will be given passages from a textbook related to this principle and a piece of code.
Analyze the code strictly through the lens of **{item['focus']}** based on the provided passages. Ignore all other potential issues.
**Format your feedback using Markdown.** Use bullet points for suggestions and backticks for code snippets.
"""
                user_prompt = f"""
**TEXTBOOK PASSAGES related to {item['focus']}:**
<passages>
{retrieved_passages}
</passages>
**CODE TO REVIEW:**
<code>
{code_to_review}
</code>
Please provide your specialized review focusing only on **{item['focus']}**, formatted in Markdown.
"""
                
                # Request a stream from the API
                stream = client.chat.completions.create(
                    model=model_to_use,
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                    stream=True
                )

                full_specialist_review = ""
                for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        content = chunk.choices[0].delta.content
                        if content:
                            full_specialist_review += content
                            yield content.encode('utf-8') # Stream each token

                # Save the full review to the file after the stream is done
                if retrieved_passages:
                     with open(output_filepath, "a", encoding="utf-8") as f:
                        f.write(heading + full_specialist_review)
                else:
                    with open(output_filepath, "a", encoding="utf-8") as f:
                        f.write(full_specialist_review)
                
                print(f"    - Completed streaming review for: '{item['focus']}'.")

            print("\n✅ All specialist reviews streamed successfully.")
        except Exception as e:
            print(f"  - An error occurred during generator execution: {e}")
            yield json.dumps({"error": str(e)}).encode('utf-8')

    # Return a streaming response
    return Response(generate_reviews(code_content, original_filename), mimetype='text/event-stream')


# --- Run Flask App ---
if __name__ == '__main__':
    # Disable reloader to prevent double-loading models during startup
    app.run(port=5000, debug=True, use_reloader=False)

