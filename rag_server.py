# rag_server.py
"""
This script runs a Flask web server for a RAG-based code reviewer.

This version implements a "Direct Specialist Reporting" model without a final consolidation step.

Features:
- Loops through a checklist to get focused reviews from a specialist LLM.
- Instructs specialists to provide feedback in Markdown.
- Combines all specialist reviews into a single Markdown report.
- Saves the final report to a timestamped .md file in 'review_outputs/'.
- Returns the complete report as a single JSON response.
- NOTE: This is a synchronous, long-running process. Your API client must have a long timeout configured.
"""
import os
import json
import chromadb
import openai
import datetime
from flask import Flask, request, jsonify
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


# --- Main API Endpoint (Synchronous, No Consolidator) ---
@app.route("/ask", methods=['POST'])
def ask_agent():
    """Handles the file upload and runs the multi-step specialist review."""
    print("\nReceived a new request for a direct specialist report...")

    if 'code_file' not in request.files:
        return jsonify({"error": "No 'code_file' part in the request"}), 400
    file = request.files['code_file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        code_content = file.read().decode('utf-8')
        print(f"  - Received file '{file.filename}' ({len(code_content)} chars).")

        with open('review_checklist.json', 'r', encoding="utf-8") as f:
            checklist = json.load(f)
        print(f"  - Loaded {len(checklist)} items from checklist.")

        # --- PHASE 1: INDIVIDUAL REVIEW LOOP ---
        individual_reviews = []
        # The tqdm progress bar has been removed from this loop
        for item in checklist:
            print(f"    - Starting review for: '{item['focus']}'...")
            
            results = collection.query(query_texts=[item['query']], n_results=2)
            retrieved_passages = ""
            if results and results.get('documents') and len(results['documents']) > 0 and results['documents'][0]:
                retrieved_passages = "\n\n---\n\n".join(results['documents'][0])
            
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
{code_content}
</code>

Please provide your specialized review focusing only on **{item['focus']}**, formatted in Markdown.
"""
            response = client.chat.completions.create(
                model=model_to_use,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            )
            focused_review = response.choices[0].message.content
            # Add a heading to each review for clarity in the final combined file
            review_with_heading = f"## Specialist Review for: {item['focus']}\n\n{focused_review}"
            individual_reviews.append(review_with_heading)
            print(f"    - Completed review for: '{item['focus']}'.")

        # --- COMBINE AND SAVE RESULTS ---
        print("  - All specialist reviews complete. Combining and saving results...")
        
        # Join all the individual markdown reviews into a single string, separated by a horizontal rule
        final_markdown_content = "\n\n---\n\n".join(individual_reviews)

        # Save the combined content to a single .md file
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{timestamp}_review_for_{file.filename}.md"
        output_filepath = os.path.join(OUTPUT_DIR, output_filename)
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(final_markdown_content)
        print(f"  - Final combined review saved to '{output_filepath}'.")

        # Return the same combined content in the JSON response
        return jsonify({"answer": final_markdown_content})

    except Exception as e:
        print(f"  - An error occurred during processing: {e}")
        return jsonify({"error": str(e)}), 500

# --- Run Flask App ---
if __name__ == '__main__':
    app.run(port=5000, debug=True)
