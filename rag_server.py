# rag_server.py
"""
This script runs a Flask web server that acts as the backend for a RAG-based code reviewer.
This version implements an advanced, multi-step "Checklist-Driven Review" process.
"""
import os
import json
import chromadb
import openai
from flask import Flask, request, jsonify
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
COLLECTION_NAME = "clean_code_book"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LOG_FILE_NAME = "last_consolidation_prompt.log"

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Global Objects - Loaded once at startup ---
print("Loading models and setting up clients...")
client = None
model_to_use = None

# --- Client Setup (THIS SECTION IS NOW CORRECTED) ---
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

try:
    print("Setting up embedding function and vector database...")
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
    db_client = chromadb.PersistentClient(path="db")
    collection = db_client.get_collection(name=COLLECTION_NAME, embedding_function=sentence_transformer_ef)
    print("Embedding function and vector database loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load models or database. Have you run index.py? Error: {e}")

print("✅ Server is fully initialized and ready to accept requests.")


# --- Main API Endpoint ---
@app.route("/ask", methods=['POST'])
def ask_agent():
    """Handles the file upload and runs the multi-step, checklist-driven review."""
    print("\nReceived a new request for a checklist-driven review...")

    if 'code_file' not in request.files:
        return jsonify({"error": "No 'code_file' part in the request"}), 400
    file = request.files['code_file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        code_content = file.read().decode('utf-8')
        print(f"  - Received file '{file.filename}' ({len(code_content)} chars).")

        try:
            with open('review_checklist.json', 'r', encoding="utf-8") as f:
                checklist = json.load(f)
            print(f"  - Successfully loaded {len(checklist)} items from review_checklist.json.")
        except FileNotFoundError:
            return jsonify({"error": "review_checklist.json not found."}), 500
        except json.JSONDecodeError:
            return jsonify({"error": "Failed to parse review_checklist.json."}), 500

        # --- PHASE 1: INDIVIDUAL REVIEW LOOP ---
        individual_reviews = []
        for item in checklist:
            print(f"    - Starting review for: '{item['focus']}'...")

            results = collection.query(query_texts=[item['query']], n_results=2)
            retrieved_passages = "\n\n---\n\n".join(results['documents'][0])

            system_prompt = f"""
You are a highly specialized code reviewer. Your ONLY focus is on reviewing code for one specific principle: **{item['focus']}**.
You will be given passages from a textbook related to this principle and a piece of code.
Analyze the code strictly through the lens of **{item['focus']}** based on the provided passages. Ignore all other potential issues.
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

Please provide your specialized review focusing only on **{item['focus']}**.
"""
            response = client.chat.completions.create(
                model=model_to_use,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            )
            focused_review = response.choices[0].message.content
            individual_reviews.append(f"## Specialist Review for: {item['focus']}\n\n{focused_review}")
            print(f"    - Completed review for: '{item['focus']}'.")

        # --- PHASE 2: FINAL CONSOLIDATION ---
        print("  - All individual reviews complete. Starting final consolidation...")

        all_reviews_text = "\n\n".join(individual_reviews)

        consolidation_system_prompt = """
You are a lead software architect responsible for finalizing a code review.
You have been provided with a set of reviews from several specialist AIs, each focusing on a different clean code principle.
Your task is to synthesize these individual reviews into a single, cohesive, de-duplicated, and actionable report for the original developer.

**Instructions:**
1.  **Synthesize and Merge:** Combine related points from different specialist reviews.
2.  **De-duplicate:** If multiple specialists commented on the same line of code for similar reasons, merge their feedback into a single, comprehensive comment.
3.  **Prioritize:** Structure the final report with the most critical issues first.
4.  **Format:** Use clear headings and bullet points. Be encouraging and constructive. Do not mention the specialist reviewers; present the feedback as a unified report from the team.
"""
        consolidation_user_prompt = f"""
Here are the raw reviews from the specialist AIs. Please consolidate them into one final report.

**RAW SPECIALIST REVIEWS:**
{all_reviews_text}
"""
        
        try:
            with open(LOG_FILE_NAME, "w", encoding="utf-8") as f:
                f.write(consolidation_user_prompt)
            print(f"  - Final consolidation prompt saved to '{LOG_FILE_NAME}'.")
        except Exception as log_e:
            print(f"  - Warning: Could not write prompt to log file. Error: {log_e}")

        print(f"  - Calling LLM for final consolidation ('{model_to_use}')...")
        final_response = client.chat.completions.create(
            model=model_to_use,
            messages=[{"role": "system", "content": consolidation_system_prompt}, {"role": "user", "content": consolidation_user_prompt}]
        )
        final_review = final_response.choices[0].message.content
        print("  - Successfully received final consolidated review.")

        return jsonify({"answer": final_review})

    except Exception as e:
        print(f"  - An error occurred during processing: {e}")
        return jsonify({"error": str(e)}), 500

# --- Run Flask App ---
if __name__ == '__main__':
    app.run(port=5000, debug=True)
