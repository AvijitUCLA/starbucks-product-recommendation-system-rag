# ☕ Starbucks Data Challenge: AI-Powered Product Recommendation System

## Overview
This repository contains an end-to-end natural language product recommendation system. The system processes unstructured, conversational customer queries and returns a ranked list of relevant Starbucks products based on specific constraints and semantic similarity.

The solution utilizes a hybrid Search/RAG (Retrieval-Augmented Generation) pipeline, combining Google Cloud Vertex AI for deterministic filtering with dense vector embeddings for semantic ranking.

## 🏆 Performance Metrics
Evaluated against a training set of 100 queries with ground-truth labels, the pipeline achieves exceptional accuracy:
* **Average NDCG:** `0.9650`
* **Average Recall:** `0.9770`
* **Top-1 Accuracy:** `0.9700`

## 🏗️ Architecture: The 3-Stage Pipeline
The recommendation engine follows a strict 3-stage workflow:

### 1. Constraint Extraction
* **Model:** Google Gemini 2.5 Flash (via GCP Vertex AI)
* **Function:** Processes the natural language query and enforces a strict JSON schema generation to extract key constraints: `category`, `temperature`, `max_calories`, `max_sugar`, `max_price`, `dairy_free`, `vegan`, and `caffeine_level`.

### 2. Deterministic Filtering
* **Tech:** Python & Pandas
* **Function:** Applies the LLM-extracted constraints against the 115-item Starbucks `products.csv` database. It strictly filters out items that violate user requirements (e.g., capping prices, removing allergens).

### 3. Semantic Relevance Ranking
* **Model:** `all-mpnet-base-v2` (SentenceTransformers)
* **Function:** Computes dense vector embeddings (768-dimensional) for the candidate products by concatenating their names, categories, ingredients, and descriptions. Calculates **Cosine Similarity** against the embedded user query to rank the filtered candidates from most to least relevant.

## 🛠️ Tech Stack
* **Language:** Python
* **Data Manipulation:** `pandas`, `numpy`
* **LLM Integration:** `vertexai`, `google.genai` (Google Cloud Platform), `gemini-2.5-flash`
* **Embeddings & Similarity:** `sentence-transformers`, `scikit-learn`
* **Environment:** Jupyter Notebook

## 🚀 How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/AvijitUCLA/starbucks-product-recommendation-system-rag.git
cd starbucks-recommendation-system
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set up Google Cloud Authentication**
This project uses Vertex AI. You must have a Google Cloud Project with the Vertex AI API enabled.
* Create a Service Account and download the JSON key.
* Save the key in the root directory as `auth.json`. *(Note: This file is git-ignored for security).*
* Update the project ID in the notebook if necessary.

**4. Execute the Pipeline**
Run all cells in the `RAG_code_final.ipynb` notebook. The script will automatically:
1. Load and pre-compute product embeddings.
2. Process the dataset through the Vertex AI extraction pipeline.
3. Calculate evaluation metrics (NDCG, Recall, Top-1).
4. Generate the final `submission.csv` in the `output/` directory.

---
*Developed by Avijit Das - MS in Business Analytics, UCLA*
