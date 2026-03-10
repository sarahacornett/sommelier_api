# Sommelier AI: Full-Stack ML Engine & RAG Chatbot

An end-to-end machine learning application that predicts wine prices using a custom-trained XGBoost model and provides highly grounded, personalized wine pairings via a Retrieval-Augmented Generation (RAG) pipeline.

## System Architecture

This project is built on a decoupled, microservice-style architecture, separating the mobile-friendly client from the heavy machine-learning backend.

* **Frontend:** Flutter (Dart) compiled to Web.
* **Backend API:** FastAPI (Python) hosted on Render.
* **Price Prediction Engine:** XGBoost Regressor trained on the Kaggle Wine Reviews dataset.
* **Chatbot Engine:** RAG architecture using ChromaDB (Vector Database), `sentence-transformers` for embeddings, and the Google Gemini API for grounded generation.

## Key Features

### 1. Robust Tabular Data Pipeline (XGBoost)
Unlike simple LLM wrappers, the price prediction engine relies on a dedicated `XGBoost` regression model. 
* **Data Engineering:** Processed a 130k+ row dataset, engineered features using frequency thresholds to handle rare categories, and utilized XGBoost's native categorical processing to elegantly handle missing user inputs without crashing.
* **Performance:** Achieved an $R^2$ baseline of 0.47, predicting price variance purely from categorical metadata (variety, region, winery) and critic scores.

### 2. Retrieval-Augmented Generation (RAG) Chatbot
To prevent the LLM from hallucinating fake vintages, the chatbot is grounded in a curated vector database of 5,000 premium wines.
* **Semantic Search:** User queries are embedded into high-dimensional vectors. ChromaDB mathematically retrieves the top 3 real-world wines that match the requested flavor profile.
* **Prompt Injection:** These real records are injected into Gemini's system prompt, forcing the AI to act as a sommelier restricted *only* to the provided inventory.

## Local Setup & Installation

### Backend (Python API)
1. Clone the repository and navigate to the backend directory.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate