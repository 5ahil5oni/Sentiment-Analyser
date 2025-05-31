Advanced Sentiment Analyzer

This project is an Advanced Sentiment Analyzer featuring a FastAPI backend for model serving and a Streamlit frontend for interactive use and analytics. It supports multiple sentiment analysis strategies, including VADER, DistilBERT (via RoBERTa CardiffNLP for better neutral handling), and an adaptive approach.

Features

*   FastAPI Backend (`main.py`):
    *   Serves sentiment analysis models.
    *   Endpoints for single text (`/analyze`) and batch text (`/analyze-batch`) analysis.
    *   Health check endpoint (`/health`).
    *   Automatic API documentation via Swagger UI (`/docs`) and ReDoc (`/redoc`).
*   Streamlit Frontend (`frontend_app.py`):
    *   Interactive UI for single and batch sentiment analysis.
    *   Visualization of sentiment scores (e.g., gauge chart).
    *   Text statistics.
    *   Analysis history and basic analytics (average sentiment, strategy usage).
    *   Settings and API access information.
*   Sentiment Analysis Logic (`app/models.py`):
    *   Supports multiple strategies:
        *   `fast`: VADER (rule-based).
        *   `balanced` / `accurate`: `cardiffnlp/twitter-roberta-base-sentiment-latest` (transformer-based, good neutral detection).
        *   `adaptive`: Chooses the best strategy based on text characteristics.
    *   Configurable thresholds for neutral sentiment detection.

Project Structure

Sentiment Analyser/
├── app/
│ ├── init.py
│ └── models.py # SentimentAnalyzer class and model logic
├── frontend_app.py # Streamlit UI application
├── main.py # FastAPI backend application
├── requirements.txt # Python dependencies
├── README.md # This file
├── .gitignore # Files to ignore for Git
└── venv/ # Python virtual environment (example)


Setup

1.  Clone the repository (if applicable):
    ```bash
    git clone <your-repository-url>
    cd "Sentiment Analyser"
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This will download necessary machine learning models from Hugging Face Hub on first run, which might take some time and require an internet connection.*

Running the Application

You need to run the FastAPI backend and the Streamlit frontend separately, typically in two different terminal windows.

1. Run the FastAPI Backend:

   Navigate to the project root directory (`Sentiment Analyser/`) in your terminal and run:
   ```bash
   uvicorn main:app --reload --port 8000
The API will be available at http://localhost:8000.

2. Run the Streamlit Frontend:
Open a new terminal, navigate to the project root directory (Sentiment Analyser/), and run:

streamlit run frontend_app.py

The Streamlit application will typically open in your web browser at http://localhost:8501.

Usage

Frontend: Open the Streamlit app URL in your browser. You can input text for single analysis, use the batch analysis features, view analytics, and get API usage information.

API:

Access API documentation at http://localhost:8000/docs (Swagger UI) or http://localhost:8000/redoc.
Example API calls are provided in the "API Access" tab of the Streamlit frontend.

Configuration (Optional)

The application uses some environment variables for configuration (defined in app/models.py and main.py with defaults):
APP_SENTIMENT_STRATEGY: Default sentiment strategy (e.g., "balanced", "adaptive"). Default: "balanced".
TRANSFORMER_NEUTRAL_THRESHOLD: Confidence threshold for transformer models. Default: 0.5.
VADER_NEUTRAL_RANGE: Compound score range for VADER to be considered neutral. Default: 0.15.
PORT: Port for the FastAPI backend (used if running python main.py). Default: 8000.
MODEL_LOG_LEVEL: Logging level for the models. Default: "INFO".

You can set these by creating a .env file in the project root (ensure it's in .gitignore) or by setting them in your shell environment. Example .env file:
APP_SENTIMENT_STRATEGY=adaptive
TRANSFORMER_NEUTRAL_THRESHOLD=0.6

Future Enhancements

Per-request strategy selection via API and UI.
More sophisticated debouncing for auto-analyze.
Persistent storage for analysis history (e.g., SQLite database).
User authentication for API.