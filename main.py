# main.py (located in project root: Sentiment Analyser/main.py)

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple, Optional, Any
from contextlib import asynccontextmanager # For lifespan events

# --- Define DummyAnalyzer outside try-except if it's referenced when import succeeds ---
# However, with Option 1 logic, we won't need to reference DummyAnalyzer if import succeeds.
# So, keep DummyAnalyzer definition within the except block where it's actually used.

MODELS_LOADED_SUCCESSFULLY = False
# Initialize placeholders that will be overridden on successful import
analyzer_instance: Any = None
APP_SENTIMENT_STRATEGY: str = "unknown_init_error"
VADER_AVAILABLE: bool = False
SentimentAnalyzer_class: Any = None # To store the class itself if needed

def get_sentiment(text: str) -> Tuple[str, float]:
    # This dummy will be replaced if import is successful
    print("Warning: Using DUMMY get_sentiment function.")
    return "error_model_not_loaded", 0.0

def get_sentiments_batch(texts: List[str], batch_size: int = 8) -> List[Tuple[str, float]]:
    # This dummy will be replaced if import is successful
    print("Warning: Using DUMMY get_sentiments_batch function.")
    return [("error_model_not_loaded", 0.0) for _ in texts]


try:
    from app.models import (
        get_sentiment as real_get_sentiment,
        get_sentiments_batch as real_get_sentiments_batch,
        analyzer_instance as real_analyzer_instance,
        APP_SENTIMENT_STRATEGY as real_APP_SENTIMENT_STRATEGY,
        VADER_AVAILABLE as real_VADER_AVAILABLE,
        SentimentAnalyzer as RealSentimentAnalyzer_class # Import the class
    )
    if real_analyzer_instance is None: # Or check specific type if you know it
        raise ImportError("analyzer_instance from app.models is None after import.")

    # Override placeholders with real imports
    get_sentiment = real_get_sentiment
    get_sentiments_batch = real_get_sentiments_batch
    analyzer_instance = real_analyzer_instance
    APP_SENTIMENT_STRATEGY = real_APP_SENTIMENT_STRATEGY
    VADER_AVAILABLE = real_VADER_AVAILABLE
    SentimentAnalyzer_class = RealSentimentAnalyzer_class

    MODELS_LOADED_SUCCESSFULLY = True
    print("Successfully imported from app.models and updated global variables.")

except ImportError as e:
    print(f"CRITICAL ERROR: Could not import from app.models: {e}")
    print("Ensure 'app/__init__.py' exists and 'app/models.py' is correct.")
    print("Application will use DUMMY sentiment analyzer functions.")
    # Dummy functions are already defined above as placeholders.
    # We can define a DummyAnalyzer class here if we want analyzer_instance to be an object
    class _DummyAnalyzerInternal: # Underscore to avoid NameError if used elsewhere by mistake
        strategy = "error_model_not_loaded_critical"
    if analyzer_instance is None: # If real_analyzer_instance wasn't even assigned due to earlier import failure
        analyzer_instance = _DummyAnalyzerInternal()


# --- Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Lifespan: Application startup...")
    if MODELS_LOADED_SUCCESSFULLY: # If this is true, analyzer_instance is the real one
        print(f"Lifespan: Sentiment Analyzer (global instance) initialized with strategy: {analyzer_instance.strategy}")
        print(f"Lifespan: Default APP_SENTIMENT_STRATEGY from env/default: {APP_SENTIMENT_STRATEGY}")
        print(f"Lifespan: VADER available: {VADER_AVAILABLE}")
    else: # This means import failed
        print("Lifespan WARNING: Sentiment Analyzer functions are DUMMIES or models failed to load.")
        if analyzer_instance: # It might be the _DummyAnalyzerInternal or still None
             print(f"Lifespan: (Dummy/Failed) Analyzer strategy/state: {getattr(analyzer_instance, 'strategy', 'unknown_state')}")
        else:
            print("Lifespan CRITICAL ERROR: analyzer_instance is None.")
    yield
    print("Lifespan: Application shutdown.")


# --- FastAPI App Instance ---
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for analyzing sentiment of text using different strategies.",
    version="1.0.0",
    lifespan=lifespan
)


# --- Pydantic Models for Request/Response ---
class SentimentRequest(BaseModel):
    text: str

class BatchSentimentRequest(BaseModel):
    texts: List[str]
    batch_size: Optional[int] = None

class SentimentResponseItem(BaseModel):
    text: str
    sentiment_label: str
    sentiment_score: float

class AnalyzeSentimentResponse(SentimentResponseItem):
    strategy_used: str

class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponseItem]
    overall_strategy_used: str
    batch_size_used: int


# --- API Endpoints ---
@app.post("/analyze", response_model=AnalyzeSentimentResponse)
async def analyze_sentiment_endpoint(request: SentimentRequest):
    if not MODELS_LOADED_SUCCESSFULLY:
        raise HTTPException(status_code=503, detail="Sentiment analyzer service is not available due to model loading issues.")

    label, score = get_sentiment(request.text) # Uses the (potentially overridden) global function
    strategy_actually_used = analyzer_instance.strategy

    return AnalyzeSentimentResponse(
        text=request.text,
        sentiment_label=label,
        sentiment_score=score,
        strategy_used=strategy_actually_used
    )

@app.post("/analyze-batch", response_model=BatchSentimentResponse)
async def analyze_batch_sentiment_endpoint(request: BatchSentimentRequest):
    if not MODELS_LOADED_SUCCESSFULLY:
        raise HTTPException(status_code=503, detail="Sentiment analyzer service is not available due to model loading issues.")

    batch_size_to_use = request.batch_size if request.batch_size and request.batch_size > 0 else 8
    results_tuples = get_sentiments_batch(request.texts, batch_size=batch_size_to_use)

    response_items: List[SentimentResponseItem] = []
    for i, (label, score) in enumerate(results_tuples):
        response_items.append(SentimentResponseItem(
            text=request.texts[i],
            sentiment_label=label,
            sentiment_score=score
        ))

    return BatchSentimentResponse(
        results=response_items,
        overall_strategy_used=analyzer_instance.strategy,
        batch_size_used=batch_size_to_use
    )

@app.get("/health", summary="Health check for the API and sentiment analyzer status.")
async def health_check():
    analyzer_status = "ok"
    current_strategy = "unknown"

    if not MODELS_LOADED_SUCCESSFULLY:
        analyzer_status = "error_loading_models"
        current_strategy = getattr(analyzer_instance, 'strategy', APP_SENTIMENT_STRATEGY) # Use APP_SENTIMENT_STRATEGY as fallback
    elif analyzer_instance:
        current_strategy = analyzer_instance.strategy
    else: # Should not happen if MODELS_LOADED_SUCCESSFULLY is True
        analyzer_status = "error_unknown_state"
        current_strategy = APP_SENTIMENT_STRATEGY

    return {
        "api_status": "ok",
        "sentiment_analyzer_status": analyzer_status,
        "configured_strategy_on_load": APP_SENTIMENT_STRATEGY, # The one from models.py's env var
        "active_analyzer_instance_strategy": current_strategy, # The strategy of the live instance
        "vader_available": VADER_AVAILABLE
    }

# --- How to Run (from the 'Sentiment Analyser' project root directory) ---
if __name__ == "__main__":
    import uvicorn
    print("Running FastAPI app with Uvicorn directly from main.py __main__ block (for local testing)...")
    port = int(os.getenv("PORT", "8000"))
    # Recommended: uvicorn main:app --reload
    # But for python main.py:
    uvicorn.run(app, host="0.0.0.0", port=port)