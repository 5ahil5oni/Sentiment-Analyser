# app/models.py

import time
import os
from typing import Tuple, List, Dict, Optional, Any
from transformers import pipeline as hf_pipeline
import logging

# Configure logger for this module
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=os.getenv("MODEL_LOG_LEVEL", "INFO").upper(),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    logger.warning("vaderSentiment package not found. Install with: pip install vaderSentiment")
    VADER_AVAILABLE = False

# --- Constants for Model Keys ---
MODEL_KEY_VADER = "vader"
MODEL_KEY_ROBERTA_CARDIFFNLP = "roberta_cardiffnlp" # Main transformer for balanced/accurate

class SentimentAnalyzer:
    """
    Refined sentiment analyzer with enhanced neutral detection, configurable thresholds,
    and clearer model strategy.
    """

    def __init__(self, strategy: str = "balanced",
                 transformer_neutral_threshold: float = 0.6, # For non-neutral predictions
                 vader_neutral_compound_range: float = 0.15):
        """
        Initialize the sentiment analyzer.
        
        Args:
            strategy (str): Analysis strategy ('fast', 'balanced', 'accurate', 'adaptive').
                            'balanced' and 'accurate' will use RoBERTa CardiffNLP.
            transformer_neutral_threshold (float): Confidence threshold for transformer's non-neutral
                                                   predictions. If confidence for 'positive' or 'negative'
                                                   is below this, it's classified as neutral. (0.0-1.0)
            vader_neutral_compound_range (float): Defines the +/- range around 0 for VADER's compound
                                                  score to be considered neutral.
        """
        if strategy not in ["fast", "balanced", "accurate", "adaptive"]:
            logger.warning(f"Invalid strategy '{strategy}'. Defaulting to 'balanced'.")
            self.strategy = "balanced"
        else:
            self.strategy = strategy

        self.transformer_neutral_threshold = max(0.0, min(1.0, transformer_neutral_threshold))
        self.vader_neutral_compound_range = max(0.0, min(0.99, vader_neutral_compound_range)) # VADER compound is -1 to 1
        
        self.pipelines: Dict[str, Optional[Any]] = {
            MODEL_KEY_ROBERTA_CARDIFFNLP: None,
        }
        self.vader_analyzer: Optional[SentimentIntensityAnalyzer] = None
        
        # Initialize models based on the primary strategy set during instantiation.
        # Other models for adaptive strategy will be loaded on demand.
        self._initialize_models(force_strategy=self.strategy)

    def _initialize_models(self, force_strategy: Optional[str] = None):
        """Initialize models required by the chosen or forced strategy."""
        strategy_to_init = force_strategy if force_strategy else self.strategy
        # This log can be verbose if called frequently. Consider DEBUG level or a flag.
        # logger.info(f"Checking/Initializing models for effective strategy: {strategy_to_init}")
        
        try:
            if strategy_to_init in ["fast", "adaptive"] and VADER_AVAILABLE:
                if not self.vader_analyzer:
                    self.vader_analyzer = SentimentIntensityAnalyzer()
                    logger.info("VADER analyzer initialized successfully.")

            # 'balanced', 'accurate', and 'adaptive' (potentially) use RoBERTa CardiffNLP
            if strategy_to_init in ["balanced", "accurate", "adaptive"]:
                if not self.pipelines[MODEL_KEY_ROBERTA_CARDIFFNLP]:
                    logger.info(f"Initializing {MODEL_KEY_ROBERTA_CARDIFFNLP} model...")
                    self.pipelines[MODEL_KEY_ROBERTA_CARDIFFNLP] = hf_pipeline(
                        "sentiment-analysis",
                        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                        top_k=None # Get all scores (positive, neutral, negative)
                    )
                    logger.info(f"{MODEL_KEY_ROBERTA_CARDIFFNLP} pipeline initialized successfully.")
        except Exception as e:
            logger.error(f"Error during model initialization for strategy '{strategy_to_init}': {e}", exc_info=True)

    def analyze_single(self, text: str) -> Tuple[str, float]:
        """Analyze sentiment of a single text."""
        if not text or not isinstance(text, str) or text.isspace():
            return "neutral", 0.0

        effective_strategy = self.strategy
        if self.strategy == "adaptive":
            effective_strategy = self._determine_adaptive_strategy(text)
            logger.debug(f"Adaptive strategy chose: {effective_strategy} for text: '{text[:30]}...'")
            # Ensure the chosen model is loaded
            self._initialize_models(force_strategy=effective_strategy)

        if effective_strategy == "fast":
            return self._analyze_vader(text)
        elif effective_strategy in ["balanced", "accurate"]: # Both use RoBERTa CardiffNLP by default
            return self._analyze_transformer(text, MODEL_KEY_ROBERTA_CARDIFFNLP)
        else: # Should not happen with current strategy definitions
            logger.warning(f"Unknown effective strategy '{effective_strategy}'. Falling back to VADER.")
            return self._analyze_vader(text)

    def analyze_batch(self, texts: List[str], batch_size: int = 8) -> List[Tuple[str, float]]:
        """Analyze sentiment of multiple texts."""
        if not texts: return []

        if self.strategy == "adaptive":
            logger.info("Batch analysis with 'adaptive' strategy processes texts individually.")
            self._initialize_models(force_strategy="adaptive") # Ensure all potential adaptive models are checked/loaded
            return [self.analyze_single(text) for text in texts]

        # For "fast", "balanced", "accurate" direct strategies
        effective_model_key = None
        if self.strategy == "fast":
            if VADER_AVAILABLE:
                self._initialize_models(force_strategy="fast")
                return [self._analyze_vader(text) for text in texts]
            else:
                logger.warning("VADER not available for 'fast' batch strategy. Returning neutral for all.")
                return [("neutral", 0.0) for _ in texts]
        
        elif self.strategy in ["balanced", "accurate"]:
            effective_model_key = MODEL_KEY_ROBERTA_CARDIFFNLP
            self._initialize_models(force_strategy=self.strategy) # Ensures RoBERTa is loaded
        
        if effective_model_key and self.pipelines.get(effective_model_key):
            pipeline_instance = self.pipelines[effective_model_key]
            logger.info(f"Processing batch of {len(texts)} texts with {effective_model_key} (API batch_size: {batch_size}).")
            try:
                transformer_batch_outputs = pipeline_instance(texts, batch_size=batch_size)
                if not isinstance(transformer_batch_outputs, list) or \
                   (transformer_batch_outputs and not isinstance(transformer_batch_outputs[0], list)):
                    logger.error(f"Unexpected batch output from {effective_model_key}: {type(transformer_batch_outputs)}. Fallback.")
                    return [self.analyze_single(text) for text in texts] # Fallback to single processing

                results = []
                for single_text_scores_list in transformer_batch_outputs:
                    label, score = self._process_transformer_result(single_text_scores_list, effective_model_key)
                    results.append((label, score))
                return results
            except Exception as e:
                logger.error(f"Error during batch processing with {effective_model_key}: {e}. Fallback.", exc_info=True)
                return [self.analyze_single(text) for text in texts]
        else:
            logger.warning(f"Appropriate model for strategy '{self.strategy}' not available for batch. Fallback to VADER.")
            if VADER_AVAILABLE: self._initialize_models(force_strategy="fast")
            return [self._analyze_vader(text) for text in texts]

    def _analyze_vader(self, text: str) -> Tuple[str, float]:
        if not VADER_AVAILABLE or not self.vader_analyzer:
            return "neutral", 0.0
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            compound = scores['compound']
            if compound >= self.vader_neutral_compound_range: return "positive", compound
            elif compound <= -self.vader_neutral_compound_range: return "negative", compound
            else: return "neutral", compound
        except Exception as e:
            logger.error(f"VADER analysis error: {e}", exc_info=True); return "neutral", 0.0

    def _analyze_transformer(self, text: str, model_key: str) -> Tuple[str, float]:
        pipeline_instance = self.pipelines.get(model_key)
        if not pipeline_instance:
            logger.warning(f"{model_key} pipeline not initialized. Trying VADER fallback.")
            if VADER_AVAILABLE: self._initialize_models(force_strategy="fast"); return self._analyze_vader(text)
            return "error_no_model", 0.0
        try:
            raw_output = pipeline_instance(text) # For single text, expects List[Dict] or List[List[Dict]]
            
            scores_list_for_text: List[Dict[str, Any]]
            if isinstance(raw_output, list) and len(raw_output) > 0 and isinstance(raw_output[0], list):
                scores_list_for_text = raw_output[0] # Unpack batch-of-one
            elif isinstance(raw_output, list):
                scores_list_for_text = raw_output # Already List[Dict]
            else:
                logger.error(f"Unexpected output from {model_key} for single text: {type(raw_output)}. Content: {str(raw_output)[:200]}"); return "error_processing", 0.0
            
            return self._process_transformer_result(scores_list_for_text, model_key)
        except Exception as e:
            logger.error(f"{model_key} single analysis error: {e}", exc_info=True)
            if VADER_AVAILABLE: self._initialize_models(force_strategy="fast"); return self._analyze_vader(text)
            return "error_exception", 0.0

    def _process_transformer_result(self, class_scores_list: List[Dict[str, Any]], model_key: str) -> Tuple[str, float]:
        """
        Processes the output from a Hugging Face sentiment analysis pipeline (when top_k=None).
        class_scores_list is expected to be like: [{'label': 'positive', 'score': 0.9}, {'label': 'neutral', ...}]
        """
        if not class_scores_list or not isinstance(class_scores_list, list) or not all(isinstance(item, dict) for item in class_scores_list):
            logger.warning(f"Invalid class_scores_list for {model_key}: {class_scores_list}. Default to neutral.")
            return "neutral", 0.0

        try:
            # Find the prediction with the highest score (model's top choice)
            # Ensure 'score' is present and numeric
            valid_predictions = [p for p in class_scores_list if 'label' in p and 'score' in p and isinstance(p['score'], (float, int))]
            if not valid_predictions:
                logger.warning(f"No valid predictions in class_scores_list for {model_key}: {class_scores_list}"); return "neutral", 0.0
            
            top_prediction = max(valid_predictions, key=lambda p: p['score'])
            model_label = top_prediction['label'].lower() # e.g., 'positive', 'negative', 'neutral', or 'label_0'
            model_confidence = top_prediction['score'] # Model's confidence in THIS specific label

            # 1. If model strongly predicts "neutral"
            if model_label == "neutral": # Handles 'neutral' from CardiffNLP
                # The confidence is in its neutrality. Polarity is 0.
                # We could use self.transformer_neutral_threshold here if we want to be skeptical of low-confidence neutral
                # e.g. if model_confidence < self.transformer_neutral_threshold: return "unknown", 0.0
                # But generally, if model says neutral, we accept it.
                return "neutral", 0.0
            
            # 2. For non-neutral predictions (positive/negative), check against our threshold
            if model_confidence < self.transformer_neutral_threshold:
                logger.debug(f"Model predicted '{model_label}' with confidence {model_confidence:.3f} for {model_key}, "
                             f"which is below threshold {self.transformer_neutral_threshold}. Classifying as neutral.")
                return "neutral", 0.0 # Confidence too low for the predicted polarity

            # 3. High confidence positive or negative prediction
            if model_label == "positive" or model_label == "label_2": # LABEL_2 often maps to positive
                return "positive", model_confidence
            elif model_label == "negative" or model_label == "label_0": # LABEL_0 often maps to negative
                return "negative", -model_confidence # Return as negative score
            # CardiffNLP also has 'LABEL_1' for neutral, already handled by model_label == "neutral"

            logger.warning(f"Unhandled model_label '{model_label}' from {model_key} with confidence {model_confidence}. Defaulting to neutral. Scores: {class_scores_list}")
            return "neutral", 0.0

        except Exception as e:
            logger.error(f"Error in _process_transformer_result for {model_key}: {e}. Scores: {class_scores_list}", exc_info=True)
            return "error_processing", 0.0


    def _determine_adaptive_strategy(self, text: str) -> str:
        """Determines the best strategy for a given text in 'adaptive' mode."""
        text_length = len(text.split())
        has_social_indicators = any(char in text for char in ['@', '#', 'http', '😊', '😢', '👍', '👎', 'lol', 'omg'])

        # Check if models are loaded before preferring them
        if VADER_AVAILABLE and self.vader_analyzer and (text_length < 15 or has_social_indicators):
            return "fast"
        # For balanced/accurate, we use the same robust RoBERTa model
        elif self.pipelines.get(MODEL_KEY_ROBERTA_CARDIFFNLP): # If RoBERTa is available
             # Could add more logic here, e.g., use "accurate" settings if text_length > 70
            return "balanced" # Default to 'balanced' which uses RoBERTa
        elif VADER_AVAILABLE and self.vader_analyzer: # Fallback if RoBERTa not yet loaded
            return "fast"
        else: # Ultimate fallback
            logger.warning("No suitable models available for adaptive strategy. Defaulting to 'fast' (which might still require initialization).")
            return "fast"

    def set_transformer_neutral_threshold(self, threshold: float):
        self.transformer_neutral_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Transformer neutral threshold set to {self.transformer_neutral_threshold}")

    def set_vader_neutral_range(self, range_val: float):
        self.vader_neutral_compound_range = max(0.0, min(0.99, range_val))
        logger.info(f"VADER neutral compound range set to +/- {self.vader_neutral_compound_range}")

# --- Global instance and helper functions ---
APP_SENTIMENT_STRATEGY = os.getenv("APP_SENTIMENT_STRATEGY", "balanced")
TRANSFORMER_NEUTRAL_THRESHOLD_ENV = float(os.getenv("TRANSFORMER_NEUTRAL_THRESHOLD", "0.5")) # Default to 0.5
VADER_NEUTRAL_RANGE_ENV = float(os.getenv("VADER_NEUTRAL_RANGE", "0.15"))

analyzer_instance = SentimentAnalyzer(
    strategy=APP_SENTIMENT_STRATEGY,
    transformer_neutral_threshold=TRANSFORMER_NEUTRAL_THRESHOLD_ENV,
    vader_neutral_compound_range=VADER_NEUTRAL_RANGE_ENV
)

def get_sentiment(text: str) -> Tuple[str, float]:
    return analyzer_instance.analyze_single(text)

def get_sentiments_batch(texts: List[str], batch_size: int = 8) -> List[Tuple[str, float]]:
    return analyzer_instance.analyze_batch(texts, batch_size=batch_size)

# --- Main block for direct execution and testing ---
if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO) # Ensure logs from this module are visible

    test_texts = [
        "I love this product! It's absolutely fantastic!",
        "This is the worst experience ever. Terrible!",
        "The weather is nice today.",
        "It's raining outside.",
        "The meeting is scheduled for 3 PM.",
        "This document contains information about the project.",
        "The product works as described in the manual.",
        "I'm not sure about this, it seems a bit off.",
        "It is what it is, can't complain too much.",
        "The results were mixed, some good, some bad.",
        "This movie was okay, not great but not terrible either.",
        "I guess it's fine.",
        "The cat sat on the mat.",
        "Excited for the new release! #awesome",
        "Feeling so-so today 😐",
        "This is just a statement of fact without any emotion.",
        "He simply stated the facts."
    ]

    print(f"--- Testing Improved Sentiment Analyzer ---")
    print(f"Initial Strategy: {analyzer_instance.strategy}")
    print(f"Initial Transformer Neutral Threshold: {analyzer_instance.transformer_neutral_threshold}")
    print(f"Initial VADER Neutral Range: +/-{analyzer_instance.vader_neutral_compound_range}")
    print("-" * 30)

    for text_input in test_texts:
        label, score = get_sentiment(text_input)
        print(f"'{text_input[:60]}...' -> {label.upper()} (Score: {score:.3f})")

    print("\n" + "-" * 30)
    print("--- Testing with 'fast' (VADER) strategy ---")
    analyzer_instance.strategy = "fast" # Change strategy of global instance for this test
    analyzer_instance._initialize_models() # Ensure VADER is loaded if not already
    for text_input in test_texts[:5]: # Test a few with VADER
        label, score = get_sentiment(text_input)
        print(f"'{text_input[:60]}...' -> {label.upper()} (Score: {score:.3f})")

    print("\n" + "-" * 30)
    print("--- Testing with 'adaptive' strategy and modified thresholds ---")
    analyzer_instance.strategy = "adaptive"
    analyzer_instance.set_transformer_neutral_threshold(0.75) # More likely to be neutral for transformers
    analyzer_instance.set_vader_neutral_range(0.25)         # Wider neutral for VADER
    print(f"New Transformer Neutral Threshold: {analyzer_instance.transformer_neutral_threshold}")
    print(f"New VADER Neutral Range: +/-{analyzer_instance.vader_neutral_compound_range}")
    
    borderline_texts = [
        "The movie was okay.", "It's an average product.", "The service was decent.",
        "The food wasn't bad.", "I have no strong feelings about this."
    ]
    for text_input in borderline_texts:
        label, score = get_sentiment(text_input) # Will use adaptive strategy
        print(f"'{text_input[:60]}...' -> {label.upper()} (Score: {score:.3f})")

    print("\n" + "-" * 30)
    print("--- Testing Batch Analysis (Balanced Strategy) ---")
    analyzer_instance.strategy = "balanced" # Set to balanced for batch test
    analyzer_instance.set_transformer_neutral_threshold(0.5) # Reset threshold
    batch_test_texts = test_texts[:6]
    batch_results = get_sentiments_batch(batch_test_texts, batch_size=2)
    for text_original, (label, score) in zip(batch_test_texts, batch_results):
        print(f"BATCH: '{text_original[:60]}...' -> {label.upper()} (Score: {score:.3f})")