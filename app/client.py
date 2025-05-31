import praw
import os
from dotenv import load_dotenv
import logging
import time
from typing import Iterator, Dict, Any, Optional, Callable, TypeVar, TypedDict
from dataclasses import dataclass # 3. Consider Using Dataclasses (from previous feedback)

# --- Configuration ---
load_dotenv()

# --- Logging Setup ---
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_API_REQUEST_DELAY_SECONDS = 1
MAX_COMMENT_PREVIEW_LENGTH = 100 # 6. Magic Numbers: Was MAX_COMMENT_PREVIEW_LENGTH = 100, changed your example 150 to this
NEWLINE_CHAR = '\n' # 6. Magic Numbers: Use '\n'

# --- Dataclasses for Structured Data ---
# 5. Type Hints Could Be More Precise
@dataclass
class CommentData:
    id: str
    body: str
    author: str
    score: int
    created_utc: float
    submission_id: str
    submission_title: str

# For TypedDict alternative (if dataclasses are not preferred for some reason)
# class CommentData(TypedDict):
#     id: str
#     body: str
#     # ... and so on

# --- Rate Limiting Wrapper ---
# 2. More Granular Rate Limiting (Conceptual - PRAW handles some internal rate limiting)
# PRAW has its own rate limit handling. Explicit sleeps are more for politeness or specific known limits.
# This function is more illustrative for operations *outside* PRAW's direct calls if needed.
_T = TypeVar('_T')
def _execute_with_delay(operation: Callable[[], _T], delay_seconds: float = DEFAULT_API_REQUEST_DELAY_SECONDS) -> _T:
    """Executes an operation and then waits for a specified delay."""
    # Note: PRAW itself handles rate limiting for its API calls.
    # This explicit delay is more about being a "good citizen" between distinct high-level operations
    # or if we were making non-PRAW HTTP calls.
    # For PRAW operations, relying on PRAW's internal handling is usually sufficient.
    # We'll keep our explicit delays between SUBMISSION processing.
    result = operation()
    time.sleep(delay_seconds) # Apply delay *after* the operation for this example
    return result

# --- Reddit Client Initialization ---
def initialize_reddit_client() -> praw.Reddit:
    """Initializes and returns a PRAW Reddit instance."""
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT")

    if not all([client_id, client_secret, user_agent]):
        logger.critical("Essential Reddit API credentials not found.")
        raise ValueError("Missing Reddit API credentials.")
    try:
        reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)
        reddit.subreddit("all").display_name # Validation call
        logger.info("Successfully initialized and validated PRAW Reddit client.")
        return reddit
    except praw.exceptions.ResponseException as e:
        logger.error(f"Reddit API Response Error (status: {e.response.status_code}): {e}")
        raise
    except praw.exceptions.PRAWException as e:
        logger.error(f"PRAW error during client initialization: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during client initialization: {e}")
        raise

# --- Core Data Fetching Logic ---
# 3. Complex Logic in Single Function - Refactoring this now
def _stream_comments_from_single_submission(
    submission: praw.models.Submission,
    comments_to_yield_per_submission: Optional[int],
    filter_keyword: Optional[str] = None
) -> Iterator[CommentData]:
    """
    Streams comments from a single PRAW submission object.
    Handles true streaming by iterating directly and expanding MoreComments if needed.

    Args:
        submission: The PRAW submission object.
        comments_to_yield_per_submission: Max comments to yield. If None, attempts all.
        filter_keyword: Optional keyword to filter comment bodies.

    Yields:
        CommentData: Structured comment data.
    """
    # 1. True Streaming Implementation
    # No .list() here. Iterate directly for true streaming.
    # replace_more(limit=0) is used to clear out existing MoreComments objects if we *don't* want to expand them.
    # If we want to expand them on demand, we handle MoreComments instances in the loop.
    # The `limit` in `replace_more` determines how many `MoreComments` objects are replaced.
    # `limit=None` replaces all, `limit=0` replaces none (just processes what's loaded).
    # `limit=N` replaces N `MoreComments` instances.

    # For this "true streaming" approach where we might expand:
    # We will iterate and if we encounter MoreComments and haven't hit our limit, we expand it.
    # This is more complex than just `replace_more(limit=X)` upfront.

    comments_yielded_count = 0
    # PRAW's comment forest can be tricky. Iterating submission.comments can give MoreComments.
    # To handle this for streaming:
    try:
        # Initial load of top-level comments.
        # replace_more(limit=0) just processes what's there, doesn't fetch more.
        # if we want to expand some, we can't use limit=0 on the main submission.comments before iteration.
        # submission.comments.replace_more(limit=0) # If we strictly only want already loaded top-level

        comment_queue = list(submission.comments) # Start with top-level items

        while comment_queue:
            if comments_to_yield_per_submission is not None and comments_yielded_count >= comments_to_yield_per_submission:
                break

            item = comment_queue.pop(0) # Process like a queue

            if isinstance(item, praw.models.Comment):
                if filter_keyword and filter_keyword.lower() not in item.body.lower():
                    continue

                yield CommentData(
                    id=item.id, body=item.body,
                    author=str(item.author) if item.author else "[deleted]",
                    score=item.score, created_utc=item.created_utc,
                    submission_id=submission.id, submission_title=submission.title
                )
                comments_yielded_count += 1
                # Add replies to the queue if we want to traverse deeper (depth-first or breadth-first)
                # For this example, let's stick to top-level and their immediate expanded children from MoreComments.
                # To get replies: comment_queue.extend(item.replies)

            elif isinstance(item, praw.models.MoreComments):
                # Expand MoreComments if we haven't met the comment yield limit
                # and if comments_to_yield_per_submission is not 0 (meaning we want some comments)
                if (comments_to_yield_per_submission is None or comments_yielded_count < comments_to_yield_per_submission) and \
                   (comments_to_yield_per_submission != 0) :
                    try:
                        # 2. Rate Limiting: Fetching more comments IS an API call
                        time.sleep(DEFAULT_API_REQUEST_DELAY_SECONDS) # Politeness delay before expanding
                        more_comments_list = item.comments(update=True) # update=True fetches
                        comment_queue.extend(more_comments_list) # Add newly fetched comments to process
                        logger.debug(f"Expanded MoreComments for submission {submission.id}, got {len(more_comments_list)} items.")
                    except praw.exceptions.PRAWException as e_more:
                        # 4. Error Handling Could Be More Specific
                        if hasattr(e_more, 'response') and e_more.response is not None and e_more.response.status_code == 429:
                            logger.warning(f"Rate limit hit while expanding MoreComments for submission {submission.id}. Skipping expansion. Error: {e_more}")
                            # Implement backoff here if desired for retries
                        else:
                            logger.error(f"PRAWException expanding MoreComments for {submission.id}: {e_more}. Skipping expansion.")
                    except Exception as e_gen_more:
                        logger.error(f"Unexpected error expanding MoreComments for {submission.id}: {e_gen_more}. Skipping expansion.")
            # Else: it's some other PRAW object we might not care about in this context
    except praw.exceptions.PRAWException as e:
        logger.error(f"PRAW error streaming comments for submission {submission.id}: {e}")
        # Decide if this error should halt further processing for this submission or be re-raised
    except Exception as e:
        logger.error(f"Unexpected error streaming comments for submission {submission.id}: {e}")


def stream_comments_from_submissions_matching_keyword(
    reddit_instance: praw.Reddit,
    keyword: str,
    subreddit_name: str = "all",
    submission_search_limit: int = 10, # Renamed for clarity
    comments_per_submission: Optional[int] = 5,
    search_sort_order: str = "relevance", # Renamed for clarity
    filter_comment_bodies_by_keyword: bool = False # Renamed for clarity
) -> Iterator[CommentData]: # 5. Type Hints: Using Dataclass
    """
    Searches for submissions matching a keyword and streams their comments.

    Performance Implications (Doc Gap):
    - Setting `comments_per_submission=None` will attempt to fetch all comments,
      including expanding "MoreComments" objects, which can be VERY API intensive and slow.
    - A small `comments_per_submission` value is much faster.
    - `filter_comment_bodies_by_keyword=True` adds processing overhead per comment.

    Args:
        reddit_instance: Initialized PRAW Reddit instance.
        keyword: Keyword to search for in submissions.
        subreddit_name: Subreddit to search in.
        submission_search_limit: Max submissions to process.
        comments_per_submission: Max comments to yield from each submission.
                                 If None, attempts to stream all available comments (can be slow).
                                 If 0, yields no comments (useful for just getting submission info).
        search_sort_order: Sort order for submission search.
        filter_comment_bodies_by_keyword: If True, only yields comments containing the keyword.

    Yields:
        CommentData: Structured data for each relevant comment.

    Raises:
        praw.exceptions.PRAWException: For underlying PRAW API errors during submission search.
    """
    if not keyword:
        logger.warning("Keyword is empty. No comments will be fetched.")
        return

    logger.info(f"Starting to stream comments from submissions matching '{keyword}' in r/{subreddit_name}")

    try:
        subreddit = reddit_instance.subreddit(subreddit_name)
        processed_submissions_count = 0
        for submission in subreddit.search(keyword, limit=submission_search_limit, sort=search_sort_order):
            if processed_submissions_count >= submission_search_limit: # Ensure strict limit
                break
            processed_submissions_count += 1

            logger.debug(f"Processing submission (ID: {submission.id}): '{submission.title}'")
            # 2. Rate Limiting: Delay between processing each submission
            time.sleep(DEFAULT_API_REQUEST_DELAY_SECONDS)

            # Delegate comment fetching for this single submission
            comment_keyword_filter = keyword if filter_comment_bodies_by_keyword else None
            yield from _stream_comments_from_single_submission(
                submission,
                comments_per_submission,
                comment_keyword_filter
            )
        logger.info("Finished streaming comments from keyword-matched submissions.")
    except praw.exceptions.PRAWException as e:
        logger.error(f"PRAWException during submission search for '{keyword}': {e}")
        raise # Re-raise critical search failures
    except Exception as e:
        logger.error(f"Unexpected error during submission search for '{keyword}': {e}")
        raise

# --- Utility / Helper Functions ---
# 4. Poor Separation of Concerns (Addressed in main execution block)
def display_comment_summary(comment: CommentData, index: int):
    """Displays a summary of a CommentData object."""
    body_preview = comment.body[:MAX_COMMENT_PREVIEW_LENGTH].replace(NEWLINE_CHAR, ' ')
    logger.info(
        f"Comment {index + 1} (ID: {comment.id}, SubID: {comment.submission_id}, Score: {comment.score}): "
        f"{body_preview}..."
    )

def _run_search_scenario(
    reddit_client: praw.Reddit,
    scenario_name: str,
    search_kwargs: Dict[str, Any], # Explicitly pass search parameters
    max_display_limit: int = 5     # Limit for display in this scenario
):
    """Helper to run a search scenario and display results."""
    logger.info(f"\n--- Scenario: {scenario_name} ---")
    total_comments_yielded_and_displayed = 0
    try:
        # Pass only the relevant kwargs to the streaming function
        comment_stream = stream_comments_from_submissions_matching_keyword(
            reddit_instance=reddit_client, **search_kwargs
        )
        for i, comment_data in enumerate(comment_stream):
            display_comment_summary(comment_data, i)
            total_comments_yielded_and_displayed += 1
            if total_comments_yielded_and_displayed >= max_display_limit:
                logger.info(f"Reached display limit of {max_display_limit} for scenario '{scenario_name}'.")
                break
        if total_comments_yielded_and_displayed == 0:
            logger.info(f"No comments yielded (or displayed up to limit) for scenario '{scenario_name}'.")
    except praw.exceptions.PRAWException as e:
        logger.error(f"PRAW error during '{scenario_name}': {e}")
    except Exception as e:
        logger.error(f"Unexpected error during '{scenario_name}': {e}", exc_info=True)


if __name__ == "__main__":
    try:
        reddit_client_instance = initialize_reddit_client()

        _run_search_scenario(
            reddit_client_instance,
            "Basic search for 'Python'",
            search_kwargs={
                "keyword": "Python",
                "submission_search_limit": 2,
                "comments_per_submission": 1,
                "filter_comment_bodies_by_keyword": False,
            },
            max_display_limit=2
        )

        _run_search_scenario(
            reddit_client_instance,
            "Search 'OpenAI' in r/technology, filter comment bodies, stream more comments",
            search_kwargs={
                "keyword": "OpenAI",
                "subreddit_name": "technology",
                "submission_search_limit": 1,
                "comments_per_submission": 5,
                "filter_comment_bodies_by_keyword": True,
            },
            max_display_limit=5
        )

        _run_search_scenario(
            reddit_client_instance,
            "Search for submissions on 'space exploration', no comments from submissions",
            search_kwargs={
                "keyword": "space exploration",
                "submission_search_limit": 2,
                "comments_per_submission": 0, # Yields no comments
            },
            max_display_limit=0 # Display limit is 0 because we expect 0 comments
        )

    except ValueError as e:
        logger.critical(f"Script halted due to configuration error: {e}")
    except praw.exceptions.PRAWException as e:
        logger.critical(f"Script halted due to a PRAW error during setup or critical operation: {e}")
    except Exception as e:
        logger.critical(f"An unexpected critical error halted the script: {e}", exc_info=True)
    finally:
        logger.info("Script execution finished.")