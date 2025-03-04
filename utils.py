# utils.py
import pickle
import logging

logger = logging.getLogger(__name__)

def save_ai_progress(data, filename):
    """Saves AI progress to a local file."""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"AI progress saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving AI progress: {e}")

def load_ai_progress(filename):
    """Loads AI progress from a local file."""
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"AI progress loaded from {filename}")
        return data
    except FileNotFoundError:
        logger.info(f"No progress file found at {filename}")
        return None  # Return None explicitly if file not found
    except Exception as e:
        logger.error(f"Error loading AI progress: {e}")
        return None
