# utils.py
import pickle
import logging

logger = logging.getLogger(__name__)

def save_ai_progress(data, filename):
    """Сохраняет прогресс AI в локальный файл."""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"AI progress saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving AI progress: {e}")

def load_ai_progress(filename):
    """Загружает прогресс AI из локального файла."""
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"AI progress loaded from {filename}")
        return data
    except
