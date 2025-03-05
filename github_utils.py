from github import Github, GithubException
import os
import base64
import logging
import time

logger = logging.getLogger(__name__)

GITHUB_USERNAME = os.environ.get("GITHUB_USERNAME", "Azerus96")
GITHUB_REPOSITORY = os.environ.get("GITHUB_REPOSITORY", "grofc")
AI_PROGRESS_FILENAME = "cfr_data.pkl"

def save_ai_progress_to_github(filename: str = AI_PROGRESS_FILENAME) -> bool:
    token = os.environ.get("AI_PROGRESS_TOKEN")
    if not token or not os.path.exists(filename):
        return False
    try:
        g = Github(token)
        repo = g.get_user(GITHUB_USERNAME).get_repo(GITHUB_REPOSITORY)
        with open(filename, "rb") as f:
            content = f.read()
        try:
            contents = repo.get_contents(filename, ref="main")
            repo.update_file(contents.path, f"Update AI progress ({time.strftime('%Y-%m-%d %H:%M:%S')})", content, contents.sha, branch="main")
        except GithubException as e:
            if e.status == 404:
                repo.create_file(filename, "Initial AI progress", content, branch="main")
            else:
                logger.error(f"GitHub API error: {e}")
                return False
        logger.info(f"Прогресс сохранён: {GITHUB_REPOSITORY}/{filename}")
        return True
    except Exception as e:
        logger.exception(f"Ошибка: {e}")
        return False

def load_ai_progress_from_github(filename: str = AI_PROGRESS_FILENAME) -> bool:
    token = os.environ.get("AI_PROGRESS_TOKEN")
    if not token:
        return False
    try:
        g = Github(token)
        repo = g.get_user(GITHUB_USERNAME).get_repo(GITHUB_REPOSITORY)
        contents = repo.get_contents(filename, ref="main")
        with open(filename, "wb") as f:
            f.write(base64.b64decode(contents.content))
        logger.info(f"Прогресс загружен: {GITHUB_REPOSITORY}/{filename}")
        return True
    except GithubException as e:
        if e.status == 404:
            logger.info(f"Файл не найден: {filename}")
        else:
            logger.error(f"GitHub API error: {e}")
        return False
