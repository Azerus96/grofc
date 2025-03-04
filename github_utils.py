from github import Github, GithubException
import os
import base64
import logging
import time
from typing import Optional

# Настройка логирования
logger = logging.getLogger(__name__)

# GitHub repository settings (can be overridden by environment variables)
GITHUB_USERNAME = os.environ.get("GITHUB_USERNAME") or "Azerus96"
GITHUB_REPOSITORY = os.environ.get("GITHUB_REPOSITORY") or "grofc"
AI_PROGRESS_FILENAME = "cfr_data.pkl"


def save_ai_progress_to_github(filename: str = AI_PROGRESS_FILENAME) -> bool:
    """
    Saves AI progress file to GitHub repository.

    Args:
        filename (str): Name of the file to save. Defaults to AI_PROGRESS_FILENAME.

    Returns:
        bool: True if save was successful, False otherwise.
    """
    token = os.environ.get("AI_PROGRESS_TOKEN")
    if not token:
        logger.warning("AI_PROGRESS_TOKEN not set. Progress saving to GitHub disabled.")
        return False

    # Проверка существования и размера локального файла
    if not os.path.exists(filename):
        logger.error(f"Local file {filename} does not exist. Cannot save to GitHub.")
        return False
        
    file_stats = os.stat(filename)
    file_size = file_stats.st_size
    file_mtime = file_stats.st_mtime
    
    if file_size == 0:
        logger.error(f"File {filename} is empty. Not saving to GitHub.")
        return False
        
    logger.info(f"Saving AI progress to GitHub. File size: {file_size} bytes, " 
                f"Last modified: {time.ctime(file_mtime)}")

    try:
        g = Github(token)
        repo = g.get_user(GITHUB_USERNAME).get_repo(GITHUB_REPOSITORY)

        try:
            # Получаем текущий файл с GitHub для сравнения
            contents = repo.get_contents(filename, ref="main")
            github_content = base64.b64decode(contents.content)
            
            # Читаем локальный файл
            with open(filename, "rb") as f:
                local_content = f.read()
                
            # Сравниваем содержимое, обновляем только если есть изменения
            if github_content == local_content:
                logger.info("Local file is identical to GitHub version. Skipping upload.")
                return True
                
            # Обновляем файл
            repo.update_file(
                contents.path,
                f"Update AI progress ({time.strftime('%Y-%m-%d %H:%M:%S')})",
                local_content,
                contents.sha,
                branch="main",
            )
            logger.info(f"AI progress updated on GitHub: {GITHUB_REPOSITORY}/{filename}")
            return True
        except GithubException as e:
            if e.status == 404:
                # Файл не существует, создаем новый
                with open(filename, "rb") as f:
                    local_content = f.read()
                repo.create_file(
                    filename,
                    f"Initial AI progress ({time.strftime('%Y-%m-%d %H:%M:%S')})",
                    local_content,
                    branch="main",
                )
                logger.info(f"Created new file for AI progress on GitHub: {GITHUB_REPOSITORY}/{filename}")
                return True
            else:
                logger.error(f"GitHub API error saving progress: status={e.status}, data={e.data}")
                return False

    except Exception as e:
        logger.exception(f"Unexpected error during saving to GitHub: {e}")
        return False


def load_ai_progress_from_github(filename: str = AI_PROGRESS_FILENAME) -> bool:
    """
    Loads AI progress file from GitHub repository.

    Args:
        filename (str): Name of the file to load. Defaults to AI_PROGRESS_FILENAME.

    Returns:
        bool: True if load was successful, False otherwise.
    """
    token = os.environ.get("AI_PROGRESS_TOKEN")
    if not token:
        logger.warning("AI_PROGRESS_TOKEN not set. Progress loading from GitHub disabled.")
        return False

    try:
        g = Github(token)
        repo = g.get_user(GITHUB_USERNAME).get_repo(GITHUB_REPOSITORY)
        
        try:
            contents = repo.get_contents(filename, ref="main")
            file_content = base64.b64decode(contents.content)
            
            # Проверяем, не пустой ли файл
            if len(file_content) == 0:
                logger.warning("GitHub file is empty. Not downloading.")
                return False
            
            # Проверяем, существует ли локальный файл
            if os.path.exists(filename):
                # Сравниваем содержимое
                with open(filename, "rb") as f:
                    local_content = f.read()
                
                if file_content == local_content:
                    logger.info("Local file is identical to GitHub version. Skipping download.")
                    return True
            
            # Сохраняем скачанный файл
            with open(filename, "wb") as f:
                f.write(file_content)
            
            logger.info(f"AI progress loaded from GitHub: {GITHUB_REPOSITORY}/{filename}, " 
                        f"Size: {len(file_content)} bytes")
            return True
            
        except GithubException as e:
            if e.status == 404:
                logger.info(f"Progress file {filename} not found in GitHub repository.")
                return False
            else:
                logger.error(f"GitHub API error loading progress: status={e.status}, data={e.data}")
                return False
    except Exception as e:
        logger.exception(f"Unexpected error during loading from GitHub: {e}")
        return False
