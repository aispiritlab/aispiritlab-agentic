import functools
from pathlib import Path
from typing import Callable, Any

from git import Repo
from git.exc import GitCommandError, InvalidGitRepositoryError, NoSuchPathError


def initial_tracking_project(path: Path) -> Repo:
    """
    Create an initial git repository for a project.

    Returns:
        Repo: The initialized git repository.
    """
    repo = Repo.init(path)
    repo.git.add(all=True)
    if not repo.is_dirty(untracked_files=True):
        return repo

    try:
        repo.git.commit(m="Initial commit")
    except GitCommandError as error:
        message = str(error).lower()
        if "nothing to commit" not in message and "no changes added to commit" not in message:
            raise
    return repo


def get_repo(path: Path) -> Repo:
    """
    Get the git repository at the specified path.

    Returns:
        Repo: The git repository at the specified path.
    """
    return Repo(path)


def commit(path: Callable[[], Path | None]) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            result = func(*args, **kwargs)
            repo_path = path()
            if repo_path is None:
                return result

            try:
                repo = get_repo(repo_path)
            except (InvalidGitRepositoryError, NoSuchPathError, TypeError):
                return result

            # Stage all changes, including newly created files.
            repo.git.add(all=True)
            if not repo.is_dirty(untracked_files=True):
                return result

            try:
                repo.git.commit(m=f"Commit from {func.__name__}")
            except GitCommandError as error:
                message = str(error).lower()
                if "nothing to commit" not in message and "no changes added to commit" not in message:
                    raise
            return result
        return wrapper
    return decorator
