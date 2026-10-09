from __future__ import annotations

"""Kaggle API client wrapper.

Provides a simplified interface to the Kaggle API for competition management.

Uses kagglehub (the modern Kaggle Python library) for downloads and data loading,
with fallback to the older kaggle package for submission functionality.
"""

import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Lazy imports for kaggle libraries
HAS_KAGGLEHUB = False
HAS_KAGGLE_LEGACY = False

try:
    import kagglehub
    HAS_KAGGLEHUB = True
except ImportError:
    pass

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    HAS_KAGGLE_LEGACY = True
except (ImportError, OSError, PermissionError):
    # OSError/PermissionError can occur if kaggle tries to create config dir
    KaggleApi = None
    pass


def _ensure_kaggle():
    """Ensure at least one kaggle library is available."""
    if not HAS_KAGGLEHUB and not HAS_KAGGLE_LEGACY:
        raise ImportError(
            "A Kaggle package is required for this functionality.\n"
            "Install the modern kagglehub: pip install kagglehub\n"
            "Or the legacy kaggle-api: pip install kaggle\n\n"
            "Then authenticate at https://www.kaggle.com/settings"
        )


def _ensure_kaggle_legacy():
    """Ensure legacy kaggle package is available (needed for submissions)."""
    if not HAS_KAGGLE_LEGACY:
        raise ImportError(
            "The kaggle package is required for submissions.\n"
            "Install with: pip install kaggle\n"
            "Then create an API token at https://www.kaggle.com/settings "
            "and place kaggle.json in ~/.kaggle/"
        )


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Read a Kaggle API field by its camelCase name; kaggle >= 1.7 (kagglesdk) uses snake_case."""
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    return getattr(obj, snake, getattr(obj, name, default))


def _slug(ref: str) -> str:
    """'https://www.kaggle.com/competitions/titanic' (kagglesdk) or 'titanic' -> 'titanic'."""
    return str(ref).rstrip("/").split("/")[-1]


def _link_or_copy(src: Path, dest_dir: Path) -> None:
    """Put the kagglehub cache file or folder `src` into `dest_dir`, hard-linking files
    (no second copy of a multi-GB dataset) and copying only across filesystems."""
    import os
    import shutil

    def place(file: Path, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            dest.unlink()
        try:
            os.link(file, dest)
        except OSError:
            shutil.copy2(file, dest)

    if src.is_file():
        place(src, dest_dir / src.name)
        return
    for file in src.rglob("*"):
        if file.is_file():
            place(file, dest_dir / file.relative_to(src))


@dataclass
class CompetitionInfo:
    """Information about a Kaggle competition.

    Attributes
    ----------
    slug : str
        Competition URL slug (e.g., 'titanic').
    title : str
        Full competition title.
    category : str
        Competition category (e.g., 'Getting Started', 'Featured').
    deadline : Optional[datetime]
        Competition deadline.
    description : str
        Short description of the competition.
    evaluation_metric : str
        Metric used for evaluation.
    reward : str
        Prize/reward description.
    team_count : int
        Number of teams participating.
    url : str
        Full URL to competition page.
    rules_url : str
        URL to competition rules.
    data_files : List[str]
        List of available data files.
    can_submit : bool
        Whether submissions are currently accepted.
    user_has_entered : bool
        Whether the authenticated user has entered.
    merger_deadline : Optional[datetime]
        Team merger deadline if applicable.
    """
    slug: str
    title: str = ""
    category: str = ""
    deadline: datetime | None = None
    description: str = ""
    evaluation_metric: str = ""
    reward: str = ""
    team_count: int = 0
    url: str = ""
    rules_url: str = ""
    data_files: list[str] = field(default_factory=list)
    can_submit: bool = True
    user_has_entered: bool = False
    merger_deadline: datetime | None = None

    def __str__(self) -> str:
        deadline_str = self.deadline.strftime("%Y-%m-%d") if self.deadline else "N/A"
        return (
            f"Competition: {self.title}\n"
            f"  Slug: {self.slug}\n"
            f"  Category: {self.category}\n"
            f"  Deadline: {deadline_str}\n"
            f"  Metric: {self.evaluation_metric}\n"
            f"  Teams: {self.team_count}\n"
            f"  Reward: {self.reward}"
        )


@dataclass
class SubmissionInfo:
    """Information about a submission.

    Attributes
    ----------
    submission_id : int
        Unique submission ID.
    date : datetime
        Submission timestamp.
    description : str
        Submission message/description.
    status : str
        Submission status (e.g., 'complete', 'pending', 'error').
    public_score : Optional[float]
        Public leaderboard score.
    private_score : Optional[float]
        Private leaderboard score (after competition ends).
    file_name : str
        Name of submitted file.
    """
    submission_id: int
    date: datetime
    description: str = ""
    status: str = "pending"
    public_score: float | None = None
    private_score: float | None = None
    file_name: str = ""

    def __str__(self) -> str:
        score_str = f"{self.public_score:.5f}" if self.public_score is not None else "N/A"
        return (
            f"Submission #{self.submission_id} ({self.date.strftime('%Y-%m-%d %H:%M')})\n"
            f"  Status: {self.status}\n"
            f"  Score: {score_str}\n"
            f"  Message: {self.description}"
        )


@dataclass
class SubmissionResult:
    """Result of a submission attempt.

    Attributes
    ----------
    success : bool
        Whether the submission was accepted.
    message : str
        Status message from Kaggle.
    submission_id : Optional[int]
        Submission ID if successful.
    public_score : Optional[float]
        Public score if immediately available.
    error : Optional[str]
        Error message if submission failed.
    """
    success: bool
    message: str = ""
    submission_id: int | None = None
    public_score: float | None = None
    error: str | None = None

    def __str__(self) -> str:
        if self.success:
            score_str = f"{self.public_score:.5f}" if self.public_score is not None else "pending"
            return f"Submission successful! Score: {score_str}"
        return f"Submission failed: {self.error or self.message}"


@dataclass
class DatasetInfo:
    """Information about a Kaggle dataset.

    Attributes
    ----------
    slug : str
        Dataset slug (owner/dataset-name).
    title : str
        Dataset title.
    size : int
        Dataset size in bytes.
    last_updated : Optional[datetime]
        Last update timestamp.
    download_count : int
        Number of downloads.
    vote_count : int
        Number of upvotes.
    usability_rating : float
        Usability rating (0-1).
    """
    slug: str
    title: str = ""
    size: int = 0
    last_updated: datetime | None = None
    download_count: int = 0
    vote_count: int = 0
    usability_rating: float = 0.0


class KaggleClient:
    """Wrapper around Kaggle APIs with convenient methods.

    Uses kagglehub (modern library) for downloads and data loading,
    with fallback to legacy kaggle package for submission functionality.

    Provides simplified access to Kaggle competition data, submissions,
    and datasets. Handles authentication automatically.

    Examples
    --------
    >>> client = KaggleClient()
    >>> client.authenticate()
    True

    >>> # List competitions (requires legacy kaggle package)
    >>> comps = client.list_competitions(search="tabular")
    >>> for c in comps[:5]:
    ...     print(c.title)

    >>> # Download competition data (uses kagglehub)
    >>> client.download_competition("titanic", path="./data")

    >>> # Submit predictions (requires legacy kaggle package)
    >>> result = client.submit("titanic", "submission.csv", "My first submission")
    >>> print(result)

    Notes
    -----
    - Downloads use kagglehub (pip install kagglehub) - faster and simpler
    - Submissions require the legacy kaggle package (pip install kaggle)
    - Authentication: run `kagglehub.login()` or place kaggle.json in ~/.kaggle/
    """

    def __init__(self):
        _ensure_kaggle()
        self._legacy_api: KaggleApi | None = None
        self._authenticated = False

    @property
    def legacy_api(self) -> KaggleApi:
        """Get authenticated legacy API instance (for submissions)."""
        _ensure_kaggle_legacy()
        if self._legacy_api is None:
            self._legacy_api = KaggleApi()
            self._legacy_api.authenticate()
        return self._legacy_api

    def authenticate(self) -> bool:
        """Authenticate with Kaggle.

        For kagglehub, this will prompt for credentials if not already set.
        For legacy API, reads from ~/.kaggle/kaggle.json.

        Returns
        -------
        bool
            True if authentication successful.

        Raises
        ------
        RuntimeError
            If authentication fails.
        """
        try:
            if HAS_KAGGLEHUB:
                # kagglehub handles auth automatically, but we can trigger login
                kagglehub.login()
            elif HAS_KAGGLE_LEGACY:
                _ = self.legacy_api  # Triggers authentication
            self._authenticated = True
            return True
        except Exception as e:
            raise RuntimeError(
                f"Kaggle authentication failed: {e}\n"
                "Options:\n"
                "  1. Run kagglehub.login() to authenticate interactively\n"
                "  2. Set KAGGLE_USERNAME and KAGGLE_KEY environment variables\n"
                "  3. Place kaggle.json in ~/.kaggle/\n"
                "Get your API token from https://www.kaggle.com/settings"
            ) from e

    def list_competitions(
        self,
        search: str | None = None,
        category: str | None = None,
        sort_by: str = "latestDeadline",
        page: int = 1,
    ) -> list[CompetitionInfo]:
        """List available Kaggle competitions.

        Parameters
        ----------
        search : str, optional
            Search term to filter competitions.
        category : str, optional
            Filter by category: 'all', 'featured', 'research',
            'recruitment', 'gettingStarted', 'masters', 'playground'.
        sort_by : str, default='latestDeadline'
            Sort order: 'latestDeadline', 'earliestDeadline',
            'recentlyCreated', 'numberOfTeams', 'prize'.
        page : int, default=1
            Page number for pagination.

        Returns
        -------
        List[CompetitionInfo]
            List of competition information objects.
        """
        competitions = self.legacy_api.competitions_list(
            search=search,
            category=category,
            sort_by=sort_by,
            page=page,
        )

        competitions = getattr(competitions, 'competitions', competitions) or []
        return [self._parse_competition(c) for c in competitions]

    def get_competition(self, competition: str) -> CompetitionInfo:
        """Get detailed information about a specific competition.

        Parameters
        ----------
        competition : str
            Competition slug (e.g., 'titanic').

        Returns
        -------
        CompetitionInfo
            Competition details including available files.
        """
        info = self._parse_competition(self._find_competition(competition))

        # Get file list
        try:
            files = self.legacy_api.competition_list_files(competition)
            info.data_files = [f.name for f in getattr(files, 'files', files) or []]
        except Exception:
            pass

        return info

    def _find_competition(self, competition: str) -> Any:
        """The API record of the competition with this exact slug."""
        competitions = self.legacy_api.competitions_list(search=competition)
        for c in getattr(competitions, 'competitions', competitions) or []:
            if _slug(c.ref) == competition:
                return c
        raise ValueError(f"Competition '{competition}' not found")

    def competition_pages(self, competition: str) -> dict[str, str]:
        """Text of a competition's website pages, which the public API does not serve.

        Keys are Kaggle's page names, e.g. 'Description' (the overview), 'Evaluation',
        'Timeline', 'Submission Requirements', 'data-description' and 'rules'.
        Uses the endpoint kaggle.com's own pages call, so it may change without notice.

        Parameters
        ----------
        competition : str
            Competition slug.

        Returns
        -------
        Dict[str, str]
            Page name -> markdown (HTML pages are reduced to text).
        """
        import html

        import requests

        comp_id = _get(self._find_competition(competition), 'id')
        with requests.Session() as http:
            http.get(f"https://www.kaggle.com/competitions/{competition}", timeout=30)
            response = http.post(
                "https://www.kaggle.com/api/i/competitions.PageService/ListPages",
                json={"competitionId": comp_id},
                headers={"X-XSRF-TOKEN": http.cookies.get("XSRF-TOKEN", "")},
                timeout=30,
            )
            response.raise_for_status()

        pages = {}
        for page in response.json().get("pages", []):
            content = page.get("content") or ""
            if "html" in (page.get("mimeType") or ""):
                content = html.unescape(re.sub(r"<[^>]+>", "", content))
            if content.strip():
                pages[page["name"]] = content
        return pages

    def _parse_competition(self, comp: Any) -> CompetitionInfo:
        """Parse competition object into CompetitionInfo."""
        deadline = _get(comp, 'deadline')
        if isinstance(deadline, str):
            try:
                deadline = datetime.fromisoformat(deadline.replace('Z', '+00:00'))
            except ValueError:
                deadline = None
        merger_deadline = _get(comp, 'mergerDeadline')
        if not isinstance(merger_deadline, datetime):
            merger_deadline = None
        slug = _slug(_get(comp, 'ref', ''))

        return CompetitionInfo(
            slug=slug,
            title=_get(comp, 'title', ''),
            category=_get(comp, 'category', ''),
            deadline=deadline,
            description=_get(comp, 'description', ''),
            evaluation_metric=_get(comp, 'evaluationMetric', ''),
            reward=_get(comp, 'reward', ''),
            team_count=_get(comp, 'teamCount', 0),
            url=_get(comp, 'url', f"https://www.kaggle.com/c/{slug}"),
            rules_url=f"https://www.kaggle.com/c/{slug}/rules",
            can_submit=_get(comp, 'canSubmit', True),
            user_has_entered=_get(comp, 'userHasEntered', False),
            merger_deadline=merger_deadline,
        )

    def list_competition_files(self, competition: str) -> list[dict[str, Any]]:
        """List files available in a competition.

        Parameters
        ----------
        competition : str
            Competition slug.

        Returns
        -------
        List[Dict[str, Any]]
            List of file info dicts with 'name', 'size', 'creationDate'.
        """
        files = self.legacy_api.competition_list_files(competition)
        files = getattr(files, 'files', files) or []
        return [
            {
                "name": f.name,
                "size": _get(f, 'totalBytes', _get(f, 'size', 0)),
                "creation_date": _get(f, 'creationDate', None),
            }
            for f in files
        ]

    def download_competition(
        self,
        competition: str,
        path: str | Path = ".",
        file_name: str | None = None,
        force: bool = False,
        quiet: bool = False,
        unzip: bool = True,
    ) -> Path:
        """Download competition data files.

        Uses kagglehub for fast, cached downloads.

        Parameters
        ----------
        competition : str
            Competition slug (e.g., 'titanic', 'digit-recognizer').
        path : str or Path, default='.'
            Directory to copy files to. If not specified, returns cache path.
        file_name : str, optional
            Specific file to download. If None, downloads all files.
        force : bool, default=False
            Force re-download even if files exist in cache.
        quiet : bool, default=False
            Suppress download progress output (legacy API only).
        unzip : bool, default=True
            Automatically unzip downloaded files (files are auto-extracted by kagglehub).

        Returns
        -------
        Path
            Path to downloaded files.

        Notes
        -----
        You must accept the competition rules on the Kaggle website
        before downloading data.
        """
        path = Path(path)

        try:
            if HAS_KAGGLEHUB:
                # Use kagglehub for downloads (faster, better caching)
                if file_name:
                    cache_path = kagglehub.competition_download(
                        competition,
                        path=file_name,
                        force_download=force,
                    )
                else:
                    cache_path = kagglehub.competition_download(
                        competition,
                        force_download=force,
                    )

                cache_path = Path(cache_path)

                if path != Path("."):
                    _link_or_copy(cache_path, path)
                    return path

                return cache_path

            else:
                # Fallback to legacy API
                path.mkdir(parents=True, exist_ok=True)

                if file_name:
                    self.legacy_api.competition_download_file(
                        competition,
                        file_name,
                        path=str(path),
                        force=force,
                        quiet=quiet,
                    )
                else:
                    self.legacy_api.competition_download_files(
                        competition,
                        path=str(path),
                        force=force,
                        quiet=quiet,
                    )

                if unzip:
                    self._unzip_files(path)

                return path

        except Exception as e:
            error_msg = str(e)
            if "403" in error_msg or "accept" in error_msg.lower() or "401" in error_msg:
                raise RuntimeError(
                    f"Access denied for competition '{competition}'.\n"
                    f"Please accept the competition rules at:\n"
                    f"https://www.kaggle.com/c/{competition}/rules"
                ) from e
            raise

    def _unzip_files(self, directory: Path) -> None:
        """Unzip all .zip files in directory."""
        import zipfile

        for zip_file in directory.glob("*.zip"):
            try:
                with zipfile.ZipFile(zip_file, 'r') as zf:
                    zf.extractall(directory)
                zip_file.unlink()  # Remove zip after extraction
            except Exception as e:
                warnings.warn(f"Failed to unzip {zip_file}: {e}")

    def submit(
        self,
        competition: str,
        file_path: str | Path,
        message: str,
        quiet: bool = False,
    ) -> SubmissionResult:
        """Submit predictions to a competition.

        Parameters
        ----------
        competition : str
            Competition slug.
        file_path : str or Path
            Path to submission file.
        message : str
            Submission description/message.
        quiet : bool, default=False
            Suppress output.

        Returns
        -------
        SubmissionResult
            Result of submission attempt.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            return SubmissionResult(
                success=False,
                error=f"Submission file not found: {file_path}"
            )

        try:
            self.legacy_api.competition_submit(
                file_name=str(file_path),
                message=message,
                competition=competition,
                quiet=quiet,
            )

            # Try to get the submission info
            submissions = self.get_submissions(competition, limit=1)

            if submissions:
                latest = submissions[0]
                return SubmissionResult(
                    success=True,
                    message="Submission successful",
                    submission_id=latest.submission_id,
                    public_score=latest.public_score,
                )

            return SubmissionResult(
                success=True,
                message="Submission uploaded successfully",
            )

        except Exception as e:
            error_msg = str(e)
            if "403" in error_msg:
                return SubmissionResult(
                    success=False,
                    error="Access denied. Please accept competition rules first."
                )
            return SubmissionResult(
                success=False,
                error=error_msg,
            )

    def get_submissions(
        self,
        competition: str,
        limit: int | None = None,
    ) -> list[SubmissionInfo]:
        """Get submission history for a competition.

        Parameters
        ----------
        competition : str
            Competition slug.
        limit : int, optional
            Maximum number of submissions to return.

        Returns
        -------
        List[SubmissionInfo]
            List of submissions, most recent first.
        """
        submissions = self.legacy_api.competition_submissions(competition)

        result = []
        for s in submissions:
            date = _get(s, 'date')
            if not isinstance(date, datetime):
                date = datetime.now()

            public_score = None
            if _get(s, 'publicScore'):
                try:
                    public_score = float(_get(s, 'publicScore'))
                except (ValueError, TypeError):
                    pass

            private_score = None
            if _get(s, 'privateScore'):
                try:
                    private_score = float(_get(s, 'privateScore'))
                except (ValueError, TypeError):
                    pass

            result.append(SubmissionInfo(
                submission_id=_get(s, 'ref', 0),
                date=date,
                description=_get(s, 'description', ''),
                status=_get(s, 'status', 'complete'),
                public_score=public_score,
                private_score=private_score,
                file_name=_get(s, 'fileName', ''),
            ))

        if limit:
            result = result[:limit]

        return result

    def get_leaderboard(
        self,
        competition: str,
        page: int = 1,
    ) -> list[dict[str, Any]]:
        """Get competition leaderboard.

        Parameters
        ----------
        competition : str
            Competition slug.
        page : int, default=1
            Page number.

        Returns
        -------
        List[Dict[str, Any]]
            Leaderboard entries with 'rank', 'team_name', 'score'.
        """
        try:
            leaderboard = self.legacy_api.competition_leaderboard_view(competition)

            result = []
            for entry in leaderboard:
                result.append({
                    "rank": _get(entry, 'rank', 0),
                    "team_name": _get(entry, 'teamName', ''),
                    "score": _get(entry, 'score', None),
                    "entries": _get(entry, 'submissionCount', 0),
                    "last_submission": _get(entry, 'lastSubmissionDate', _get(entry, 'submissionDate')),
                })

            return result

        except Exception as e:
            warnings.warn(f"Failed to get leaderboard: {e}")
            return []

    # Notebook (kernel) methods

    def list_notebooks(
        self,
        competition: str | None = None,
        search: str | None = None,
        sort_by: str = "hotness",
        page_size: int = 20,
        language: str | None = None,
    ) -> list[dict[str, Any]]:
        """List public notebooks, e.g. the hottest ones for a competition.

        Parameters
        ----------
        competition : str, optional
            Competition slug to filter by.
        search : str, optional
            Search term.
        sort_by : str, default='hotness'
            'hotness', 'voteCount', 'dateRun', 'dateCreated', 'commentCount',
            'viewCount', 'scoreDescending', 'scoreAscending' or 'relevance'.
        page_size : int, default=20
            Number of notebooks to return (max 200).
        language : str, optional
            'python', 'r', ... (default: all).

        Returns
        -------
        List[Dict[str, Any]]
            Notebooks with 'ref', 'title', 'author', 'votes', 'last_run', 'url'.
        """
        kernels = self.legacy_api.kernels_list(
            competition=competition,
            search=search,
            sort_by=sort_by,
            page_size=page_size,
            language=language,
        ) or []
        return [
            {
                "ref": k.ref,
                "title": k.title,
                "author": k.author,
                "votes": k.total_votes,
                "last_run": k.last_run_time,
                "url": f"https://www.kaggle.com/code/{k.ref}",
            }
            for k in kernels
            if k is not None
        ]

    def read_notebook(self, ref: str) -> dict[str, Any]:
        """Download a notebook and return its source as text.

        Markdown cells are kept as-is and code cells are fenced; cell outputs
        (images, tables) are dropped.

        Parameters
        ----------
        ref : str
            Notebook reference, 'owner/slug' (a full kaggle.com/code URL also works).

        Returns
        -------
        Dict[str, Any]
            'ref', 'title', 'language', 'data_sources' and 'source' (text).
        """
        import json
        import tempfile

        ref = ref.split("kaggle.com/code/")[-1].strip("/")
        with tempfile.TemporaryDirectory() as tmp:
            self.legacy_api.kernels_pull(ref, path=tmp, metadata=True)
            meta = json.loads((Path(tmp) / "kernel-metadata.json").read_text())
            code_file = Path(tmp) / meta["code_file"]
            raw = code_file.read_text()

        if code_file.suffix == ".ipynb":
            nb = json.loads(raw)
            fence = "```r" if meta.get("language") == "r" else "```python"
            parts = []
            for cell in nb.get("cells", []):
                text = "".join(cell.get("source", []))
                if not text.strip():
                    continue
                parts.append(text if cell.get("cell_type") == "markdown" else f"{fence}\n{text}\n```")
            raw = "\n\n".join(parts)

        return {
            "ref": ref,
            "title": meta.get("title", ""),
            "language": meta.get("language", ""),
            "data_sources": {
                "competitions": meta.get("competition_sources", []),
                "datasets": meta.get("dataset_sources", []),
                "notebooks": meta.get("kernel_sources", []),
            },
            "source": raw,
        }

    def push_notebook(
        self,
        code_file: str | Path,
        title: str,
        competition: str | None = None,
        datasets: list[str] | None = None,
        public: bool = False,
        enable_gpu: bool = False,
        enable_internet: bool = False,
    ) -> dict[str, Any]:
        """Upload a notebook (.ipynb) or script (.py) to Kaggle and run it there.

        Pushing the same title again creates a new version of the same notebook.

        Parameters
        ----------
        code_file : str or Path
            Local .ipynb or .py file.
        title : str
            Notebook title; its slug becomes the notebook id.
        competition : str, optional
            Competition whose data is attached (under /kaggle/input/).
        datasets : list of str, optional
            Dataset slugs ('owner/name') to attach.
        public : bool, default=False
            Make the notebook public. Private by default.
        enable_gpu, enable_internet : bool, default=False
            Kaggle runtime settings.

        Returns
        -------
        Dict[str, Any]
            'ref', 'url', 'version' and 'error' (empty on success).
        """
        import json
        import re
        import shutil
        import tempfile

        code_file = Path(code_file)
        if code_file.suffix not in (".ipynb", ".py"):
            raise ValueError(f"Expected a .ipynb or .py file, got {code_file.name}")
        username = self.legacy_api.get_config_value("username")
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        meta = {
            "id": f"{username}/{slug}",
            "title": title,
            "code_file": code_file.name,
            "language": "python",
            "kernel_type": "notebook" if code_file.suffix == ".ipynb" else "script",
            "is_private": not public,
            "enable_gpu": enable_gpu,
            "enable_internet": enable_internet,
            "competition_sources": [competition] if competition else [],
            "dataset_sources": datasets or [],
            "kernel_sources": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy2(code_file, Path(tmp) / code_file.name)
            (Path(tmp) / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
            resp = self.legacy_api.kernels_push(tmp)

        return {
            "ref": resp.ref or meta["id"],
            "url": resp.url or f"https://www.kaggle.com/code/{meta['id']}",
            "version": resp.version_number,
            "error": resp.error or "",
        }

    # Dataset methods

    def list_datasets(
        self,
        search: str | None = None,
        sort_by: str = "hottest",
        file_type: str | None = None,
        page: int = 1,
    ) -> list[DatasetInfo]:
        """List available Kaggle datasets.

        Parameters
        ----------
        search : str, optional
            Search term.
        sort_by : str, default='hottest'
            Sort order: 'hottest', 'votes', 'updated', 'active'.
        file_type : str, optional
            Filter by file type: 'csv', 'json', 'sqlite', etc.
        page : int, default=1
            Page number.

        Returns
        -------
        List[DatasetInfo]
            List of dataset information objects.
        """
        datasets = self.legacy_api.dataset_list(
            search=search,
            sort_by=sort_by,
            file_type=file_type,
            page=page,
        )

        result = []
        for d in datasets:
            last_updated = _get(d, 'lastUpdated')
            if not isinstance(last_updated, datetime):
                last_updated = None

            result.append(DatasetInfo(
                slug=_get(d, 'ref', ''),
                title=_get(d, 'title', ''),
                size=_get(d, 'totalBytes', 0),
                last_updated=last_updated,
                download_count=_get(d, 'downloadCount', 0),
                vote_count=_get(d, 'voteCount', 0),
                usability_rating=_get(d, 'usabilityRating', 0.0),
            ))

        return result

    def download_dataset(
        self,
        dataset: str,
        path: str | Path = ".",
        file_name: str | None = None,
        force: bool = False,
        quiet: bool = False,
        unzip: bool = True,
    ) -> Path:
        """Download a Kaggle dataset.

        Uses kagglehub for fast, cached downloads when available.

        Parameters
        ----------
        dataset : str
            Dataset slug (owner/dataset-name, e.g., 'bricevergnou/spotify-recommendation').
        path : str or Path, default='.'
            Directory to download to.
        file_name : str, optional
            Specific file to download.
        force : bool, default=False
            Force re-download.
        quiet : bool, default=False
            Suppress output (legacy API only).
        unzip : bool, default=True
            Automatically unzip files.

        Returns
        -------
        Path
            Path to downloaded files.
        """
        path = Path(path)

        if HAS_KAGGLEHUB:
            # Use kagglehub for downloads
            if file_name:
                cache_path = kagglehub.dataset_download(dataset, path=file_name, force_download=force)
            else:
                cache_path = kagglehub.dataset_download(dataset, force_download=force)

            cache_path = Path(cache_path)

            if path != Path("."):
                _link_or_copy(cache_path, path)
                return path

            return cache_path

        else:
            # Fallback to legacy API
            path.mkdir(parents=True, exist_ok=True)

            if file_name:
                self.legacy_api.dataset_download_file(
                    dataset,
                    file_name,
                    path=str(path),
                    force=force,
                    quiet=quiet,
                )
            else:
                self.legacy_api.dataset_download_files(
                    dataset,
                    path=str(path),
                    force=force,
                    quiet=quiet,
                    unzip=unzip,
                )

            if unzip and not file_name:
                self._unzip_files(path)

            return path
