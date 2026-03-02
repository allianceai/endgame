from __future__ import annotations

"""Kaggle API client wrapper.

Provides a simplified interface to the Kaggle API for competition management.

Uses kagglehub (the modern Kaggle Python library) for downloads and data loading,
with fallback to the older kaggle package for submission functionality.
"""

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
        # Get competition list filtered by exact name
        competitions = self.legacy_api.competitions_list(search=competition)

        # Find exact match
        comp_data = None
        for c in competitions:
            if c.ref == competition:
                comp_data = c
                break

        if comp_data is None:
            raise ValueError(f"Competition '{competition}' not found")

        info = self._parse_competition(comp_data)

        # Get file list
        try:
            files = self.legacy_api.competition_list_files(competition)
            info.data_files = [f.name for f in files]
        except Exception:
            pass

        return info

    def _parse_competition(self, comp: Any) -> CompetitionInfo:
        """Parse competition object into CompetitionInfo."""
        deadline = None
        if hasattr(comp, 'deadline') and comp.deadline:
            if isinstance(comp.deadline, datetime):
                deadline = comp.deadline
            elif isinstance(comp.deadline, str):
                try:
                    deadline = datetime.fromisoformat(comp.deadline.replace('Z', '+00:00'))
                except Exception:
                    pass

        merger_deadline = None
        if hasattr(comp, 'mergerDeadline') and comp.mergerDeadline:
            if isinstance(comp.mergerDeadline, datetime):
                merger_deadline = comp.mergerDeadline

        return CompetitionInfo(
            slug=getattr(comp, 'ref', ''),
            title=getattr(comp, 'title', ''),
            category=getattr(comp, 'category', ''),
            deadline=deadline,
            description=getattr(comp, 'description', ''),
            evaluation_metric=getattr(comp, 'evaluationMetric', ''),
            reward=getattr(comp, 'reward', ''),
            team_count=getattr(comp, 'teamCount', 0),
            url=getattr(comp, 'url', f"https://www.kaggle.com/c/{getattr(comp, 'ref', '')}"),
            rules_url=f"https://www.kaggle.com/c/{getattr(comp, 'ref', '')}/rules",
            can_submit=getattr(comp, 'canSubmit', True),
            user_has_entered=getattr(comp, 'userHasEntered', False),
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
        return [
            {
                "name": f.name,
                "size": getattr(f, 'size', 0),
                "creation_date": getattr(f, 'creationDate', None),
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

                # Copy to specified path if different from cache
                if path != Path("."):
                    path.mkdir(parents=True, exist_ok=True)
                    import shutil

                    if cache_path.is_file():
                        shutil.copy2(cache_path, path / cache_path.name)
                    else:
                        # Copy entire directory contents
                        for item in cache_path.iterdir():
                            dest = path / item.name
                            if item.is_file():
                                shutil.copy2(item, dest)
                            else:
                                shutil.copytree(item, dest, dirs_exist_ok=True)
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
            date = s.date if isinstance(s.date, datetime) else datetime.now()

            public_score = None
            if hasattr(s, 'publicScore') and s.publicScore:
                try:
                    public_score = float(s.publicScore)
                except (ValueError, TypeError):
                    pass

            private_score = None
            if hasattr(s, 'privateScore') and s.privateScore:
                try:
                    private_score = float(s.privateScore)
                except (ValueError, TypeError):
                    pass

            result.append(SubmissionInfo(
                submission_id=getattr(s, 'ref', 0),
                date=date,
                description=getattr(s, 'description', ''),
                status=getattr(s, 'status', 'complete'),
                public_score=public_score,
                private_score=private_score,
                file_name=getattr(s, 'fileName', ''),
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
                    "rank": getattr(entry, 'rank', 0),
                    "team_name": getattr(entry, 'teamName', ''),
                    "score": getattr(entry, 'score', None),
                    "entries": getattr(entry, 'submissionCount', 0),
                    "last_submission": getattr(entry, 'lastSubmissionDate', None),
                })

            return result

        except Exception as e:
            warnings.warn(f"Failed to get leaderboard: {e}")
            return []

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
            last_updated = None
            if hasattr(d, 'lastUpdated') and d.lastUpdated:
                if isinstance(d.lastUpdated, datetime):
                    last_updated = d.lastUpdated

            result.append(DatasetInfo(
                slug=getattr(d, 'ref', ''),
                title=getattr(d, 'title', ''),
                size=getattr(d, 'totalBytes', 0),
                last_updated=last_updated,
                download_count=getattr(d, 'downloadCount', 0),
                vote_count=getattr(d, 'voteCount', 0),
                usability_rating=getattr(d, 'usabilityRating', 0.0),
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

            # Copy to specified path if different from cache
            if path != Path("."):
                path.mkdir(parents=True, exist_ok=True)
                import shutil

                if cache_path.is_file():
                    shutil.copy2(cache_path, path / cache_path.name)
                else:
                    for item in cache_path.iterdir():
                        dest = path / item.name
                        if item.is_file():
                            shutil.copy2(item, dest)
                        else:
                            shutil.copytree(item, dest, dirs_exist_ok=True)
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
