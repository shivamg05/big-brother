"""Big Brother: queryable worker memory from egocentric video windows."""

from .episode import EpisodeBoundaryConfig, EpisodeBuilder
from .extractor import GeminiExtractor, HeuristicExtractor
from .gating import GateDecision, GatingConfig, GatingEngine
from .pipeline import PipelineConfig, WorkerMemoryPipeline
from .runner import AnalysisSummary, analyze_video
from .schema import Episode, SubtaskEvent
from .storage import MemoryStore
from .video import VideoIngestConfig, iter_video_windows

__all__ = [
    "Episode",
    "EpisodeBoundaryConfig",
    "EpisodeBuilder",
    "AnalysisSummary",
    "GateDecision",
    "GeminiExtractor",
    "GatingConfig",
    "GatingEngine",
    "HeuristicExtractor",
    "MemoryStore",
    "PipelineConfig",
    "SubtaskEvent",
    "VideoIngestConfig",
    "WorkerMemoryPipeline",
    "analyze_video",
    "iter_video_windows",
]
