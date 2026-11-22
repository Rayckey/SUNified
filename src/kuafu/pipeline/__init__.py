"""End-to-end workflow orchestration and artifact handoffs."""

from .config import PipelineConfig
from .pipeline import Pipeline, STAGES

__all__ = ["Pipeline", "PipelineConfig", "STAGES"]
__version__ = "0.1.0"
