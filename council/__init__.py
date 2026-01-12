"""
LLM Council - Open Source Multi-Model Consensus System
A production-ready implementation for collaborative AI reasoning,
evaluation, and consensus building using open-source LLMs.
"""
__version__ = "1.0.0"
__author__ = "Your Name"
__license__ = "MIT"
from .council import LLMCouncil, CouncilResponse
from .models import CouncilMember, ModelManager, ModelRole
from .voting import VotingSystem, VotingMethod, Vote, VotingResult
from .evaluator import ResponseEvaluator, EvaluationResult, AggregatedEvaluation
from .consensus import ConsensusBuilder, ConsensusResult, DebateManager
from .calibration import ConfidenceCalibrator, CalibrationResult
from .persistence import CouncilDatabase, SessionRecord
from .streaming import StreamingHandler, StreamChunk
from .backends import (
    LLMBackend,
    OllamaBackend,
    VLLMBackend,
    LlamaCppBackend,
    BackendManager
)
__all__ = [
    "LLMCouncil",
    "CouncilResponse",
    "CouncilMember",
    "ModelManager",
    "ModelRole",
    "VotingSystem",
    "VotingMethod",
    "Vote",
    "VotingResult",
    "ResponseEvaluator",
    "EvaluationResult",
    "AggregatedEvaluation",
    "ConsensusBuilder",
    "ConsensusResult",
    "ConfidenceCalibrator",
    "CalibrationResult",
    "CouncilDatabase",
    "SessionRecord",
    "StreamingHandler",
    "StreamChunk",
    "LLMBackend",
    "OllamaBackend",
    "VLLMBackend",
    "LlamaCppBackend",
    "BackendManager",
]
