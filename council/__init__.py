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

# Backend imports
from .backends import (
    LLMBackend,
    OllamaBackend,
    VLLMBackend,
    LlamaCppBackend,
    BackendManager
)

__all__ = [
    # Core
    "LLMCouncil",
    "CouncilResponse",
    
    # Models
    "CouncilMember",
    "ModelManager",
    "ModelRole",
    
    # Voting
    "VotingSystem",
    "VotingMethod",
    "Vote",
    "VotingResult",
    
    # Evaluation
    "ResponseEvaluator",
    "EvaluationResult",
    "AggregatedEvaluation",
    
    # Consensus
    "ConsensusBuilder",
    "ConsensusResult",
    
    # Calibration
    "ConfidenceCalibrator",
    "CalibrationResult",
    
    # Persistence
    "CouncilDatabase",
    "SessionRecord",
    
    # Streaming
    "StreamingHandler",
    "StreamChunk",
    
    # Backends
    "LLMBackend",
    "OllamaBackend",
    "VLLMBackend",
    "LlamaCppBackend",
    "BackendManager",
]
