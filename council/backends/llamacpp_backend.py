"""
LlamaCpp backend stub - requires llama-cpp-python installation.
"""
import time
from typing import Optional, List, Dict, Any, AsyncIterator
from pathlib import Path
from loguru import logger
from .base import (
    LLMBackend,
    BackendConfig,
    BackendType,
    GenerationResult,
    StreamChunk
)
class LlamaCppBackend(LLMBackend):
    """Backend implementation for llama.cpp (requires llama-cpp-python)."""
    def __init__(self, model_path: str, **kwargs):
        config = BackendConfig(backend_type=BackendType.LLAMACPP, model_path=model_path, **kwargs)
        super().__init__(config)
        self._is_connected = False
    async def connect(self) -> bool:
        logger.warning("LlamaCpp backend requires llama-cpp-python package")
        return False
    async def disconnect(self) -> None:
        pass
    async def generate(self, prompt: str, model: str, **kwargs) -> GenerationResult:
        return GenerationResult(success=False, content="", model=model, error="LlamaCpp not available")
    async def generate_stream(self, prompt: str, model: str, **kwargs) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(model=model, content="LlamaCpp not available", is_final=True)
    async def list_models(self) -> List[str]:
        return []
    async def model_info(self, model: str) -> Dict[str, Any]:
        return {}
    async def health_check(self) -> bool:
        return False
