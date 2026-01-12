"""
Abstract base class for LLM backends.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, AsyncIterator, Iterator
from enum import Enum
import time
class BackendType(Enum):
    """Supported backend types."""
    OLLAMA = "ollama"
    VLLM = "vllm"
    LLAMACPP = "llamacpp"
    OPENAI_COMPATIBLE = "openai_compatible"
    TRANSFORMERS = "transformers"
@dataclass
class BackendConfig:
    """Configuration for an LLM backend."""
    backend_type: BackendType
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model_path: Optional[str] = None
    timeout: int = 120
    max_retries: int = 3
    retry_delay: float = 2.0
    default_temperature: float = 0.7
    default_max_tokens: int = 2048
    default_top_p: float = 0.9
    num_ctx: int = 4096
    num_gpu: int = 1
    num_threads: int = 8
    extra_options: Dict[str, Any] = field(default_factory=dict)
@dataclass
class GenerationResult:
    """Result from a generation request."""
    success: bool
    content: str
    model: str
    tokens_generated: int = 0
    tokens_prompt: int = 0
    response_time: float = 0.0
    finish_reason: str = ""
    error: Optional[str] = None
    error_type: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
    def __post_init__(self):
        self.timestamp = time.time()
    @property
    def total_tokens(self) -> int:
        return self.tokens_generated + self.tokens_prompt
    @property
    def tokens_per_second(self) -> float:
        if self.response_time > 0:
            return self.tokens_generated / self.response_time
        return 0.0
@dataclass
class StreamChunk:
    """A chunk from streaming generation."""
    content: str
    is_final: bool = False
    tokens: int = 0
    model: str = ""
    chunk_time: float = 0.0
    total_time: float = 0.0
class LLMBackend(ABC):
    """
    Abstract base class for LLM inference backends.
    Implementations should handle:
    - Connection management
    - Request/response formatting
    - Error handling and retries
    - Streaming support
    """
    def __init__(self, config: BackendConfig):
        self.config = config
        self._is_connected = False
        self._available_models: List[str] = []
    @property
    def backend_type(self) -> BackendType:
        return self.config.backend_type
    @property
    def is_connected(self) -> bool:
        return self._is_connected
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the backend."""
        pass
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the backend."""
        pass
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> GenerationResult:
        """Generate a response from the model."""
        pass
    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        """Generate a streaming response from the model."""
        pass
    @abstractmethod
    async def list_models(self) -> List[str]:
        """List available models."""
        pass
    @abstractmethod
    async def model_info(self, model: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        pass
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the backend is healthy and responsive."""
        pass
    def generate_sync(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> GenerationResult:
        """Synchronous wrapper for generate."""
        import asyncio
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.generate(prompt, model, system_prompt, **kwargs)
        )
    def generate_stream_sync(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Iterator[StreamChunk]:
        """Synchronous wrapper for streaming generation."""
        import asyncio
        async def collect_chunks():
            chunks = []
            async for chunk in self.generate_stream(prompt, model, system_prompt, **kwargs):
                chunks.append(chunk)
            return chunks
        loop = asyncio.get_event_loop()
        chunks = loop.run_until_complete(collect_chunks())
        return iter(chunks)
    def is_model_available(self, model: str) -> bool:
        """Check if a specific model is available."""
        return model in self._available_models
    def get_default_options(self) -> Dict[str, Any]:
        """Get default generation options."""
        return {
            "temperature": self.config.default_temperature,
            "max_tokens": self.config.default_max_tokens,
            "top_p": self.config.default_top_p,
        }
    def format_messages(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Format prompt into message format."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (model-specific implementations can override)."""
        return len(text) // 4
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.backend_type.value}, connected={self.is_connected})"
