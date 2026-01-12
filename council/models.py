"""
Model management for LLM Council.
"""
import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from .backends import BackendManager, OllamaBackend, GenerationResult, BackendType
class ModelRole(Enum):
    """Predefined roles for council members."""
    GENERAL_REASONER = "general_reasoner"
    ANALYTICAL_THINKER = "analytical_thinker"
    CREATIVE_THINKER = "creative_thinker"
    CONCISE_RESPONDER = "concise_responder"
    TECHNICAL_EXPERT = "technical_expert"
    DEVIL_ADVOCATE = "devil_advocate"
    SYNTHESIZER = "synthesizer"
    FACT_CHECKER = "fact_checker"
    ETHICAL_REVIEWER = "ethical_reviewer"
@dataclass
class ModelStats:
    """Statistics for a model."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_tokens: int = 0
    total_response_time: float = 0.0
    response_times: List[float] = field(default_factory=list)
    @property
    def success_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls
    @property
    def avg_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)
    @property
    def tokens_per_second(self) -> float:
        if self.total_response_time == 0:
            return 0.0
        return self.total_tokens / self.total_response_time
    def record_call(self, result: GenerationResult):
        """Record a call result."""
        self.total_calls += 1
        self.response_times.append(result.response_time)
        self.total_response_time += result.response_time
        if result.success:
            self.successful_calls += 1
            self.total_tokens += result.total_tokens
        else:
            self.failed_calls += 1
        if len(self.response_times) > 100:
            self.response_times = self.response_times[-100:]
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": round(self.success_rate, 3),
            "total_tokens": self.total_tokens,
            "avg_response_time": round(self.avg_response_time, 3),
            "tokens_per_second": round(self.tokens_per_second, 1)
        }
@dataclass
class CouncilMember:
    """
    Represents a single LLM in the council.
    """
    name: str
    role: str
    weight: float = 1.0
    enabled: bool = True
    temperature: float = 0.7
    max_tokens: int = 2048
    priority: int = 1
    tags: List[str] = field(default_factory=list)
    description: str = ""
    _backend: Optional[Any] = field(default=None, repr=False)
    _stats: ModelStats = field(default_factory=ModelStats, repr=False)
    def __post_init__(self):
        if isinstance(self.role, ModelRole):
            self.role = self.role.value
    async def generate_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        format: Optional[str] = None,
        **kwargs
    ) -> GenerationResult:
        """Generate response asynchronously."""
        if not self._backend:
            return GenerationResult(
                success=False,
                content="",
                model=self.name,
                error="No backend configured",
                error_type="no_backend"
            )
        result = await self._backend.generate(
            prompt=prompt,
            model=self.name,
            system_prompt=system_prompt,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            format=format,
            **kwargs
        )
        result.model = self.name
        self._stats.record_call(result)
        return result
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        format: Optional[str] = None,
        **kwargs
    ) -> GenerationResult:
        """Generate response synchronously."""
        if not self._backend:
            return GenerationResult(
                success=False,
                content="",
                model=self.name,
                error="No backend configured",
                error_type="no_backend"
            )
        if hasattr(self._backend, 'generate_sync'):
            result = self._backend.generate_sync(
                prompt=prompt,
                model=self.name,
                system_prompt=system_prompt,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                format=format,
                **kwargs
            )
        else:
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(
                    self.generate_async(
                        prompt, system_prompt, temperature, max_tokens, format, **kwargs
                    )
                )
            finally:
                loop.close()
        result.model = self.name
        self._stats.record_call(result)
        return result
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        """Generate streaming response."""
        if self._backend and hasattr(self._backend, 'generate_stream'):
            async for chunk in self._backend.generate_stream(
                prompt=prompt,
                model=self.name,
                system_prompt=system_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **kwargs
            ):
                yield chunk
    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        return {
            "name": self.name,
            "role": self.role,
            "weight": self.weight,
            "enabled": self.enabled,
            **self._stats.to_dict()
        }
    def reset_stats(self):
        """Reset statistics."""
        self._stats = ModelStats()
    @property
    def is_available(self) -> bool:
        """Check if model is available."""
        if not self._backend:
            return False
        return self._backend.is_model_available(self.name)
class ModelManager:
    """
    Manages all council members and their backends.
    """
    def __init__(self):
        self.members: Dict[str, CouncilMember] = {}
        self.backend_manager = BackendManager()
        self._executor = ThreadPoolExecutor(max_workers=10)
    async def initialize(
        self,
        ollama_url: str = "http://localhost:11434",
        vllm_url: Optional[str] = None,
        llamacpp_path: Optional[str] = None
    ):
        """Initialize backends."""
        await self.backend_manager.add_backend(
            name="ollama",
            backend_type=BackendType.OLLAMA,
            is_primary=True,
            base_url=ollama_url
        )
        if vllm_url:
            await self.backend_manager.add_backend(
                name="vllm",
                backend_type=BackendType.VLLM,
                api_url=vllm_url
            )
        if llamacpp_path:
            await self.backend_manager.add_backend(
                name="llamacpp",
                backend_type=BackendType.LLAMACPP,
                model_path=llamacpp_path
            )
    async def add_member(
        self,
        name: str,
        role: str = "general_reasoner",
        weight: float = 1.0,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> bool:
        """Add a council member."""
        backend = self.backend_manager.get_backend_for_model(name)
        if not backend:
            logger.warning(f"No backend available for model: {name}")
            return False
        if not backend.is_model_available(name):
            logger.warning(f"Model not available: {name}")
            if hasattr(backend, 'pull_model'):
                logger.info(f"Attempting to pull model: {name}")
                success = await backend.pull_model(name)
                if not success:
                    return False
            else:
                return False
        member = CouncilMember(
            name=name,
            role=role,
            weight=weight,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        member._backend = backend
        self.members[name] = member
        logger.info(f"Added council member: {name} ({role})")
        return True
    def add_member_sync(
        self,
        name: str,
        role: str = "general_reasoner",
        **kwargs
    ) -> bool:
        """Synchronous version of add_member."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.add_member(name, role, **kwargs)
            )
        finally:
            loop.close()
    def remove_member(self, name: str) -> bool:
        """Remove a council member."""
        if name in self.members:
            del self.members[name]
            logger.info(f"Removed council member: {name}")
            return True
        return False
    def get_member(self, name: str) -> Optional[CouncilMember]:
        """Get a specific council member."""
        return self.members.get(name)
    def get_active_members(self) -> List[CouncilMember]:
        """Get all enabled council members."""
        return [m for m in self.members.values() if m.enabled]
    def get_members_by_role(self, role: str) -> List[CouncilMember]:
        """Get members with a specific role."""
        return [m for m in self.members.values() if m.role == role and m.enabled]
    def get_members_by_tag(self, tag: str) -> List[CouncilMember]:
        """Get members with a specific tag."""
        return [m for m in self.members.values() if tag in m.tags and m.enabled]
    async def parallel_generate(
        self,
        prompt: str,
        system_prompts: Optional[Dict[str, str]] = None,
        members: Optional[List[CouncilMember]] = None
    ) -> List[Dict[str, Any]]:
        """Generate responses from multiple members in parallel."""
        if members is None:
            members = self.get_active_members()
        if system_prompts is None:
            system_prompts = {}
        async def generate_one(member: CouncilMember) -> Dict[str, Any]:
            system_prompt = system_prompts.get(member.role) or system_prompts.get(member.name)
            result = await member.generate_async(prompt, system_prompt=system_prompt)
            return {
                "model": member.name,
                "role": member.role,
                "weight": member.weight,
                "success": result.success,
                "content": result.content,
                "response_time": result.response_time,
                "tokens": result.total_tokens,
                "error": result.error
            }
        tasks = [generate_one(m) for m in members]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        processed = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed.append({
                    "model": members[i].name,
                    "role": members[i].role,
                    "weight": members[i].weight,
                    "success": False,
                    "content": "",
                    "error": str(result)
                })
            else:
                processed.append(result)
        return processed
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all members."""
        return {
            "members": [m.get_stats() for m in self.members.values()],
            "total_members": len(self.members),
            "active_members": len(self.get_active_members()),
            "backends": list(self.backend_manager.backends.keys())
        }
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all members."""
        results = {}
        for name, member in self.members.items():
            try:
                result = await member.generate_async("test", max_tokens=5)
                results[name] = result.success
            except Exception:
                results[name] = False
        return results
    async def shutdown(self):
        """Shutdown the manager."""
        await self.backend_manager.disconnect_all()
        self._executor.shutdown(wait=False)
