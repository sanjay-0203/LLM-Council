"""
vLLM backend implementation for high-throughput inference.
"""

import time
from typing import Optional, List, Dict, Any, AsyncIterator

import httpx
from loguru import logger

from .base import (
    LLMBackend,
    BackendConfig,
    BackendType,
    GenerationResult,
    StreamChunk
)


class VLLMBackend(LLMBackend):
    """
    Backend implementation for vLLM.
    
    vLLM provides high-throughput serving with:
    - Continuous batching
    - PagedAttention
    - OpenAI-compatible API
    """
    
    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 120,
        **kwargs
    ):
        config = BackendConfig(
            backend_type=BackendType.VLLM,
            base_url=api_url,
            api_key=api_key,
            timeout=timeout,
            **kwargs
        )
        super().__init__(config)
        
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self._client: Optional[httpx.AsyncClient] = None
        self._model_name: Optional[str] = None
    
    async def connect(self) -> bool:
        """Connect to vLLM server."""
        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            self._client = httpx.AsyncClient(
                base_url=self.api_url,
                timeout=httpx.Timeout(self.config.timeout),
                headers=headers
            )
            
            # Check available models
            response = await self._client.get("/v1/models")
            if response.status_code == 200:
                models = response.json().get("data", [])
                self._available_models = [m["id"] for m in models]
                if self._available_models:
                    self._model_name = self._available_models[0]
                self._is_connected = True
                logger.info(f"Connected to vLLM at {self.api_url}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to connect to vLLM: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from vLLM server."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._is_connected = False
    
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
        """Generate a response from vLLM."""
        
        if not self._client:
            await self.connect()
        
        start_time = time.time()
        
        messages = self.format_messages(prompt, system_prompt)
        
        request_data = {
            "model": model or self._model_name,
            "messages": messages,
            "temperature": temperature or self.config.default_temperature,
            "max_tokens": max_tokens or self.config.default_max_tokens,
            "stream": False,
        }
        
        if stop:
            request_data["stop"] = stop
        
        try:
            response = await self._client.post(
                "/v1/chat/completions",
                json=request_data
            )
            
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                choice = data["choices"][0]
                usage = data.get("usage", {})
                
                return GenerationResult(
                    success=True,
                    content=choice["message"]["content"],
                    model=model,
                    tokens_generated=usage.get("completion_tokens", 0),
                    tokens_prompt=usage.get("prompt_tokens", 0),
                    response_time=elapsed_time,
                    finish_reason=choice.get("finish_reason", "stop"),
                    raw_response=data
                )
            else:
                return GenerationResult(
                    success=False,
                    content="",
                    model=model,
                    response_time=elapsed_time,
                    error=f"HTTP {response.status_code}: {response.text}",
                    error_type="http_error"
                )
                
        except Exception as e:
            return GenerationResult(
                success=False,
                content="",
                model=model,
                response_time=time.time() - start_time,
                error=str(e),
                error_type="unknown"
            )
    
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
        """Generate a streaming response from vLLM."""
        
        if not self._client:
            await self.connect()
        
        start_time = time.time()
        
        messages = self.format_messages(prompt, system_prompt)
        
        request_data = {
            "model": model or self._model_name,
            "messages": messages,
            "temperature": temperature or self.config.default_temperature,
            "max_tokens": max_tokens or self.config.default_max_tokens,
            "stream": True,
        }
        
        if stop:
            request_data["stop"] = stop
        
        try:
            async with self._client.stream(
                "POST",
                "/v1/chat/completions",
                json=request_data
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            yield StreamChunk(
                                content="",
                                is_final=True,
                                model=model,
                                total_time=time.time() - start_time
                            )
                            break
                        
                        import json
                        try:
                            data = json.loads(data_str)
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            
                            yield StreamChunk(
                                content=content,
                                is_final=False,
                                model=model,
                                total_time=time.time() - start_time
                            )
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            yield StreamChunk(
                content=f"Error: {str(e)}",
                is_final=True,
                model=model
            )
    
    async def list_models(self) -> List[str]:
        """List available models."""
        if not self._client:
            await self.connect()
        
        try:
            response = await self._client.get("/v1/models")
            if response.status_code == 200:
                models = response.json().get("data", [])
                self._available_models = [m["id"] for m in models]
                return self._available_models
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
        
        return []
    
    async def model_info(self, model: str) -> Dict[str, Any]:
        """Get model information."""
        if not self._client:
            await self.connect()
        
        try:
            response = await self._client.get(f"/v1/models/{model}")
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
        
        return {}
    
    async def health_check(self) -> bool:
        """Check if vLLM is healthy."""
        try:
            if not self._client:
                await self.connect()
            
            response = await self._client.get("/health")
            return response.status_code == 200
        except Exception:
            return False
