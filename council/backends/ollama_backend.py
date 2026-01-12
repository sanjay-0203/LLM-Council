"""
Ollama backend implementation.
"""

import asyncio
import time
from typing import Optional, List, Dict, Any, AsyncIterator

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from loguru import logger

from .base import (
    LLMBackend,
    BackendConfig,
    BackendType,
    GenerationResult,
    StreamChunk
)


class OllamaBackend(LLMBackend):
    """
    Backend implementation for Ollama.
    
    Supports:
    - Async and sync generation
    - Streaming responses
    - Model management
    - Health checking
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
        **kwargs
    ):
        config = BackendConfig(
            backend_type=BackendType.OLLAMA,
            base_url=base_url,
            timeout=timeout,
            **kwargs
        )
        super().__init__(config)
        
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None
        self._sync_client: Optional[httpx.Client] = None
    
    async def connect(self) -> bool:
        """Connect to Ollama server."""
        try:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.config.timeout)
            )
            
            # Verify connection
            response = await self._client.get("/api/tags")
            if response.status_code == 200:
                self._is_connected = True
                self._available_models = [
                    m["name"] for m in response.json().get("models", [])
                ]
                logger.info(f"Connected to Ollama at {self.base_url}")
                logger.info(f"Available models: {self._available_models}")
                return True
            else:
                logger.error(f"Failed to connect to Ollama: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Ollama server."""
        if self._client:
            await self._client.aclose()
            self._client = None
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None
        self._is_connected = False
        logger.info("Disconnected from Ollama")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError))
    )
    async def generate(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        format: Optional[str] = None,
        **kwargs
    ) -> GenerationResult:
        """Generate a response from Ollama."""
        
        if not self._client:
            await self.connect()
        
        start_time = time.time()
        
        # Build messages
        messages = self.format_messages(prompt, system_prompt)
        
        # Build options
        options = {
            "temperature": temperature or self.config.default_temperature,
            "num_predict": max_tokens or self.config.default_max_tokens,
            "num_ctx": self.config.num_ctx,
        }
        
        if stop:
            options["stop"] = stop
        
        # Build request
        request_data = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        
        if format:
            request_data["format"] = format
        
        try:
            response = await self._client.post(
                "/api/chat",
                json=request_data
            )
            
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                return GenerationResult(
                    success=True,
                    content=data["message"]["content"],
                    model=model,
                    tokens_generated=data.get("eval_count", 0),
                    tokens_prompt=data.get("prompt_eval_count", 0),
                    response_time=elapsed_time,
                    finish_reason=data.get("done_reason", "stop"),
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
                
        except httpx.TimeoutException as e:
            return GenerationResult(
                success=False,
                content="",
                model=model,
                response_time=time.time() - start_time,
                error=f"Timeout: {str(e)}",
                error_type="timeout"
            )
        except Exception as e:
            logger.error(f"Generation error: {e}")
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
        """Generate a streaming response from Ollama."""
        
        if not self._client:
            await self.connect()
        
        start_time = time.time()
        chunk_start = start_time
        
        messages = self.format_messages(prompt, system_prompt)
        
        options = {
            "temperature": temperature or self.config.default_temperature,
            "num_predict": max_tokens or self.config.default_max_tokens,
            "num_ctx": self.config.num_ctx,
        }
        
        if stop:
            options["stop"] = stop
        
        request_data = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": options,
        }
        
        try:
            async with self._client.stream(
                "POST",
                "/api/chat",
                json=request_data
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        import json
                        try:
                            data = json.loads(line)
                            chunk_time = time.time()
                            
                            yield StreamChunk(
                                content=data.get("message", {}).get("content", ""),
                                is_final=data.get("done", False),
                                tokens=data.get("eval_count", 0),
                                model=model,
                                chunk_time=chunk_time - chunk_start,
                                total_time=chunk_time - start_time
                            )
                            
                            chunk_start = chunk_time
                            
                            if data.get("done"):
                                break
                                
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield StreamChunk(
                content=f"Error: {str(e)}",
                is_final=True,
                model=model
            )
    
    async def list_models(self) -> List[str]:
        """List available models in Ollama."""
        if not self._client:
            await self.connect()
        
        try:
            response = await self._client.get("/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                self._available_models = [m["name"] for m in models]
                return self._available_models
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
        
        return []
    
    async def model_info(self, model: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        if not self._client:
            await self.connect()
        
        try:
            response = await self._client.post(
                "/api/show",
                json={"name": model}
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
        
        return {}
    
    async def pull_model(self, model: str) -> bool:
        """Pull a model from Ollama registry."""
        if not self._client:
            await self.connect()
        
        try:
            logger.info(f"Pulling model: {model}")
            
            async with self._client.stream(
                "POST",
                "/api/pull",
                json={"name": model}
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        import json
                        try:
                            data = json.loads(line)
                            status = data.get("status", "")
                            if "pulling" in status.lower():
                                logger.debug(f"Pull status: {status}")
                            if data.get("status") == "success":
                                logger.info(f"Successfully pulled: {model}")
                                return True
                        except json.JSONDecodeError:
                            continue
            
            # Refresh available models
            await self.list_models()
            return model in self._available_models
            
        except Exception as e:
            logger.error(f"Failed to pull model: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check if Ollama is healthy."""
        try:
            if not self._client:
                self._client = httpx.AsyncClient(
                    base_url=self.base_url,
                    timeout=httpx.Timeout(10.0)
                )
            
            response = await self._client.get("/api/tags")
            return response.status_code == 200
            
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
    
    # Synchronous methods for convenience
    def generate_sync(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> GenerationResult:
        """Synchronous generation."""
        import ollama
        
        start_time = time.time()
        
        messages = self.format_messages(prompt, system_prompt)
        
        options = {
            "temperature": kwargs.get("temperature", self.config.default_temperature),
            "num_predict": kwargs.get("max_tokens", self.config.default_max_tokens),
            "num_ctx": self.config.num_ctx,
        }
        
        try:
            response = ollama.chat(
                model=model,
                messages=messages,
                options=options,
                format=kwargs.get("format")
            )
            
            return GenerationResult(
                success=True,
                content=response["message"]["content"],
                model=model,
                tokens_generated=response.get("eval_count", 0),
                tokens_prompt=response.get("prompt_eval_count", 0),
                response_time=time.time() - start_time,
                finish_reason=response.get("done_reason", "stop"),
                raw_response=response
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
