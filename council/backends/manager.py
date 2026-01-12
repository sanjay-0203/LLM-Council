"""
Backend manager for handling multiple LLM backends.
"""

from typing import Dict, Optional, List, Type
from loguru import logger

from .base import LLMBackend, BackendType, GenerationResult
from .ollama_backend import OllamaBackend
from .vllm_backend import VLLMBackend
from .llamacpp_backend import LlamaCppBackend


class BackendManager:
    """
    Manages multiple LLM backends with automatic failover.
    """
    
    BACKEND_CLASSES: Dict[BackendType, Type[LLMBackend]] = {
        BackendType.OLLAMA: OllamaBackend,
        BackendType.VLLM: VLLMBackend,
        BackendType.LLAMACPP: LlamaCppBackend,
    }
    
    def __init__(self):
        self.backends: Dict[str, LLMBackend] = {}
        self.primary_backend: Optional[str] = None
        self.model_to_backend: Dict[str, str] = {}
    
    async def add_backend(
        self,
        name: str,
        backend_type: BackendType,
        is_primary: bool = False,
        **config
    ) -> bool:
        """Add a backend to the manager."""
        
        backend_class = self.BACKEND_CLASSES.get(backend_type)
        if not backend_class:
            logger.error(f"Unknown backend type: {backend_type}")
            return False
        
        try:
            backend = backend_class(**config)
            connected = await backend.connect()
            
            if connected:
                self.backends[name] = backend
                
                if is_primary or self.primary_backend is None:
                    self.primary_backend = name
                
                # Map models to this backend
                models = await backend.list_models()
                for model in models:
                    if model not in self.model_to_backend:
                        self.model_to_backend[model] = name
                
                logger.info(f"Added backend: {name} ({backend_type.value})")
                return True
            else:
                logger.warning(f"Failed to connect backend: {name}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding backend {name}: {e}")
            return False
    
    def get_backend(self, name: str) -> Optional[LLMBackend]:
        """Get a specific backend."""
        return self.backends.get(name)
    
    def get_backend_for_model(self, model: str) -> Optional[LLMBackend]:
        """Get the backend that has a specific model."""
        backend_name = self.model_to_backend.get(model)
        if backend_name:
            return self.backends.get(backend_name)
        
        # Check all backends
        for name, backend in self.backends.items():
            if backend.is_model_available(model):
                self.model_to_backend[model] = name
                return backend
        
        # Return primary backend as fallback
        if self.primary_backend:
            return self.backends.get(self.primary_backend)
        
        return None
    
    async def generate(
        self,
        prompt: str,
        model: str,
        **kwargs
    ) -> GenerationResult:
        """Generate using the appropriate backend."""
        
        backend = self.get_backend_for_model(model)
        
        if not backend:
            return GenerationResult(
                success=False,
                content="",
                model=model,
                error="No backend available for model",
                error_type="no_backend"
            )
        
        result = await backend.generate(prompt, model, **kwargs)
        
        # Try failover on failure
        if not result.success and len(self.backends) > 1:
            for name, fallback in self.backends.items():
                if fallback != backend and fallback.is_model_available(model):
                    logger.warning(f"Trying failover to {name}")
                    result = await fallback.generate(prompt, model, **kwargs)
                    if result.success:
                        break
        
        return result
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all backends."""
        results = {}
        for name, backend in self.backends.items():
            try:
                results[name] = await backend.health_check()
            except Exception:
                results[name] = False
        return results
    
    def list_all_models(self) -> Dict[str, List[str]]:
        """List all models from all backends."""
        result = {}
        for name, backend in self.backends.items():
            result[name] = backend._available_models
        return result
    
    async def disconnect_all(self) -> None:
        """Disconnect all backends."""
        for backend in self.backends.values():
            await backend.disconnect()
        self.backends.clear()
        self.model_to_backend.clear()
        self.primary_backend = None
