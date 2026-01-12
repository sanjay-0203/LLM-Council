"""
Streaming support for LLM Council.
"""

import asyncio
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Optional, Dict, Any, List
from collections import defaultdict

from loguru import logger


@dataclass
class StreamChunk:
    """A chunk from streaming generation."""
    model: str
    content: str
    is_final: bool = False
    tokens: int = 0
    chunk_time: float = 0.0
    total_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiStreamState:
    """State for multi-model streaming."""
    model: str
    content: str = ""
    is_complete: bool = False
    error: Optional[str] = None
    tokens: int = 0
    response_time: float = 0.0


class StreamingHandler:
    """
    Handles streaming responses from multiple models.
    """
    
    def __init__(self):
        self.active_streams: Dict[str, MultiStreamState] = {}
        self.callbacks: List[Callable] = []
    
    def add_callback(self, callback: Callable):
        """Add a callback for stream updates."""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable):
        """Remove a callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    async def _notify_callbacks(self, model: str, chunk: StreamChunk):
        """Notify all callbacks of a new chunk."""
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(model, chunk)
                else:
                    callback(model, chunk)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    async def stream_single(
        self,
        model_member: Any,  # CouncilMember
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> AsyncIterator[StreamChunk]:
        """Stream from a single model."""
        model = model_member.name
        self.active_streams[model] = MultiStreamState(model=model)
        
        try:
            async for chunk in model_member.generate_stream(prompt, system_prompt):
                self.active_streams[model].content += chunk.content
                self.active_streams[model].tokens += chunk.tokens
                self.active_streams[model].response_time = chunk.total_time
                
                stream_chunk = StreamChunk(
                    model=model,
                    content=chunk.content,
                    is_final=chunk.is_final,
                    tokens=chunk.tokens,
                    chunk_time=chunk.chunk_time,
                    total_time=chunk.total_time
                )
                
                await self._notify_callbacks(model, stream_chunk)
                yield stream_chunk
                
                if chunk.is_final:
                    self.active_streams[model].is_complete = True
                    break
        
        except Exception as e:
            self.active_streams[model].error = str(e)
            self.active_streams[model].is_complete = True
            yield StreamChunk(
                model=model,
                content=f"Error: {e}",
                is_final=True
            )
    
    async def stream_parallel(
        self,
        members: List[Any],  # List of CouncilMember
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream from multiple models in parallel.
        Yields chunks as they arrive from any model.
        """
        # Initialize state for all models
        for member in members:
            self.active_streams[member.name] = MultiStreamState(model=member.name)
        
        # Create async generators for each model
        async def stream_one(member):
            chunks = []
            async for chunk in self.stream_single(member, prompt, system_prompt):
                chunks.append(chunk)
            return chunks
        
        # Use asyncio.as_completed-style approach
        # Create tasks
        tasks = {
            asyncio.create_task(stream_one(member)): member.name
            for member in members
        }
        
        # Collect results as they complete
        for task in asyncio.as_completed(tasks.keys()):
            try:
                chunks = await task
                for chunk in chunks:
                    yield chunk
            except Exception as e:
                model_name = tasks.get(task, "unknown")
                yield StreamChunk(
                    model=model_name,
                    content=f"Error: {e}",
                    is_final=True
                )
    
    async def stream_sequential(
        self,
        members: List[Any],
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> AsyncIterator[StreamChunk]:
        """Stream from models one at a time."""
        for member in members:
            async for chunk in self.stream_single(member, prompt, system_prompt):
                yield chunk
    
    async def stream_round_robin(
        self,
        members: List[Any],
        prompt: str,
        system_prompt: Optional[str] = None,
        chunk_size: int = 1
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream from models in round-robin fashion.
        Gets chunk_size chunks from each before moving to next.
        """
        # Initialize generators
        generators = {
            member.name: self.stream_single(member, prompt, system_prompt)
            for member in members
        }
        
        active_models = list(generators.keys())
        
        while active_models:
            models_to_remove = []
            
            for model in active_models:
                try:
                    for _ in range(chunk_size):
                        chunk = await anext(generators[model])
                        yield chunk
                        
                        if chunk.is_final:
                            models_to_remove.append(model)
                            break
                
                except StopAsyncIteration:
                    models_to_remove.append(model)
            
            for model in models_to_remove:
                if model in active_models:
                    active_models.remove(model)
    
    def get_stream_state(self, model: str) -> Optional[MultiStreamState]:
        """Get current state of a model's stream."""
        return self.active_streams.get(model)
    
    def get_all_states(self) -> Dict[str, MultiStreamState]:
        """Get states of all active streams."""
        return self.active_streams.copy()
    
    def get_combined_content(self) -> Dict[str, str]:
        """Get combined content from all streams."""
        return {
            model: state.content
            for model, state in self.active_streams.items()
        }
    
    def is_complete(self) -> bool:
        """Check if all streams are complete."""
        if not self.active_streams:
            return True
        return all(state.is_complete for state in self.active_streams.values())
    
    def reset(self):
        """Reset all stream states."""
        self.active_streams.clear()


class StreamAggregator:
    """
    Aggregates streaming content for display.
    """
    
    def __init__(self):
        self.model_contents: Dict[str, str] = defaultdict(str)
        self.model_tokens: Dict[str, int] = defaultdict(int)
        self.model_complete: Dict[str, bool] = defaultdict(bool)
    
    def update(self, chunk: StreamChunk):
        """Update with a new chunk."""
        self.model_contents[chunk.model] += chunk.content
        self.model_tokens[chunk.model] += chunk.tokens
        
        if chunk.is_final:
            self.model_complete[chunk.model] = True
    
    def get_display_state(self) -> Dict[str, Dict[str, Any]]:
        """Get current state for display."""
        return {
            model: {
                "content": content,
                "tokens": self.model_tokens[model],
                "complete": self.model_complete[model]
            }
            for model, content in self.model_contents.items()
        }
    
    def get_content(self, model: str) -> str:
        """Get content for a specific model."""
        return self.model_contents.get(model, "")
    
    def is_all_complete(self) -> bool:
        """Check if all models have completed."""
        if not self.model_contents:
            return False
        return all(self.model_complete.values())
    
    def reset(self):
        """Reset aggregator state."""
        self.model_contents.clear()
        self.model_tokens.clear()
        self.model_complete.clear()
