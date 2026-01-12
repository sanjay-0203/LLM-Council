"""
Utility functions for LLM Council.
"""

import json
import re
import hashlib
import time
from typing import Any, Dict, List, Optional

from loguru import logger


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Robustly extract JSON from LLM output."""
    text = text.strip()
    
    # Look for ```json ... ```
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end == -1:
            end = len(text)
        text = text[start:end].strip()
    
    # Find first { and last }
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    
    json_str = text[start:end]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.debug(f"JSON parse failed: {e}\nContent: {json_str}")
        return None


def create_session_id(question: str) -> str:
    """Create a deterministic session ID."""
    return hashlib.sha256(question.encode()).hexdigest()[:16]


def truncate(text: str, max_chars: int = 500) -> str:
    """Truncate with ellipsis."""
    return text[:max_chars] + "..." if len(text) > max_chars else text


def format_duration(seconds: float) -> str:
    """Format seconds into human readable."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        return f"{seconds/60:.1f}m"


def clean_response(text: str) -> str:
    """Clean common LLM artifacts."""
    text = re.sub(r"^Assistant:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^Response:\s*", "", text, flags=re.IGNORECASE)
    text = text.strip()
    return text
