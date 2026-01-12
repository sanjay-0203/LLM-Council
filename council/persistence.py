"""
Database persistence for LLM Council.
"""
import json
import asyncio
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from pathlib import Path
from loguru import logger
try:
    import aiosqlite
    HAS_AIOSQLITE = True
except ImportError:
    HAS_AIOSQLITE = False
    logger.warning("aiosqlite not installed. Persistence features limited.")
@dataclass
class SessionRecord:
    """A record of a council session."""
    id: Optional[int] = None
    session_id: str = ""
    timestamp: str = ""
    question: str = ""
    responses: str = ""  
    votes: str = ""  
    consensus: str = ""
    evaluation: str = ""  
    metadata: str = ""  
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "question": self.question,
            "responses": json.loads(self.responses) if self.responses else [],
            "votes": json.loads(self.votes) if self.votes else {},
            "consensus": self.consensus,
            "evaluation": json.loads(self.evaluation) if self.evaluation else {},
            "metadata": json.loads(self.metadata) if self.metadata else {}
        }
class CouncilDatabase:
    """
    SQLite-based persistence for council sessions.
    """
    def __init__(self, db_path: str = "data/council.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False
    async def initialize(self):
        """Initialize database tables."""
        if not HAS_AIOSQLITE:
            logger.warning("aiosqlite not available. Using in-memory storage.")
            self._memory_store: List[SessionRecord] = []
            self._initialized = True
            return
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE,
                    timestamp TEXT,
                    question TEXT,
                    responses TEXT,
                    votes TEXT,
                    consensus TEXT,
                    evaluation TEXT,
                    metadata TEXT
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS model_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT,
                    timestamp TEXT,
                    total_calls INTEGER,
                    successful_calls INTEGER,
                    avg_response_time REAL,
                    avg_tokens REAL
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS calibration_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT,
                    timestamp TEXT,
                    predicted_confidence REAL,
                    was_correct INTEGER,
                    category TEXT
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_id ON sessions(session_id)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON sessions(timestamp)
            """)
            await db.commit()
        self._initialized = True
        logger.info(f"Database initialized at {self.db_path}")
    async def save_session(
        self,
        session_id: str,
        question: str,
        responses: List[Dict[str, Any]],
        votes: Optional[Dict[str, Any]] = None,
        consensus: str = "",
        evaluation: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Save a council session."""
        if not self._initialized:
            await self.initialize()
        record = SessionRecord(
            session_id=session_id,
            timestamp=datetime.utcnow().isoformat(),
            question=question,
            responses=json.dumps(responses),
            votes=json.dumps(votes or {}),
            consensus=consensus,
            evaluation=json.dumps(evaluation or {}),
            metadata=json.dumps(metadata or {})
        )
        if not HAS_AIOSQLITE:
            self._memory_store.append(record)
            return True
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO sessions
                    (session_id, timestamp, question, responses, votes, consensus, evaluation, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.session_id,
                    record.timestamp,
                    record.question,
                    record.responses,
                    record.votes,
                    record.consensus,
                    record.evaluation,
                    record.metadata
                ))
                await db.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            return False
    async def get_session(self, session_id: str) -> Optional[SessionRecord]:
        """Get a session by ID."""
        if not self._initialized:
            await self.initialize()
        if not HAS_AIOSQLITE:
            for record in self._memory_store:
                if record.session_id == session_id:
                    return record
            return None
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT * FROM sessions WHERE session_id = ?",
                    (session_id,)
                ) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        return SessionRecord(**dict(row))
        except Exception as e:
            logger.error(f"Failed to get session: {e}")
        return None
    async def get_recent_sessions(
        self,
        limit: int = 10,
        offset: int = 0
    ) -> List[SessionRecord]:
        """Get recent sessions."""
        if not self._initialized:
            await self.initialize()
        if not HAS_AIOSQLITE:
            return self._memory_store[-limit-offset:-offset] if offset else self._memory_store[-limit:]
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT * FROM sessions ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                    (limit, offset)
                ) as cursor:
                    rows = await cursor.fetchall()
                    return [SessionRecord(**dict(row)) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get recent sessions: {e}")
            return []
    async def search_sessions(
        self,
        query: str,
        limit: int = 10
    ) -> List[SessionRecord]:
        """Search sessions by question content."""
        if not self._initialized:
            await self.initialize()
        if not HAS_AIOSQLITE:
            return [
                r for r in self._memory_store
                if query.lower() in r.question.lower()
            ][:limit]
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT * FROM sessions WHERE question LIKE ? ORDER BY timestamp DESC LIMIT ?",
                    (f"%{query}%", limit)
                ) as cursor:
                    rows = await cursor.fetchall()
                    return [SessionRecord(**dict(row)) for row in rows]
        except Exception as e:
            logger.error(f"Failed to search sessions: {e}")
            return []
    async def save_calibration_sample(
        self,
        model_name: str,
        predicted_confidence: float,
        was_correct: bool,
        category: str = "general"
    ):
        """Save a calibration sample."""
        if not HAS_AIOSQLITE:
            return
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO calibration_samples
                    (model_name, timestamp, predicted_confidence, was_correct, category)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    model_name,
                    datetime.utcnow().isoformat(),
                    predicted_confidence,
                    1 if was_correct else 0,
                    category
                ))
                await db.commit()
        except Exception as e:
            logger.error(f"Failed to save calibration sample: {e}")
    async def get_calibration_samples(
        self,
        model_name: str,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get calibration samples for a model."""
        if not HAS_AIOSQLITE:
            return []
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    """SELECT * FROM calibration_samples
                       WHERE model_name = ?
                       ORDER BY timestamp DESC
                       LIMIT ?""",
                    (model_name, limit)
                ) as cursor:
                    rows = await cursor.fetchall()
                    return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get calibration samples: {e}")
            return []
    async def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        if not HAS_AIOSQLITE:
            return {
                "total_sessions": len(getattr(self, '_memory_store', [])),
                "storage": "memory"
            }
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute("SELECT COUNT(*) FROM sessions") as cursor:
                    total_sessions = (await cursor.fetchone())[0]
                async with db.execute("""
                    SELECT DATE(timestamp) as date, COUNT(*) as count
                    FROM sessions
                    WHERE timestamp > datetime('now', '-7 days')
                    GROUP BY DATE(timestamp)
                    ORDER BY date
                """) as cursor:
                    daily_counts = {row[0]: row[1] for row in await cursor.fetchall()}
                async with db.execute("""
                    SELECT COUNT(*) FROM calibration_samples
                """) as cursor:
                    total_samples = (await cursor.fetchone())[0]
                return {
                    "total_sessions": total_sessions,
                    "daily_sessions": daily_counts,
                    "total_calibration_samples": total_samples,
                    "storage": "sqlite",
                    "db_path": str(self.db_path)
                }
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {"error": str(e)}
    async def cleanup_old_records(self, days: int = 90):
        """Remove records older than specified days."""
        if not HAS_AIOSQLITE:
            return
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "DELETE FROM sessions WHERE timestamp < datetime('now', ?)",
                    (f'-{days} days',)
                )
                await db.execute(
                    "DELETE FROM calibration_samples WHERE timestamp < datetime('now', ?)",
                    (f'-{days} days',)
                )
                await db.commit()
            logger.info(f"Cleaned up records older than {days} days")
        except Exception as e:
            logger.error(f"Failed to cleanup old records: {e}")
