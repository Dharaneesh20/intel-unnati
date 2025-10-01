"""
Memory and state management for the AI Agent Framework
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json
import asyncio
from collections import defaultdict


class MemoryType(Enum):
    """Types of memory storage"""
    SHORT_TERM = "short_term"    # Session-based memory
    LONG_TERM = "long_term"      # Persistent memory
    WORKING = "working"          # Current execution context
    EPISODIC = "episodic"        # Event-based memory


@dataclass
class MemoryEntry:
    """Single memory entry"""
    id: str
    key: str
    value: Any
    memory_type: MemoryType
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "key": self.key,
            "value": self.value,
            "memory_type": self.memory_type.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """Create from dictionary"""
        return cls(
            id=data["id"],
            key=data["key"],
            value=data["value"],
            memory_type=MemoryType(data["memory_type"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            metadata=data.get("metadata", {}),
        )


class Memory(ABC):
    """
    Abstract base class for memory implementations
    """
    
    @abstractmethod
    async def initialize(self):
        """Initialize the memory storage"""
        pass
    
    @abstractmethod
    async def close(self):
        """Close the memory storage"""
        pass
    
    @abstractmethod
    async def store(
        self, 
        key: str, 
        value: Any, 
        memory_type: MemoryType = MemoryType.WORKING,
        expires_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a value in memory"""
        pass
    
    @abstractmethod
    async def retrieve(self, key: str) -> Optional[MemoryEntry]:
        """Retrieve a value from memory"""
        pass
    
    @abstractmethod
    async def update(
        self, 
        key: str, 
        value: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update a value in memory"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a value from memory"""
        pass
    
    @abstractmethod
    async def list_keys(
        self, 
        memory_type: Optional[MemoryType] = None,
        pattern: Optional[str] = None
    ) -> List[str]:
        """List all keys, optionally filtered"""
        pass
    
    @abstractmethod
    async def clear(self, memory_type: Optional[MemoryType] = None) -> int:
        """Clear memory, optionally by type"""
        pass
    
    @abstractmethod
    async def cleanup_expired(self) -> int:
        """Clean up expired entries"""
        pass


class InMemoryStorage(Memory):
    """
    In-memory storage implementation using dictionaries
    """
    
    def __init__(self):
        self._storage: Dict[str, MemoryEntry] = {}
        self._lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize the memory storage"""
        # For in-memory storage, no special initialization needed
        pass
    
    async def close(self):
        """Close the memory storage"""
        # Clear all data when closing
        async with self._lock:
            self._storage.clear()
    
    async def store(
        self, 
        key: str, 
        value: Any, 
        memory_type: MemoryType = MemoryType.WORKING,
        expires_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a value in memory"""
        async with self._lock:
            now = datetime.utcnow()
            entry_id = str(uuid4())
            
            entry = MemoryEntry(
                id=entry_id,
                key=key,
                value=value,
                memory_type=memory_type,
                created_at=now,
                updated_at=now,
                expires_at=expires_at,
                metadata=metadata or {}
            )
            
            self._storage[key] = entry
            return entry_id
    
    async def retrieve(self, key: str) -> Optional[MemoryEntry]:
        """Retrieve a value from memory"""
        async with self._lock:
            entry = self._storage.get(key)
            
            if entry is None:
                return None
            
            # Check if expired
            if entry.expires_at and datetime.utcnow() > entry.expires_at:
                del self._storage[key]
                return None
            
            return entry
    
    async def update(
        self, 
        key: str, 
        value: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update a value in memory"""
        async with self._lock:
            if key not in self._storage:
                return False
            
            entry = self._storage[key]
            entry.value = value
            entry.updated_at = datetime.utcnow()
            
            if metadata:
                entry.metadata.update(metadata)
            
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete a value from memory"""
        async with self._lock:
            if key in self._storage:
                del self._storage[key]
                return True
            return False
    
    async def list_keys(
        self, 
        memory_type: Optional[MemoryType] = None,
        pattern: Optional[str] = None
    ) -> List[str]:
        """List all keys, optionally filtered"""
        async with self._lock:
            keys = []
            
            for key, entry in self._storage.items():
                # Check memory type filter
                if memory_type and entry.memory_type != memory_type:
                    continue
                
                # Check pattern filter (simple wildcard matching)
                if pattern:
                    import fnmatch
                    if not fnmatch.fnmatch(key, pattern):
                        continue
                
                # Check if expired
                if entry.expires_at and datetime.utcnow() > entry.expires_at:
                    continue
                
                keys.append(key)
            
            return keys
    
    async def clear(self, memory_type: Optional[MemoryType] = None) -> int:
        """Clear memory, optionally by type"""
        async with self._lock:
            if memory_type is None:
                count = len(self._storage)
                self._storage.clear()
                return count
            
            keys_to_delete = []
            for key, entry in self._storage.items():
                if entry.memory_type == memory_type:
                    keys_to_delete.append(key)
            
            for key in keys_to_delete:
                del self._storage[key]
            
            return len(keys_to_delete)
    
    async def cleanup_expired(self) -> int:
        """Clean up expired entries"""
        async with self._lock:
            now = datetime.utcnow()
            expired_keys = []
            
            for key, entry in self._storage.items():
                if entry.expires_at and now > entry.expires_at:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._storage[key]
            
            return len(expired_keys)


class RedisMemory(Memory):
    """
    Redis-based memory implementation (requires redis package)
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self._redis = None
    
    async def _get_redis(self):
        """Get Redis connection"""
        if self._redis is None:
            import redis.asyncio as redis
            self._redis = redis.from_url(self.redis_url)
        return self._redis
    
    async def store(
        self, 
        key: str, 
        value: Any, 
        memory_type: MemoryType = MemoryType.WORKING,
        expires_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a value in Redis"""
        redis = await self._get_redis()
        
        now = datetime.utcnow()
        entry_id = str(uuid4())
        
        entry = MemoryEntry(
            id=entry_id,
            key=key,
            value=value,
            memory_type=memory_type,
            created_at=now,
            updated_at=now,
            expires_at=expires_at,
            metadata=metadata or {}
        )
        
        # Serialize entry
        entry_data = json.dumps(entry.to_dict(), default=str)
        
        # Calculate TTL if expires_at is set
        ttl = None
        if expires_at:
            ttl = int((expires_at - now).total_seconds())
            if ttl <= 0:
                return entry_id  # Already expired
        
        # Store in Redis
        if ttl:
            await redis.setex(key, ttl, entry_data)
        else:
            await redis.set(key, entry_data)
        
        return entry_id
    
    async def retrieve(self, key: str) -> Optional[MemoryEntry]:
        """Retrieve a value from Redis"""
        redis = await self._get_redis()
        
        entry_data = await redis.get(key)
        if entry_data is None:
            return None
        
        # Deserialize entry
        entry_dict = json.loads(entry_data)
        return MemoryEntry.from_dict(entry_dict)
    
    async def update(
        self, 
        key: str, 
        value: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update a value in Redis"""
        redis = await self._get_redis()
        
        # Get existing entry
        existing_entry = await self.retrieve(key)
        if existing_entry is None:
            return False
        
        # Update entry
        existing_entry.value = value
        existing_entry.updated_at = datetime.utcnow()
        
        if metadata:
            existing_entry.metadata.update(metadata)
        
        # Store updated entry
        entry_data = json.dumps(existing_entry.to_dict(), default=str)
        
        # Preserve TTL
        ttl = await redis.ttl(key)
        if ttl > 0:
            await redis.setex(key, ttl, entry_data)
        else:
            await redis.set(key, entry_data)
        
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete a value from Redis"""
        redis = await self._get_redis()
        result = await redis.delete(key)
        return result > 0
    
    async def list_keys(
        self, 
        memory_type: Optional[MemoryType] = None,
        pattern: Optional[str] = None
    ) -> List[str]:
        """List all keys, optionally filtered"""
        redis = await self._get_redis()
        
        # Get all keys matching pattern
        search_pattern = pattern or "*"
        all_keys = await redis.keys(search_pattern)
        
        if memory_type is None:
            return [key.decode() for key in all_keys]
        
        # Filter by memory type
        filtered_keys = []
        for key in all_keys:
            entry = await self.retrieve(key.decode())
            if entry and entry.memory_type == memory_type:
                filtered_keys.append(key.decode())
        
        return filtered_keys
    
    async def clear(self, memory_type: Optional[MemoryType] = None) -> int:
        """Clear memory, optionally by type"""
        redis = await self._get_redis()
        
        if memory_type is None:
            # Clear all keys
            all_keys = await redis.keys("*")
            if all_keys:
                return await redis.delete(*all_keys)
            return 0
        
        # Clear keys by memory type
        all_keys = await redis.keys("*")
        keys_to_delete = []
        
        for key in all_keys:
            entry = await self.retrieve(key.decode())
            if entry and entry.memory_type == memory_type:
                keys_to_delete.append(key)
        
        if keys_to_delete:
            return await redis.delete(*keys_to_delete)
        return 0
    
    async def cleanup_expired(self) -> int:
        """Clean up expired entries (Redis handles this automatically)"""
        # Redis automatically removes expired keys
        return 0


class MemoryManager:
    """
    High-level memory manager that coordinates different memory types
    """
    
    def __init__(self, storage: Memory):
        self.storage = storage
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_interval = 300  # 5 minutes
    
    async def start_cleanup_task(self):
        """Start background cleanup task"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def stop_cleanup_task(self):
        """Stop background cleanup task"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self.storage.cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception:
                # Log error but continue
                pass
    
    # Delegate methods to storage
    async def store(self, *args, **kwargs) -> str:
        return await self.storage.store(*args, **kwargs)
    
    async def retrieve(self, *args, **kwargs) -> Optional[MemoryEntry]:
        return await self.storage.retrieve(*args, **kwargs)
    
    async def update(self, *args, **kwargs) -> bool:
        return await self.storage.update(*args, **kwargs)
    
    async def delete(self, *args, **kwargs) -> bool:
        return await self.storage.delete(*args, **kwargs)
    
    async def list_keys(self, *args, **kwargs) -> List[str]:
        return await self.storage.list_keys(*args, **kwargs)
    
    async def clear(self, *args, **kwargs) -> int:
        return await self.storage.clear(*args, **kwargs)
    
    # Convenience methods for different memory types
    async def store_short_term(self, key: str, value: Any, **kwargs) -> str:
        """Store in short-term memory"""
        return await self.store(key, value, MemoryType.SHORT_TERM, **kwargs)
    
    async def store_long_term(self, key: str, value: Any, **kwargs) -> str:
        """Store in long-term memory"""
        return await self.store(key, value, MemoryType.LONG_TERM, **kwargs)
    
    async def store_working(self, key: str, value: Any, **kwargs) -> str:
        """Store in working memory"""
        return await self.store(key, value, MemoryType.WORKING, **kwargs)
    
    async def store_episodic(self, key: str, value: Any, **kwargs) -> str:
        """Store in episodic memory"""
        return await self.store(key, value, MemoryType.EPISODIC, **kwargs)