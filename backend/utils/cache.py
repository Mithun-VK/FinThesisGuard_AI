# backend/utils/cache.py
"""
FinThesisGuard AI — Redis Caching Layer
ArkAngel Financial Solutions

Async Redis cache with graceful degradation, namespaced keys,
cache-aside pattern, and full hit/miss telemetry.
Every agent and pipeline imports from here to avoid duplicate LLM calls.
"""

import asyncio
import hashlib
import json
import time
import re
from typing import Any, Callable, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

import redis.asyncio as aioredis
from redis.asyncio import Redis
from redis.exceptions import (
    ConnectionError as RedisConnectionError,
    TimeoutError as RedisTimeoutError,
    ResponseError as RedisResponseError,
)

from backend.config import (
    settings,
    CACHE_TTL,
    CACHE_TTL_THESIS,
    CACHE_TTL_EMBEDDINGS,
    CACHE_TTL_HEALTH,
    CACHE_NAMESPACES,
)
from backend.utils.logger import logger, log_metric


# ─────────────────────────────────────────────
# CACHE NAMESPACE ENUM
# ─────────────────────────────────────────────

class CacheNamespace(str, Enum):
    RAG     = "rag:"
    THESIS  = "thesis:"
    EMBED   = "embed:"
    HEALTH  = "health:"
    AGENT   = "agent:"
    INGEST  = "ingest:"
    QUERY   = "query:"


# ─────────────────────────────────────────────
# DEFAULT TTLs PER NAMESPACE
# ─────────────────────────────────────────────

NAMESPACE_TTL: dict[str, int] = {
    CacheNamespace.RAG:    CACHE_TTL,
    CacheNamespace.THESIS: CACHE_TTL_THESIS,
    CacheNamespace.EMBED:  CACHE_TTL_EMBEDDINGS,
    CacheNamespace.HEALTH: CACHE_TTL_HEALTH,
    CacheNamespace.AGENT:  CACHE_TTL,
    CacheNamespace.INGEST: CACHE_TTL * 24,           # 24 hours for ingestion state
    CacheNamespace.QUERY:  CACHE_TTL,
}


# ─────────────────────────────────────────────
# CACHE STATS DATACLASS
# ─────────────────────────────────────────────

@dataclass
class CacheStats:
    """Tracks cache performance metrics across the session."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0
    bypasses: int = 0                                 # Times Redis was down
    total_get_time_ms: float = 0.0
    total_set_time_ms: float = 0.0

    @property
    def total_gets(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        if self.total_gets == 0:
            return 0.0
        return round(self.hits / self.total_gets * 100, 2)

    @property
    def avg_get_time_ms(self) -> float:
        if self.total_gets == 0:
            return 0.0
        return round(self.total_get_time_ms / self.total_gets, 2)

    def to_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "deletes": self.deletes,
            "errors": self.errors,
            "bypasses": self.bypasses,
            "hit_rate_pct": self.hit_rate,
            "avg_get_time_ms": self.avg_get_time_ms,
            "total_gets": self.total_gets,
        }


# ─────────────────────────────────────────────
# CACHE CLIENT CLASS
# ─────────────────────────────────────────────

class CacheClient:
    """
    Async Redis cache client with graceful degradation.

    If Redis is unavailable, all operations silently bypass cache
    and return None/False — the application continues without caching.

    Usage:
        from backend.utils.cache import cache

        # Simple get/set
        await cache.set_cache("rag:query123", {"answer": "..."}, ttl=3600)
        result = await cache.get_cached("rag:query123")

        # Cache-aside pattern
        result = await cache.get_or_compute(
            key="rag:query123",
            compute_fn=lambda: expensive_llm_call(),
            ttl=3600,
        )

        # Namespaced key generation
        key = cache.generate_cache_key("HDFC NIM Q3", CacheNamespace.RAG)
    """

    def __init__(self):
        self._client: Optional[Redis] = None
        self._connected: bool = False
        self._connect_attempted: bool = False
        self._last_error_time: float = 0.0
        self._error_cooldown: float = 30.0            # Retry connect after 30s
        self._stats = CacheStats()

    # ─────────────────────────────────────────
    # CONNECTION MANAGEMENT
    # ─────────────────────────────────────────

    async def _get_client(self) -> Optional[Redis]:
        """
        Returns Redis client if connected, or attempts reconnection.
        Returns None if Redis is unavailable (graceful degradation).
        """
        if not settings.enable_cache:
            return None

        # If connected, return existing client
        if self._connected and self._client:
            return self._client

        # Cooldown: don't hammer a failing Redis
        now = time.monotonic()
        if (
            self._connect_attempted
            and not self._connected
            and (now - self._last_error_time) < self._error_cooldown
        ):
            return None

        # Attempt (re)connection
        try:
            client = aioredis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=3,
                retry_on_timeout=False,
                health_check_interval=30,
            )
            # Ping to verify connection
            await client.ping()
            self._client = client
            self._connected = True
            self._connect_attempted = True
            logger.info(
                f"[CACHE] Redis connected | url={settings.redis_url}"
            )
            return self._client

        except (RedisConnectionError, RedisTimeoutError, OSError) as e:
            self._connected = False
            self._connect_attempted = True
            self._last_error_time = time.monotonic()
            logger.warning(
                f"[CACHE] Redis unavailable — cache disabled | "
                f"error={type(e).__name__}: {e} | "
                f"retry in {self._error_cooldown}s"
            )
            return None

        except Exception as e:
            self._connected = False
            self._connect_attempted = True
            self._last_error_time = time.monotonic()
            logger.error(
                f"[CACHE] Unexpected Redis error: {type(e).__name__}: {e}"
            )
            return None

    async def connect(self) -> bool:
        """
        Explicitly connects to Redis.
        Called at app startup. Returns True if connected.
        """
        client = await self._get_client()
        return client is not None

    async def disconnect(self) -> None:
        """Closes Redis connection. Called at app shutdown."""
        if self._client:
            try:
                # Try aclose() first (newer redis-py), fall back to close()
                if hasattr(self._client, 'aclose'):
                    await self._client.aclose()
                else:
                    await self._client.close()
                self._connected = False
                logger.info("[CACHE] Redis connection closed")
            except Exception as e:
                logger.warning(f"[CACHE] Error closing Redis: {e}")

    async def ping(self) -> bool:
        """Health check — returns True if Redis is responding."""
        client = await self._get_client()
        if client is None:
            return False
        try:
            result = await client.ping()
            return result is True
        except Exception:
            self._connected = False
            return False

    # ─────────────────────────────────────────
    # CORE: get_cached
    # ─────────────────────────────────────────

    async def get_cached(
        self,
        key: str,
        default: Any = None,
    ) -> Optional[Any]:
        """
        Retrieves a cached value by key.
        Returns None (or default) on miss or Redis failure.

        Args:
            key:     Full cache key (include namespace prefix)
            default: Value to return on miss (default: None)

        Returns:
            Deserialized cached value, or default on miss

        Examples:
            result = await cache.get_cached("rag:abc123")
            result = await cache.get_cached("embed:query_hash", default=[])
        """
        start = time.perf_counter()
        client = await self._get_client()

        if client is None:
            self._stats.bypasses += 1
            return default

        try:
            raw = await client.get(key)
            duration_ms = (time.perf_counter() - start) * 1000
            self._stats.total_get_time_ms += duration_ms

            if raw is None:
                self._stats.misses += 1
                logger.debug(f"[CACHE MISS] key={key} | {duration_ms:.1f}ms")
                return default

            value = json.loads(raw)
            self._stats.hits += 1
            logger.debug(
                f"[CACHE HIT] key={key} | {duration_ms:.1f}ms"
            )
            return value

        except (RedisConnectionError, RedisTimeoutError) as e:
            self._connected = False
            self._stats.errors += 1
            self._stats.bypasses += 1
            logger.warning(f"[CACHE GET ERROR] key={key} | {type(e).__name__}: {e}")
            return default

        except json.JSONDecodeError as e:
            self._stats.errors += 1
            logger.error(f"[CACHE DESERIALIZE ERROR] key={key} | {e}")
            # Delete corrupt entry
            await self._silent_delete(client, key)
            return default

        except Exception as e:
            self._stats.errors += 1
            logger.error(f"[CACHE GET UNEXPECTED] key={key} | {type(e).__name__}: {e}")
            return default

    # ─────────────────────────────────────────
    # CORE: set_cache
    # ─────────────────────────────────────────

    async def set_cache(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Stores a value in cache with TTL.
        Silently returns False if Redis is unavailable.

        Args:
            key:   Full cache key (include namespace prefix)
            value: Any JSON-serializable value
            ttl:   TTL in seconds (default: CACHE_TTL from config)

        Returns:
            True on success, False on failure

        Examples:
            await cache.set_cache("rag:abc123", {"answer": "..."}, ttl=3600)
            await cache.set_cache("embed:query", [0.1, 0.2, ...], ttl=86400)
        """
        start = time.perf_counter()
        client = await self._get_client()

        if client is None:
            self._stats.bypasses += 1
            return False

        _ttl = ttl if ttl is not None else CACHE_TTL

        try:
            serialized = json.dumps(value, default=str)
            await client.setex(key, _ttl, serialized)
            duration_ms = (time.perf_counter() - start) * 1000
            self._stats.sets += 1
            self._stats.total_set_time_ms += duration_ms
            logger.debug(
                f"[CACHE SET] key={key} | ttl={_ttl}s | "
                f"size={len(serialized)}B | {duration_ms:.1f}ms"
            )
            return True

        except (RedisConnectionError, RedisTimeoutError) as e:
            self._connected = False
            self._stats.errors += 1
            self._stats.bypasses += 1
            logger.warning(f"[CACHE SET ERROR] key={key} | {type(e).__name__}: {e}")
            return False

        except (TypeError, ValueError) as e:
            self._stats.errors += 1
            logger.error(f"[CACHE SERIALIZE ERROR] key={key} | {e}")
            return False

        except Exception as e:
            self._stats.errors += 1
            logger.error(f"[CACHE SET UNEXPECTED] key={key} | {type(e).__name__}: {e}")
            return False

    # ─────────────────────────────────────────
    # CORE: delete_cache
    # ─────────────────────────────────────────

    async def delete_cache(self, key: str) -> bool:
        """
        Deletes a cache entry by key.
        Returns True if deleted, False on failure or miss.

        Args:
            key: Full cache key to delete

        Returns:
            True if key existed and was deleted

        Examples:
            await cache.delete_cache("rag:abc123")
        """
        client = await self._get_client()

        if client is None:
            self._stats.bypasses += 1
            return False

        try:
            count = await client.delete(key)
            deleted = count > 0
            if deleted:
                self._stats.deletes += 1
                logger.debug(f"[CACHE DELETE] key={key}")
            return deleted

        except (RedisConnectionError, RedisTimeoutError) as e:
            self._connected = False
            self._stats.errors += 1
            logger.warning(f"[CACHE DELETE ERROR] key={key} | {type(e).__name__}: {e}")
            return False

        except Exception as e:
            self._stats.errors += 1
            logger.error(f"[CACHE DELETE UNEXPECTED] key={key} | {type(e).__name__}: {e}")
            return False

    # ─────────────────────────────────────────
    # CORE: get_or_compute (cache-aside)
    # ─────────────────────────────────────────

    async def get_or_compute(
        self,
        key: str,
        compute_fn: Callable,
        ttl: Optional[int] = None,
        force_refresh: bool = False,
    ) -> Any:
        """
        Cache-aside pattern: get from cache or compute and store.
        The compute_fn can be sync or async.

        Args:
            key:           Full cache key
            compute_fn:    Callable (sync or async) returning the value
            ttl:           TTL in seconds (default: CACHE_TTL)
            force_refresh: If True, bypass cache and recompute

        Returns:
            Cached or freshly computed value

        Examples:
            result = await cache.get_or_compute(
                key=cache.generate_cache_key(query, CacheNamespace.RAG),
                compute_fn=lambda: pipeline.run(query),
                ttl=3600,
            )
        """
        # Step 1: Try cache (unless force refresh)
        if not force_refresh:
            cached = await self.get_cached(key)
            if cached is not None:
                return cached

        # Step 2: Compute fresh value
        compute_start = time.perf_counter()
        try:
            if asyncio.iscoroutinefunction(compute_fn):
                value = await compute_fn()
            else:
                value = compute_fn()
        except Exception as e:
            logger.error(
                f"[CACHE COMPUTE ERROR] key={key} | "
                f"compute_fn={getattr(compute_fn, '__name__', 'lambda')} | "
                f"{type(e).__name__}: {e}"
            )
            raise

        compute_duration_ms = (time.perf_counter() - compute_start) * 1000
        logger.debug(
            f"[CACHE COMPUTED] key={key} | {compute_duration_ms:.0f}ms"
        )

        # Step 3: Store in cache (don't block on failure)
        await self.set_cache(key, value, ttl=ttl)

        return value

    # ─────────────────────────────────────────
    # CORE: flush_namespace
    # ─────────────────────────────────────────

    async def flush_namespace(self, namespace: str) -> int:
        """
        Deletes all keys matching a namespace prefix.
        Uses SCAN for safety (never KEYS in production).

        Args:
            namespace: Prefix string (e.g., "rag:", "thesis:", "embed:")

        Returns:
            Number of keys deleted

        Examples:
            deleted = await cache.flush_namespace(CacheNamespace.RAG)
            deleted = await cache.flush_namespace("embed:")
        """
        client = await self._get_client()

        if client is None:
            self._stats.bypasses += 1
            return 0

        namespace_str = namespace.value if hasattr(namespace, 'value') else str(namespace)
        pattern = f"{namespace_str}*"        
        deleted_count = 0

        try:
            # SCAN iterates safely without blocking Redis
            async for key in client.scan_iter(match=pattern, count=100):
                await client.delete(key)
                deleted_count += 1
                self._stats.deletes += 1

            logger.info(
                f"[CACHE FLUSH] namespace={namespace} | "
                f"deleted={deleted_count} keys"
            )
            return deleted_count

        except (RedisConnectionError, RedisTimeoutError) as e:
            self._connected = False
            self._stats.errors += 1
            logger.warning(
                f"[CACHE FLUSH ERROR] namespace={namespace} | "
                f"{type(e).__name__}: {e}"
            )
            return 0

        except Exception as e:
            self._stats.errors += 1
            logger.error(
                f"[CACHE FLUSH UNEXPECTED] namespace={namespace} | "
                f"{type(e).__name__}: {e}"
            )
            return 0

    # ─────────────────────────────────────────
    # UTILITY: generate_cache_key
    # ─────────────────────────────────────────

    @staticmethod
    def generate_cache_key(
        query: str,
        namespace: str = CacheNamespace.RAG,
        extra: Optional[dict] = None,
    ) -> str:
        """
        Generates a deterministic, normalized cache key from a query string.
        Uses MD5 hash of normalized + lowercased query.

        Args:
            query:     Input string to hash (query, thesis text, etc.)
            namespace: Cache namespace prefix
            extra:     Optional extra params to include in hash (top_k, filters etc.)

        Returns:
            Cache key string like "rag:a1b2c3d4e5f6..."

        Examples:
            key = CacheClient.generate_cache_key("HDFC NIM Q3", CacheNamespace.RAG)
            key = CacheClient.generate_cache_key(thesis, CacheNamespace.THESIS)
            key = CacheClient.generate_cache_key(
                "HDFC NIM",
                CacheNamespace.RAG,
                extra={"top_k": 10}
            )
        """
        # Normalize: lowercase, strip whitespace, collapse spaces
        normalized = re.sub(r'\s+', ' ', query.lower().strip())

        # Include extra params in hash if provided
        hash_input = normalized
        if extra:
            extra_str = json.dumps(extra, sort_keys=True)
            hash_input = f"{normalized}|{extra_str}"

        key_hash = hashlib.md5(hash_input.encode("utf-8")).hexdigest()
        namespace_str = namespace.value if hasattr(namespace, 'value') else str(namespace)
        return f"{namespace_str}{key_hash}"

    # ─────────────────────────────────────────
    # UTILITY: get_ttl
    # ─────────────────────────────────────────

    async def get_ttl(self, key: str) -> int:
        """
        Returns remaining TTL of a cache key in seconds.
        Returns -1 if key has no TTL, -2 if key doesn't exist.
        """
        client = await self._get_client()
        if client is None:
            return -2
        try:
            return await client.ttl(key)
        except Exception:
            return -2

    # ─────────────────────────────────────────
    # UTILITY: exists
    # ─────────────────────────────────────────

    async def exists(self, key: str) -> bool:
        """Checks if a key exists in cache without retrieving value."""
        client = await self._get_client()
        if client is None:
            return False
        try:
            count = await client.exists(key)
            return count > 0
        except Exception:
            return False

    # ─────────────────────────────────────────
    # UTILITY: get_many / set_many
    # ─────────────────────────────────────────

    async def get_many(self, keys: list[str]) -> dict[str, Any]:
        """
        Retrieves multiple keys in a single round-trip using MGET.

        Args:
            keys: List of cache keys

        Returns:
            Dict of {key: value} for keys that exist in cache
        """
        client = await self._get_client()
        if client is None or not keys:
            return {}

        try:
            raw_values = await client.mget(keys)
            result = {}
            for key, raw in zip(keys, raw_values):
                if raw is not None:
                    try:
                        result[key] = json.loads(raw)
                        self._stats.hits += 1
                    except json.JSONDecodeError:
                        self._stats.errors += 1
                else:
                    self._stats.misses += 1
            logger.debug(
                f"[CACHE MGET] keys={len(keys)} | "
                f"hits={len(result)} | misses={len(keys)-len(result)}"
            )
            return result
        except Exception as e:
            self._stats.errors += 1
            logger.error(f"[CACHE MGET ERROR] {type(e).__name__}: {e}")
            return {}

    async def set_many(
        self,
        items: dict[str, Any],
        ttl: Optional[int] = None,
    ) -> int:
        """
        Stores multiple key-value pairs using pipeline for efficiency.

        Args:
            items: Dict of {key: value} to store
            ttl:   TTL in seconds for all keys

        Returns:
            Number of keys successfully stored
        """
        client = await self._get_client()
        if client is None or not items:
            return 0

        _ttl = ttl if ttl is not None else CACHE_TTL
        success_count = 0

        try:
            pipe = client.pipeline(transaction=False)
            for key, value in items.items():
                try:
                    serialized = json.dumps(value, default=str)
                    pipe.setex(key, _ttl, serialized)
                except (TypeError, ValueError) as e:
                    logger.error(f"[CACHE PIPELINE SERIALIZE] key={key} | {e}")

            results = await pipe.execute()
            success_count = sum(1 for r in results if r)
            self._stats.sets += success_count
            logger.debug(
                f"[CACHE MSET] attempted={len(items)} | "
                f"stored={success_count} | ttl={_ttl}s"
            )
            return success_count

        except Exception as e:
            self._stats.errors += 1
            logger.error(f"[CACHE MSET ERROR] {type(e).__name__}: {e}")
            return 0

    # ─────────────────────────────────────────
    # EMBEDDING-SPECIFIC: encode vectors
    # ─────────────────────────────────────────

    async def get_embedding(self, text: str) -> Optional[list[float]]:
        """
        Retrieves cached embedding vector for text.

        Args:
            text: Text whose embedding to look up

        Returns:
            List of floats (embedding vector), or None on miss
        """
        key = self.generate_cache_key(text, CacheNamespace.EMBED)
        return await self.get_cached(key)

    async def set_embedding(
        self,
        text: str,
        vector: list[float],
        ttl: int = CACHE_TTL_EMBEDDINGS,
    ) -> bool:
        """
        Caches an embedding vector for text.

        Args:
            text:   Text that was embedded
            vector: Embedding vector (list of floats)
            ttl:    TTL in seconds (default: 24 hours)

        Returns:
            True on success
        """
        key = self.generate_cache_key(text, CacheNamespace.EMBED)
        return await self.set_cache(key, vector, ttl=ttl)

    # ─────────────────────────────────────────
    # STATS & MONITORING
    # ─────────────────────────────────────────

    def get_stats(self) -> dict:
        """Returns cache performance statistics."""
        stats = self._stats.to_dict()
        stats["connected"] = self._connected
        stats["cache_enabled"] = settings.enable_cache
        return stats

    def emit_stats(self) -> None:
        """Emits cache stats as structured metrics to metrics.log."""
        stats = self._stats.to_dict()
        log_metric("cache_hit_rate", self._stats.hit_rate, unit="pct")
        log_metric("cache_hits", self._stats.hits, unit="count")
        log_metric("cache_misses", self._stats.misses, unit="count")
        log_metric("cache_errors", self._stats.errors, unit="count")
        log_metric("cache_avg_get_ms", self._stats.avg_get_time_ms, unit="ms")

    async def get_redis_info(self) -> dict:
        """
        Returns Redis server info for health endpoint.
        Safe subset of Redis INFO command output.
        """
        client = await self._get_client()
        if client is None:
            return {"status": "unavailable"}

        try:
            info = await client.info()
            return {
                "status": "ok",
                "redis_version": info.get("redis_version"),
                "connected_clients": info.get("connected_clients"),
                "used_memory_human": info.get("used_memory_human"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits"),
                "keyspace_misses": info.get("keyspace_misses"),
                "uptime_in_seconds": info.get("uptime_in_seconds"),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def count_namespace(self, namespace: str) -> int:
        """Returns count of keys in a namespace."""
        client = await self._get_client()
        if client is None:
            return 0
        try:
            count = 0
            async for _ in client.scan_iter(
                match=f"{namespace}*", count=100
            ):
                count += 1
            return count
        except Exception:
            return 0

    # ─────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────

    async def _silent_delete(
        self,
        client: Redis,
        key: str,
    ) -> None:
        """Deletes a key without raising exceptions — for cleanup use."""
        try:
            await client.delete(key)
        except Exception:
            pass

    def _get_namespace_ttl(self, namespace: str) -> int:
        """Returns default TTL for a given namespace."""
        return NAMESPACE_TTL.get(namespace, CACHE_TTL)


# ─────────────────────────────────────────────
# CONVENIENCE FUNCTIONS
# Module-level wrappers around the singleton
# ─────────────────────────────────────────────

async def get_cached(key: str, default: Any = None) -> Optional[Any]:
    """Module-level wrapper for cache.get_cached()"""
    return await cache.get_cached(key, default)


async def set_cache(
    key: str,
    value: Any,
    ttl: Optional[int] = None,
) -> bool:
    """Module-level wrapper for cache.set_cache()"""
    return await cache.set_cache(key, value, ttl)


async def delete_cache(key: str) -> bool:
    """Module-level wrapper for cache.delete_cache()"""
    return await cache.delete_cache(key)


def generate_cache_key(
    query: str,
    namespace: str = CacheNamespace.RAG,
    extra: Optional[dict] = None,
) -> str:
    """Module-level wrapper for CacheClient.generate_cache_key()"""
    return CacheClient.generate_cache_key(query, namespace, extra)


async def get_or_compute(
    key: str,
    compute_fn: Callable,
    ttl: Optional[int] = None,
    force_refresh: bool = False,
) -> Any:
    """Module-level wrapper for cache.get_or_compute()"""
    return await cache.get_or_compute(key, compute_fn, ttl, force_refresh)


async def flush_namespace(namespace: str) -> int:
    """Module-level wrapper for cache.flush_namespace()"""
    return await cache.flush_namespace(namespace)


# ─────────────────────────────────────────────
# SINGLETON INSTANCE
# All modules import: from backend.utils.cache import cache
# ─────────────────────────────────────────────

cache = CacheClient()


# ─────────────────────────────────────────────
# MODULE EXPORTS
# ─────────────────────────────────────────────

__all__ = [
    # Singleton
    "cache",
    "CacheClient",

    # Enums & Types
    "CacheNamespace",
    "CacheStats",

    # Convenience functions
    "get_cached",
    "set_cache",
    "delete_cache",
    "generate_cache_key",
    "get_or_compute",
    "flush_namespace",
]
