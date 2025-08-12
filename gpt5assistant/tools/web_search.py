import logging
from typing import Dict, Any, Optional
import asyncio
from datetime import datetime, timedelta

logger = logging.getLogger("red.gpt5assistant.tools.web_search")


class WebSearchTool:
    def __init__(self):
        self._cache = {}
        self._cache_ttl = timedelta(minutes=30)  # Cache web search results for 30 minutes
    
    def is_enabled_in_tools(self, tools_config: Dict[str, Any]) -> bool:
        return tools_config.get("web_search", False)
    
    def get_tool_config(self) -> Dict[str, Any]:
        return {"type": "web_search"}
    
    async def get_cached_result(self, query: str) -> Optional[Dict[str, Any]]:
        cache_key = query.lower().strip()
        
        if cache_key in self._cache:
            cached_result, timestamp = self._cache[cache_key]
            
            # Check if cache is still valid
            if datetime.now() - timestamp < self._cache_ttl:
                logger.debug(f"Using cached web search result for query: {query[:50]}...")
                return cached_result
            else:
                # Remove expired cache entry
                del self._cache[cache_key]
        
        return None
    
    async def cache_result(self, query: str, result: Dict[str, Any]) -> None:
        cache_key = query.lower().strip()
        self._cache[cache_key] = (result, datetime.now())
        
        # Clean up old cache entries to prevent memory leaks
        await self._cleanup_cache()
    
    async def _cleanup_cache(self) -> None:
        current_time = datetime.now()
        expired_keys = []
        
        for cache_key, (_, timestamp) in self._cache.items():
            if current_time - timestamp >= self._cache_ttl:
                expired_keys.append(cache_key)
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def get_search_guidelines(self) -> str:
        return """
        Web search is automatically enabled when this tool is active. The AI will:
        - Search for current information when needed
        - Provide citations and sources
        - Use up-to-date data for responses
        - Avoid making requests without context
        """
    
    def get_usage_stats(self) -> Dict[str, Any]:
        return {
            "cached_queries": len(self._cache),
            "cache_ttl_minutes": int(self._cache_ttl.total_seconds() / 60)
        }
    
    async def clear_cache(self) -> int:
        cleared_count = len(self._cache)
        self._cache.clear()
        logger.info(f"Cleared {cleared_count} cached web search results")
        return cleared_count