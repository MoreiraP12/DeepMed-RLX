# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import os
from typing import List, Dict, Any
from enum import Enum

class SearchEngine(Enum):
    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"
    BRAVE_SEARCH = "brave_search"
    ARXIV = "arxiv"

class MultiSearchConfig:
    """Configuration for multi-search API support."""
    
    def __init__(self):
        # Parse enabled search engines from environment
        self.enabled_engines = self._parse_enabled_engines()
        self.primary_engine = os.getenv("PRIMARY_SEARCH_API", SearchEngine.TAVILY.value)
        self.fallback_enabled = os.getenv("SEARCH_FALLBACK_ENABLED", "true").lower() == "true"
        self.parallel_search = os.getenv("PARALLEL_SEARCH_ENABLED", "false").lower() == "true"
        self.max_engines_per_query = int(os.getenv("MAX_SEARCH_ENGINES", "3"))
        
    def _parse_enabled_engines(self) -> List[str]:
        """Parse which search engines are enabled from environment variables."""
        # Method 1: Comma-separated list
        engines_str = os.getenv("ENABLED_SEARCH_APIS", "")
        if engines_str:
            return [engine.strip() for engine in engines_str.split(",")]
        
        # Method 2: Individual flags
        enabled = []
        
        if os.getenv("ENABLE_TAVILY", "true").lower() == "true" and os.getenv("TAVILY_API_KEY"):
            enabled.append(SearchEngine.TAVILY.value)
            
        if os.getenv("ENABLE_BRAVE", "true").lower() == "true" and os.getenv("BRAVE_SEARCH_API_KEY"):
            enabled.append(SearchEngine.BRAVE_SEARCH.value)
            
        if os.getenv("ENABLE_DUCKDUCKGO", "true").lower() == "true":
            enabled.append(SearchEngine.DUCKDUCKGO.value)
            
        if os.getenv("ENABLE_ARXIV", "true").lower() == "true":
            enabled.append(SearchEngine.ARXIV.value)
        
        # Default to Tavily if nothing is explicitly enabled
        if not enabled:
            if os.getenv("TAVILY_API_KEY"):
                enabled.append(SearchEngine.TAVILY.value)
            else:
                enabled.append(SearchEngine.DUCKDUCKGO.value)
        
        return enabled
    
    def get_search_strategy(self, query_type: str = "general") -> List[str]:
        """Get the search strategy based on query type and configuration."""
        if not self.enabled_engines:
            return [SearchEngine.DUCKDUCKGO.value]  # Fallback
        
        # If only one engine is enabled, use it
        if len(self.enabled_engines) == 1:
            return self.enabled_engines
        
        # Query-type based strategy
        strategies = {
            "academic": [SearchEngine.ARXIV.value, SearchEngine.TAVILY.value, SearchEngine.BRAVE_SEARCH.value],
            "news": [SearchEngine.BRAVE_SEARCH.value, SearchEngine.TAVILY.value, SearchEngine.DUCKDUCKGO.value],
            "general": [SearchEngine.TAVILY.value, SearchEngine.BRAVE_SEARCH.value, SearchEngine.DUCKDUCKGO.value],
            "privacy": [SearchEngine.DUCKDUCKGO.value, SearchEngine.TAVILY.value],
        }
        
        preferred_order = strategies.get(query_type, strategies["general"])
        
        # Filter by enabled engines and respect max_engines_per_query
        available_engines = [eng for eng in preferred_order if eng in self.enabled_engines]
        
        # Add any remaining enabled engines not in the preferred order
        for engine in self.enabled_engines:
            if engine not in available_engines:
                available_engines.append(engine)
        
        return available_engines[:self.max_engines_per_query]
    
    def should_use_parallel_search(self) -> bool:
        """Whether to run searches in parallel."""
        return self.parallel_search and len(self.enabled_engines) > 1
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration."""
        return {
            "enabled_engines": self.enabled_engines,
            "primary_engine": self.primary_engine,
            "fallback_enabled": self.fallback_enabled,
            "parallel_search": self.parallel_search,
            "max_engines_per_query": self.max_engines_per_query,
        }

# Global configuration instance
search_config = MultiSearchConfig()

# Example .env configuration:
"""
# Multi-Search API Configuration

# Method 1: Comma-separated list of enabled engines
ENABLED_SEARCH_APIS=tavily,brave_search,duckduckgo,arxiv

# Method 2: Individual engine flags (if ENABLED_SEARCH_APIS not set)
ENABLE_TAVILY=true
ENABLE_BRAVE=true
ENABLE_DUCKDUCKGO=true
ENABLE_ARXIV=true

# Primary search engine (used first, unless query-specific routing applies)
PRIMARY_SEARCH_API=tavily

# Enable fallback to other engines if primary fails
SEARCH_FALLBACK_ENABLED=true

# Enable parallel searching (faster but uses more resources)
PARALLEL_SEARCH_ENABLED=false

# Maximum number of search engines to use per query
MAX_SEARCH_ENGINES=3

# API Keys (required for respective engines)
TAVILY_API_KEY=your_tavily_key
BRAVE_SEARCH_API_KEY=your_brave_key
# DuckDuckGo and Arxiv don't require keys
""" 