# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import json
import logging
import os
from typing import Dict, List, Optional, Union
import asyncio

from langchain_core.tools import BaseTool
from langchain_community.tools import BraveSearch, DuckDuckGoSearchResults
from langchain_community.tools.arxiv import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper, BraveSearchWrapper

from src.config import SEARCH_MAX_RESULTS
from src.tools.tavily_search.tavily_search_results_with_images import (
    TavilySearchResultsWithImages,
)
from src.tools.decorators import create_logged_tool

logger = logging.getLogger(__name__)


class IntelligentSearchTool(BaseTool):
    """
    An intelligent search tool that can use multiple search APIs with:
    - Automatic fallbacks if one API fails
    - Query routing based on query type
    - Result aggregation and deduplication
    """
    
    name = "intelligent_search"
    description = """
    Intelligent web search that automatically selects the best search engine(s) based on query type.
    Supports Tavily (with images), Brave Search, DuckDuckGo, and Arxiv.
    Automatically falls back to other engines if primary choice fails.
    """
    
    def __init__(self):
        super().__init__()
        self._initialize_search_engines()
    
    def _initialize_search_engines(self):
        """Initialize all available search engines."""
        self.search_engines = {}
        
        # Tavily - best for general web search with images
        if os.getenv("TAVILY_API_KEY"):
            LoggedTavilySearch = create_logged_tool(TavilySearchResultsWithImages)
            self.search_engines["tavily"] = LoggedTavilySearch(
                name="tavily_internal",
                max_results=SEARCH_MAX_RESULTS,
                include_raw_content=True,
                include_images=True,
                include_image_descriptions=True,
            )
        
        # Brave Search - good for recent/news content
        if os.getenv("BRAVE_SEARCH_API_KEY"):
            LoggedBraveSearch = create_logged_tool(BraveSearch)
            self.search_engines["brave"] = LoggedBraveSearch(
                name="brave_internal",
                search_wrapper=BraveSearchWrapper(
                    api_key=os.getenv("BRAVE_SEARCH_API_KEY"),
                    search_kwargs={"count": SEARCH_MAX_RESULTS},
                ),
            )
        
        # DuckDuckGo - always available, privacy-focused
        LoggedDuckDuckGoSearch = create_logged_tool(DuckDuckGoSearchResults)
        self.search_engines["duckduckgo"] = LoggedDuckDuckGoSearch(
            name="duckduckgo_internal",
            max_results=SEARCH_MAX_RESULTS
        )
        
        # Arxiv - for academic/scientific content
        LoggedArxivSearch = create_logged_tool(ArxivQueryRun)
        self.search_engines["arxiv"] = LoggedArxivSearch(
            name="arxiv_internal",
            api_wrapper=ArxivAPIWrapper(
                top_k_results=SEARCH_MAX_RESULTS,
                load_max_docs=SEARCH_MAX_RESULTS,
                load_all_available_meta=True,
            ),
        )
    
    def _determine_search_strategy(self, query: str) -> List[str]:
        """Determine which search engines to use based on query content."""
        query_lower = query.lower()
        
        # Academic/research queries
        academic_keywords = ["paper", "research", "study", "academic", "journal", "arxiv", "doi"]
        if any(keyword in query_lower for keyword in academic_keywords):
            return ["arxiv", "tavily", "brave", "duckduckgo"]
        
        # News/recent events
        news_keywords = ["news", "latest", "recent", "today", "yesterday", "breaking"]
        if any(keyword in query_lower for keyword in news_keywords):
            return ["brave", "tavily", "duckduckgo"]
        
        # General queries - prioritize Tavily for images, then others
        return ["tavily", "brave", "duckduckgo", "arxiv"]
    
    def _deduplicate_results(self, all_results: List[Dict]) -> List[Dict]:
        """Remove duplicate results based on URL and title similarity."""
        seen_urls = set()
        seen_titles = set()
        deduplicated = []
        
        for result in all_results:
            if isinstance(result, dict):
                url = result.get("url", "")
                title = result.get("title", "")
                
                # Skip if we've seen this URL or very similar title
                if url and url not in seen_urls:
                    if title and title not in seen_titles:
                        deduplicated.append(result)
                        seen_urls.add(url)
                        seen_titles.add(title)
        
        return deduplicated
    
    def _run(self, query: str) -> List[Dict]:
        """Execute search using multiple engines with fallbacks."""
        search_order = self._determine_search_strategy(query)
        all_results = []
        successful_engines = []
        
        for engine_name in search_order:
            if engine_name not in self.search_engines:
                continue
                
            try:
                logger.info(f"Searching with {engine_name} for query: {query}")
                engine = self.search_engines[engine_name]
                
                if engine_name == "arxiv":
                    # Arxiv expects different input format
                    results = engine.invoke(query)
                else:
                    results = engine.invoke(query)
                
                if results and isinstance(results, list):
                    # Add source information to each result
                    for result in results:
                        if isinstance(result, dict):
                            result["search_engine"] = engine_name
                    
                    all_results.extend(results)
                    successful_engines.append(engine_name)
                    
                    # If we have enough results, we can stop early
                    if len(all_results) >= SEARCH_MAX_RESULTS * 2:
                        break
                        
            except Exception as e:
                logger.warning(f"Search engine {engine_name} failed: {str(e)}")
                continue
        
        if not all_results:
            logger.error("All search engines failed")
            return []
        
        # Deduplicate and limit results
        deduplicated_results = self._deduplicate_results(all_results)
        final_results = deduplicated_results[:SEARCH_MAX_RESULTS * 2]  # Allow more results since we're aggregating
        
        logger.info(f"Search completed using engines: {successful_engines}, returned {len(final_results)} results")
        return final_results
    
    async def _arun(self, query: str) -> List[Dict]:
        """Async version - run searches in parallel for faster results."""
        search_order = self._determine_search_strategy(query)
        
        # Create tasks for all available engines
        tasks = []
        for engine_name in search_order[:3]:  # Limit parallel searches to avoid overwhelming
            if engine_name in self.search_engines:
                engine = self.search_engines[engine_name]
                task = asyncio.create_task(self._search_with_engine(engine, engine_name, query))
                tasks.append(task)
        
        # Wait for all searches to complete (with timeout)
        try:
            results_list = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=30)
        except asyncio.TimeoutError:
            logger.warning("Some search engines timed out")
            results_list = []
        
        # Combine all successful results
        all_results = []
        for results in results_list:
            if isinstance(results, list):
                all_results.extend(results)
        
        if not all_results:
            # Fallback to synchronous if async failed
            return self._run(query)
        
        # Deduplicate and return
        deduplicated_results = self._deduplicate_results(all_results)
        return deduplicated_results[:SEARCH_MAX_RESULTS * 2]
    
    async def _search_with_engine(self, engine, engine_name: str, query: str) -> List[Dict]:
        """Helper method to search with a single engine asynchronously."""
        try:
            if hasattr(engine, '_arun'):
                results = await engine._arun(query)
            else:
                # Run sync method in thread pool
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(None, engine.invoke, query)
            
            if results and isinstance(results, list):
                for result in results:
                    if isinstance(result, dict):
                        result["search_engine"] = engine_name
                return results
            return []
            
        except Exception as e:
            logger.warning(f"Async search with {engine_name} failed: {str(e)}")
            return []


# Create the intelligent search tool instance
intelligent_search_tool = IntelligentSearchTool()

if __name__ == "__main__":
    # Test the intelligent search
    tool = IntelligentSearchTool()
    results = tool._run("latest AI research papers")
    print(f"Found {len(results)} results")
    for i, result in enumerate(results[:3]):
        print(f"{i+1}. {result.get('title', 'No title')} (via {result.get('search_engine', 'unknown')})") 