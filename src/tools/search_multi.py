# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import json
import logging
import os

from langchain_community.tools import BraveSearch, DuckDuckGoSearchResults
from langchain_community.tools.arxiv import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper, BraveSearchWrapper

from src.config import SEARCH_MAX_RESULTS, SearchEngine
from src.tools.tavily_search.tavily_search_results_with_images import (
    TavilySearchResultsWithImages,
)

from src.tools.decorators import create_logged_tool

logger = logging.getLogger(__name__)

# Create all search tools with unique names
LoggedTavilySearch = create_logged_tool(TavilySearchResultsWithImages)
tavily_search_tool = LoggedTavilySearch(
    name="tavily_search",  # Unique name
    max_results=SEARCH_MAX_RESULTS,
    include_raw_content=True,
    include_images=True,
    include_image_descriptions=True,
) if os.getenv("TAVILY_API_KEY") else None

LoggedDuckDuckGoSearch = create_logged_tool(DuckDuckGoSearchResults)
duckduckgo_search_tool = LoggedDuckDuckGoSearch(
    name="duckduckgo_search",  # Unique name
    max_results=SEARCH_MAX_RESULTS
)

LoggedBraveSearch = create_logged_tool(BraveSearch)
brave_search_tool = LoggedBraveSearch(
    name="brave_search",  # Unique name
    search_wrapper=BraveSearchWrapper(
        api_key=os.getenv("BRAVE_SEARCH_API_KEY", ""),
        search_kwargs={"count": SEARCH_MAX_RESULTS},
    ),
) if os.getenv("BRAVE_SEARCH_API_KEY") else None

LoggedArxivSearch = create_logged_tool(ArxivQueryRun)
arxiv_search_tool = LoggedArxivSearch(
    name="arxiv_search",  # Unique name
    api_wrapper=ArxivAPIWrapper(
        top_k_results=SEARCH_MAX_RESULTS,
        load_max_docs=SEARCH_MAX_RESULTS,
        load_all_available_meta=True,
    ),
)

# Create a list of all available search tools
def get_available_search_tools():
    """Return a list of all available search tools based on API keys."""
    tools = []
    
    if tavily_search_tool:
        tools.append(tavily_search_tool)
    
    tools.append(duckduckgo_search_tool)  # Always available
    
    if brave_search_tool:
        tools.append(brave_search_tool)
    
    tools.append(arxiv_search_tool)  # Always available
    
    return tools

# For backward compatibility
web_search_tools = get_available_search_tools()

if __name__ == "__main__":
    available_tools = get_available_search_tools()
    print(f"Available search tools: {[tool.name for tool in available_tools]}") 