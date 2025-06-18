# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import json
import re
import logging
import threading
from typing import List, Dict, Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Global storage for temporarily holding Tavily sources during tool execution
_temp_sources_store = []
_temp_sources_lock = threading.Lock()

def extract_tavily_sources_from_response(response_content: str) -> List[Dict[str, Any]]:
    """
    Extract Tavily search results from agent response content.
    
    This looks for JSON arrays containing Tavily results that are typically
    printed by the Tavily tool during agent execution.
    
    Args:
        response_content: The full response content from an agent
        
    Returns:
        List of Tavily source dictionaries with type="page" and valid URLs
    """
    sources = []
    
    # Pattern 1: Look for "sync" or "async" followed by JSON array
    sync_async_pattern = r'(?:sync|async)\s+(\[.*?\])'
    matches = re.findall(sync_async_pattern, response_content, re.DOTALL)
    
    for match in matches:
        try:
            parsed = json.loads(match)
            if isinstance(parsed, list):
                for item in parsed:
                    if (isinstance(item, dict) and 
                        item.get("type") == "page" and 
                        item.get("url")):
                        sources.append(item)
                        logger.debug(f"Found Tavily source: {item.get('title', 'No title')}")
        except json.JSONDecodeError:
            continue
    
    # Pattern 2: Look for standalone JSON arrays (fallback)
    if not sources:
        json_pattern = r'\[(?:[^[\]]*|\[[^[\]]*\])*\]'
        json_matches = re.findall(json_pattern, response_content, re.DOTALL)
        
        for match in json_matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, list):
                    for item in parsed:
                        if (isinstance(item, dict) and 
                            item.get("type") == "page" and 
                            item.get("url")):
                            sources.append(item)
                            logger.debug(f"Found Tavily source via fallback: {item.get('title', 'No title')}")
            except json.JSONDecodeError:
                continue
    
    return sources

def generate_sources_section_from_state(tavily_sources: List[Dict[str, Any]]) -> str:
    """
    Generate a markdown sources section from Tavily sources stored in state.
    
    Args:
        tavily_sources: List of Tavily source dictionaries from state
        
    Returns:
        Formatted markdown section with sources and favicons
    """
    if not tavily_sources:
        return ""
    
    # Remove duplicates based on URL
    unique_sources = []
    seen_urls = set()
    
    for source in tavily_sources:
        url = source.get('url', '')
        if url and url not in seen_urls:
            unique_sources.append(source)
            seen_urls.add(url)
    
    if not unique_sources:
        return ""
    
    # Sort by score if available
    unique_sources.sort(key=lambda x: x.get('score', 0.0), reverse=True)
    
    section = "\n\n---\n\n## 📚 Research Sources\n\n"
    section += f"The following {len(unique_sources)} sources were consulted during this research:\n\n"
    
    for i, source in enumerate(unique_sources, 1):
        title = source.get('title', 'Untitled Source')
        url = source.get('url', '')
        score = source.get('score', 0.0)
        content = source.get('content', '')
        
        # Extract domain for favicon with multiple fallback options
        domain = ""
        favicon_url = ""
        if url:
            try:
                parsed_url = urlparse(url)
                domain = parsed_url.netloc
                # Try multiple favicon approaches with fallback
                # First try standard favicon.ico
                favicon_url = f"https://www.google.com/s2/favicons?domain={domain}&sz=16"
            except Exception:
                domain = "unknown"
                favicon_url = ""
        
        # Content preview (first 120 chars)
        preview = content[:120] + "..." if len(content) > 120 else content
        
        # Quality indicator based on score
        if score >= 0.8:
            quality = "🟢 High Quality"
        elif score >= 0.6:
            quality = "🟡 Medium Quality"
        elif score > 0:
            quality = "🔴 Lower Quality"
        else:
            quality = ""
        
        # Format source entry - use markdown image syntax
        if favicon_url:
            section += f"### {i}. ![{domain} favicon]({favicon_url}) [{title}]({url})\n\n"
        else:
            section += f"### {i}. 🌐 [{title}]({url})\n\n"
        
        if domain and domain != "unknown":
            section += f"**🌐 Domain:** `{domain}`\n\n"
        
        if score > 0 and quality:
            section += f"**📊 Relevance:** {score:.2f}/1.0 {quality}\n\n"
        
        if preview.strip():
            section += f"**📝 Preview:** {preview}\n\n"
        
        section += "---\n\n"
    
    # Domain summary
    domains = {}
    for source in unique_sources:
        url = source.get('url', '')
        if url:
            try:
                domain = urlparse(url).netloc
                if domain:
                    domains[domain] = domains.get(domain, 0) + 1
            except Exception:
                continue
    
    if domains and len(domains) > 1:
        section += "### 🏢 Sources by Domain\n\n"
        sorted_domains = sorted(domains.items(), key=lambda x: x[1], reverse=True)
        
        for domain, count in sorted_domains:
            favicon_url = f"https://www.google.com/s2/favicons?domain={domain}&sz=16"
            plural = "source" if count == 1 else "sources"
            section += f"- ![{domain} favicon]({favicon_url}) **{domain}** ({count} {plural})\n"
        
        section += "\n"
    
    section += "### ℹ️ About These Sources\n\n"
    section += "- **Search Technology:** Tavily AI-powered web search\n"
    section += "- **Quality Scoring:** Automated relevance assessment (when available)\n"
    section += "- **Logos:** Automatically extracted from source domains\n\n"
    
    return section

def store_tavily_sources_from_results(cleaned_results: List[Dict[str, Any]]) -> None:
    """
    Store Tavily sources from tool results into temporary global storage.
    
    Args:
        cleaned_results: List of cleaned Tavily results from tool execution
    """
    with _temp_sources_lock:
        for result in cleaned_results:
            if (isinstance(result, dict) and 
                result.get("type") == "page" and 
                result.get("url")):
                _temp_sources_store.append(result)
                logger.debug(f"Stored Tavily source: {result.get('title', 'No title')}")

def get_and_clear_temp_sources() -> List[Dict[str, Any]]:
    """
    Retrieve and clear all temporarily stored Tavily sources.
    
    Returns:
        List of stored Tavily source dictionaries
    """
    with _temp_sources_lock:
        sources = _temp_sources_store.copy()
        _temp_sources_store.clear()
        logger.debug(f"Retrieved and cleared {len(sources)} temp sources")
        return sources

def clear_temp_sources() -> None:
    """Clear all temporarily stored Tavily sources."""
    with _temp_sources_lock:
        count = len(_temp_sources_store)
        _temp_sources_store.clear()
        logger.debug(f"Cleared {count} temp sources")