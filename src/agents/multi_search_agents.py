# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
Example agents demonstrating multiple search API usage.
This shows how to modify existing agents to use multiple search engines.
"""

from langgraph.prebuilt import create_react_agent

from src.prompts import apply_prompt_template
from src.tools import (
    crawl_tool,
    python_repl_tool,
)

# Import multi-search capabilities
from src.tools.search_multi import get_available_search_tools
from src.tools.intelligent_search import intelligent_search_tool

from src.llms.llm import get_llm_by_type
from src.config.agents import AGENT_LLM_MAP


def create_multi_search_agent(agent_name: str, agent_type: str, extra_tools: list, prompt_template: str):
    """Factory function to create agents with multiple search tools."""
    # Get all available search tools
    search_tools = get_available_search_tools()
    
    # Combine with other tools
    all_tools = search_tools + extra_tools
    
    return create_react_agent(
        name=agent_name,
        model=get_llm_by_type(AGENT_LLM_MAP[agent_type]),
        tools=all_tools,
        prompt=lambda state: apply_prompt_template(prompt_template, state),
    )


def create_intelligent_search_agent(agent_name: str, agent_type: str, extra_tools: list, prompt_template: str):
    """Factory function to create agents with intelligent search capability."""
    # Use the intelligent search tool that automatically selects best engines
    all_tools = [intelligent_search_tool] + extra_tools
    
    return create_react_agent(
        name=agent_name,
        model=get_llm_by_type(AGENT_LLM_MAP[agent_type]),
        tools=all_tools,
        prompt=lambda state: apply_prompt_template(prompt_template, state),
    )


# Example agents with different multi-search approaches

# Approach 1: Research agent with access to all individual search tools
multi_search_research_agent = create_multi_search_agent(
    "multi_researcher", 
    "researcher", 
    [crawl_tool], 
    "researcher"
)

# Approach 2: Research agent with intelligent search tool
intelligent_research_agent = create_intelligent_search_agent(
    "intelligent_researcher", 
    "researcher", 
    [crawl_tool], 
    "researcher"
)

# Approach 3: Specialized agents for different types of research
academic_research_agent = create_react_agent(
    name="academic_researcher",
    model=get_llm_by_type(AGENT_LLM_MAP["researcher"]),
    tools=[
        # Prioritize academic sources
        tool for tool in get_available_search_tools() 
        if tool.name in ["arxiv_search", "tavily_search"]
    ] + [crawl_tool],
    prompt=lambda state: apply_prompt_template("researcher", state),
)

news_research_agent = create_react_agent(
    name="news_researcher", 
    model=get_llm_by_type(AGENT_LLM_MAP["researcher"]),
    tools=[
        # Prioritize current/news sources
        tool for tool in get_available_search_tools() 
        if tool.name in ["brave_search", "tavily_search", "duckduckgo_search"]
    ] + [crawl_tool],
    prompt=lambda state: apply_prompt_template("researcher", state),
)

# Example of how to use these agents in the graph nodes
def create_adaptive_research_agent(query_type: str = "general"):
    """Create a research agent based on the type of query."""
    if "academic" in query_type.lower() or "research" in query_type.lower():
        return academic_research_agent
    elif "news" in query_type.lower() or "current" in query_type.lower():
        return news_research_agent
    else:
        return intelligent_research_agent


# Utility function to get search tool information
def get_search_tools_info():
    """Get information about available search tools for debugging/logging."""
    tools = get_available_search_tools()
    return {
        "available_tools": [tool.name for tool in tools],
        "tool_count": len(tools),
        "intelligent_search_available": True,  # Since we created it
    } 