from langchain_community.tools.tavily_search import TavilySearchResults

search_tavily = TavilySearchResults(max_results=2)
search_tavily.name = "search_tavily"
search_tavily.description = "A search engine useful for when you need to answer questions about any information, and recent events. Input should be a search query." 