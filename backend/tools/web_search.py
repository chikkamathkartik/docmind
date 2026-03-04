
import sys
import os
import requests
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.settings import SERPER_API_KEY

class WebSearchTool:
    """
    Searches the web using Serper API.
    This is Tool 2 in the agent's tool registry.
    """

    name = "web_search"
    description = """Search the internet for current information not 
    available in uploaded documents. Use this when documents don't 
    contain the answer or when current/recent information is needed.
    Input should be a search query string."""

    def __init__(self):
        self.api_key = SERPER_API_KEY
        self.endpoint = "https://google.serper.dev/search"

    def run(self, query: str, num_results: int = 5) -> dict:
        """
        Search the web for the given query.
        Returns top search results with snippets.
        """
        try:
            headers = {
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json"
            }

            payload = {
                "q": query,
                "num": num_results
            }

            response = requests.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=10
            )

            if response.status_code != 200:
                return {
                    "success": False,
                    "message": f"Web search failed with status {response.status_code}",
                    "results": []
                }

            data = response.json()

            # extract organic results
            results = []
            for item in data.get("organic", []):
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "link": item.get("link", ""),
                    "position": item.get("position", 0)
                })

            return {
                "success": True,
                "query": query,
                "results": results,
                "total_found": len(results)
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Web search error: {str(e)}",
                "results": []
            }

    def format_for_agent(self, query: str) -> str:
        """
        Run search and format results as text the agent can read.
        """
        result = self.run(query)

        if not result["success"]:
            return f"Web search failed: {result['message']}"

        if not result["results"]:
            return "No web results found for this query."

        output = f"Web search results for '{query}':\n\n"
        for i, r in enumerate(result["results"]):
            output += f"[Web Result {i+1}]\n"
            output += f"Title  : {r['title']}\n"
            output += f"Source : {r['link']}\n"
            output += f"Summary: {r['snippet']}\n"
            output += "-" * 40 + "\n"

        return output


# quick test
if __name__ == "__main__":
    tool = WebSearchTool()
    print("Testing web search tool...")
    result = tool.format_for_agent("What is Agentic RAG in AI?")
    print(result)