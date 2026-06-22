import json
import logging
from typing import List, Dict, Any

from ..routers.legal import _SECTIONS, _JUDGEMENTS

logger = logging.getLogger(__name__)

def search_legal_sections(query: str) -> str:
    """
    Search the internal database of Indian consumer law sections.
    """
    try:
        if not query:
            return "Please provide a search query."
        
        q = query.lower()
        results = [
            s for s in _SECTIONS
            if (
                q in s["title"].lower()
                or q in s["description"].lower()
                or q in s["number"].lower()
                or any(q in kw.lower() for kw in s["keywords"])
            )
        ]
        
        if not results:
            return f"No legal sections found matching '{query}'."
            
        # Format the top 3 results to return to the LLM
        output = []
        for s in results[:3]:
            output.append(
                f"Act: {s['act']}\n"
                f"Section: {s['number']}\n"
                f"Title: {s['title']}\n"
                f"Description: {s['description']}\n"
                f"Keywords: {', '.join(s['keywords'])}\n"
            )
            
        return "Found the following sections:\n\n" + "\n---\n".join(output)
        
    except Exception as e:
        logger.error(f"Error in search_legal_sections tool: {e}")
        return f"Error executing search: {str(e)}"

def search_judgements(query: str, sector: str = "") -> str:
    """
    Search landmark Indian consumer court judgements.
    """
    try:
        if not query:
            return "Please provide a search query."
            
        q = query.lower()
        sec = sector.lower() if sector else ""
        
        results = _JUDGEMENTS.copy()
        
        if sec:
            results = [j for j in results if sec in j["sector"].lower()]
            
        results = [
            j for j in results
            if (
                q in j["title"].lower()
                or q in j["summary"].lower()
                or q in j["key_principle"].lower()
                or any(q in tag.lower() for tag in j["tags"])
                or q in j["sector"].lower()
            )
        ]
        
        if not results:
            return f"No judgements found matching query '{query}' and sector '{sector}'."
            
        # Format top 3 results
        output = []
        for j in results[:3]:
            output.append(
                f"Case: {j['title']} ({j['court']}, {j['year']})\n"
                f"Citation: {j['citation']}\n"
                f"Outcome: {j['outcome']}\n"
                f"Key Principle: {j['key_principle']}\n"
                f"Summary: {j['summary']}\n"
                f"Sections: {', '.join(j['sections'])}\n"
            )
            
        return "Found the following landmark judgements:\n\n" + "\n---\n".join(output)
        
    except Exception as e:
        logger.error(f"Error in search_judgements tool: {e}")
        return f"Error executing search: {str(e)}"

# Define the tools schema for Anthropic
ANTHROPIC_TOOLS = [
    {
        "name": "search_legal_sections",
        "description": "Search the Judgelytics internal database of Indian consumer law sections (e.g. COPRA, IPC, BNS). Use this when the user asks about specific laws, penalties, or rules.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keyword (e.g., 'deficiency of service', 'section 35', 'medical negligence')"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "search_judgements",
        "description": "Search the Judgelytics internal database for landmark Indian consumer court judgements (Supreme Court, NCDRC). Use this when the user asks if there are any previous cases or precedents for their situation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keyword for the case (e.g., 'insurance delay', 'builder possession')"
                },
                "sector": {
                    "type": "string",
                    "description": "Optional. The sector to filter by (e.g., 'Healthcare', 'Real Estate', 'E-commerce', 'Banking', 'Insurance', 'Automobile', 'Travel', 'Telecom')"
                }
            },
            "required": ["query"]
        }
    }
]

def execute_tool(tool_name: str, tool_args: Dict[str, Any]) -> str:
    """Execute the requested tool and return the result as a string."""
    if tool_name == "search_legal_sections":
        return search_legal_sections(tool_args.get("query", ""))
    elif tool_name == "search_judgements":
        return search_judgements(tool_args.get("query", ""), tool_args.get("sector", ""))
    else:
        return f"Tool {tool_name} is not recognized."
