import os
import requests
import frappe

uid = "web_search"
name = "web_search"
title = "Web Search"
description = "Search the internet for information using Google Search. Use this tool to find current events, facts, or data not in your training set."
is_destructive = False
is_idempotent = True

properties = {
    "query": {
        "type": "string",
        "description": "The search query to execute"
    }
}
required = ["query"]

def web_search(query: str):
    """
    Search the web for the given query.
    Supports 'SERPER_API_KEY' or 'GOOGLE_SEARCH_API_KEY' + 'GOOGLE_SEARCH_CX'.
    """
    
    # Check for Serper API Key first
    serper_key = os.environ.get("SERPER_API_KEY") or frappe.conf.get("SERPER_API_KEY")
    if serper_key:
        return _serper_search(query, serper_key)
        
    # Check for Google custom search
    google_key = os.environ.get("GOOGLE_SEARCH_API_KEY") or frappe.conf.get("GOOGLE_SEARCH_API_KEY")
    google_cx = os.environ.get("GOOGLE_SEARCH_CX") or frappe.conf.get("GOOGLE_SEARCH_CX")
    
    if google_key and google_cx:
        return _google_custom_search(query, google_key, google_cx)

    # Fallback/Error
    available_methods = []
    if not serper_key: available_methods.append("Serper (missing SERPER_API_KEY)")
    if not (google_key and google_cx): available_methods.append("Google (missing GOOGLE_SEARCH_API_KEY/CX)")
    
    return {
        "error": "No search API keys configured.",
        "details": "Please configure SERPER_API_KEY or GOOGLE_SEARCH_API_KEY+CX in bench site config or environment variables."
    }

def _serper_search(query: str, api_key: str):
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query, "num": 10})
    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        data = response.json()
        
        results = []
        if "organic" in data:
            for item in data["organic"]:
                results.append({
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "snippet": item.get("snippet"),
                    "date": item.get("date")
                })
        return {"results": results}
    except Exception as e:
        return {"error": f"Serper API failed: {str(e)}"}

def _google_custom_search(query: str, api_key: str, cx: str):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cx,
        "q": query,
        "num": 10
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        results = []
        if "items" in data:
            for item in data["items"]:
                results.append({
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "snippet": item.get("snippet")
                })
        return {"results": results}
    except Exception as e:
        return {"error": f"Google API failed: {str(e)}"}

import json
