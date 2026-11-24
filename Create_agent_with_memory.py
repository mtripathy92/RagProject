from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain.tools import tool
from pathlib import Path
from typing import Any
import yaml
from yfinance import Ticker
import requests
from langgraph.prebuilt import create_react_agent
import wikipedia
import yfinance as yf
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
# Load environment variables from .env
load_dotenv()

def load_config(config_path: Path | None = None) -> dict[str, Any]:
    resolved_path = Path("config.yaml")
    with resolved_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
config = load_config()
# Access environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")

def get_llm():
    # Initialize the ChatGroq model with the API key
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=config["GROQ_MODEL"],
        temperature=0.7,
        reasoning_format="hidden",
        max_retries=2,
        )
    return llm

@tool
def get_stock_fundamentals(ticker_symbol: str) -> str:
    """
    Use this to get current stock price, P/E ratio, market cap, and revenue growth for a ticker.
    """
    try:
        stock = Ticker(ticker_symbol)
        info = stock.info
        return {
            "price": info.get("currentPrice"),
            "pe_ratio": info.get("trailingPE"),
            "market_cap": info.get("marketCap"),
            "revenue_growth": info.get("revenueGrowth"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
        }
    except Exception as e:
        return f"An error occurred while fetching the stock price: {str(e)}"

@tool
def get_stock_news(ticker: str) -> str:
    """Get recent Yahoo Finance news headlines for a stock ticker (e.g. NVDA, AMD)."""
    try:
        items = yf.Ticker(ticker).news or []
    except Exception as exc:                      # network, bad ticker, rate limit
        return f"Could not fetch news for {ticker}: {exc}"

    if not items:
        return f"No recent Yahoo Finance news for {ticker}."

    lines = []
    for item in items[:5]:
        # Current yfinance nests the fields under "content"; fall back to the
        # older flat shape so this keeps working across versions.
        content = item.get("content", item)

        title = content.get("title") or "(no title)"

        provider = content.get("provider")
        publisher = (
            provider.get("displayName") if isinstance(provider, dict)
            else provider or item.get("publisher") or "unknown source"
        )

        canonical = content.get("canonicalUrl")
        url = (
            canonical.get("url") if isinstance(canonical, dict)
            else canonical or item.get("link") or ""
        )

        published = content.get("pubDate") or ""

        lines.append(f"- {title} ({publisher}{', ' + published if published else ''})")
        if url:
            lines.append(f"  {url}")

    return "\n".join(lines)


@tool
def search_wikipedia(query: str) -> str:
    """Look up background information on a company, person, or concept on Wikipedia."""
    try:
        titles = wikipedia.search(query, results=3)
    except Exception as exc:
        return f"Wikipedia search failed: {exc}"

    if not titles:
        return f"No Wikipedia article found for {query!r}."

    summaries = []
    for title in titles:
        try:
            page = wikipedia.page(title, auto_suggest=False)
        except wikipedia.DisambiguationError as exc:
            summaries.append(
                f"Page: {title}\n(ambiguous - options include "
                f"{', '.join(exc.options[:5])})"
            )
            continue
        except Exception:
            continue          # redirect or missing page - just skip it
        summaries.append(f"Page: {page.title}\n{page.summary[:1000]}")

    if not summaries:
        return f"No readable Wikipedia article for {query!r}."
    return "\n\n".join(summaries)


@tool
def search_news(query: str) -> str:
    """Search the last 24 hours of Google News for a query. Returns headlines with URLs."""
    api_key = os.environ.get("SERPAPI_API_KEY")
    if not api_key:
        return "News search unavailable: SERPAPI_API_KEY is not set."

    try:
        response = requests.get(
            config["SERPAPI_ENDPOINT"],
            params={
                "q": query,
                "engine": "google",
                "tbm": "nws",       # Google News vertical
                "tbs": "qdr:d",     # past 24 hours
                "api_key": api_key,
            },
            timeout=20,             # never let a tool hang the agent loop
        )
        response.raise_for_status()
        results = response.json().get("news_results", [])
    except requests.RequestException as exc:
        return f"News search failed: {exc}"
    except ValueError:
        return "News search failed: SerpAPI returned a non-JSON response."

    if not results:
        return f"No news found for {query!r} in the last 24 hours."

    lines = []
    for item in results[:5]:
        source = item.get("source")
        if isinstance(source, dict):
            source = source.get("name")
        lines.append(f"- {item.get('title') or '(no title)'} ({source or 'unknown source'})")
        if item.get("date"):
            lines.append(f"  {item['date']}")
        if item.get("link"):
            lines.append(f"  {item['link']}")

    return "\n".join(lines)

SYSTEM_PROMPT = """
You are FinBot, an expert equity research analyst with deep knowledge of financial markets,
valuation methodologies, and macroeconomic trends.

## Task
Given a stock ticker or company name, produce a concise, structured analyst brief that helps users evaluate the investment. Do not give buy/sell advice. Present data-driven signals only.

## Rules
1. Gather data before analysis. Never rely on memory for numbers.
2. If a tool fails or returns empty data, state it and proceed.
3. Never fabricate prices, ratios, or news.
4. Always follow the output format.
5. Flag notable risks or red flags.

## Output Format

**[TICKER] — Analyst Brief**
🗓️ Data As Of: [DATE] | Analysis Date: [DATE]
📊 Fundamentals: price, P/E, market cap, revenue growth (one line)
📈 Valuation Signal: OVERVALUED / FAIRLY VALUED / UNDERVALUED + reason
📰 News Sentiment: bullish / neutral / bearish + key headline
⚠️ Key Risks: 1–2 bullets
🧭 Outlook: 1–2 sentence synthesis, no advice
"""
def create_agent_with_tools(llm):

    agent = create_react_agent(
    model=llm,
    tools=[
        get_stock_fundamentals,
        get_stock_news,
        search_wikipedia,
        search_news,
    ],
    prompt=SYSTEM_PROMPT,
    checkpointer=InMemorySaver()
)
    return agent



if __name__ == "__main__":
    llm = get_llm()
    my_finance_agent = create_agent_with_tools(llm)
    config = {"configurable": {"thread_id": "user-alice-session-1"}}
    resp_1 = my_finance_agent.invoke(
    {"messages": [{"role": "user", "content": "Give me a quick brief on NVDA"}]},
    config
)

    print(resp_1["messages"][-1].content)
    