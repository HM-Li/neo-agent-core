# Fetch News from Hacker News

A simple example demonstrating Neo's agentic framework with Playwright MCP to fetch and summarize news from Hacker News.

## Quick Start

```bash
cd src/neo/examples/use_cases/fetch_news
jupyter notebook fetch_news.ipynb
```

Run all cells in order. The notebook will:
1. Initialize Playwright MCP client
2. Navigate to Hacker News
3. Extract top 10 news items
4. Present results in natural language

## What It Does

This example shows the basics of Neo + MCP integration:
- **Simple web navigation** - No cookies or authentication needed
- **Content extraction** - Uses `browser_snapshot` to capture page content
- **Natural language output** - AI formats results in readable prose

## How It Works

```python
# 1. Initialize MCP client (simple!)
playwright_client = MCPClient(
    name="playwright",
    command="npx",
    args=["-y", "@playwright/mcp@latest"]
)

# 2. Create Neo task
task = ModelTask(
    user_input="Navigate to HN and extract top 10 news...",
    instruction=Instruction(
        model_configs=ModelConfigs(model="claude-3-7-sonnet-20250219"),
        other_configs=OtherConfigs(mcp_clients=[playwright_client])
    )
)

# 3. Run and get results
neo = Neo(tasks=task)
result = await neo.run_all()
```

## Expected Output

The AI will provide a natural language summary like:

> "Today's top stories on Hacker News include:
>
> 1. **New Programming Language Released** (485 points) - A discussion about...
> 2. **Startup Raises $50M Series B** (342 points) - Covering recent funding...
> ..."

## Comparison with movie_tickets Example

| Feature | fetch_news | movie_tickets |
|---------|------------|---------------|
| **Complexity** | Simple | Complex |
| **Cookies** | Not needed | Required |
| **Anti-bot** | Not needed | Essential |
| **Use case** | Learning Neo basics | Real-world scraping |
| **Setup time** | < 1 minute | 5-10 minutes |

**Recommendation:** Start with this example to learn Neo and MCP basics, then move to movie_tickets for advanced techniques.

## Files

- `fetch_news.ipynb` - Main notebook
- `README.md` - This file

## Requirements

- Neo framework installed
- Node.js (for Playwright MCP via npx)
- Jupyter notebook

## Troubleshooting

### MCP Connection Fails
- Ensure Node.js is installed: `node --version`
- The first run downloads Playwright MCP (may take 30 seconds)

### No Results Returned
- Check your internet connection
- Hacker News may be down (rare)
- Try running the cells again

### Output Format Issues
- The AI's natural language output may vary
- This is expected - different runs may phrase results differently
