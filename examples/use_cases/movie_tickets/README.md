# Movie Ticket Search Example with AMC Theaters

This example demonstrates using Neo's agentic framework with Playwright MCP to automate movie ticket searches while bypassing bot protection.

## Quick Start

### 1. Replace Cookies (REQUIRED)

The `amc_cookies.json` file contains placeholder values. You **MUST** replace it with fresh cookies from your browser:

#### Using Browser Extension (Easiest)
1. Install "EditThisCookie" (Chrome) or "Cookie-Editor" (Firefox)
2. Visit https://www.amctheatres.com
3. Click the extension icon → Export
4. Copy the JSON
5. **Paste into `amc_cookies.json`** (overwrite the entire file)

#### Using DevTools
1. Visit https://www.amctheatres.com
2. Press F12 → Application/Storage → Cookies
3. Manually export cookies for `.amctheatres.com`
4. Format as JSON array and save to `amc_cookies.json`

### 2. Run the Notebook

```bash
cd src/neo/examples/use_cases/movie_tickets
jupyter notebook movie_tickets.ipynb
```

### 3. Critical Cookies

Make sure your `amc_cookies.json` includes:
- `__cf_bm` - Cloudflare bot management (expires in ~30 min)
- `osano_consentmanager_uuid`
- `osano_consentmanager`
- `connect.sid` - Session ID
- `_ga`, `_gcl_au` - Analytics

## How It Works

The notebook uses:
1. **Neo Agentic Framework** - Manages task execution with AI models
2. **Playwright MCP Server** - Provides browser automation tools via `npx @playwright/mcp@latest`
3. **Cookie Management**:
   - Cookies are loaded from `amc_cookies.json`
   - Passed to MCP server via `PLAYWRIGHT_COOKIES` environment variable
   - AI agent loads them using `browser_run_code` before navigating
4. **Anti-Bot Techniques**:
   - Pre-loading session cookies (especially Cloudflare's `__cf_bm`)
   - Random delays (3-6 seconds between actions)
   - Human-like mouse movements
   - Direct navigation to theater pages
   - Proper wait strategies for page loads

### Cookie Flow
```python
# 1. Load cookies from JSON
with open("amc_cookies.json") as f:
    cookies = json.load(f)

# 2. Pass to MCP client via environment
playwright_client = MCPClient(
    command="npx",
    args=["-y", "@playwright/mcp@latest"],
    env={"PLAYWRIGHT_COOKIES": json.dumps(cookies)}
)

# 3. AI agent loads them in browser
# (Automatically done via browser_run_code in the task)
```

## Troubleshooting

### Redirected to `about:blank`
- Your cookies expired (especially `__cf_bm`)
- Get fresh cookies and try again

### Cloudflare Challenge Appears
- Increase delays in the notebook
- Add more mouse movements
- Ensure cookies are less than 30 minutes old

### No Showtimes Found
- Check theater slug in URL is correct
- Verify the date parameter
- Ensure you're navigating to the direct theater showtime page

## Files

- `movie_tickets.ipynb` - Main notebook
- `amc_cookies.json` - Cookie storage (replace with your cookies!)
- `README.md` - This file

## Based On

This implementation uses techniques from the working AMC crawler at:
`src/discord-bot/discord_bot/amc_crawler/acrawler.py`
