# Cloudflare Browser Rendering Scripts

Bash scripts to fetch web content using Cloudflare's Browser Rendering API. Includes both markdown conversion and raw HTML content fetching.

## Setup

### 1. Install Dependencies

The script requires:
- `curl` (usually pre-installed)
- `python3` (for JSON parsing)

### 2. Configure Environment Variables

Create a `.env` file in the project directory with your Cloudflare credentials:

```bash
ACCOUNT_ID=your_account_id_here
CLOUDFLARE_EMAIL=your_email@example.com
CLOUDFLARE_API_KEY=your_api_key_here
```

**Security Note:** The `.env` file is already included in `.gitignore` to prevent accidental commits.

### 3. Make Scripts Executable

```bash
chmod +x fetch_markdown.sh fetch_content.sh
```

## Available Scripts

### 1. Fetch Markdown (`fetch_markdown.sh`)

Fetches a webpage and converts it to clean markdown format.

**Usage:**
```bash
./fetch_markdown.sh <url> <output_file>
```

**Examples:**
```bash
# Fetch a webpage and save to markdown
./fetch_markdown.sh https://example.com output.md

# Fetch an article
./fetch_markdown.sh https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html induction-heads.md
```

### 2. Fetch Content (`fetch_content.sh`)

Fetches the raw HTML content of a webpage.

**Usage:**
```bash
./fetch_content.sh <url> [output_file]
```

**Examples:**
```bash
# Fetch HTML content and save to file
./fetch_content.sh https://example.com output.html

# Print HTML to stdout
./fetch_content.sh https://example.com
```

## How It Works

Both scripts follow the same workflow:
1. Load credentials from `.env` file
2. Make a POST request to Cloudflare's Browser Rendering API
3. Extract the content from the JSON response
4. Save the result to the specified output file (or print to stdout)

## API Reference

### Markdown API
- **Endpoint:** `https://api.cloudflare.com/client/v4/accounts/{account_id}/browser-rendering/markdown`
- **Method:** POST
- **Body:** `{"url": "https://example.com"}`
- **Returns:** Cleaned markdown text

### Content API
- **Endpoint:** `https://api.cloudflare.com/client/v4/accounts/{account_id}/browser-rendering/content`
- **Method:** POST
- **Body:** `{"url": "https://example.com"}`
- **Returns:** Raw HTML content

## Troubleshooting

### "Error: .env file not found"
Make sure you've created a `.env` file in the same directory as the script.

### "Error: Required environment variables are not set"
Check that your `.env` file contains all three required variables: `ACCOUNT_ID`, `CLOUDFLARE_EMAIL`, and `CLOUDFLARE_API_KEY`.

### Empty or small output files
Verify that your Cloudflare API credentials are correct and that your account has access to the Browser Rendering API.

## Security Best Practices

- Never commit the `.env` file to version control
- Keep your API key secure and rotate it regularly
- Consider using Cloudflare API tokens instead of API keys for better security
- Set restrictive permissions on `.env`: `chmod 600 .env`
