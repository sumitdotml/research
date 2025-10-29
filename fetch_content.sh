#!/bin/bash

# Check if URL argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <url> [output_file]"
    echo "Example: $0 https://example.com output.html"
    exit 1
fi

URL="$1"
OUTPUT_FILE="$2"

# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "Error: .env file not found"
    exit 1
fi

# Verify variables are loaded
if [ -z "$ACCOUNT_ID" ] || [ -z "$CLOUDFLARE_EMAIL" ] || [ -z "$CLOUDFLARE_API_KEY" ]; then
    echo "Error: Required environment variables are not set"
    exit 1
fi

# Run the curl command
RESPONSE=$(curl -s "https://api.cloudflare.com/client/v4/accounts/$ACCOUNT_ID/browser-rendering/content" \
    -X POST \
    -H 'Content-Type: application/json' \
    -H "X-Auth-Email: $CLOUDFLARE_EMAIL" \
    -H "X-Auth-Key: $CLOUDFLARE_API_KEY" \
    -d "{\"url\":\"$URL\"}")

# Check if response is successful
if echo "$RESPONSE" | grep -q '"success":true'; then
    # Extract content from response using Python
    CONTENT=$(echo "$RESPONSE" | python3 -c "import sys, json; data = json.load(sys.stdin); print(data.get('result', ''))")

    if [ -n "$OUTPUT_FILE" ]; then
        echo "$CONTENT" > "$OUTPUT_FILE"
        echo "Content saved to: $OUTPUT_FILE"
    else
        echo "$CONTENT"
    fi
else
    echo "Error: Request failed"
    echo "$RESPONSE"
fi
