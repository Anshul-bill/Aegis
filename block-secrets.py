import sys
import re
import os

def scan_content(content):
    # Regex for potential secrets
    patterns = [
        r"(?i)api_key\s*=\s*['\"][a-zA-Z0-9]{32,128}['\"]",  # Generic API keys
        r"(?i)secret_key\s*=\s*['\"][a-zA-Z0-9]{32,128}['\"]", # Generic Secret keys
        r"(?i)bearer\s+[a-zA-Z0-9.-_]{32,}",                # Bearer tokens
        r"(?i)https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?::\d+)?/api/v1/", # Unauthorized API endpoints (placeholder logic)
        r"(?i)password\s*=\s*['\"][^'\"]{8,}['\"]",          # Hardcoded passwords
    ]
    
    for pattern in patterns:
        if re.search(pattern, content):
            return True, pattern
    return False, None

def main():
    # The BeforeTool hook passes the tool input as context if needed, 
    # but for write_file/replace, we usually want to scan the 'content' or 'new_string' fields.
    # In Gemini CLI, hooks receive information via environment variables or stdin.
    # According to docs, the payload is often available.
    
    # For now, let's assume we scan the files that were just written if the hook is post-write, 
    # or scan the input if it's pre-write. 
    # The blueprint says "intercept and validate agent actions prior to execution".
    
    # If this is a BeforeTool hook, we can check the command line arguments or the environment.
    # Gemini CLI BeforeTool hooks receive the tool name and its arguments.
    
    # We will look for 'content' or 'new_string' in the process arguments or environment.
    # Actually, simpler: check the files being modified in the current directory if they are about to be written.
    
    # For a robust implementation, we'd parse the JSON payload from GEMINI_HOOK_PAYLOAD.
    payload = os.environ.get("GEMINI_HOOK_PAYLOAD")
    if not payload:
        # Fallback for manual testing or misconfiguration
        sys.exit(0)
        
    # Check for secrets in the payload string
    found, pattern = scan_content(payload)
    if found:
        print(f"SECURITY VIOLATION: Hardcoded secret or unauthorized endpoint detected matching pattern: {pattern}", file=sys.stderr)
        sys.exit(2) # System Block

    sys.exit(0)

if __name__ == "__main__":
    main()
