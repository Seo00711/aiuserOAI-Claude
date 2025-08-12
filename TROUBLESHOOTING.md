# GPT5Assistant Troubleshooting Guide

## Common Loading Issues

### 1. Cog Failed to Load

**Error**: "Failed to load cog gpt5assistant"

**Solutions**:
1. **Check Python Version**:
   - Requires Python 3.11+
   - Check with: `python --version`

2. **Install Dependencies**:
   ```bash
   [p]pipinstall openai>=2.0.0
   [p]pipinstall httpx
   [p]pipinstall tenacity
   [p]pipinstall aiofiles
   [p]pipinstall tiktoken
   ```

3. **Reload the Repository**:
   ```bash
   [p]repo remove aiuser-claude
   [p]repo add aiuser-claude https://github.com/mjnhbg3/aiuserOAI-Claude
   [p]cog install aiuser-claude gpt5assistant
   [p]load gpt5assistant
   ```

### 2. Import Errors

**Error**: "No module named 'openai'" or similar

**Solution**: Install missing dependencies manually:
```bash
[p]pipinstall openai httpx tenacity aiofiles tiktoken
```

### 3. API Key Issues

**Error**: "OpenAI client not initialized"

**Solution**:
```bash
[p]set api openai api_key,sk-your-actual-api-key-here
[p]restart
[p]load gpt5assistant
```

### 4. Permission Errors

**Error**: Commands not working

**Solutions**:
1. Enable the cog:
   ```bash
   [p]gpt5 config enable true
   ```

2. Check channel permissions:
   ```bash
   [p]gpt5 config channels add #your-channel
   ```

3. Check user opt-in (if required):
   ```bash
   [p]gpt5 config require_opt_in false
   ```

## Debug Steps

### 1. Check Cog Status
```bash
[p]cogs
[p]coginfo gpt5assistant
[p]gpt5 status
```

### 2. Check Logs
Look in your Red-DiscordBot logs for detailed error messages:
- Search for "gpt5assistant" or "GPT5Assistant"
- Look for import errors, API errors, or permission issues

### 3. Test Basic Functionality
```bash
[p]gpt5 config show
[p]gpt5ask Hello, are you working?
```

### 4. Test Individual Features
```bash
# Test variables
[p]gpt5 variables test Hello {username}\!

# Test image analysis (attach an image)
[p]gpt5 image analyze

# Test batch processing (attach files)
[p]gpt5 batch upload
```

## Known Issues

### 1. Red-DiscordBot Version
- Requires Red-DiscordBot 3.5.0 or higher
- Some features may not work on older versions

### 2. OpenAI API Limits
- Requires OpenAI API key with sufficient credits
- GPT-5 access may be limited based on your account tier

### 3. Large File Processing
- Files over 100MB may cause issues
- Batch processing is limited to 50 files per operation

## Getting Help

1. **Check Red-DiscordBot Logs**: Most issues will show detailed error messages in logs
2. **Verify Dependencies**: Ensure all required packages are installed
3. **Test API Key**: Verify your OpenAI API key is valid and has credits
4. **Check Permissions**: Ensure the bot has necessary Discord permissions

## Manual Installation (Alternative)

If repository installation fails, you can manually install:

1. Download the repository files
2. Place in your Red-DiscordBot cogs folder
3. Install dependencies manually
4. Load the cog

```bash
# Navigate to cogs folder
cd /path/to/redbot/data/cogs

# Clone repository
git clone https://github.com/mjnhbg3/aiuserOAI-Claude.git gpt5assistant

# Install dependencies
pip install openai>=2.0.0 httpx tenacity aiofiles tiktoken

# Load in Red-DiscordBot
[p]load gpt5assistant
```

## Support

- Repository: https://github.com/mjnhbg3/aiuserOAI-Claude
- Issues: https://github.com/mjnhbg3/aiuserOAI-Claude/issues
- Red-DiscordBot Documentation: https://docs.discord.red/
EOF < /dev/null