# Migration Guide: aiuser → gpt5assistant

This guide helps server administrators migrate from the bz-cogs/aiuser cog to the new gpt5assistant cog.

## 🔄 Why Migrate?

### Key Improvements in gpt5assistant
- **GPT-5 Access**: Latest OpenAI models with enhanced capabilities
- **Native Tools**: Built-in web search, file search, code interpreter  
- **Responses API**: Stateful conversations, better tool coordination
- **Enhanced UX**: Improved streaming, chunking, error handling
- **Better Performance**: Optimized for Discord, async throughout

### What Changes
- Custom tool integrations → Native OpenAI tools
- ChatCompletions API → Responses API  
- Limited models → GPT-5 model family
- Basic error handling → Comprehensive error management

## 📋 Pre-Migration Checklist

### 1. Document Current Settings
Run these commands in your server to document current aiuser configuration:

```bash
[p]aiuser settings              # General settings
[p]aiuser model                # Current model
[p]aiuser channels list        # Channel restrictions  
[p]aiuser prompt show          # System prompt
[p]aiuser tools list           # Enabled tools
```

### 2. Note Custom Configurations
- Custom system prompts
- Channel allow/deny lists
- Tool preferences
- Model settings
- Temperature/creativity settings

### 3. Backup Configuration
Consider screenshotting or copying your current settings before migration.

## 🚀 Migration Steps

### Step 1: Install gpt5assistant

```bash
# Add the repository
[p]repo add red-gpt5assistant https://github.com/your-username/red-gpt5assistant

# Install the cog
[p]cog install red-gpt5assistant gpt5assistant

# Load the cog
[p]load gpt5assistant
```

### Step 2: Configure API Access

```bash
# Set OpenAI API key (same as aiuser)
[p]set api openai api_key,sk-your-api-key-here
```

**Important**: Ensure your API key has access to GPT-5 models.

### Step 3: Basic Configuration

```bash
# Enable the assistant
[p]gpt5 config enable

# Set preferred model (upgrade from GPT-4)
[p]gpt5 config model gpt-5-mini  # or gpt-5 for best performance

# Configure verbosity (new feature)
[p]gpt5 config verbosity medium

# Set reasoning effort (new feature) 
[p]gpt5 config reasoning medium
```

### Step 4: Migrate Settings

#### System Prompt
```bash
# aiuser
[p]aiuser prompt set Your custom prompt here

# gpt5assistant
[p]gpt5 config system Your custom prompt here
```

#### Temperature
```bash
# aiuser
[p]aiuser temperature 0.8

# gpt5assistant  
[p]gpt5 config temperature 0.8
```

#### Channel Management
```bash
# aiuser
[p]aiuser channels allow #general #chat
[p]aiuser channels deny #admin

# gpt5assistant
[p]gpt5 config channels allow #general
[p]gpt5 config channels allow #chat  
[p]gpt5 config channels deny #admin
```

#### Tool Configuration
```bash
# Enable all tools (replaces aiuser's custom tools)
[p]gpt5 config tools enable web_search
[p]gpt5 config tools enable file_search
[p]gpt5 config tools enable code_interpreter
[p]gpt5 config tools enable image
```

### Step 5: Test Functionality

Test each feature to ensure it works as expected:

```bash
# Test basic chat
@YourBot Hello, how are you?

# Test web search (replaces aiuser search function)
@YourBot What's the latest news about AI?

# Test code execution (enhanced from aiuser)
@YourBot Calculate fibonacci(20) and show the results

# Test image generation (new feature)
/gpt5 image A beautiful sunset over mountains

# Test file upload (new feature)  
/gpt5 upload [attach a PDF]
@YourBot Summarize this document
```

### Step 6: Remove aiuser (Optional)

Once satisfied with gpt5assistant:

```bash
[p]unload aiuser
[p]cog uninstall bz-cogs aiuser
```

## 📊 Setting Mapping Reference

| aiuser Command | gpt5assistant Command | Notes |
|----------------|----------------------|-------|
| `[p]aiuser enable` | `[p]gpt5 config enable` | Same functionality |
| `[p]aiuser disable` | `[p]gpt5 config disable` | Same functionality |
| `[p]aiuser model gpt-4` | `[p]gpt5 config model gpt-5` | Upgraded models |
| `[p]aiuser temperature 0.8` | `[p]gpt5 config temperature 0.8` | Same range |
| `[p]aiuser prompt set <text>` | `[p]gpt5 config system <text>` | Same functionality |
| `[p]aiuser channels allow #ch` | `[p]gpt5 config channels allow #ch` | Same functionality |
| `[p]aiuser channels deny #ch` | `[p]gpt5 config channels deny #ch` | Same functionality |
| `[p]aiuser tools enable search` | `[p]gpt5 config tools enable web_search` | Native OpenAI search |
| `[p]aiuser settings` | `[p]gpt5 config show` | Enhanced display |

## 🆕 New Features Available

### Advanced Model Controls
```bash
# Control reasoning depth (new)
[p]gpt5 config reasoning high

# Control response length (new)
[p]gpt5 config verbosity low
```

### Enhanced Tools
```bash
# File search with documents (new)
/gpt5 upload [files...]

# Advanced code interpreter (enhanced)
@YourBot Create a data visualization showing...

# Native web search (improved)
@YourBot Search for recent developments in...
```

### Better Commands
```bash
# Modern slash commands (new)
/gpt5 ask What is quantum computing?
/gpt5 image A logo for my startup
/gpt5 upload [documents for analysis]

# Enhanced status (new)
[p]gpt5 status
```

## ⚠️ Important Differences

### Tool Integration
- **aiuser**: Custom search/scrape functions
- **gpt5assistant**: Native OpenAI tools (more reliable, current data)

### API Usage
- **aiuser**: ChatCompletions API
- **gpt5assistant**: Responses API (stateful, better tool coordination)

### Cost Considerations
- GPT-5 models have different pricing than GPT-4
- New tools have usage costs (file search, code interpreter)
- Web search included with model usage

### Feature Parity
Some aiuser features may not have direct equivalents:
- Custom function definitions → Use native tools instead
- Specific third-party integrations → Replaced with OpenAI native tools
- Legacy model support → Only GPT-5 family supported

## 🐛 Common Migration Issues

### "API key invalid"
- Ensure your OpenAI key has GPT-5 access
- Some older keys may need upgrading

### "Model not available"
- Start with `gpt-5-mini` if `gpt-5` isn't available yet
- Check OpenAI model availability in your region

### "Higher costs than expected"
- GPT-5 has different pricing than GPT-4
- Monitor usage with `[p]gpt5 status`
- Consider `gpt-5-mini` for cost optimization

### "Missing custom functions"
- Use native tools instead of custom functions
- File search replaces document analysis
- Web search replaces custom scraping

## 💡 Optimization Tips

### Cost Optimization
```bash
# Use cheaper model for simple queries
[p]gpt5 config model gpt-5-mini

# Reduce verbosity for shorter responses
[p]gpt5 config verbosity low

# Lower reasoning for faster, cheaper responses
[p]gpt5 config reasoning minimal
```

### Performance Optimization
```bash
# Limit message history for faster context
[p]gpt5 config show  # Check max_message_history

# Use channel restrictions to control usage
[p]gpt5 config channels allow #ai-chat
```

## 📞 Getting Help

### Configuration Issues
1. Run `[p]gpt5 config show` to verify settings
2. Check `[p]gpt5 status` for API connectivity
3. Review Red logs for detailed errors

### Feature Questions  
1. Check the README.md for full feature documentation
2. Compare with aiuser documentation for equivalent features
3. Open GitHub issues for missing functionality

### Support Resources
- [GitHub Issues](https://github.com/your-username/red-gpt5assistant/issues)
- [Red-DiscordBot Discord](https://discord.gg/red)
- [OpenAI Documentation](https://platform.openai.com/docs)

## 🎯 Migration Success Checklist

- [ ] gpt5assistant installed and loaded
- [ ] OpenAI API key configured
- [ ] Assistant enabled in target server
- [ ] System prompt migrated
- [ ] Channel restrictions configured
- [ ] Tools enabled as desired
- [ ] Model and parameters set
- [ ] Basic functionality tested
- [ ] Advanced features explored
- [ ] aiuser unloaded (if desired)
- [ ] Users informed of new features
- [ ] Monitor costs and usage

Congratulations! You've successfully migrated to gpt5assistant. Enjoy the enhanced AI capabilities! 🎉