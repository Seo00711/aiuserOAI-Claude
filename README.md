# GPT5Assistant - Red-DiscordBot Cog

A production-ready AI assistant cog for Red-DiscordBot that replaces aiuser with modern GPT-5 capabilities.

## 🚀 Features

### Core AI Capabilities
- **GPT-5 Integration**: Latest OpenAI models with reasoning and advanced text capabilities
- **Voice Transcription**: OpenAI Whisper integration for voice message processing
- **Vision Analysis**: Comprehensive image analysis, comparison, and generation
- **Batch Processing**: Upload and analyze multiple files simultaneously
- **Dynamic Variables**: Full aiuser variable compatibility (`{botname}`, `{username}`, etc.)

### Advanced Features
- **Smart Conversation Management**: Token-aware history with configurable timeframes
- **Multi-Modal Processing**: Text, voice, images, and documents in one interface
- **File Type Support**: PDF, Word, Excel, PowerPoint, images, code, archives, videos
- **Automatic Summarization**: AI-powered document analysis and key point extraction
- **Per-Guild Configuration**: Granular control over features and permissions

### aiuser Compatibility
- **Random Responses**: Configurable response percentage for ambient interaction
- **Channel Controls**: Allow/deny lists and per-channel overrides
- **User Opt-in/Opt-out**: Flexible permission system
- **Context Awareness**: Maintains conversation context like aiuser
- **Variable System**: Drop-in replacement for aiuser variables

## 📦 Installation

### Prerequisites
- Red-DiscordBot 3.5.0+
- Python 3.11+
- OpenAI API key

### Install from GitHub
```bash
[p]repo add aiuser-claude https://github.com/mjnhbg3/aiuserOAI-Claude
[p]cog install aiuser-claude gpt5assistant
[p]load gpt5assistant
```

### Setup
1. **Set OpenAI API Key**:
   ```bash
   [p]set api openai api_key,sk-your-openai-api-key-here
   ```

2. **Enable the Cog**:
   ```bash
   [p]gpt5 config enable true
   ```

3. **Configure Channels** (optional):
   ```bash
   [p]gpt5 config channels add #general #ai-chat
   ```

## 🎯 Usage

### Basic Commands
- `[p]gpt5` - Show help and available commands
- `[p]gpt5 ask <question>` - Ask the AI a question
- `[p]gpt5 status` - Show bot status and statistics
- `@BotName <message>` - Mention the bot to get a response

### Voice Features
- **Voice Messages**: Attach voice messages - they'll be automatically transcribed
- **Transcription Settings**: Configure per-guild voice processing

### Image Features
- `[p]gpt5 image analyze` - Analyze attached images
- `[p]gpt5 image compare` - Compare two attached images
- **Auto-Analysis**: Images in conversations are automatically analyzed for context

### File Processing
- `[p]gpt5 batch upload` - Process multiple files with analysis
- `[p]gpt5 batch info` - Show supported file types and limits
- **Supported Formats**: PDF, Word, Excel, PowerPoint, images, code, archives

### Variables
- `[p]gpt5 variables list` - Show all available variables
- `[p]gpt5 variables test <text>` - Test variable substitution
- **Available Variables**: `{botname}`, `{username}`, `{servername}`, `{date}`, `{time}`, etc.

### Conversation Management
- `[p]gpt5 forget` - Clear conversation history for current channel
- `[p]gpt5 forgetall` - Clear all conversation histories (admin only)
- `[p]gpt5 config conversation_timeframe <seconds>` - Set memory duration

## ⚙️ Configuration

### Basic Settings
```bash
[p]gpt5 config show                    # View current configuration
[p]gpt5 config enable <true/false>     # Enable/disable the cog
[p]gpt5 config model <model_name>      # Set GPT model
[p]gpt5 config system_prompt <prompt>  # Set custom system prompt
```

### aiuser-like Features
```bash
[p]gpt5 config response_percentage <0-100>  # Random response chance
[p]gpt5 config require_opt_in <true/false>  # Require user opt-in
[p]gpt5 config random_messages <true/false> # Enable random messages
```

### Advanced Settings
```bash
[p]gpt5 config conversation_timeframe <seconds>  # Memory duration
[p]gpt5 config token_limit <number>             # Context token limit
[p]gpt5 config tools <tool_name> <true/false>   # Enable/disable tools
```

## 🔧 Tools & Capabilities

### Built-in Tools
- **Web Search**: Real-time web search capabilities
- **File Search**: Knowledge base search through uploaded files
- **Code Interpreter**: Execute and analyze code
- **Image Generation**: Create and edit images with gpt-image-1

### File Support
- **Documents**: PDF, Word (.doc, .docx), RTF, ODT, Pages, EPUB
- **Spreadsheets**: Excel (.xls, .xlsx), CSV, TSV, ODS
- **Presentations**: PowerPoint (.ppt, .pptx), ODP
- **Images**: PNG, JPEG, GIF, WebP, BMP
- **Code**: Python, JavaScript, Java, C++, and 20+ languages
- **Archives**: ZIP, TAR, RAR, 7Z
- **Audio**: MP3, WAV, M4A, OGG, WebM, FLAC

## 🚨 Migration from aiuser

This cog is designed as a drop-in replacement for aiuser:

1. **Backup your aiuser settings**
2. **Unload aiuser**: `[p]unload aiuser`
3. **Install GPT5Assistant** (see installation above)
4. **Configure similar settings** using the config commands
5. **Test functionality** in a private channel first

### Key Differences
- **Enhanced AI**: GPT-5 vs older models
- **Voice Support**: Built-in voice transcription
- **Vision Capabilities**: Full image analysis
- **Better File Handling**: More formats and batch processing
- **Improved Context**: Token-aware conversation management

## 📋 Requirements

### System Requirements
- Python 3.11+
- Red-DiscordBot 3.5.0+
- 2GB+ RAM recommended for large file processing

### Python Dependencies
```
openai>=2.0.0
httpx
tenacity
aiofiles
tiktoken
```

### API Requirements
- OpenAI API key with GPT-5 access
- Sufficient OpenAI credits for usage

## 🔒 Privacy & Security

- **Local Processing**: No data sent to third parties except OpenAI
- **Temporary Storage**: Conversation history stored temporarily for context
- **No Permanent Data**: Files and transcriptions processed temporarily
- **Configurable**: All features can be disabled per-guild
- **Opt-in System**: Users can control their participation

## 🆘 Support

### Common Issues
1. **"OpenAI client not initialized"**: Set your API key with `[p]set api openai api_key,<key>`
2. **Commands not working**: Ensure the cog is loaded with `[p]load gpt5assistant`
3. **No responses**: Check if the cog is enabled with `[p]gpt5 config show`

### Getting Help
- Use `[p]gpt5 status` to check system status
- Check Red-DiscordBot logs for detailed error messages
- Ensure your OpenAI API key has sufficient credits and permissions

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Built for the Red-DiscordBot ecosystem
- Inspired by the original aiuser cog
- Powered by OpenAI's GPT-5 and Whisper APIs

---

**Note**: This cog requires an OpenAI API key and will make API calls that consume OpenAI credits. Please monitor your usage and set appropriate limits.
EOF < /dev/null