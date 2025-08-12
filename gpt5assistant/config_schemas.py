from typing import Dict, List, Optional, Literal, Any
from pydantic import BaseModel, Field


class ToolConfig(BaseModel):
    web_search: bool = True
    file_search: bool = True
    code_interpreter: bool = True
    image: bool = True
    voice_transcription: bool = True


class ReasoningConfig(BaseModel):
    effort: Literal["minimal", "medium", "high"] = "medium"


class TextConfig(BaseModel):
    verbosity: Literal["low", "medium", "high"] = "medium"


class ModelConfig(BaseModel):
    name: str = "gpt-5"
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)
    text: TextConfig = Field(default_factory=TextConfig)


class ChannelConfig(BaseModel):
    enabled: bool = True
    tools: Optional[ToolConfig] = None
    model: Optional[ModelConfig] = None
    system_prompt: Optional[str] = None
    response_percentage: Optional[float] = None  # Override guild response percentage
    random_messages: Optional[bool] = None       # Override guild random message setting


class GuildConfig(BaseModel):
    enabled: bool = True
    model: ModelConfig = Field(default_factory=ModelConfig)
    tools: ToolConfig = Field(default_factory=ToolConfig)
    system_prompt: str = "You are a helpful AI assistant in a Discord server."
    allowed_channels: List[int] = Field(default_factory=list)
    denied_channels: List[int] = Field(default_factory=list)
    channel_overrides: Dict[int, ChannelConfig] = Field(default_factory=dict)
    ephemeral_responses: bool = False
    max_message_history: int = 10
    cooldown_seconds: int = 0
    file_search_kb_id: Optional[str] = None
    
    # New aiuser-like features
    response_percentage: float = 0.0         # 0-100, percentage of messages to respond to
    opted_in_users: List[int] = Field(default_factory=list)  # Users who opted in
    require_opt_in: bool = True             # Require users to opt-in
    random_messages: bool = False           # Send random messages when idle
    random_message_cooldown: int = 1980     # 33 minutes in seconds
    idle_timeout: int = 3600                # 1 hour before considering channel idle
    random_message_topics: List[str] = Field(default_factory=lambda: [
        "Ask about everyone's day", "Share an interesting fact", "Start a conversation"
    ])
    
    # Conversation management
    conversation_timeframe: int = 3600  # seconds (1 hour)
    token_limit: int = 8000            # max tokens for conversation context
    auto_forget: bool = False          # automatically forget old conversations


class GlobalConfig(BaseModel):
    openai_api_key: Optional[str] = None
    default_guild_config: GuildConfig = Field(default_factory=GuildConfig)


GUILD_CONFIG_SCHEMA = {
    "enabled": True,
    "model": {
        "name": "gpt-5",
        "max_tokens": None,
        "temperature": 0.7,
        "reasoning": {"effort": "medium"},
        "text": {"verbosity": "medium"}
    },
    "tools": {
        "web_search": True,
        "file_search": True,
        "code_interpreter": True,
        "image": True,
        "voice_transcription": True
    },
    "system_prompt": "You are a helpful AI assistant in a Discord server.",
    "allowed_channels": [],
    "denied_channels": [],
    "channel_overrides": {},
    "ephemeral_responses": False,
    "max_message_history": 10,
    "cooldown_seconds": 0,
    "file_search_kb_id": None,
    "response_percentage": 0.0,
    "opted_in_users": [],
    "require_opt_in": True,
    "random_messages": False,
    "random_message_cooldown": 1980,
    "idle_timeout": 3600,
    "random_message_topics": [
        "Ask about everyone's day", "Share an interesting fact", "Start a conversation"
    ],
    "conversation_timeframe": 3600,
    "token_limit": 8000,
    "auto_forget": False
}

CHANNEL_CONFIG_SCHEMA = {
    "enabled": True,
    "tools": None,
    "model": None,
    "system_prompt": None,
    "response_percentage": None,
    "random_messages": None
}