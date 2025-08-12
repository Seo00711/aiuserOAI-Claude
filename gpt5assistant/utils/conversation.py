import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
import discord
from redbot.core import Config
import logging
import tiktoken

logger = logging.getLogger("red.gpt5assistant.conversation")


class ConversationManager:
    """Manages conversation history with token-aware truncation and timeframe limits"""
    
    def __init__(self, config: Config):
        self.config = config
        self._conversation_cache: Dict[str, List[Dict[str, Any]]] = {}  # channel_id -> messages
        self._conversation_timestamps: Dict[str, float] = {}  # channel_id -> last_activity
        self._lock = asyncio.Lock()
        
        # Token counting (approximation for GPT models)
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4/GPT-5 tokenizer
        except Exception:
            self.tokenizer = None
            logger.warning("Could not load tiktoken encoder, using character approximation")
    
    async def get_conversation_history(
        self,
        channel: discord.TextChannel,
        guild_config: Dict[str, Any],
        max_tokens: Optional[int] = None,
        max_messages: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get conversation history with token-aware truncation"""
        
        channel_key = str(channel.id)
        conversation_timeframe = guild_config.get("conversation_timeframe", 3600)
        token_limit = max_tokens or guild_config.get("token_limit", 8000)
        max_msg_limit = max_messages or guild_config.get("max_message_history", 10)
        
        async with self._lock:
            # Check if we have cached conversation
            if channel_key in self._conversation_cache:
                last_activity = self._conversation_timestamps.get(channel_key, 0)
                
                # Check if conversation has expired
                if time.time() - last_activity > conversation_timeframe:
                    logger.debug(f"Conversation expired for channel {channel.name}, clearing cache")
                    self._conversation_cache.pop(channel_key, None)
                    self._conversation_timestamps.pop(channel_key, None)
                else:
                    # Return cached conversation
                    cached_messages = self._conversation_cache[channel_key]
                    return self._truncate_by_tokens(cached_messages, token_limit, max_msg_limit)
            
            # Fetch fresh conversation from Discord
            messages = await self._fetch_discord_history(channel, max_msg_limit * 2)  # Fetch more to allow for truncation
            
            # Cache the conversation
            self._conversation_cache[channel_key] = messages
            self._conversation_timestamps[channel_key] = time.time()
            
            return self._truncate_by_tokens(messages, token_limit, max_msg_limit)
    
    async def add_message_to_history(
        self,
        channel: discord.TextChannel,
        role: str,
        content: str,
        guild_config: Dict[str, Any]
    ) -> None:
        """Add a message to the conversation history cache"""
        
        channel_key = str(channel.id)
        token_limit = guild_config.get("token_limit", 8000)
        max_msg_limit = guild_config.get("max_message_history", 10)
        
        async with self._lock:
            if channel_key not in self._conversation_cache:
                self._conversation_cache[channel_key] = []
            
            # Add new message
            self._conversation_cache[channel_key].append({
                "role": role,
                "content": content,
                "timestamp": time.time()
            })
            
            # Update activity timestamp
            self._conversation_timestamps[channel_key] = time.time()
            
            # Truncate if necessary
            self._conversation_cache[channel_key] = self._truncate_by_tokens(
                self._conversation_cache[channel_key], 
                token_limit, 
                max_msg_limit
            )
    
    async def forget_conversation(self, channel: discord.TextChannel) -> bool:
        """Clear conversation history for a channel"""
        
        channel_key = str(channel.id)
        
        async with self._lock:
            cleared = False
            if channel_key in self._conversation_cache:
                del self._conversation_cache[channel_key]
                cleared = True
            
            if channel_key in self._conversation_timestamps:
                del self._conversation_timestamps[channel_key]
                cleared = True
            
            return cleared
    
    async def forget_all_conversations(self, guild_id: int) -> int:
        """Clear all conversation histories for a guild"""
        
        async with self._lock:
            cleared_count = 0
            
            # For simplicity, clear all conversations
            # In production, you'd want to filter by guild channels
            keys_to_remove = list(self._conversation_cache.keys())
            
            for channel_key in keys_to_remove:
                del self._conversation_cache[channel_key]
                cleared_count += 1
            
            for channel_key in list(self._conversation_timestamps.keys()):
                del self._conversation_timestamps[channel_key]
            
            return cleared_count
    
    def _truncate_by_tokens(
        self, 
        messages: List[Dict[str, Any]], 
        token_limit: int, 
        max_messages: int
    ) -> List[Dict[str, Any]]:
        """Truncate messages to fit within token and message limits"""
        
        if not messages:
            return []
        
        # First apply message count limit
        if len(messages) > max_messages:
            messages = messages[-max_messages:]
        
        # Then apply token limit
        if self.tokenizer:
            # Use actual token counting
            total_tokens = 0
            result_messages = []
            
            # Process messages in reverse order (newest first)
            for message in reversed(messages):
                message_tokens = len(self.tokenizer.encode(message["content"]))
                
                if total_tokens + message_tokens <= token_limit:
                    result_messages.insert(0, message)
                    total_tokens += message_tokens
                else:
                    break
            
            return result_messages
        else:
            # Use character approximation (roughly 4 chars per token)
            total_chars = 0
            result_messages = []
            
            for message in reversed(messages):
                message_chars = len(message["content"])
                
                if total_chars + message_chars <= token_limit * 4:
                    result_messages.insert(0, message)
                    total_chars += message_chars
                else:
                    break
            
            return result_messages
    
    async def _fetch_discord_history(
        self, 
        channel: discord.TextChannel, 
        limit: int
    ) -> List[Dict[str, Any]]:
        """Fetch message history from Discord"""
        
        messages = []
        
        try:
            async for message in channel.history(limit=limit):
                if message.author.bot:
                    # Bot message
                    if message.author.id == channel.guild.me.id:
                        content = self._clean_bot_message(message.content)
                        if content:
                            messages.append({
                                "role": "assistant",
                                "content": content,
                                "timestamp": message.created_at.timestamp()
                            })
                else:
                    # User message
                    content = self._clean_user_message(message)
                    if content:
                        messages.append({
                            "role": "user", 
                            "content": f"{message.author.display_name}: {content}",
                            "timestamp": message.created_at.timestamp()
                        })
        
        except discord.Forbidden:
            logger.warning(f"Cannot read message history in {channel.name}")
        except Exception as e:
            logger.error(f"Error fetching Discord history: {e}")
        
        # Return in chronological order (oldest first)
        return list(reversed(messages))
    
    def _clean_user_message(self, message: discord.Message) -> str:
        """Clean and format user message"""
        content = message.clean_content
        
        # Handle attachments
        if message.attachments:
            attachment_info = []
            for attachment in message.attachments:
                if attachment.content_type:
                    if attachment.content_type.startswith('image/'):
                        attachment_info.append(f"[Image: {attachment.filename}]")
                    elif attachment.content_type.startswith('audio/'):
                        # Voice attachments are handled separately via transcription
                        attachment_info.append(f"[Voice message: {attachment.filename}]")
                    else:
                        attachment_info.append(f"[File: {attachment.filename}]")
                else:
                    attachment_info.append(f"[File: {attachment.filename}]")
            
            if attachment_info:
                content += " " + " ".join(attachment_info)
        
        return content.strip()
    
    def _clean_bot_message(self, content: str) -> str:
        """Clean bot message content"""
        # Remove code block indicators if they're formatting artifacts
        import re
        content = re.sub(r'```\w*\n?$', '', content)
        content = re.sub(r'^```\w*\n?', '', content)
        
        # Remove thinking/processing indicators
        content = re.sub(r'^\*.*?\*\s*', '', content, flags=re.MULTILINE)
        
        return content.strip()
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation manager statistics"""
        return {
            "active_conversations": len(self._conversation_cache),
            "total_cached_messages": sum(len(messages) for messages in self._conversation_cache.values()),
            "has_tokenizer": self.tokenizer is not None
        }
    
    async def cleanup_expired_conversations(self) -> int:
        """Clean up expired conversations (background task)"""
        
        current_time = time.time()
        expired_count = 0
        
        async with self._lock:
            expired_channels = []
            
            for channel_key, last_activity in self._conversation_timestamps.items():
                # Use a default 1 hour expiry for cleanup
                if current_time - last_activity > 3600:
                    expired_channels.append(channel_key)
            
            for channel_key in expired_channels:
                self._conversation_cache.pop(channel_key, None)
                self._conversation_timestamps.pop(channel_key, None)
                expired_count += 1
        
        if expired_count > 0:
            logger.debug(f"Cleaned up {expired_count} expired conversations")
        
        return expired_count