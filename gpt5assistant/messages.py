import re
from typing import List, Dict, Any, Optional
import discord
import logging

from .utils.variables import variable_processor

logger = logging.getLogger("red.gpt5assistant.messages")


class MessageBuilder:
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
    
    async def build_message_list(
        self,
        channel: discord.TextChannel,
        user_message: str,
        system_prompt: str,
        include_history: bool = True,
        bot: Optional[discord.Client] = None,
        user: Optional[discord.User] = None,
        conversation_manager = None,
        guild_config: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        messages = []
        
        if system_prompt:
            # Process variables in system prompt
            processed_system_prompt = await variable_processor.process_variables(
                system_prompt,
                bot=bot,
                guild=channel.guild if channel else None,
                channel=channel,
                user=user
            )
            messages.append({
                "role": "system",
                "content": processed_system_prompt
            })
        
        if include_history and conversation_manager and guild_config:
            # Use advanced conversation manager
            history = await conversation_manager.get_conversation_history(
                channel,
                guild_config,
                max_messages=self.max_history
            )
            messages.extend(history)
        elif include_history:
            # Fallback to basic history
            history = await self._get_channel_history(channel)
            messages.extend(history)
        
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        return messages
    
    async def _get_channel_history(self, channel: discord.TextChannel) -> List[Dict[str, Any]]:
        messages = []
        
        try:
            async for message in channel.history(limit=self.max_history * 2):
                if message.author.bot:
                    if message.author.id == channel.guild.me.id:
                        # This is our bot's message
                        content = self._clean_bot_message(message.content)
                        if content:
                            messages.append({
                                "role": "assistant",
                                "content": content
                            })
                else:
                    # User message
                    content = self._clean_user_message(message)
                    if content:
                        messages.append({
                            "role": "user",
                            "content": f"{message.author.display_name}: {content}"
                        })
                
                # Stop if we have enough messages
                if len(messages) >= self.max_history:
                    break
        
        except discord.Forbidden:
            logger.warning(f"Cannot read message history in {channel.name}")
        except Exception as e:
            logger.error(f"Error reading message history: {e}")
        
        # Reverse to get chronological order (oldest first)
        return list(reversed(messages))
    
    def _clean_user_message(self, message: discord.Message) -> str:
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
        
        # Handle embeds
        if message.embeds:
            content += " [Contains embeds]"
        
        return content.strip()
    
    def _clean_bot_message(self, content: str) -> str:
        # Remove common bot prefixes/suffixes and clean up the content
        # Remove code block indicators if they're formatting artifacts
        content = re.sub(r'```\w*\n?$', '', content)  # Remove trailing code block markers
        content = re.sub(r'^```\w*\n?', '', content)  # Remove leading code block markers
        
        # Remove thinking/processing indicators
        content = re.sub(r'^\*.*?\*\s*', '', content, flags=re.MULTILINE)
        
        return content.strip()


class ImageDetector:
    def __init__(self):
        self.image_keywords = {
            'generate', 'create', 'make', 'draw', 'design', 'paint', 'sketch',
            'illustration', 'picture', 'image', 'photo', 'artwork', 'graphic',
            'render', 'visualize', 'produce'
        }
        
        self.image_subjects = {
            'logo', 'icon', 'banner', 'poster', 'wallpaper', 'avatar',
            'diagram', 'chart', 'graph', 'map', 'scene', 'landscape',
            'portrait', 'character', 'concept', 'design'
        }
    
    def is_image_request(self, text: str) -> bool:
        text_lower = text.lower()
        
        # Direct patterns
        direct_patterns = [
            r'\b(generate|create|make|draw|design)\s+(an?\s+)?(image|picture|photo|artwork|graphic|illustration)',
            r'\bshow\s+me\s+(an?\s+)?(image|picture|photo)',
            r'\bi\s+(want|need)\s+(an?\s+)?(image|picture|photo|artwork)',
            r'\bcan\s+you\s+(generate|create|make|draw|design)',
        ]
        
        for pattern in direct_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Keyword-based detection
        has_action = any(keyword in text_lower for keyword in self.image_keywords)
        has_subject = any(subject in text_lower for subject in self.image_subjects)
        
        # Image request if has both action and subject keywords
        if has_action and has_subject:
            return True
        
        # Look for "image of", "picture of", etc.
        if re.search(r'\b(image|picture|photo|artwork|graphic)\s+of\s+', text_lower):
            return True
        
        return False
    
    def extract_image_prompt(self, text: str) -> str:
        # Clean up the text to extract the core image description
        text = re.sub(r'^(please\s+)?(generate|create|make|draw|design|show\s+me)\s+(an?\s+)?(image|picture|photo|artwork|graphic|illustration)\s+(of\s+)?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^(can\s+you\s+)?(generate|create|make|draw|design)\s+', '', text, flags=re.IGNORECASE)
        
        return text.strip()


class MessageDispatcher:
    def __init__(self):
        self.image_detector = ImageDetector()
        self.message_builder = MessageBuilder()
    
    async def classify_and_route(
        self,
        message: discord.Message,
        system_prompt: str,
        max_history: int = 10,
        bot: Optional[discord.Client] = None,
        conversation_manager = None,
        guild_config: Optional[Dict] = None,
        voice_transcription: Optional[str] = None
    ) -> Dict[str, Any]:
        content = message.clean_content
        
        # If we have voice transcription, include it in the content
        if voice_transcription:
            if content.strip():
                content = f"{content}\n\n{voice_transcription}"
            else:
                content = voice_transcription
        
        # Check for image attachments that might need analysis
        image_attachments = [att for att in message.attachments if att.content_type and att.content_type.startswith('image/')]
        
        # Check for image request (generation/editing)
        if self.image_detector.is_image_request(content):
            # Process variables in image prompt
            processed_prompt = await variable_processor.process_variables(
                content,
                bot=bot,
                guild=message.guild,
                channel=message.channel,
                user=message.author
            )
            return {
                "type": "image",
                "prompt": self.image_detector.extract_image_prompt(processed_prompt),
                "attachments": image_attachments
            }
        
        # If images are attached but no image generation request, include them in chat context
        image_context = ""
        if image_attachments:
            image_context = f"\n\n[User attached {len(image_attachments)} image(s): {', '.join(att.filename for att in image_attachments)}]"
        
        # Default to chat
        self.message_builder.max_history = max_history
        
        # Add image context to content if there are attached images
        final_content = content + image_context
        
        messages = await self.message_builder.build_message_list(
            message.channel,
            final_content,
            system_prompt,
            bot=bot,
            user=message.author,
            conversation_manager=conversation_manager,
            guild_config=guild_config
        )
        
        return {
            "type": "chat",
            "messages": messages,
            "image_attachments": image_attachments  # Pass images for potential analysis
        }