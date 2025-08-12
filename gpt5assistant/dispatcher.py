import asyncio
import logging
import random
import time
from typing import Dict, Any, Optional, AsyncGenerator
import discord
from redbot.core import Config

from .openai_client import OpenAIClient
from .messages import MessageDispatcher
from .config_schemas import GuildConfig, ChannelConfig
from .tools.image import ImageTool
from .tools.file_search import FileSearchTool
from .tools.web_search import WebSearchTool
from .tools.code_interpreter import CodeInterpreterTool
from .utils.discord_io import DiscordStreamer, send_image_result
from .utils.variables import variable_processor
from .utils.conversation import ConversationManager
from .utils.voice import VoiceProcessor, VoiceMessageHandler
from .utils.batch_processor import BatchFileProcessor
from .errors import error_handler, RateLimitError, GPT5AssistantError

logger = logging.getLogger("red.gpt5assistant.dispatcher")


class GPTDispatcher:
    def __init__(self, config: Config, bot):
        self.config = config
        self.bot = bot
        self.openai_client: Optional[OpenAIClient] = None
        self.message_dispatcher = MessageDispatcher()
        
        # Tools
        self.image_tool: Optional[ImageTool] = None
        self.file_search_tool: Optional[FileSearchTool] = None
        self.web_search_tool = WebSearchTool()
        self.code_interpreter_tool = CodeInterpreterTool()
        
        # Voice processing
        self.voice_processor: Optional[VoiceProcessor] = None
        self.voice_message_handler: Optional[VoiceMessageHandler] = None
        
        # Batch processing
        self.batch_processor: Optional[BatchFileProcessor] = None
        
        # Concurrency controls
        self._channel_locks: Dict[int, asyncio.Lock] = {}
        self._active_requests: Dict[int, asyncio.Task] = {}
        
        # aiuser-like features
        self._channel_last_activity: Dict[int, float] = {}  # Track last activity per channel
        self._random_message_tasks: Dict[int, asyncio.Task] = {}  # Random message tasks per channel
        
        # Conversation management
        self.conversation_manager = ConversationManager(self.config)
    
    async def initialize(self, api_key: str) -> None:
        if self.openai_client:
            await self.openai_client.close()
        
        self.openai_client = OpenAIClient(api_key)
        self.image_tool = ImageTool(self.openai_client)
        self.file_search_tool = FileSearchTool(self.openai_client)
        
        # Initialize voice processing
        self.voice_processor = VoiceProcessor(self.openai_client)
        self.voice_message_handler = VoiceMessageHandler(self.voice_processor)
        
        # Initialize batch processing
        self.batch_processor = BatchFileProcessor(self.openai_client)
        
        logger.info("GPT Dispatcher initialized with OpenAI client")
    
    async def shutdown(self) -> None:
        # Cancel active requests
        for task in self._active_requests.values():
            if not task.done():
                task.cancel()
        
        # Cancel random message tasks
        for task in self._random_message_tasks.values():
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        all_tasks = list(self._active_requests.values()) + list(self._random_message_tasks.values())
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)
        
        if self.openai_client:
            await self.openai_client.close()
        
        logger.info("GPT Dispatcher shut down")
    
    async def handle_all_messages(self, message: discord.Message) -> None:
        """Handle all messages for random response checking and activity tracking"""
        if not message.guild or message.author.bot:
            return
        
        channel_id = message.channel.id
        guild_id = message.guild.id
        
        # Update activity tracking
        self._channel_last_activity[channel_id] = time.time()
        
        # Get configuration
        guild_config = await self._get_guild_config(guild_id)
        channel_config = await self._get_channel_config(guild_id, channel_id)
        
        if not self._is_channel_allowed(guild_config, channel_config, channel_id):
            return
        
        # Check if user has opted in (if required)
        if guild_config.require_opt_in and message.author.id not in guild_config.opted_in_users:
            return
        
        # Check for random response
        should_respond_randomly = self._should_respond_randomly(guild_config, channel_config, message)
        
        if should_respond_randomly:
            await self.handle_message(message)
    
    async def handle_message(self, message: discord.Message) -> None:
        if not self.openai_client:
            await message.reply("❌ OpenAI client not initialized. Please set API key.")
            return
        
        channel_id = message.channel.id
        guild_id = message.guild.id if message.guild else None
        
        # Get channel lock
        if channel_id not in self._channel_locks:
            self._channel_locks[channel_id] = asyncio.Lock()
        
        async with self._channel_locks[channel_id]:
            # Cancel any existing request for this channel
            if channel_id in self._active_requests:
                existing_task = self._active_requests[channel_id]
                if not existing_task.done():
                    existing_task.cancel()
                    try:
                        await existing_task
                    except asyncio.CancelledError:
                        pass
            
            # Start new request
            task = asyncio.create_task(self._process_message(message))
            self._active_requests[channel_id] = task
            
            try:
                await task
            finally:
                self._active_requests.pop(channel_id, None)
    
    async def _process_message(self, message: discord.Message) -> None:
        try:
            guild_id = message.guild.id if message.guild else None
            if not guild_id:
                await message.reply("❌ This cog only works in servers.")
                return
            
            # Get configuration
            guild_config = await self._get_guild_config(guild_id)
            channel_config = await self._get_channel_config(guild_id, message.channel.id)
            
            if not self._is_channel_allowed(guild_config, channel_config, message.channel.id):
                return
            
            # Handle voice messages if present and enabled
            voice_transcription = None
            tools_config = self._get_effective_tools_config(guild_config, channel_config)
            if (self.voice_message_handler and 
                tools_config.voice_transcription and 
                self.voice_processor.is_voice_message(message)):
                try:
                    voice_transcription = await self.voice_message_handler.process_voice_message(message)
                    if voice_transcription:
                        logger.info(f"Transcribed voice message in {message.channel.name}")
                except Exception as e:
                    logger.error(f"Error processing voice message: {e}")
                    # Continue processing even if voice transcription fails
            
            # Route the message (with voice transcription if available)
            route_result = await self.message_dispatcher.classify_and_route(
                message,
                self._get_effective_system_prompt(guild_config, channel_config),
                self._get_effective_max_history(guild_config, channel_config),
                bot=self.bot,
                conversation_manager=self.conversation_manager,
                guild_config=guild_config.__dict__ if hasattr(guild_config, '__dict__') else guild_config,
                voice_transcription=voice_transcription
            )
            
            if route_result["type"] == "image":
                await self._handle_image_request(message, route_result, guild_config, channel_config)
            else:
                await self._handle_chat_request(message, route_result, guild_config, channel_config)
        
        except GPT5AssistantError as e:
            await error_handler.send_error_message(message, e)
        except Exception as e:
            await error_handler.send_error_message(message, e)
    
    async def _handle_chat_request(
        self,
        message: discord.Message,
        route_result: Dict[str, Any],
        guild_config: GuildConfig,
        channel_config: Optional[ChannelConfig]
    ) -> None:
        model_config = self._get_effective_model_config(guild_config, channel_config)
        tools_config = self._get_effective_tools_config(guild_config, channel_config)
        
        # Check if there are image attachments that need analysis
        image_attachments = route_result.get("image_attachments", [])
        messages = route_result["messages"]
        
        # If there are images, analyze them and add to context
        if image_attachments and self.image_tool:
            try:
                # Analyze the first image (for simplicity, could be extended for multiple)
                first_image = image_attachments[0]
                
                if self.image_tool.validate_image_attachment(first_image):
                    logger.info(f"Analyzing attached image {first_image.filename} for context")
                    
                    # Analyze the image
                    analysis_result = await self.image_tool.analyze_image(
                        first_image,
                        f"Analyze this image that the user has shared. Provide a helpful description that will give context for the conversation."
                    )
                    
                    if analysis_result["success"]:
                        # Add image analysis to the conversation
                        image_analysis = f"\n\n[Image Analysis of {first_image.filename}]:\n{analysis_result['analysis']}"
                        
                        # Add to the last user message
                        if messages and messages[-1]["role"] == "user":
                            messages[-1]["content"] += image_analysis
                        else:
                            messages.append({
                                "role": "user",
                                "content": image_analysis
                            })
                        
                        logger.info(f"Added image analysis to conversation context")
                    
            except Exception as e:
                logger.error(f"Failed to analyze attached image: {e}")
                # Continue without image analysis
        
        streamer = DiscordStreamer(message)
        
        try:
            content_generator = self.openai_client.respond_chat(
                messages=messages,
                model_config=model_config,
                tool_config=tools_config,
                guild_id=message.guild.id
            )
            
            # Collect response for conversation history
            response_text = ""
            async def response_collector():
                nonlocal response_text
                async for chunk in content_generator:
                    response_text += chunk
                    yield chunk
            
            sent_message = await streamer.stream_response(
                response_collector(),
                ephemeral=guild_config.ephemeral_responses
            )
            
            # Add response to conversation history
            if response_text.strip():
                guild_config_dict = guild_config.__dict__ if hasattr(guild_config, '__dict__') else guild_config
                await self.conversation_manager.add_message_to_history(
                    message.channel,
                    "assistant",
                    response_text,
                    guild_config_dict
                )
        
        except Exception as e:
            await streamer.stop_typing()
            raise e
    
    async def _handle_image_request(
        self,
        message: discord.Message,
        route_result: Dict[str, Any],
        guild_config: GuildConfig,
        channel_config: Optional[ChannelConfig]
    ) -> None:
        tools_config = self._get_effective_tools_config(guild_config, channel_config)
        
        if not tools_config.image:
            await message.reply("❌ Image generation is disabled in this server.")
            return
        
        prompt = route_result["prompt"]
        attachments = route_result.get("attachments", [])
        
        try:
            if attachments:
                # Image editing
                image_attachment = attachments[0]
                if not self.image_tool.validate_image_attachment(image_attachment):
                    await message.reply("❌ Invalid image format. Supported: PNG, JPEG, WebP")
                    return
                
                result = await self.image_tool.edit_image(
                    attachment=image_attachment,
                    prompt=prompt
                )
            else:
                # Image generation
                result = await self.image_tool.generate_image(prompt=prompt)
            
            await send_image_result(
                target=message,
                image_url=result["url"],
                prompt=prompt,
                metadata=result,
                ephemeral=guild_config.ephemeral_responses
            )
        
        except Exception as e:
            await error_handler.send_error_message(message, e)
    
    async def handle_slash_command(self, interaction: discord.Interaction, command_data: Dict[str, Any]) -> None:
        if not self.openai_client:
            await interaction.response.send_message(
                "❌ OpenAI client not initialized. Please set API key.",
                ephemeral=True
            )
            return
        
        command_type = command_data.get("type")
        guild_id = interaction.guild_id
        
        try:
            if command_type == "ask":
                await self._handle_slash_ask(interaction, command_data)
            elif command_type == "image":
                await self._handle_slash_image(interaction, command_data)
            elif command_type == "upload":
                await self._handle_slash_upload(interaction, command_data)
            elif command_type == "batch":
                await self._handle_slash_batch(interaction, command_data)
            else:
                await interaction.response.send_message("❌ Unknown command type.", ephemeral=True)
        
        except Exception as e:
            logger.error(f"Error handling slash command: {e}", exc_info=True)
            if not interaction.response.is_done():
                await interaction.response.send_message("❌ An error occurred.", ephemeral=True)
    
    async def _handle_slash_ask(self, interaction: discord.Interaction, command_data: Dict[str, Any]) -> None:
        guild_config = await self._get_guild_config(interaction.guild_id)
        channel_config = await self._get_channel_config(interaction.guild_id, interaction.channel_id)
        
        prompt = command_data.get("prompt", "")
        if not prompt:
            await interaction.response.send_message("❌ No prompt provided.", ephemeral=True)
            return
        
        messages = [{
            "role": "system",
            "content": self._get_effective_system_prompt(guild_config, channel_config)
        }, {
            "role": "user",
            "content": prompt
        }]
        
        model_config = self._get_effective_model_config(guild_config, channel_config)
        tools_config = self._get_effective_tools_config(guild_config, channel_config)
        
        streamer = DiscordStreamer(interaction)
        
        content_generator = self.openai_client.respond_chat(
            messages=messages,
            model_config=model_config,
            tool_config=tools_config,
            guild_id=interaction.guild_id
        )
        
        await streamer.stream_response(
            content_generator,
            ephemeral=guild_config.ephemeral_responses
        )
    
    async def _handle_slash_image(self, interaction: discord.Interaction, command_data: Dict[str, Any]) -> None:
        guild_config = await self._get_guild_config(interaction.guild_id)
        tools_config = self._get_effective_tools_config(guild_config, None)
        
        if not tools_config.image:
            await interaction.response.send_message("❌ Image generation is disabled.", ephemeral=True)
            return
        
        prompt = command_data.get("prompt", "")
        edit_attachment = command_data.get("edit_attachment")
        
        if not prompt:
            await interaction.response.send_message("❌ No prompt provided.", ephemeral=True)
            return
        
        await interaction.response.defer(ephemeral=guild_config.ephemeral_responses)
        
        try:
            if edit_attachment:
                if not self.image_tool.validate_image_attachment(edit_attachment):
                    await interaction.followup.send("❌ Invalid image format.", ephemeral=True)
                    return
                
                result = await self.image_tool.edit_image(
                    attachment=edit_attachment,
                    prompt=prompt
                )
            else:
                result = await self.image_tool.generate_image(prompt=prompt)
            
            await send_image_result(
                target=interaction,
                image_url=result["url"],
                prompt=prompt,
                metadata=result,
                ephemeral=guild_config.ephemeral_responses
            )
        
        except Exception as e:
            logger.error(f"Slash image command error: {e}")
            await interaction.followup.send(f"❌ Image generation failed: {str(e)}", ephemeral=True)
    
    async def _handle_slash_upload(self, interaction: discord.Interaction, command_data: Dict[str, Any]) -> None:
        guild_config = await self._get_guild_config(interaction.guild_id)
        tools_config = self._get_effective_tools_config(guild_config, None)
        
        if not tools_config.file_search:
            await interaction.response.send_message("❌ File search is disabled.", ephemeral=True)
            return
        
        files = command_data.get("files", [])
        if not files:
            await interaction.response.send_message("❌ No files provided.", ephemeral=True)
            return
        
        await interaction.response.defer(ephemeral=True)
        
        try:
            result = await self.file_search_tool.upload_files(files, interaction.guild_id)
            
            # Update guild config with KB ID
            async with self.config.guild(interaction.guild).all() as guild_data:
                guild_data["file_search_kb_id"] = result["knowledge_base_id"]
            
            await interaction.followup.send(
                f"✅ Uploaded {result['file_count']} files to knowledge base.\n"
                f"Files: {', '.join(result['uploaded_files'][:5])}{'...' if len(result['uploaded_files']) > 5 else ''}",
                ephemeral=True
            )
        
        except Exception as e:
            logger.error(f"File upload error: {e}")
            await interaction.followup.send(f"❌ File upload failed: {str(e)}", ephemeral=True)
    
    async def _handle_slash_batch(self, interaction: discord.Interaction, command_data: Dict[str, Any]) -> None:
        guild_config = await self._get_guild_config(interaction.guild_id)
        tools_config = self._get_effective_tools_config(guild_config, None)
        
        if not tools_config.file_search:
            await interaction.response.send_message("❌ File processing is disabled.", ephemeral=True)
            return
        
        files = command_data.get("files", [])
        if not files:
            await interaction.response.send_message("❌ No files provided.", ephemeral=True)
            return
        
        if len(files) > self.batch_processor.get_batch_limits()["max_files"]:
            await interaction.response.send_message(
                f"❌ Too many files. Maximum: {self.batch_processor.get_batch_limits()['max_files']}", 
                ephemeral=True
            )
            return
        
        await interaction.response.defer(ephemeral=True)
        
        try:
            # Process options
            options = {
                "generate_summaries": command_data.get("summaries", True),
                "extract_key_points": command_data.get("key_points", True),
                "include_content": command_data.get("include_content", False)
            }
            
            # Process the batch
            result = await self.batch_processor.process_batch(files, options)
            
            # Upload successful files to knowledge base if any
            successful_files = [
                f for f in result["processed_files"] 
                if f.processed and not f.error
            ]
            
            if successful_files:
                # Convert to attachments for knowledge base upload
                file_attachments = [
                    att for att in files 
                    if any(att.filename == f.filename for f in successful_files)
                ]
                
                if file_attachments:
                    kb_result = await self.file_search_tool.upload_files(file_attachments, interaction.guild_id)
                    
                    # Update guild config with KB ID
                    async with self.config.guild(interaction.guild).all() as guild_data:
                        guild_data["file_search_kb_id"] = kb_result["knowledge_base_id"]
            
            # Create response embed
            embed = await self._create_batch_result_embed(result)
            
            await interaction.followup.send(embed=embed, ephemeral=True)
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            await interaction.followup.send(f"❌ Batch processing failed: {str(e)}", ephemeral=True)
    
    # Configuration helpers
    async def _get_guild_config(self, guild_id: int) -> GuildConfig:
        guild_data = await self.config.guild_from_id(guild_id).all()
        return GuildConfig(**guild_data)
    
    async def _get_channel_config(self, guild_id: int, channel_id: int) -> Optional[ChannelConfig]:
        guild_data = await self.config.guild_from_id(guild_id).all()
        channel_overrides = guild_data.get("channel_overrides", {})
        
        if str(channel_id) in channel_overrides:
            return ChannelConfig(**channel_overrides[str(channel_id)])
        return None
    
    def _is_channel_allowed(self, guild_config: GuildConfig, channel_config: Optional[ChannelConfig], channel_id: int) -> bool:
        if not guild_config.enabled:
            return False
        
        if channel_config and not channel_config.enabled:
            return False
        
        if guild_config.denied_channels and channel_id in guild_config.denied_channels:
            return False
        
        if guild_config.allowed_channels and channel_id not in guild_config.allowed_channels:
            return False
        
        return True
    
    def _get_effective_system_prompt(self, guild_config: GuildConfig, channel_config: Optional[ChannelConfig]) -> str:
        if channel_config and channel_config.system_prompt:
            return channel_config.system_prompt
        return guild_config.system_prompt
    
    def _get_effective_max_history(self, guild_config: GuildConfig, channel_config: Optional[ChannelConfig]) -> int:
        return guild_config.max_message_history
    
    def _get_effective_model_config(self, guild_config: GuildConfig, channel_config: Optional[ChannelConfig]):
        if channel_config and channel_config.model:
            return channel_config.model
        return guild_config.model
    
    def _get_effective_tools_config(self, guild_config: GuildConfig, channel_config: Optional[ChannelConfig]):
        if channel_config and channel_config.tools:
            return channel_config.tools
        return guild_config.tools
    
    def _should_respond_randomly(self, guild_config: GuildConfig, channel_config: Optional[ChannelConfig], message: discord.Message) -> bool:
        """Determine if we should respond to this message randomly"""
        # Get effective response percentage
        response_percentage = guild_config.response_percentage
        if channel_config and channel_config.response_percentage is not None:
            response_percentage = channel_config.response_percentage
        
        if response_percentage <= 0:
            return False
        
        # Random chance check
        return random.random() * 100 < response_percentage
    
    def _get_effective_response_percentage(self, guild_config: GuildConfig, channel_config: Optional[ChannelConfig]) -> float:
        """Get effective response percentage for a channel"""
        if channel_config and channel_config.response_percentage is not None:
            return channel_config.response_percentage
        return guild_config.response_percentage
    
    def _get_effective_random_messages(self, guild_config: GuildConfig, channel_config: Optional[ChannelConfig]) -> bool:
        """Get effective random messages setting for a channel"""
        if channel_config and channel_config.random_messages is not None:
            return channel_config.random_messages
        return guild_config.random_messages
    
    async def start_random_message_loop(self, guild_id: int, channel_id: int) -> None:
        """Start the random message loop for a channel"""
        if channel_id in self._random_message_tasks:
            # Already running
            return
        
        task = asyncio.create_task(self._random_message_loop(guild_id, channel_id))
        self._random_message_tasks[channel_id] = task
    
    async def stop_random_message_loop(self, channel_id: int) -> None:
        """Stop the random message loop for a channel"""
        if channel_id in self._random_message_tasks:
            task = self._random_message_tasks.pop(channel_id)
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
    
    async def _random_message_loop(self, guild_id: int, channel_id: int) -> None:
        """Background task that sends random messages when channel is idle"""
        try:
            while True:
                guild_config = await self._get_guild_config(guild_id)
                channel_config = await self._get_channel_config(guild_id, channel_id)
                
                if not self._get_effective_random_messages(guild_config, channel_config):
                    break
                
                # Wait for the cooldown period
                await asyncio.sleep(guild_config.random_message_cooldown)
                
                # Check if channel has been idle
                last_activity = self._channel_last_activity.get(channel_id, 0)
                if time.time() - last_activity < guild_config.idle_timeout:
                    continue  # Channel not idle long enough
                
                # Send random message
                try:
                    channel = self.bot.get_channel(channel_id)
                    if channel:
                        topic = random.choice(guild_config.random_message_topics)
                        await self._send_random_message(channel, topic, guild_config, channel_config)
                except Exception as e:
                    logger.error(f"Error sending random message: {e}")
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in random message loop: {e}")
    
    async def _send_random_message(self, channel: discord.TextChannel, topic: str, guild_config: GuildConfig, channel_config: Optional[ChannelConfig]) -> None:
        """Send a random message based on a topic"""
        system_prompt = self._get_effective_system_prompt(guild_config, channel_config)
        model_config = self._get_effective_model_config(guild_config, channel_config)
        tools_config = self._get_effective_tools_config(guild_config, channel_config)
        
        # Process variables in the topic and system prompt
        processed_topic = await variable_processor.process_variables(
            topic,
            bot=self.bot,
            guild=channel.guild,
            channel=channel
        )
        
        processed_system_prompt = await variable_processor.process_variables(
            system_prompt,
            bot=self.bot,
            guild=channel.guild,
            channel=channel
        )
        
        messages = [
            {"role": "system", "content": f"{processed_system_prompt} Generate a brief, natural message to {processed_topic}. Keep it conversational and engaging, as if you're a member of this Discord community."},
            {"role": "user", "content": f"Please {processed_topic.lower()}"}
        ]
        
        try:
            content_generator = self.openai_client.respond_chat(
                messages=messages,
                model_config=model_config,
                tool_config=tools_config,
                guild_id=channel.guild.id
            )
            
            # Send the random message
            response_text = ""
            async for chunk in content_generator:
                response_text += chunk
            
            if response_text.strip():
                await channel.send(response_text[:2000])  # Discord limit
                
        except Exception as e:
            logger.error(f"Error generating random message: {e}")
    
    # Conversation management methods
    async def forget_conversation(self, channel: discord.TextChannel) -> bool:
        """Clear conversation history for a channel"""
        return await self.conversation_manager.forget_conversation(channel)
    
    async def forget_all_conversations(self, guild_id: int) -> int:
        """Clear all conversation histories for a guild"""
        return await self.conversation_manager.forget_all_conversations(guild_id)
    
    async def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation manager statistics"""
        return self.conversation_manager.get_conversation_stats()
    
    async def _create_batch_result_embed(self, result: Dict[str, Any]) -> discord.Embed:
        """Create an embed showing batch processing results"""
        
        stats = result["stats"]
        processed_files = result["processed_files"]
        errors = result["errors"]
        batch_summary = result.get("batch_summary")
        
        # Create main embed
        embed = discord.Embed(
            title="📁 Batch File Processing Results",
            description=f"Processed {stats['processed_successfully']}/{stats['total_files']} files successfully",
            color=0x00ff00 if stats['failed'] == 0 else 0xffaa00
        )
        
        # Add statistics
        embed.add_field(
            name="📊 Statistics",
            value=f"✅ Successful: {stats['processed_successfully']}\n"
                  f"❌ Failed: {stats['failed']}\n"
                  f"💾 Total Size: {self._format_bytes(stats['total_size'])}",
            inline=True
        )
        
        # Add batch summary if available
        if batch_summary:
            embed.add_field(
                name="📋 Batch Summary",
                value=batch_summary[:1000] + ("..." if len(batch_summary) > 1000 else ""),
                inline=False
            )
        
        # Add processed files (limited)
        if processed_files:
            file_list = []
            for file_meta in processed_files[:5]:  # Show first 5 files
                status = "✅" if file_meta.processed else "❌"
                summary = file_meta.summary[:100] + "..." if file_meta.summary and len(file_meta.summary) > 100 else file_meta.summary or "No summary"
                file_list.append(f"{status} **{file_meta.filename}** ({file_meta.file_type})\n{summary}")
            
            if len(processed_files) > 5:
                file_list.append(f"... and {len(processed_files) - 5} more files")
            
            embed.add_field(
                name="📄 Processed Files",
                value="\n\n".join(file_list)[:1000],
                inline=False
            )
        
        # Add errors if any
        if errors:
            error_list = []
            for error in errors[:3]:  # Show first 3 errors
                error_list.append(f"❌ **{error['filename']}**: {error['error'][:100]}")
            
            if len(errors) > 3:
                error_list.append(f"... and {len(errors) - 3} more errors")
            
            embed.add_field(
                name="⚠️ Errors",
                value="\n".join(error_list)[:1000],
                inline=False
            )
        
        embed.set_footer(text="Files uploaded to knowledge base for future searches")
        return embed
    
    def _format_bytes(self, bytes_size: int) -> str:
        """Format bytes into human-readable size"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024
        return f"{bytes_size:.1f} TB"