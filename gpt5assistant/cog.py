import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, Literal
import discord
from discord.ext import commands
from redbot.core import commands as red_commands, Config, checks
from redbot.core.bot import Red
from redbot.core.utils.chat_formatting import box, pagify

from .dispatcher import GPTDispatcher
from .config_schemas import GUILD_CONFIG_SCHEMA, CHANNEL_CONFIG_SCHEMA, GuildConfig, ChannelConfig
from .utils.discord_io import create_allowed_mentions
from .utils.variables import variable_processor

logger = logging.getLogger("red.gpt5assistant.cog")


class GPT5Assistant(red_commands.Cog):
    def __init__(self, bot: Red):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=1234567890, force_registration=True)
        
        # Set up config defaults
        self.config.register_guild(**GUILD_CONFIG_SCHEMA)
        self.config.register_global(
            openai_api_key=None
        )
        
        self.dispatcher = GPTDispatcher(self.config, bot)
        self._initialization_task: Optional[asyncio.Task] = None
        
        # Set up allowed mentions
        bot.allowed_mentions = create_allowed_mentions()
    
    async def cog_load(self) -> None:
        api_key = await self.bot.get_shared_api_tokens("openai")
        if api_key and api_key.get("api_key"):
            self._initialization_task = asyncio.create_task(
                self.dispatcher.initialize(api_key["api_key"])
            )
        else:
            logger.warning("No OpenAI API key found. Set with `[p]set api openai api_key,<key>`")
    
    async def cog_unload(self) -> None:
        if self._initialization_task and not self._initialization_task.done():
            self._initialization_task.cancel()
            try:
                await self._initialization_task
            except asyncio.CancelledError:
                pass
        
        await self.dispatcher.shutdown()
    
    # Event handlers
    @red_commands.Cog.listener()
    async def on_message_without_command(self, message: discord.Message) -> None:
        if message.author.bot:
            return
        
        if not message.guild:
            return
        
        # Always handle messages for activity tracking and random responses
        await self.dispatcher.handle_all_messages(message)
        
        # Check if bot was mentioned for direct responses
        if self.bot.user in message.mentions:
            await self.dispatcher.handle_message(message)
    
    # Commands Group
    @red_commands.group(name="gpt5", invoke_without_command=True)
    async def gpt5(self, ctx: red_commands.Context) -> None:
        """GPT-5 Assistant commands"""
        await ctx.send_help(ctx.command)
    
    # Configuration commands
    @gpt5.group(name="config", invoke_without_command=True)
    @checks.admin_or_permissions(manage_guild=True)
    async def config(self, ctx: red_commands.Context) -> None:
        """Configuration commands for GPT-5 Assistant"""
        await ctx.send_help(ctx.command)
    
    @config.command(name="show")
    async def config_show(self, ctx: red_commands.Context) -> None:
        """Show current configuration"""
        guild_data = await self.config.guild(ctx.guild).all()
        guild_config = GuildConfig(**guild_data)
        
        embed = discord.Embed(
            title="🤖 GPT-5 Assistant Configuration",
            color=0x00ff00 if guild_config.enabled else 0xff0000
        )
        
        # Basic settings
        embed.add_field(
            name="Basic Settings",
            value=f"Enabled: {'✅' if guild_config.enabled else '❌'}\n"
                  f"Model: `{guild_config.model.name}`\n"
                  f"Verbosity: `{guild_config.model.text.verbosity}`\n"
                  f"Reasoning: `{guild_config.model.reasoning.effort}`\n"
                  f"Temperature: `{guild_config.model.temperature}`",
            inline=True
        )
        
        # Tools
        tools_status = []
        tools_status.append(f"Web Search: {'✅' if guild_config.tools.web_search else '❌'}")
        tools_status.append(f"File Search: {'✅' if guild_config.tools.file_search else '❌'}")
        tools_status.append(f"Code Interpreter: {'✅' if guild_config.tools.code_interpreter else '❌'}")
        tools_status.append(f"Image Generation: {'✅' if guild_config.tools.image else '❌'}")
        
        embed.add_field(
            name="Tools",
            value="\n".join(tools_status),
            inline=True
        )
        
        # Channel settings
        channel_info = []
        if guild_config.allowed_channels:
            channels = [f"<#{ch}>" for ch in guild_config.allowed_channels[:3]]
            channel_info.append(f"Allowed: {', '.join(channels)}{'...' if len(guild_config.allowed_channels) > 3 else ''}")
        
        if guild_config.denied_channels:
            channels = [f"<#{ch}>" for ch in guild_config.denied_channels[:3]]
            channel_info.append(f"Denied: {', '.join(channels)}{'...' if len(guild_config.denied_channels) > 3 else ''}")
        
        if guild_config.channel_overrides:
            channel_info.append(f"Overrides: {len(guild_config.channel_overrides)} channels")
        
        embed.add_field(
            name="Channels",
            value="\n".join(channel_info) if channel_info else "All channels allowed",
            inline=False
        )
        
        # Response behavior
        embed.add_field(
            name="Response Behavior",
            value=f"Response %: `{guild_config.response_percentage}%`\n"
                  f"Require Opt-in: {'✅' if guild_config.require_opt_in else '❌'}\n"
                  f"Random Messages: {'✅' if guild_config.random_messages else '❌'}\n"
                  f"Opted-in Users: `{len(guild_config.opted_in_users)}`",
            inline=True
        )
        
        # Additional settings
        embed.add_field(
            name="Other Settings",
            value=f"Ephemeral Responses: {'✅' if guild_config.ephemeral_responses else '❌'}\n"
                  f"Message History: `{guild_config.max_message_history}`\n"
                  f"Cooldown: `{guild_config.cooldown_seconds}s`\n"
                  f"KB ID: `{guild_config.file_search_kb_id or 'None'}`",
            inline=True
        )
        
        await ctx.send(embed=embed)
    
    @config.command(name="model")
    async def config_model(self, ctx: red_commands.Context, model: str = "gpt-5") -> None:
        """Set the model to use"""
        valid_models = ["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4o", "gpt-4o-mini"]
        
        if model not in valid_models:
            await ctx.send(f"❌ Invalid model. Valid options: {', '.join(valid_models)}")
            return
        
        async with self.config.guild(ctx.guild).all() as guild_data:
            guild_data["model"]["name"] = model
        
        await ctx.send(f"✅ Model set to `{model}`")
    
    @config.command(name="verbosity")
    async def config_verbosity(self, ctx: red_commands.Context, level: Literal["low", "medium", "high"]) -> None:
        """Set response verbosity level"""
        async with self.config.guild(ctx.guild).all() as guild_data:
            guild_data["model"]["text"]["verbosity"] = level
        
        await ctx.send(f"✅ Verbosity set to `{level}`")
    
    @config.command(name="reasoning")
    async def config_reasoning(self, ctx: red_commands.Context, effort: Literal["minimal", "medium", "high"]) -> None:
        """Set reasoning effort level"""
        async with self.config.guild(ctx.guild).all() as guild_data:
            guild_data["model"]["reasoning"]["effort"] = effort
        
        await ctx.send(f"✅ Reasoning effort set to `{effort}`")
    
    @config.command(name="temperature")
    async def config_temperature(self, ctx: red_commands.Context, temperature: float) -> None:
        """Set model temperature (0.0-2.0)"""
        if not 0.0 <= temperature <= 2.0:
            await ctx.send("❌ Temperature must be between 0.0 and 2.0")
            return
        
        async with self.config.guild(ctx.guild).all() as guild_data:
            guild_data["model"]["temperature"] = temperature
        
        await ctx.send(f"✅ Temperature set to `{temperature}`")
    
    @config.command(name="system")
    async def config_system(self, ctx: red_commands.Context, *, prompt: str) -> None:
        """Set the system prompt"""
        if len(prompt) > 2000:
            await ctx.send("❌ System prompt too long (max 2000 characters)")
            return
        
        await self.config.guild(ctx.guild).system_prompt.set(prompt)
        await ctx.send("✅ System prompt updated")
    
    @config.command(name="enable")
    async def config_enable(self, ctx: red_commands.Context) -> None:
        """Enable the assistant in this server"""
        await self.config.guild(ctx.guild).enabled.set(True)
        await ctx.send("✅ GPT-5 Assistant enabled")
    
    @config.command(name="disable")
    async def config_disable(self, ctx: red_commands.Context) -> None:
        """Disable the assistant in this server"""
        await self.config.guild(ctx.guild).enabled.set(False)
        await ctx.send("❌ GPT-5 Assistant disabled")
    
    @config.group(name="tools", invoke_without_command=True)
    async def config_tools(self, ctx: red_commands.Context) -> None:
        """Tool configuration commands"""
        await ctx.send_help(ctx.command)
    
    @config_tools.command(name="enable")
    async def tools_enable(self, ctx: red_commands.Context, tool: Literal["web_search", "file_search", "code_interpreter", "image"]) -> None:
        """Enable a specific tool"""
        async with self.config.guild(ctx.guild).all() as guild_data:
            guild_data["tools"][tool] = True
        
        await ctx.send(f"✅ {tool.replace('_', ' ').title()} tool enabled")
    
    @config_tools.command(name="disable")
    async def tools_disable(self, ctx: red_commands.Context, tool: Literal["web_search", "file_search", "code_interpreter", "image"]) -> None:
        """Disable a specific tool"""
        async with self.config.guild(ctx.guild).all() as guild_data:
            guild_data["tools"][tool] = False
        
        await ctx.send(f"❌ {tool.replace('_', ' ').title()} tool disabled")
    
    @config.group(name="channels", invoke_without_command=True)
    async def config_channels(self, ctx: red_commands.Context) -> None:
        """Channel configuration commands"""
        await ctx.send_help(ctx.command)
    
    @config_channels.command(name="allow")
    async def channels_allow(self, ctx: red_commands.Context, channel: discord.TextChannel) -> None:
        """Allow the assistant in a specific channel"""
        async with self.config.guild(ctx.guild).all() as guild_data:
            if channel.id not in guild_data["allowed_channels"]:
                guild_data["allowed_channels"].append(channel.id)
            
            if channel.id in guild_data["denied_channels"]:
                guild_data["denied_channels"].remove(channel.id)
        
        await ctx.send(f"✅ Assistant allowed in {channel.mention}")
    
    @config_channels.command(name="deny")
    async def channels_deny(self, ctx: red_commands.Context, channel: discord.TextChannel) -> None:
        """Deny the assistant in a specific channel"""
        async with self.config.guild(ctx.guild).all() as guild_data:
            if channel.id not in guild_data["denied_channels"]:
                guild_data["denied_channels"].append(channel.id)
            
            if channel.id in guild_data["allowed_channels"]:
                guild_data["allowed_channels"].remove(channel.id)
        
        await ctx.send(f"❌ Assistant denied in {channel.mention}")
    
    @config_channels.command(name="clear")
    async def channels_clear(self, ctx: red_commands.Context) -> None:
        """Clear all channel restrictions"""
        await self.config.guild(ctx.guild).allowed_channels.set([])
        await self.config.guild(ctx.guild).denied_channels.set([])
        await ctx.send("✅ Channel restrictions cleared")
    
    # User management commands
    @gpt5.command(name="optin")
    async def optin(self, ctx: red_commands.Context) -> None:
        """Opt in to AI responses (required if opt-in is enabled)"""
        async with self.config.guild(ctx.guild).all() as guild_data:
            if ctx.author.id not in guild_data["opted_in_users"]:
                guild_data["opted_in_users"].append(ctx.author.id)
        
        await ctx.send(f"✅ {ctx.author.mention} has opted in to AI responses!")
    
    @gpt5.command(name="optout")
    async def optout(self, ctx: red_commands.Context) -> None:
        """Opt out of AI responses"""
        async with self.config.guild(ctx.guild).all() as guild_data:
            if ctx.author.id in guild_data["opted_in_users"]:
                guild_data["opted_in_users"].remove(ctx.author.id)
        
        await ctx.send(f"❌ {ctx.author.mention} has opted out of AI responses")
    
    # Response behavior commands
    @config.command(name="response_percentage")
    async def config_response_percentage(self, ctx: red_commands.Context, percentage: float) -> None:
        """Set the percentage of messages to respond to randomly (0-100)"""
        if not 0 <= percentage <= 100:
            await ctx.send("❌ Percentage must be between 0 and 100")
            return
        
        await self.config.guild(ctx.guild).response_percentage.set(percentage)
        await ctx.send(f"✅ Response percentage set to `{percentage}%`")
    
    @config.command(name="require_optin")
    async def config_require_optin(self, ctx: red_commands.Context, enabled: bool) -> None:
        """Set whether users must opt-in to receive responses"""
        await self.config.guild(ctx.guild).require_opt_in.set(enabled)
        status = "enabled" if enabled else "disabled"
        await ctx.send(f"✅ User opt-in requirement {status}")
    
    @config.command(name="random_messages")
    async def config_random_messages(self, ctx: red_commands.Context, enabled: bool) -> None:
        """Enable/disable random messages when channels are idle"""
        await self.config.guild(ctx.guild).random_messages.set(enabled)
        status = "enabled" if enabled else "disabled"
        await ctx.send(f"✅ Random messages {status}")
        
        # Start/stop random message loops for all allowed channels
        if enabled:
            guild_data = await self.config.guild(ctx.guild).all()
            allowed_channels = guild_data.get("allowed_channels", [])
            if not allowed_channels:
                # If no specific channels, start for all accessible channels
                for channel in ctx.guild.text_channels:
                    if channel.permissions_for(ctx.guild.me).send_messages:
                        await self.dispatcher.start_random_message_loop(ctx.guild.id, channel.id)
            else:
                for channel_id in allowed_channels:
                    await self.dispatcher.start_random_message_loop(ctx.guild.id, channel_id)
        else:
            # Stop all random message loops for this guild
            for channel in ctx.guild.text_channels:
                await self.dispatcher.stop_random_message_loop(channel.id)
    
    @config.command(name="random_topics")
    async def config_random_topics(self, ctx: red_commands.Context, *, topics: str) -> None:
        """Set random message topics (comma-separated)"""
        topic_list = [topic.strip() for topic in topics.split(",")]
        if len(topic_list) > 20:
            await ctx.send("❌ Maximum 20 topics allowed")
            return
        
        await self.config.guild(ctx.guild).random_message_topics.set(topic_list)
        topics_display = "\n".join(f"• {topic}" for topic in topic_list[:10])
        if len(topic_list) > 10:
            topics_display += f"\n... and {len(topic_list) - 10} more"
        
        await ctx.send(f"✅ Random message topics updated:\n{topics_display}")
    
    # Channel-specific overrides
    @config.group(name="channel", invoke_without_command=True)
    async def config_channel(self, ctx: red_commands.Context) -> None:
        """Channel-specific configuration commands"""
        await ctx.send_help(ctx.command)
    
    @config_channel.command(name="response_percentage")
    async def channel_response_percentage(self, ctx: red_commands.Context, channel: discord.TextChannel, percentage: Optional[float] = None) -> None:
        """Set response percentage for a specific channel (None to use guild default)"""
        if percentage is not None and not 0 <= percentage <= 100:
            await ctx.send("❌ Percentage must be between 0 and 100")
            return
        
        async with self.config.guild(ctx.guild).all() as guild_data:
            if str(channel.id) not in guild_data["channel_overrides"]:
                guild_data["channel_overrides"][str(channel.id)] = {}
            
            guild_data["channel_overrides"][str(channel.id)]["response_percentage"] = percentage
        
        if percentage is None:
            await ctx.send(f"✅ {channel.mention} will use guild default response percentage")
        else:
            await ctx.send(f"✅ {channel.mention} response percentage set to `{percentage}%`")
    
    # Variable system commands
    @gpt5.group(name="variables", invoke_without_command=True)
    async def variables(self, ctx: red_commands.Context) -> None:
        """Dynamic variable system commands"""
        await ctx.send_help(ctx.command)
    
    @variables.command(name="list")
    async def variables_list(self, ctx: red_commands.Context) -> None:
        """List all available dynamic variables"""
        variables = variable_processor.get_available_variables()
        
        embed = discord.Embed(
            title="📝 Available Dynamic Variables",
            description="Variables can be used in system prompts and random message topics",
            color=0x00ff00
        )
        
        for var_name, description in variables.items():
            embed.add_field(
                name=f"{{{var_name}}}",
                value=description,
                inline=True
            )
        
        embed.add_field(
            name="Usage Examples",
            value="```\nHello {username}! Welcome to {servername}!\nToday is {date} and the time is {time}\nRandom number: {random}\n```",
            inline=False
        )
        
        await ctx.send(embed=embed)
    
    @variables.command(name="test")
    async def variables_test(self, ctx: red_commands.Context, *, text: str) -> None:
        """Test variable substitution in text"""
        if not variable_processor.has_variables(text):
            await ctx.send("❌ No variables found in the provided text.")
            return
        
        processed_text = await variable_processor.process_variables(
            text,
            bot=self.bot,
            guild=ctx.guild,
            channel=ctx.channel,
            user=ctx.author
        )
        
        embed = discord.Embed(title="🧪 Variable Test Results", color=0x00ff00)
        embed.add_field(name="Original", value=f"```{text[:1000]}```", inline=False)
        embed.add_field(name="Processed", value=f"```{processed_text[:1000]}```", inline=False)
        
        # Show which variables were found
        found_vars = variable_processor.extract_variables(text)
        if found_vars:
            embed.add_field(
                name="Variables Found",
                value=", ".join(f"`{{{var}}}`" for var in found_vars),
                inline=False
            )
        
        await ctx.send(embed=embed)
    
    # Conversation management commands
    @gpt5.command(name="forget")
    async def forget(self, ctx: red_commands.Context, channel: Optional[discord.TextChannel] = None) -> None:
        """Clear conversation history for current or specified channel"""
        target_channel = channel or ctx.channel
        
        cleared = await self.dispatcher.forget_conversation(target_channel)
        
        if cleared:
            await ctx.send(f"🧠 Conversation history cleared for {target_channel.mention}")
        else:
            await ctx.send(f"💭 No conversation history found for {target_channel.mention}")
    
    @gpt5.command(name="forgetall")
    @checks.admin_or_permissions(manage_guild=True)
    async def forget_all(self, ctx: red_commands.Context) -> None:
        """Clear all conversation histories in this server"""
        cleared_count = await self.dispatcher.forget_all_conversations(ctx.guild.id)
        
        if cleared_count > 0:
            await ctx.send(f"🧠 Cleared conversation history for {cleared_count} channels")
        else:
            await ctx.send("💭 No conversation histories found to clear")
    
    @config.command(name="conversation_timeframe")
    async def config_conversation_timeframe(self, ctx: red_commands.Context, seconds: int) -> None:
        """Set how long conversations are remembered (in seconds)"""
        if seconds < 60:
            await ctx.send("❌ Minimum timeframe is 60 seconds")
            return
        
        if seconds > 86400:  # 24 hours
            await ctx.send("❌ Maximum timeframe is 86400 seconds (24 hours)")
            return
        
        await self.config.guild(ctx.guild).conversation_timeframe.set(seconds)
        
        # Format time nicely
        if seconds >= 3600:
            time_str = f"{seconds // 3600} hours"
        elif seconds >= 60:
            time_str = f"{seconds // 60} minutes"
        else:
            time_str = f"{seconds} seconds"
        
        await ctx.send(f"🕐 Conversation timeframe set to {time_str}")
    
    @config.command(name="token_limit")
    async def config_token_limit(self, ctx: red_commands.Context, tokens: int) -> None:
        """Set maximum tokens for conversation context"""
        if tokens < 1000:
            await ctx.send("❌ Minimum token limit is 1000")
            return
        
        if tokens > 32000:
            await ctx.send("❌ Maximum token limit is 32000")
            return
        
        await self.config.guild(ctx.guild).token_limit.set(tokens)
        await ctx.send(f"🔢 Token limit set to {tokens:,} tokens")
    
    # Interaction commands
    @gpt5.command(name="ask")
    async def ask_command(self, ctx: red_commands.Context, *, prompt: str) -> None:
        """Ask the assistant a question"""
        await self.dispatcher.handle_slash_command(
            ctx.interaction or ctx,
            {"type": "ask", "prompt": prompt}
        )
    
    @gpt5.command(name="status")
    async def status_command(self, ctx: red_commands.Context) -> None:
        """Show assistant status and usage"""
        embed = discord.Embed(title="🤖 GPT-5 Assistant Status", color=0x00ff00)
        
        # API Status
        api_status = "✅ Connected" if self.dispatcher.openai_client else "❌ Not Connected"
        embed.add_field(name="API Status", value=api_status, inline=True)
        
        # Active requests
        active_requests = len(self.dispatcher._active_requests)
        embed.add_field(name="Active Requests", value=str(active_requests), inline=True)
        
        # Cache info
        if self.dispatcher.web_search_tool:
            cache_info = self.dispatcher.web_search_tool.get_usage_stats()
            embed.add_field(
                name="Web Search Cache",
                value=f"{cache_info['cached_queries']} entries\n{cache_info['cache_ttl_minutes']}min TTL",
                inline=True
            )
        
        # File search info
        if self.dispatcher.file_search_tool:
            kb_info = await self.dispatcher.file_search_tool.get_knowledge_base_info(ctx.guild.id)
            if kb_info["has_knowledge_base"]:
                embed.add_field(
                    name="Knowledge Base",
                    value=f"ID: `{kb_info['knowledge_base_id'][:16]}...`\nFiles: {kb_info['file_count']}",
                    inline=True
                )
        
        # Conversation stats
        conv_stats = await self.dispatcher.get_conversation_stats()
        embed.add_field(
            name="Conversations",
            value=f"Active: {conv_stats['active_conversations']}\nMessages: {conv_stats['total_cached_messages']}\nTokenizer: {'✅' if conv_stats['has_tokenizer'] else '❌'}",
            inline=True
        )
        
        await ctx.send(embed=embed)
    
    # Slash commands
    async def cog_app_command_error(self, interaction: discord.Interaction, error: Exception) -> None:
        logger.error(f"Slash command error: {error}", exc_info=True)
        
        message = "An error occurred while processing your request."
        if isinstance(error, commands.CommandError):
            message = str(error)
        
        try:
            if interaction.response.is_done():
                await interaction.followup.send(f"❌ {message}", ephemeral=True)
            else:
                await interaction.response.send_message(f"❌ {message}", ephemeral=True)
        except Exception:
            pass
    
    @red_commands.command(name="gpt5ask")
    async def ask_slash(self, ctx: red_commands.Context, *, prompt: str) -> None:
        """Ask GPT-5 a question"""
        # Convert to interaction format for dispatcher
        fake_interaction_data = {
            "type": "ask",
            "prompt": prompt
        }
        
        # Create a simple message-like interface
        class FakeMessage:
            def __init__(self, ctx):
                self.ctx = ctx
                self.clean_content = prompt
                self.channel = ctx.channel
                self.guild = ctx.guild
                self.author = ctx.author
                self.mentions = []
                self.attachments = []
            
            async def reply(self, content, **kwargs):
                return await self.ctx.send(content, **kwargs)
        
        fake_message = FakeMessage(ctx)
        await self.dispatcher.handle_message(fake_message)
    
    # Batch processing commands
    @gpt5.group(name="batch", invoke_without_command=True)
    async def batch(self, ctx: red_commands.Context) -> None:
        """Batch file processing commands"""
        await ctx.send_help(ctx.command)
    
    @batch.command(name="upload")
    async def batch_upload(self, ctx: red_commands.Context, summaries: bool = True, key_points: bool = True) -> None:
        """Upload and process multiple files with batch analysis
        
        Parameters:
        - summaries: Generate summaries for each file (default: True)
        - key_points: Extract key points from each file (default: True)
        
        Attach files to your message to process them.
        """
        if not ctx.message.attachments:
            await ctx.send("❌ No files attached. Please attach files to process.")
            return
        
        # Check file limits
        limits = self.dispatcher.batch_processor.get_batch_limits()
        
        if len(ctx.message.attachments) > limits["max_files"]:
            await ctx.send(f"❌ Too many files. Maximum: {limits['max_files']}")
            return
        
        total_size = sum(att.size for att in ctx.message.attachments)
        if total_size > limits["max_total_size"]:
            await ctx.send(f"❌ Total file size too large. Maximum: {self._format_bytes(limits['max_total_size'])}")
            return
        
        # Send processing message
        processing_msg = await ctx.send("🔄 Processing files... This may take a moment.")
        
        try:
            # Process options
            options = {
                "generate_summaries": summaries,
                "extract_key_points": key_points,
                "include_content": False
            }
            
            # Process the batch
            result = await self.dispatcher.batch_processor.process_batch(ctx.message.attachments, options)
            
            # Upload successful files to knowledge base
            successful_files = [
                f for f in result["processed_files"] 
                if f.processed and not f.error
            ]
            
            if successful_files:
                file_attachments = [
                    att for att in ctx.message.attachments
                    if any(att.filename == f.filename for f in successful_files)
                ]
                
                if file_attachments:
                    kb_result = await self.dispatcher.file_search_tool.upload_files(file_attachments, ctx.guild.id)
                    
                    # Update guild config with KB ID
                    async with self.config.guild(ctx.guild).all() as guild_data:
                        guild_data["file_search_kb_id"] = kb_result["knowledge_base_id"]
            
            # Create response embed
            embed = await self.dispatcher._create_batch_result_embed(result)
            
            await processing_msg.edit(content="", embed=embed)
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            await processing_msg.edit(content=f"❌ Batch processing failed: {str(e)}")
    
    @batch.command(name="info")
    async def batch_info(self, ctx: red_commands.Context) -> None:
        """Show information about batch processing capabilities"""
        
        supported_types = self.dispatcher.batch_processor.get_supported_file_types()
        limits = self.dispatcher.batch_processor.get_batch_limits()
        
        embed = discord.Embed(
            title="📁 Batch File Processing Information",
            description="Process multiple files at once with automatic summarization and analysis",
            color=0x3498db
        )
        
        # Supported file types
        for category, extensions in supported_types.items():
            embed.add_field(
                name=f"📄 {category.title()}",
                value=", ".join(f"`{ext}`" for ext in extensions[:10]) + 
                      ("..." if len(extensions) > 10 else ""),
                inline=True
            )
        
        # Processing limits
        embed.add_field(
            name="📊 Limits",
            value=f"Max files: {limits['max_files']}\n"
                  f"Max file size: {self._format_bytes(limits['max_file_size'])}\n"
                  f"Max total size: {self._format_bytes(limits['max_total_size'])}",
            inline=False
        )
        
        # Features
        embed.add_field(
            name="🔧 Features",
            value="• Automatic file type detection\n"
                  "• Individual file summaries\n"
                  "• Key point extraction\n"
                  "• Batch overview analysis\n"
                  "• Knowledge base integration\n"
                  "• Parallel processing",
            inline=False
        )
        
        embed.add_field(
            name="💡 Usage",
            value=f"`{ctx.prefix}gpt5 batch upload` - Upload and process files\n"
                  f"`{ctx.prefix}gpt5 batch upload false true` - Skip summaries, extract key points only",
            inline=False
        )
        
        await ctx.send(embed=embed)
    
    # Image analysis commands
    @gpt5.group(name="image", invoke_without_command=True)
    async def image(self, ctx: red_commands.Context) -> None:
        """Image analysis and processing commands"""
        await ctx.send_help(ctx.command)
    
    @image.command(name="analyze")
    async def image_analyze(self, ctx: red_commands.Context, *, custom_prompt: str = None) -> None:
        """Analyze an image using GPT-5 vision
        
        Attach an image to your message to analyze it.
        Optionally provide a custom analysis prompt.
        """
        if not ctx.message.attachments:
            await ctx.send("❌ No image attached. Please attach an image to analyze.")
            return
        
        # Find the first image attachment
        image_attachment = None
        for attachment in ctx.message.attachments:
            if self.dispatcher.image_tool.validate_image_attachment(attachment):
                image_attachment = attachment
                break
        
        if not image_attachment:
            await ctx.send("❌ No valid image found. Supported formats: PNG, JPEG, WebP")
            return
        
        # Send processing message
        processing_msg = await ctx.send("🔍 Analyzing image... This may take a moment.")
        
        try:
            # Analyze the image
            result = await self.dispatcher.image_tool.analyze_image(image_attachment, custom_prompt)
            
            if result["success"]:
                # Create response embed
                embed = discord.Embed(
                    title="🖼️ Image Analysis",
                    description=result["analysis"][:2000],  # Discord embed description limit
                    color=0x3498db
                )
                
                metadata = result["metadata"]
                embed.add_field(
                    name="📋 Image Details",
                    value=f"**File**: {metadata['filename']}\n"
                          f"**Size**: {metadata['size_formatted']}\n"
                          f"**Format**: {metadata['format']}\n"
                          f"**Type**: {metadata.get('content_type', 'Unknown')}",
                    inline=True
                )
                
                # Add thumbnail if possible
                if "url" in metadata:
                    embed.set_thumbnail(url=metadata["url"])
                
                embed.set_footer(text="Analyzed using GPT-5 Vision")
                
                await processing_msg.edit(content="", embed=embed)
            else:
                await processing_msg.edit(content=f"❌ Analysis failed: {result['analysis']}")
                
        except Exception as e:
            logger.error(f"Image analysis command error: {e}")
            await processing_msg.edit(content=f"❌ Analysis failed: {str(e)}")
    
    @image.command(name="compare")
    async def image_compare(self, ctx: red_commands.Context, *, comparison_prompt: str = None) -> None:
        """Compare two images using GPT-5 vision
        
        Attach exactly 2 images to your message to compare them.
        Optionally provide a custom comparison prompt.
        """
        if len(ctx.message.attachments) < 2:
            await ctx.send("❌ Please attach exactly 2 images to compare.")
            return
        
        # Find the first two image attachments
        image_attachments = []
        for attachment in ctx.message.attachments:
            if self.dispatcher.image_tool.validate_image_attachment(attachment):
                image_attachments.append(attachment)
                if len(image_attachments) == 2:
                    break
        
        if len(image_attachments) < 2:
            await ctx.send("❌ Please attach at least 2 valid images. Supported formats: PNG, JPEG, WebP")
            return
        
        # Send processing message
        processing_msg = await ctx.send("🔍 Comparing images... This may take a moment.")
        
        try:
            # Compare the images
            result = await self.dispatcher.image_tool.compare_images(
                image_attachments[0], 
                image_attachments[1], 
                comparison_prompt
            )
            
            if result["success"]:
                # Create response embed
                embed = discord.Embed(
                    title="🔍 Image Comparison",
                    description=result["comparison"][:2000],  # Discord embed description limit
                    color=0x9b59b6
                )
                
                # Add image details
                img1_meta = result["image1_metadata"]
                img2_meta = result["image2_metadata"]
                
                embed.add_field(
                    name="🖼️ Image 1",
                    value=f"**File**: {img1_meta['filename']}\n"
                          f"**Size**: {img1_meta['size']}\n"
                          f"**Format**: {img1_meta['format']}",
                    inline=True
                )
                
                embed.add_field(
                    name="🖼️ Image 2", 
                    value=f"**File**: {img2_meta['filename']}\n"
                          f"**Size**: {img2_meta['size']}\n"
                          f"**Format**: {img2_meta['format']}",
                    inline=True
                )
                
                embed.set_footer(text="Compared using GPT-5 Vision")
                
                await processing_msg.edit(content="", embed=embed)
            else:
                await processing_msg.edit(content=f"❌ Comparison failed: {result['comparison']}")
                
        except Exception as e:
            logger.error(f"Image comparison command error: {e}")
            await processing_msg.edit(content=f"❌ Comparison failed: {str(e)}")
    
    @image.command(name="info")
    async def image_info(self, ctx: red_commands.Context) -> None:
        """Show information about image analysis capabilities"""
        
        embed = discord.Embed(
            title="🖼️ Image Analysis & Processing",
            description="Analyze and process images using GPT-5 vision capabilities",
            color=0x3498db
        )
        
        # Supported formats
        embed.add_field(
            name="📄 Supported Formats",
            value="• PNG\n• JPEG/JPG\n• WebP\n• GIF\n• BMP",
            inline=True
        )
        
        # Analysis features
        embed.add_field(
            name="🔧 Analysis Features",
            value="• Object & scene detection\n• Text & symbol recognition\n• Style & composition analysis\n• Context & purpose identification\n• Mood & atmosphere assessment\n• Technical characteristics",
            inline=True
        )
        
        # Commands
        embed.add_field(
            name="💡 Commands",
            value=f"`{ctx.prefix}gpt5 image analyze` - Analyze a single image\n"
                  f"`{ctx.prefix}gpt5 image compare` - Compare two images\n"
                  f"`{ctx.prefix}gpt5 image analyze [custom prompt]` - Custom analysis\n"
                  f"`{ctx.prefix}gpt5 image compare [custom prompt]` - Custom comparison",
            inline=False
        )
        
        # Usage tips
        embed.add_field(
            name="💡 Usage Tips",
            value="• Attach images directly to your command message\n"
                  "• Use custom prompts for specific analysis needs\n"
                  "• Higher resolution images provide better analysis\n"
                  "• Analysis works best with clear, well-lit images",
            inline=False
        )
        
        await ctx.send(embed=embed)
    
    def _format_bytes(self, bytes_size: int) -> str:
        """Format bytes into human-readable size"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024
        return f"{bytes_size:.1f} TB"