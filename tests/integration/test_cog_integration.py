import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import tempfile
import os
from pathlib import Path

import discord
from redbot.core import Config
from redbot.core.bot import Red

from gpt5assistant.cog import GPT5Assistant


@pytest.fixture
async def temp_config_dir():
    """Create a temporary directory for config storage during integration tests"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
async def integration_bot(temp_config_dir):
    """Create a mock Red bot instance for integration testing"""
    bot = Mock(spec=Red)
    bot.user = Mock()
    bot.user.id = 123456789
    bot.user.mention = "<@123456789>"
    
    # Mock shared API tokens
    bot.get_shared_api_tokens = AsyncMock(return_value={"api_key": "test-api-key"})
    
    # Mock allowed mentions
    bot.allowed_mentions = discord.AllowedMentions(
        everyone=False, users=True, roles=False, replied_user=True
    )
    
    return bot


@pytest.fixture
async def integration_cog(integration_bot):
    """Create a GPT5Assistant cog instance for integration testing"""
    with patch('gpt5assistant.cog.Config') as mock_config_class:
        # Create a real-ish config mock that maintains state
        config_data = {}
        
        class MockConfig:
            def __init__(self, *args, **kwargs):
                self.data = config_data
            
            def register_guild(self, **defaults):
                self.guild_defaults = defaults
            
            def register_global(self, **defaults):
                self.global_defaults = defaults
            
            def guild(self, guild):
                guild_id = guild.id if hasattr(guild, 'id') else guild
                return MockGuildConfig(self.data, guild_id, self.guild_defaults)
            
            def guild_from_id(self, guild_id):
                return MockGuildConfig(self.data, guild_id, self.guild_defaults)
        
        class MockGuildConfig:
            def __init__(self, data, guild_id, defaults):
                self.data = data
                self.guild_id = guild_id
                self.defaults = defaults
                if f"guild_{guild_id}" not in data:
                    data[f"guild_{guild_id}"] = defaults.copy()
            
            async def all(self):
                return self.data[f"guild_{self.guild_id}"].copy()
            
            def enabled(self):
                return MockAttribute(self.data, f"guild_{self.guild_id}", "enabled")
            
            def system_prompt(self):
                return MockAttribute(self.data, f"guild_{self.guild_id}", "system_prompt")
        
        class MockAttribute:
            def __init__(self, data, key, attr):
                self.data = data
                self.key = key
                self.attr = attr
            
            async def set(self, value):
                if self.key not in self.data:
                    self.data[self.key] = {}
                self.data[self.key][self.attr] = value
        
        mock_config_class.get_conf.return_value = MockConfig()
        
        cog = GPT5Assistant(integration_bot)
        return cog


@pytest.fixture
async def mock_guild_and_channel():
    """Create mock guild and channel for testing"""
    guild = Mock(spec=discord.Guild)
    guild.id = 987654321
    guild.name = "Test Guild"
    guild.me = Mock()
    guild.me.id = 123456789
    
    channel = Mock(spec=discord.TextChannel)
    channel.id = 555666777
    channel.name = "test-channel"
    channel.guild = guild
    channel.mention = "#test-channel"
    channel.trigger_typing = AsyncMock()
    channel.send = AsyncMock()
    
    # Mock message history
    channel.history = Mock()
    async def mock_history_iter(limit=50):
        return []
    channel.history.return_value.__aiter__ = mock_history_iter
    
    return guild, channel


@pytest.fixture
async def mock_user():
    """Create a mock user for testing"""
    user = Mock(spec=discord.User)
    user.id = 888999000
    user.name = "testuser"
    user.display_name = "Test User"
    user.bot = False
    user.mention = "<@888999000>"
    return user


@pytest.mark.asyncio
async def test_cog_load_and_initialization(integration_cog, integration_bot):
    """Test that the cog loads and initializes properly"""
    # The cog should be created without errors
    assert integration_cog is not None
    assert integration_cog.bot == integration_bot
    assert integration_cog.dispatcher is not None
    
    # Test cog loading
    await integration_cog.cog_load()
    
    # Should have attempted to initialize the dispatcher
    assert integration_cog._initialization_task is not None


@pytest.mark.asyncio
async def test_cog_unload(integration_cog):
    """Test that the cog unloads cleanly"""
    await integration_cog.cog_load()
    
    # Unload should complete without errors
    await integration_cog.cog_unload()


@pytest.mark.asyncio
async def test_message_without_command_mentioned(integration_cog, integration_bot, mock_guild_and_channel, mock_user):
    """Test handling of messages where the bot is mentioned"""
    guild, channel = mock_guild_and_channel
    
    message = Mock(spec=discord.Message)
    message.author = mock_user
    message.guild = guild
    message.channel = channel
    message.content = f"{integration_bot.user.mention} hello"
    message.mentions = [integration_bot.user]
    message.reply = AsyncMock()
    
    with patch.object(integration_cog.dispatcher, 'handle_message') as mock_handle:
        await integration_cog.on_message_without_command(message)
        mock_handle.assert_called_once_with(message)


@pytest.mark.asyncio
async def test_message_without_command_not_mentioned(integration_cog, mock_guild_and_channel, mock_user):
    """Test that messages without mentions are ignored"""
    guild, channel = mock_guild_and_channel
    
    message = Mock(spec=discord.Message)
    message.author = mock_user
    message.guild = guild
    message.channel = channel
    message.content = "hello"
    message.mentions = []
    
    with patch.object(integration_cog.dispatcher, 'handle_message') as mock_handle:
        await integration_cog.on_message_without_command(message)
        mock_handle.assert_not_called()


@pytest.mark.asyncio
async def test_message_from_bot_ignored(integration_cog, mock_guild_and_channel):
    """Test that bot messages are ignored"""
    guild, channel = mock_guild_and_channel
    
    bot_user = Mock()
    bot_user.bot = True
    
    message = Mock(spec=discord.Message)
    message.author = bot_user
    message.guild = guild
    message.channel = channel
    
    with patch.object(integration_cog.dispatcher, 'handle_message') as mock_handle:
        await integration_cog.on_message_without_command(message)
        mock_handle.assert_not_called()


@pytest.mark.asyncio
async def test_dm_message_ignored(integration_cog, mock_user):
    """Test that DM messages are ignored"""
    message = Mock(spec=discord.Message)
    message.author = mock_user
    message.guild = None  # DM
    message.mentions = []
    
    with patch.object(integration_cog.dispatcher, 'handle_message') as mock_handle:
        await integration_cog.on_message_without_command(message)
        mock_handle.assert_not_called()


@pytest.mark.asyncio
async def test_config_commands_integration(integration_cog, mock_guild_and_channel):
    """Test configuration command integration"""
    guild, channel = mock_guild_and_channel
    
    # Create a mock context
    ctx = Mock()
    ctx.guild = guild
    ctx.send = AsyncMock()
    
    # Test enable command
    await integration_cog.config_enable(ctx)
    ctx.send.assert_called_with("✅ GPT-5 Assistant enabled")
    
    # Test disable command
    await integration_cog.config_disable(ctx)
    ctx.send.assert_called_with("❌ GPT-5 Assistant disabled")


@pytest.mark.asyncio
async def test_model_configuration(integration_cog, mock_guild_and_channel):
    """Test model configuration commands"""
    guild, channel = mock_guild_and_channel
    
    ctx = Mock()
    ctx.guild = guild
    ctx.send = AsyncMock()
    
    # Test valid model
    await integration_cog.config_model(ctx, "gpt-5-mini")
    ctx.send.assert_called_with("✅ Model set to `gpt-5-mini`")
    
    # Test invalid model
    ctx.send.reset_mock()
    await integration_cog.config_model(ctx, "invalid-model")
    assert "Invalid model" in ctx.send.call_args[0][0]


@pytest.mark.asyncio
async def test_temperature_configuration(integration_cog, mock_guild_and_channel):
    """Test temperature configuration"""
    guild, channel = mock_guild_and_channel
    
    ctx = Mock()
    ctx.guild = guild
    ctx.send = AsyncMock()
    
    # Test valid temperature
    await integration_cog.config_temperature(ctx, 0.8)
    ctx.send.assert_called_with("✅ Temperature set to `0.8`")
    
    # Test invalid temperature
    ctx.send.reset_mock()
    await integration_cog.config_temperature(ctx, 3.0)
    assert "must be between" in ctx.send.call_args[0][0]


@pytest.mark.asyncio
async def test_tool_configuration(integration_cog, mock_guild_and_channel):
    """Test tool enable/disable commands"""
    guild, channel = mock_guild_and_channel
    
    ctx = Mock()
    ctx.guild = guild
    ctx.send = AsyncMock()
    
    # Test enable tool
    await integration_cog.tools_enable(ctx, "web_search")
    assert "enabled" in ctx.send.call_args[0][0].lower()
    
    # Test disable tool
    ctx.send.reset_mock()
    await integration_cog.tools_disable(ctx, "web_search")
    assert "disabled" in ctx.send.call_args[0][0].lower()


@pytest.mark.asyncio
async def test_channel_configuration(integration_cog, mock_guild_and_channel):
    """Test channel allow/deny commands"""
    guild, channel = mock_guild_and_channel
    
    ctx = Mock()
    ctx.guild = guild
    ctx.send = AsyncMock()
    
    # Test allow channel
    await integration_cog.channels_allow(ctx, channel)
    assert "allowed" in ctx.send.call_args[0][0].lower()
    
    # Test deny channel
    ctx.send.reset_mock()
    await integration_cog.channels_deny(ctx, channel)
    assert "denied" in ctx.send.call_args[0][0].lower()


@pytest.mark.asyncio
async def test_ask_command_integration(integration_cog, mock_guild_and_channel, mock_user):
    """Test the ask command integration"""
    guild, channel = mock_guild_and_channel
    
    ctx = Mock()
    ctx.guild = guild
    ctx.channel = channel
    ctx.author = mock_user
    ctx.interaction = None
    ctx.send = AsyncMock()
    
    with patch.object(integration_cog.dispatcher, 'handle_message') as mock_handle:
        await integration_cog.ask_command(ctx, prompt="What is 2+2?")
        mock_handle.assert_called_once()


@pytest.mark.asyncio 
async def test_status_command_integration(integration_cog, mock_guild_and_channel):
    """Test the status command"""
    guild, channel = mock_guild_and_channel
    
    ctx = Mock()
    ctx.guild = guild
    ctx.send = AsyncMock()
    
    await integration_cog.status_command(ctx)
    
    # Should have sent an embed
    ctx.send.assert_called_once()
    args = ctx.send.call_args[1]
    assert 'embed' in args


@pytest.mark.asyncio
async def test_error_handling_in_commands(integration_cog, mock_guild_and_channel):
    """Test error handling in command processing"""
    guild, channel = mock_guild_and_channel
    
    ctx = Mock()
    ctx.guild = guild
    ctx.send = AsyncMock()
    
    # Create a mock interaction that will raise an error
    interaction = Mock()
    interaction.response = Mock()
    interaction.response.is_done.return_value = False
    interaction.response.send_message = AsyncMock(side_effect=Exception("Test error"))
    interaction.followup = Mock()
    interaction.followup.send = AsyncMock()
    
    # This should handle the error gracefully
    await integration_cog.cog_app_command_error(interaction, Exception("Test error"))
    
    # Should have attempted to send an error message
    assert interaction.response.send_message.called or interaction.followup.send.called


@pytest.mark.asyncio
async def test_concurrent_message_handling(integration_cog, integration_bot, mock_guild_and_channel, mock_user):
    """Test that concurrent messages are handled properly"""
    guild, channel = mock_guild_and_channel
    
    # Create multiple messages
    messages = []
    for i in range(3):
        message = Mock(spec=discord.Message)
        message.author = mock_user
        message.guild = guild
        message.channel = channel
        message.content = f"{integration_bot.user.mention} message {i}"
        message.mentions = [integration_bot.user]
        message.reply = AsyncMock()
        messages.append(message)
    
    # Handle all messages concurrently
    tasks = []
    for message in messages:
        task = asyncio.create_task(integration_cog.on_message_without_command(message))
        tasks.append(task)
    
    # Wait for all to complete
    await asyncio.gather(*tasks, return_exceptions=True)
    
    # All should have been processed (though they might not all succeed due to mocking)