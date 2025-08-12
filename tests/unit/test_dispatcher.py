import pytest
from unittest.mock import Mock, AsyncMock, patch
import asyncio

from gpt5assistant.dispatcher import GPTDispatcher
from gpt5assistant.config_schemas import GuildConfig, ChannelConfig
from gpt5assistant.errors import GPT5AssistantError


@pytest.fixture
async def gpt_dispatcher(mock_config):
    dispatcher = GPTDispatcher(mock_config)
    await dispatcher.initialize("test-api-key")
    return dispatcher


@pytest.mark.asyncio
async def test_dispatcher_initialization(mock_config):
    dispatcher = GPTDispatcher(mock_config)
    
    assert dispatcher.openai_client is None
    assert dispatcher.image_tool is None
    assert dispatcher.file_search_tool is None
    assert dispatcher.web_search_tool is not None
    assert dispatcher.code_interpreter_tool is not None
    
    await dispatcher.initialize("test-api-key")
    
    assert dispatcher.openai_client is not None
    assert dispatcher.image_tool is not None
    assert dispatcher.file_search_tool is not None


@pytest.mark.asyncio
async def test_dispatcher_shutdown(gpt_dispatcher):
    # Add a fake active request
    mock_task = Mock()
    mock_task.done.return_value = False
    mock_task.cancel = Mock()
    gpt_dispatcher._active_requests[123] = mock_task
    
    await gpt_dispatcher.shutdown()
    
    mock_task.cancel.assert_called_once()


@pytest.mark.asyncio
async def test_handle_message_no_client(mock_config, mock_message):
    dispatcher = GPTDispatcher(mock_config)  # Not initialized
    
    await dispatcher.handle_message(mock_message)
    
    mock_message.reply.assert_called_once_with("❌ OpenAI client not initialized. Please set API key.")


@pytest.mark.asyncio
async def test_handle_message_no_guild(gpt_dispatcher, mock_message):
    mock_message.guild = None
    
    await gpt_dispatcher.handle_message(mock_message)
    
    # Should not process DM messages
    mock_message.reply.assert_called_once_with("❌ This cog only works in servers.")


@pytest.mark.asyncio
async def test_handle_message_chat_request(gpt_dispatcher, mock_message, mock_config):
    # Mock the config to return enabled guild
    guild_config_data = {
        "enabled": True,
        "model": {
            "name": "gpt-5",
            "temperature": 0.7,
            "reasoning": {"effort": "medium"},
            "text": {"verbosity": "medium"}
        },
        "tools": {
            "web_search": True,
            "file_search": True,
            "code_interpreter": True,
            "image": True
        },
        "system_prompt": "You are helpful",
        "allowed_channels": [],
        "denied_channels": [],
        "channel_overrides": {},
        "ephemeral_responses": False,
        "max_message_history": 10,
        "cooldown_seconds": 0,
        "file_search_kb_id": None
    }
    
    mock_config.guild_from_id.return_value.all.return_value = guild_config_data
    mock_message.clean_content = "Hello, how are you?"
    
    with patch.object(gpt_dispatcher.openai_client, 'respond_chat') as mock_respond:
        async def mock_generator():
            yield "I'm fine, thank you!"
        
        mock_respond.return_value = mock_generator()
        
        await gpt_dispatcher.handle_message(mock_message)
        
        # Should have called the OpenAI client
        mock_respond.assert_called_once()


@pytest.mark.asyncio
async def test_handle_message_image_request(gpt_dispatcher, mock_message, mock_config):
    # Mock the config
    guild_config_data = {
        "enabled": True,
        "model": {
            "name": "gpt-5",
            "temperature": 0.7,
            "reasoning": {"effort": "medium"},
            "text": {"verbosity": "medium"}
        },
        "tools": {
            "web_search": True,
            "file_search": True,
            "code_interpreter": True,
            "image": True
        },
        "system_prompt": "You are helpful",
        "allowed_channels": [],
        "denied_channels": [],
        "channel_overrides": {},
        "ephemeral_responses": False,
        "max_message_history": 10,
        "cooldown_seconds": 0,
        "file_search_kb_id": None
    }
    
    mock_config.guild_from_id.return_value.all.return_value = guild_config_data
    mock_message.clean_content = "generate an image of a cat"
    
    with patch.object(gpt_dispatcher.image_tool, 'generate_image') as mock_generate:
        mock_generate.return_value = {
            "url": "https://example.com/cat.png",
            "revised_prompt": "A cute cat",
            "size": "1024x1024"
        }
        
        with patch('gpt5assistant.dispatcher.send_image_result') as mock_send:
            await gpt_dispatcher.handle_message(mock_message)
            
            # Should have generated an image
            mock_generate.assert_called_once()
            mock_send.assert_called_once()


@pytest.mark.asyncio
async def test_handle_message_disabled_guild(gpt_dispatcher, mock_message, mock_config):
    # Mock disabled guild config
    guild_config_data = {
        "enabled": False,
        "model": {"name": "gpt-5", "temperature": 0.7, "reasoning": {"effort": "medium"}, "text": {"verbosity": "medium"}},
        "tools": {"web_search": True, "file_search": True, "code_interpreter": True, "image": True},
        "system_prompt": "You are helpful",
        "allowed_channels": [],
        "denied_channels": [],
        "channel_overrides": {},
        "ephemeral_responses": False,
        "max_message_history": 10,
        "cooldown_seconds": 0,
        "file_search_kb_id": None
    }
    
    mock_config.guild_from_id.return_value.all.return_value = guild_config_data
    
    await gpt_dispatcher.handle_message(mock_message)
    
    # Should not have responded (guild disabled)
    mock_message.reply.assert_not_called()


@pytest.mark.asyncio
async def test_handle_message_denied_channel(gpt_dispatcher, mock_message, mock_config):
    # Mock guild config with denied channel
    guild_config_data = {
        "enabled": True,
        "model": {"name": "gpt-5", "temperature": 0.7, "reasoning": {"effort": "medium"}, "text": {"verbosity": "medium"}},
        "tools": {"web_search": True, "file_search": True, "code_interpreter": True, "image": True},
        "system_prompt": "You are helpful",
        "allowed_channels": [],
        "denied_channels": [mock_message.channel.id],  # Deny this channel
        "channel_overrides": {},
        "ephemeral_responses": False,
        "max_message_history": 10,
        "cooldown_seconds": 0,
        "file_search_kb_id": None
    }
    
    mock_config.guild_from_id.return_value.all.return_value = guild_config_data
    
    await gpt_dispatcher.handle_message(mock_message)
    
    # Should not have responded (channel denied)
    mock_message.reply.assert_not_called()


@pytest.mark.asyncio
async def test_handle_slash_command_ask(gpt_dispatcher, mock_interaction, mock_config):
    guild_config_data = {
        "enabled": True,
        "model": {"name": "gpt-5", "temperature": 0.7, "reasoning": {"effort": "medium"}, "text": {"verbosity": "medium"}},
        "tools": {"web_search": True, "file_search": True, "code_interpreter": True, "image": True},
        "system_prompt": "You are helpful",
        "allowed_channels": [],
        "denied_channels": [],
        "channel_overrides": {},
        "ephemeral_responses": False,
        "max_message_history": 10,
        "cooldown_seconds": 0,
        "file_search_kb_id": None
    }
    
    mock_config.guild_from_id.return_value.all.return_value = guild_config_data
    
    command_data = {
        "type": "ask",
        "prompt": "What is 2+2?"
    }
    
    with patch.object(gpt_dispatcher.openai_client, 'respond_chat') as mock_respond:
        async def mock_generator():
            yield "2+2 equals 4"
        
        mock_respond.return_value = mock_generator()
        
        await gpt_dispatcher.handle_slash_command(mock_interaction, command_data)
        
        mock_respond.assert_called_once()


@pytest.mark.asyncio
async def test_handle_slash_command_image(gpt_dispatcher, mock_interaction, mock_config):
    guild_config_data = {
        "enabled": True,
        "model": {"name": "gpt-5", "temperature": 0.7, "reasoning": {"effort": "medium"}, "text": {"verbosity": "medium"}},
        "tools": {"web_search": True, "file_search": True, "code_interpreter": True, "image": True},
        "system_prompt": "You are helpful",
        "allowed_channels": [],
        "denied_channels": [],
        "channel_overrides": {},
        "ephemeral_responses": False,
        "max_message_history": 10,
        "cooldown_seconds": 0,
        "file_search_kb_id": None
    }
    
    mock_config.guild_from_id.return_value.all.return_value = guild_config_data
    
    command_data = {
        "type": "image",
        "prompt": "A beautiful sunset"
    }
    
    with patch.object(gpt_dispatcher.image_tool, 'generate_image') as mock_generate:
        mock_generate.return_value = {
            "url": "https://example.com/sunset.png",
            "revised_prompt": "A beautiful sunset",
            "size": "1024x1024"
        }
        
        with patch('gpt5assistant.dispatcher.send_image_result') as mock_send:
            await gpt_dispatcher.handle_slash_command(mock_interaction, command_data)
            
            mock_generate.assert_called_once()
            mock_send.assert_called_once()


@pytest.mark.asyncio
async def test_handle_slash_command_upload(gpt_dispatcher, mock_interaction, mock_config, mock_attachment):
    guild_config_data = {
        "enabled": True,
        "model": {"name": "gpt-5", "temperature": 0.7, "reasoning": {"effort": "medium"}, "text": {"verbosity": "medium"}},
        "tools": {"web_search": True, "file_search": True, "code_interpreter": True, "image": True},
        "system_prompt": "You are helpful",
        "allowed_channels": [],
        "denied_channels": [],
        "channel_overrides": {},
        "ephemeral_responses": False,
        "max_message_history": 10,
        "cooldown_seconds": 0,
        "file_search_kb_id": None
    }
    
    mock_config.guild_from_id.return_value.all.return_value = guild_config_data
    
    command_data = {
        "type": "upload",
        "files": [mock_attachment]
    }
    
    with patch.object(gpt_dispatcher.file_search_tool, 'upload_files') as mock_upload:
        mock_upload.return_value = {
            "knowledge_base_id": "kb-123",
            "uploaded_files": ["test.png"],
            "file_count": 1
        }
        
        await gpt_dispatcher.handle_slash_command(mock_interaction, command_data)
        
        mock_upload.assert_called_once()
        mock_interaction.followup.send.assert_called_once()


@pytest.mark.asyncio
async def test_channel_locking(gpt_dispatcher, mock_message, mock_config):
    # Test that concurrent requests to the same channel are handled properly
    guild_config_data = {
        "enabled": True,
        "model": {"name": "gpt-5", "temperature": 0.7, "reasoning": {"effort": "medium"}, "text": {"verbosity": "medium"}},
        "tools": {"web_search": True, "file_search": True, "code_interpreter": True, "image": True},
        "system_prompt": "You are helpful",
        "allowed_channels": [],
        "denied_channels": [],
        "channel_overrides": {},
        "ephemeral_responses": False,
        "max_message_history": 10,
        "cooldown_seconds": 0,
        "file_search_kb_id": None
    }
    
    mock_config.guild_from_id.return_value.all.return_value = guild_config_data
    
    # Start first request
    task1 = asyncio.create_task(gpt_dispatcher.handle_message(mock_message))
    await asyncio.sleep(0.01)  # Let it start
    
    # Start second request - should cancel the first
    task2 = asyncio.create_task(gpt_dispatcher.handle_message(mock_message))
    
    # Wait for both to complete
    await asyncio.gather(task1, task2, return_exceptions=True)
    
    # Check that a lock was created for the channel
    assert mock_message.channel.id in gpt_dispatcher._channel_locks


@pytest.mark.asyncio
async def test_error_handling_in_message_processing(gpt_dispatcher, mock_message, mock_config):
    guild_config_data = {
        "enabled": True,
        "model": {"name": "gpt-5", "temperature": 0.7, "reasoning": {"effort": "medium"}, "text": {"verbosity": "medium"}},
        "tools": {"web_search": True, "file_search": True, "code_interpreter": True, "image": True},
        "system_prompt": "You are helpful",
        "allowed_channels": [],
        "denied_channels": [],
        "channel_overrides": {},
        "ephemeral_responses": False,
        "max_message_history": 10,
        "cooldown_seconds": 0,
        "file_search_kb_id": None
    }
    
    mock_config.guild_from_id.return_value.all.return_value = guild_config_data
    
    # Mock an error in the OpenAI client
    with patch.object(gpt_dispatcher.openai_client, 'respond_chat') as mock_respond:
        mock_respond.side_effect = GPT5AssistantError("Test error", "User error message")
        
        await gpt_dispatcher.handle_message(mock_message)
        
        # Should have handled the error gracefully - error handler will send message
        # The specific assertion depends on how error_handler.send_error_message is mocked


def test_is_channel_allowed(gpt_dispatcher):
    guild_config = GuildConfig(
        enabled=True,
        allowed_channels=[123, 456],
        denied_channels=[789]
    )
    
    # Allowed channel
    assert gpt_dispatcher._is_channel_allowed(guild_config, None, 123)
    
    # Denied channel
    assert not gpt_dispatcher._is_channel_allowed(guild_config, None, 789)
    
    # Channel not in allowed list
    assert not gpt_dispatcher._is_channel_allowed(guild_config, None, 999)
    
    # Disabled guild
    guild_config.enabled = False
    assert not gpt_dispatcher._is_channel_allowed(guild_config, None, 123)


def test_effective_config_resolution(gpt_dispatcher):
    guild_config = GuildConfig(system_prompt="Guild prompt")
    channel_config = ChannelConfig(system_prompt="Channel prompt")
    
    # Channel config should override guild config
    assert gpt_dispatcher._get_effective_system_prompt(guild_config, channel_config) == "Channel prompt"
    
    # Without channel config, should use guild config
    assert gpt_dispatcher._get_effective_system_prompt(guild_config, None) == "Guild prompt"