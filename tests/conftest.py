import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, MagicMock, patch
import discord
from redbot.core import Config
from redbot.core.bot import Red

from gpt5assistant.config_schemas import GUILD_CONFIG_SCHEMA


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def mock_bot():
    bot = Mock(spec=Red)
    bot.user = Mock()
    bot.user.id = 123456789
    bot.get_shared_api_tokens = AsyncMock(return_value={"api_key": "test-api-key"})
    bot.allowed_mentions = discord.AllowedMentions(everyone=False, users=True, roles=False, replied_user=True)
    return bot


@pytest.fixture
async def mock_config():
    config = Mock(spec=Config)
    
    # Mock guild config
    guild_config_mock = Mock()
    guild_config_mock.all = AsyncMock(return_value=GUILD_CONFIG_SCHEMA)
    guild_config_mock.enabled = Mock()
    guild_config_mock.enabled.set = AsyncMock()
    guild_config_mock.system_prompt = Mock()
    guild_config_mock.system_prompt.set = AsyncMock()
    
    config.guild = Mock(return_value=guild_config_mock)
    config.guild_from_id = Mock(return_value=guild_config_mock)
    
    # Mock global config
    config.openai_api_key = AsyncMock(return_value="test-api-key")
    
    return config


@pytest.fixture
async def mock_guild():
    guild = Mock(spec=discord.Guild)
    guild.id = 987654321
    guild.name = "Test Guild"
    guild.me = Mock()
    guild.me.id = 123456789
    return guild


@pytest.fixture
async def mock_channel():
    channel = Mock(spec=discord.TextChannel)
    channel.id = 123123123
    channel.name = "test-channel"
    channel.mention = "#test-channel"
    channel.trigger_typing = AsyncMock()
    channel.send = AsyncMock()
    
    # Mock message history
    channel.history = Mock()
    async def mock_history(limit=50):
        # Return empty history for simplicity
        return []
    
    channel.history.return_value.__aiter__ = mock_history
    
    return channel


@pytest.fixture
async def mock_user():
    user = Mock(spec=discord.User)
    user.id = 555666777
    user.name = "testuser"
    user.display_name = "Test User"
    user.bot = False
    user.mention = "@testuser"
    return user


@pytest.fixture
async def mock_message(mock_guild, mock_channel, mock_user):
    message = Mock(spec=discord.Message)
    message.id = 111222333
    message.guild = mock_guild
    message.channel = mock_channel
    message.author = mock_user
    message.content = "Test message"
    message.clean_content = "Test message"
    message.mentions = []
    message.attachments = []
    message.embeds = []
    message.reply = AsyncMock()
    return message


@pytest.fixture
async def mock_interaction(mock_guild, mock_channel, mock_user):
    interaction = Mock(spec=discord.Interaction)
    interaction.guild_id = mock_guild.id
    interaction.guild = mock_guild
    interaction.channel_id = mock_channel.id
    interaction.channel = mock_channel
    interaction.user = mock_user
    
    # Mock response
    interaction.response = Mock()
    interaction.response.is_done = Mock(return_value=False)
    interaction.response.send_message = AsyncMock()
    interaction.response.defer = AsyncMock()
    
    # Mock followup
    interaction.followup = Mock()
    interaction.followup.send = AsyncMock()
    
    # Mock original_response
    interaction.original_response = AsyncMock(return_value=mock_message)
    
    return interaction


@pytest.fixture
async def mock_openai_client():
    with patch('gpt5assistant.openai_client.AsyncOpenAI') as mock_client_class:
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock responses.create
        mock_response = Mock()
        mock_response.content = [Mock(text="Test response chunk 1"), Mock(text=" chunk 2")]
        mock_client.responses = Mock()
        mock_client.responses.create = AsyncMock(return_value=mock_response)
        
        # Mock images.generate
        mock_image_response = Mock()
        mock_image_response.data = [Mock(url="https://example.com/image.png", revised_prompt="Test image")]
        mock_client.images = Mock()
        mock_client.images.generate = AsyncMock(return_value=mock_image_response)
        mock_client.images.edit = AsyncMock(return_value=mock_image_response)
        
        # Mock files.create
        mock_file_response = Mock()
        mock_file_response.id = "file-123"
        mock_client.files = Mock()
        mock_client.files.create = AsyncMock(return_value=mock_file_response)
        
        # Mock assistants
        mock_assistant = Mock()
        mock_assistant.id = "asst-123"
        mock_client.beta = Mock()
        mock_client.beta.assistants = Mock()
        mock_client.beta.assistants.create = AsyncMock(return_value=mock_assistant)
        mock_client.beta.assistants.files = Mock()
        mock_client.beta.assistants.files.create = AsyncMock()
        
        yield mock_client


@pytest.fixture
def mock_attachment():
    attachment = Mock(spec=discord.Attachment)
    attachment.id = 444555666
    attachment.filename = "test.png"
    attachment.size = 1024
    attachment.content_type = "image/png"
    attachment.url = "https://cdn.discordapp.com/attachments/123/456/test.png"
    attachment.save = AsyncMock()
    return attachment


@pytest.fixture
def sample_guild_config():
    return GUILD_CONFIG_SCHEMA.copy()


class MockAsyncIterator:
    def __init__(self, items):
        self.items = items
        self.index = 0
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item