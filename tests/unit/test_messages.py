import pytest
from unittest.mock import Mock, AsyncMock
import discord

from gpt5assistant.messages import MessageBuilder, ImageDetector, MessageDispatcher


@pytest.fixture
def message_builder():
    return MessageBuilder(max_history=5)


@pytest.fixture
def image_detector():
    return ImageDetector()


@pytest.fixture
def message_dispatcher():
    return MessageDispatcher()


@pytest.mark.asyncio
async def test_message_builder_simple_message(message_builder, mock_channel):
    mock_channel.history.return_value.__aiter__ = lambda: iter([])
    
    messages = await message_builder.build_message_list(
        mock_channel,
        "Hello AI",
        "You are a helpful assistant"
    )
    
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a helpful assistant"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Hello AI"


@pytest.mark.asyncio
async def test_message_builder_with_history(message_builder, mock_channel, mock_user):
    # Mock message history
    old_message = Mock(spec=discord.Message)
    old_message.author = mock_user
    old_message.clean_content = "Previous message"
    old_message.attachments = []
    old_message.embeds = []
    
    bot_message = Mock(spec=discord.Message)
    bot_user = Mock()
    bot_user.id = 123456789  # Bot's ID from conftest
    bot_message.author = Mock()
    bot_message.author.bot = True
    bot_message.author.id = 123456789
    bot_message.content = "Previous bot response"
    
    mock_channel.guild.me.id = 123456789
    mock_channel.history.return_value.__aiter__ = lambda: iter([bot_message, old_message])
    
    messages = await message_builder.build_message_list(
        mock_channel,
        "Current message",
        "System prompt"
    )
    
    assert len(messages) == 4  # system + old_user + old_bot + current
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    assert messages[3]["role"] == "user"
    assert messages[3]["content"] == "Current message"


def test_image_detector_positive_cases(image_detector):
    test_cases = [
        "generate an image of a cat",
        "create a logo for my company",
        "draw me a picture of a sunset",
        "make an illustration of a dragon",
        "can you generate a chart showing data?",
        "I need a picture of a mountain",
        "show me an image of a car"
    ]
    
    for case in test_cases:
        assert image_detector.is_image_request(case), f"Failed for: {case}"


def test_image_detector_negative_cases(image_detector):
    test_cases = [
        "tell me about cats",
        "what is the weather today?",
        "explain machine learning",
        "write a story",
        "help me with my code",
        "what's 2+2?",
        "how are you today?"
    ]
    
    for case in test_cases:
        assert not image_detector.is_image_request(case), f"Failed for: {case}"


def test_image_detector_extract_prompt(image_detector):
    test_cases = [
        ("generate an image of a cat", "a cat"),
        ("create a picture of a sunset over mountains", "a sunset over mountains"),
        ("can you make an illustration of a dragon?", "a dragon?"),
        ("draw a logo for my tech startup", "a logo for my tech startup")
    ]
    
    for input_text, expected in test_cases:
        result = image_detector.extract_image_prompt(input_text)
        assert expected in result.lower()


@pytest.mark.asyncio
async def test_message_dispatcher_image_classification(message_dispatcher, mock_message):
    mock_message.clean_content = "generate an image of a cat"
    
    result = await message_dispatcher.classify_and_route(
        mock_message,
        "You are helpful"
    )
    
    assert result["type"] == "image"
    assert "cat" in result["prompt"]
    assert result["attachments"] == []


@pytest.mark.asyncio
async def test_message_dispatcher_chat_classification(message_dispatcher, mock_message, mock_channel):
    mock_message.clean_content = "What is the weather today?"
    mock_message.channel = mock_channel
    mock_channel.history.return_value.__aiter__ = lambda: iter([])
    
    result = await message_dispatcher.classify_and_route(
        mock_message,
        "You are helpful"
    )
    
    assert result["type"] == "chat"
    assert "messages" in result
    assert len(result["messages"]) >= 2  # system + user


@pytest.mark.asyncio
async def test_message_dispatcher_image_with_attachment(message_dispatcher, mock_message, mock_attachment):
    mock_message.clean_content = "edit this image to be more colorful"
    mock_attachment.content_type = "image/png"
    mock_message.attachments = [mock_attachment]
    
    result = await message_dispatcher.classify_and_route(
        mock_message,
        "You are helpful"
    )
    
    assert result["type"] == "image"
    assert len(result["attachments"]) == 1
    assert result["attachments"][0] == mock_attachment


def test_clean_bot_message(message_builder):
    test_cases = [
        ("```python\nprint('hello')\n```", "print('hello')"),
        ("*thinking...*\nHere's the answer", "Here's the answer"),
        ("Normal response", "Normal response"),
        ("```\ncode\n", "code"),
    ]
    
    for input_text, expected in test_cases:
        result = message_builder._clean_bot_message(input_text)
        assert expected in result


def test_clean_user_message(message_builder, mock_message, mock_attachment):
    mock_message.clean_content = "Hello world"
    mock_message.attachments = [mock_attachment]
    mock_message.embeds = []
    
    result = message_builder._clean_user_message(mock_message)
    
    assert "Hello world" in result
    assert "[Image: test.png]" in result


@pytest.mark.asyncio
async def test_message_builder_history_limit(message_builder, mock_channel, mock_user):
    # Create more messages than the limit
    messages = []
    for i in range(10):
        msg = Mock(spec=discord.Message)
        msg.author = mock_user
        msg.clean_content = f"Message {i}"
        msg.attachments = []
        msg.embeds = []
        messages.append(msg)
    
    mock_channel.history.return_value.__aiter__ = lambda: iter(reversed(messages))
    
    result = await message_builder.build_message_list(
        mock_channel,
        "Current message",
        "System prompt"
    )
    
    # Should have system + max_history + current = 1 + 5 + 1 = 7
    assert len(result) <= 7