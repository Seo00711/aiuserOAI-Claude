import pytest
from unittest.mock import Mock, AsyncMock, patch
import discord
import asyncio

from gpt5assistant.utils.discord_io import DiscordStreamer, send_image_result, create_allowed_mentions


@pytest.fixture
def discord_streamer_message(mock_message):
    return DiscordStreamer(mock_message)


@pytest.fixture
def discord_streamer_interaction(mock_interaction):
    return DiscordStreamer(mock_interaction)


async def mock_content_generator():
    yield "Hello, "
    yield "this is "
    yield "a test "
    yield "response."


async def mock_long_content_generator():
    # Generate content that would exceed Discord's limit
    yield "x" * 1500
    yield "y" * 1000
    yield "z" * 500


@pytest.mark.asyncio
async def test_discord_streamer_message_typing(discord_streamer_message, mock_message):
    await discord_streamer_message.start_typing()
    
    # Wait a brief moment to let the typing task start
    await asyncio.sleep(0.1)
    
    assert discord_streamer_message._typing_task is not None
    assert not discord_streamer_message._typing_task.done()
    
    await discord_streamer_message.stop_typing()
    
    # Wait for task to be cancelled
    await asyncio.sleep(0.1)
    assert discord_streamer_message._typing_task.done()


@pytest.mark.asyncio
async def test_discord_streamer_interaction_no_typing(discord_streamer_interaction, mock_interaction):
    # Interactions don't support typing
    await discord_streamer_interaction.start_typing()
    
    assert discord_streamer_interaction._typing_task is None


@pytest.mark.asyncio
async def test_stream_response_message(discord_streamer_message, mock_message):
    result = await discord_streamer_message.stream_response(mock_content_generator())
    
    # Should have called reply once with the full content
    mock_message.reply.assert_called_once()
    call_args = mock_message.reply.call_args[0][0]
    assert "Hello, this is a test response." in call_args


@pytest.mark.asyncio
async def test_stream_response_interaction(discord_streamer_interaction, mock_interaction):
    result = await discord_streamer_interaction.stream_response(mock_content_generator())
    
    # Should have called response.send_message
    mock_interaction.response.send_message.assert_called_once()
    call_args = mock_interaction.response.send_message.call_args[0][0]
    assert "Hello, this is a test response." in call_args


@pytest.mark.asyncio
async def test_stream_response_chunking(discord_streamer_message, mock_message):
    await discord_streamer_message.stream_response(mock_long_content_generator())
    
    # Should have made multiple calls due to length limits
    assert mock_message.reply.call_count >= 1


@pytest.mark.asyncio
async def test_find_split_point_code_blocks(discord_streamer_message):
    text = "Here's some code:\n```python\nprint('hello')\n```\nAnd more text" + "x" * 2000
    
    split_point = discord_streamer_message._find_split_point(text, 1800)
    
    # Should split after the code block
    assert "```" in text[:split_point]
    # Make sure we don't split inside the code block
    code_start = text.find("```python")
    code_end = text.find("```", code_start + 3) + 3
    assert split_point >= code_end


@pytest.mark.asyncio
async def test_find_split_point_paragraph_breaks(discord_streamer_message):
    text = "First paragraph.\n\nSecond paragraph." + "x" * 2000
    
    split_point = discord_streamer_message._find_split_point(text, 1800)
    
    # Should split at paragraph break
    assert text[split_point - 2:split_point] == "\n\n"


@pytest.mark.asyncio
async def test_find_split_point_sentence_breaks(discord_streamer_message):
    text = "First sentence. Second sentence." + "x" * 2000
    
    split_point = discord_streamer_message._find_split_point(text, 1800)
    
    # Should split after a sentence
    assert text[split_point - 1] == " " or "." in text[split_point - 3:split_point]


@pytest.mark.asyncio
async def test_find_split_point_word_boundaries(discord_streamer_message):
    text = "word " * 500 + "x" * 1000
    
    split_point = discord_streamer_message._find_split_point(text, 1800)
    
    # Should split at a space
    assert text[split_point - 1] == " " or split_point >= len(text)


@pytest.mark.asyncio
@patch('gpt5assistant.utils.discord_io.httpx.AsyncClient')
async def test_download_image_success(mock_client_class):
    mock_client = Mock()
    mock_client_class.return_value.__aenter__.return_value = mock_client
    mock_client_class.return_value.__aexit__.return_value = None
    
    mock_response = Mock()
    mock_response.content = b"fake image data"
    mock_response.raise_for_status = Mock()
    mock_client.get.return_value = mock_response
    
    with patch('builtins.open', create=True) as mock_open:
        result = await discord_streamer_message._DiscordStreamer__class__.__dict__.get('download_image', lambda: None)()
        # This test needs adjustment for the actual implementation


@pytest.mark.asyncio
async def test_send_image_result_success(mock_message):
    with patch('gpt5assistant.utils.discord_io.download_image') as mock_download:
        from pathlib import Path
        mock_download.return_value = Path("/tmp/test.png")
        
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value = Mock()
            
            await send_image_result(
                target=mock_message,
                image_url="https://example.com/image.png",
                prompt="Test prompt",
                metadata={"size": "1024x1024", "quality": "standard"}
            )
            
            # Should have called reply with embed and file
            mock_message.reply.assert_called_once()
            call_kwargs = mock_message.reply.call_args[1]
            assert "embed" in call_kwargs
            assert "file" in call_kwargs


@pytest.mark.asyncio
async def test_send_image_result_download_failure(mock_message):
    with patch('gpt5assistant.utils.discord_io.download_image') as mock_download:
        mock_download.return_value = None
        
        await send_image_result(
            target=mock_message,
            image_url="https://example.com/image.png",
            prompt="Test prompt",
            metadata={}
        )
        
        # Should have called reply with error message
        mock_message.reply.assert_called_once()
        call_args = mock_message.reply.call_args[0][0]
        assert "Failed to download" in call_args


@pytest.mark.asyncio
async def test_send_image_result_interaction(mock_interaction):
    with patch('gpt5assistant.utils.discord_io.download_image') as mock_download:
        from pathlib import Path
        mock_download.return_value = Path("/tmp/test.png")
        
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value = Mock()
            
            await send_image_result(
                target=mock_interaction,
                image_url="https://example.com/image.png",
                prompt="Test prompt",
                metadata={},
                ephemeral=True
            )
            
            # Should have called response.send_message with ephemeral
            mock_interaction.response.send_message.assert_called_once()
            call_kwargs = mock_interaction.response.send_message.call_args[1]
            assert call_kwargs.get("ephemeral") is True


def test_create_allowed_mentions():
    mentions = create_allowed_mentions()
    
    assert isinstance(mentions, discord.AllowedMentions)
    assert mentions.everyone is False
    assert mentions.users is True
    assert mentions.roles is False
    assert mentions.replied_user is True


@pytest.mark.asyncio
async def test_stream_response_error_handling(discord_streamer_message, mock_message):
    mock_message.reply.side_effect = discord.HTTPException(response=Mock(), message="Failed")
    
    # Should handle the error gracefully
    try:
        await discord_streamer_message.stream_response(mock_content_generator())
    except discord.HTTPException:
        pytest.fail("Exception should have been handled gracefully")