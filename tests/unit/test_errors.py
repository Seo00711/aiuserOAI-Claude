import pytest
from unittest.mock import Mock, AsyncMock
import discord
import httpx
from openai import OpenAIError, RateLimitError as OpenAIRateLimitError, AuthenticationError

from gpt5assistant.errors import ErrorHandler, GPT5AssistantError, APIError, RateLimitError, error_handler


@pytest.fixture
def error_handler_instance():
    return ErrorHandler()


def test_handle_openai_rate_limit_error(error_handler_instance):
    openai_error = OpenAIRateLimitError("Rate limit exceeded for requests")
    
    result = error_handler_instance.handle_openai_error(openai_error)
    
    assert isinstance(result, RateLimitError)
    assert "Rate limit exceeded" in result.args[0]
    assert "Too many requests" in result.user_message


def test_handle_openai_auth_error(error_handler_instance):
    openai_error = AuthenticationError("Invalid API key")
    
    result = error_handler_instance.handle_openai_error(openai_error)
    
    assert isinstance(result, GPT5AssistantError)
    assert "Invalid API key" in result.user_message


def test_handle_openai_quota_error(error_handler_instance):
    openai_error = OpenAIError("You have exceeded your quota")
    
    result = error_handler_instance.handle_openai_error(openai_error)
    
    assert isinstance(result, GPT5AssistantError)
    assert "quota exceeded" in result.user_message.lower()


def test_handle_openai_content_policy_error(error_handler_instance):
    openai_error = OpenAIError("Your request violates our content_policy")
    
    result = error_handler_instance.handle_openai_error(openai_error)
    
    assert isinstance(result, GPT5AssistantError)
    assert "content policy" in result.user_message.lower()


def test_handle_http_429_error(error_handler_instance):
    response = Mock()
    response.status_code = 429
    http_error = httpx.HTTPStatusError("Rate limit", request=Mock(), response=response)
    
    result = error_handler_instance.handle_http_error(http_error)
    
    assert isinstance(result, RateLimitError)
    assert "Too many requests" in result.user_message


def test_handle_http_401_error(error_handler_instance):
    response = Mock()
    response.status_code = 401
    http_error = httpx.HTTPStatusError("Unauthorized", request=Mock(), response=response)
    
    result = error_handler_instance.handle_http_error(http_error)
    
    assert isinstance(result, GPT5AssistantError)
    assert "Invalid API key" in result.user_message


def test_handle_http_413_error(error_handler_instance):
    response = Mock()
    response.status_code = 413
    http_error = httpx.HTTPStatusError("Payload too large", request=Mock(), response=response)
    
    result = error_handler_instance.handle_http_error(http_error)
    
    assert isinstance(result, GPT5AssistantError)
    assert "too large" in result.user_message.lower()


def test_handle_file_error_large_file(error_handler_instance):
    file_size = 50 * 1024 * 1024  # 50MB
    
    result = error_handler_instance.handle_file_error(Exception("File error"), file_size)
    
    assert isinstance(result, GPT5AssistantError)
    assert "too large" in result.user_message.lower()


def test_handle_discord_forbidden_error(error_handler_instance):
    discord_error = discord.Forbidden(response=Mock(), message="Missing permissions")
    
    result = error_handler_instance.handle_discord_error(discord_error)
    
    assert isinstance(result, GPT5AssistantError)
    assert "permission" in result.user_message.lower()


def test_handle_discord_http_413_error(error_handler_instance):
    discord_error = discord.HTTPException(response=Mock(), message="Message too long")
    discord_error.status = 413
    
    result = error_handler_instance.handle_discord_error(discord_error)
    
    assert isinstance(result, GPT5AssistantError)
    assert "too long" in result.user_message.lower()


def test_get_user_message_gpt5_error(error_handler_instance):
    error = GPT5AssistantError("Internal error", "User friendly message")
    
    result = error_handler_instance.get_user_message(error)
    
    assert result == "User friendly message"


def test_get_user_message_openai_error(error_handler_instance):
    error = OpenAIRateLimitError("Rate limit exceeded")
    
    result = error_handler_instance.get_user_message(error)
    
    assert "Too many requests" in result


def test_get_user_message_unknown_error(error_handler_instance):
    error = ValueError("Some random error")
    
    result = error_handler_instance.get_user_message(error)
    
    assert "unexpected error" in result.lower()


@pytest.mark.asyncio
async def test_send_error_message_to_interaction(error_handler_instance, mock_interaction):
    error = GPT5AssistantError("Test error", "User message")
    
    await error_handler_instance.send_error_message(mock_interaction, error, ephemeral=True)
    
    mock_interaction.response.send_message.assert_called_once_with("User message", ephemeral=True)


@pytest.mark.asyncio
async def test_send_error_message_to_message(error_handler_instance, mock_message):
    error = GPT5AssistantError("Test error", "User message")
    
    await error_handler_instance.send_error_message(mock_message, error)
    
    mock_message.reply.assert_called_once_with("User message")


@pytest.mark.asyncio
async def test_send_error_message_interaction_done(error_handler_instance, mock_interaction):
    mock_interaction.response.is_done.return_value = True
    error = GPT5AssistantError("Test error", "User message")
    
    await error_handler_instance.send_error_message(mock_interaction, error)
    
    mock_interaction.followup.send.assert_called_once_with("User message", ephemeral=False)


def test_error_handler_singleton():
    # Test that error_handler is accessible as a module-level instance
    assert error_handler is not None
    assert isinstance(error_handler, ErrorHandler)


@pytest.mark.asyncio
async def test_send_error_message_failure_handling(error_handler_instance, mock_message):
    # Test that send_error_message handles failures gracefully
    mock_message.reply.side_effect = discord.HTTPException(response=Mock(), message="Failed")
    error = GPT5AssistantError("Test error", "User message")
    
    # Should not raise exception
    await error_handler_instance.send_error_message(mock_message, error)


def test_error_messages_completeness(error_handler_instance):
    # Test that all expected error message types are defined
    expected_keys = [
        "rate_limit_exceeded", "quota_exceeded", "invalid_api_key", "content_policy",
        "model_unavailable", "timeout", "connection_error", "invalid_request",
        "server_error", "file_too_large", "unsupported_file", "upload_failed",
        "image_generation_failed", "invalid_image_format", "image_too_large",
        "unknown_error", "configuration_error", "permission_denied", "feature_disabled"
    ]
    
    for key in expected_keys:
        assert key in error_handler_instance.error_messages
        assert len(error_handler_instance.error_messages[key]) > 0