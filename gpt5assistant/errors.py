import logging
from typing import Dict, Any, Optional
import discord
import httpx
from openai import OpenAIError, RateLimitError as OpenAIRateLimitError, APIConnectionError, AuthenticationError

logger = logging.getLogger("red.gpt5assistant.errors")


class GPT5AssistantError(Exception):
    """Base exception for GPT5Assistant errors"""
    def __init__(self, message: str, user_message: Optional[str] = None):
        super().__init__(message)
        self.user_message = user_message or message


class ConfigurationError(GPT5AssistantError):
    """Configuration-related errors"""
    pass


class APIError(GPT5AssistantError):
    """OpenAI API-related errors"""
    pass


class RateLimitError(APIError):
    """Rate limit exceeded"""
    pass


class QuotaExceededError(APIError):
    """API quota exceeded"""
    pass


class ContentPolicyError(APIError):
    """Content violates OpenAI policy"""
    pass


class ModelUnavailableError(APIError):
    """Requested model is unavailable"""
    pass


class FileTooLargeError(GPT5AssistantError):
    """File is too large for processing"""
    pass


class UnsupportedFileError(GPT5AssistantError):
    """File type is not supported"""
    pass


class ErrorHandler:
    """Handles and maps various errors to user-friendly messages"""
    
    def __init__(self):
        self.error_messages = {
            # OpenAI API errors
            "rate_limit_exceeded": "⏳ Too many requests. Please wait a moment and try again.",
            "quota_exceeded": "💳 API quota exceeded. Please contact server administrator.",
            "invalid_api_key": "🔑 Invalid API key. Please contact server administrator.",
            "content_policy": "🚫 Your request violates content policy. Please rephrase.",
            "model_unavailable": "🤖 Requested model is currently unavailable.",
            "timeout": "⏰ Request timed out. Please try again.",
            "connection_error": "🔌 Connection error. Please try again.",
            "invalid_request": "❌ Invalid request format.",
            "server_error": "🛠️ Server error. Please try again later.",
            
            # File errors
            "file_too_large": "📁 File is too large. Maximum size: 32MB.",
            "unsupported_file": "📄 Unsupported file type.",
            "upload_failed": "⬆️ File upload failed. Please try again.",
            
            # Image errors
            "image_generation_failed": "🎨 Image generation failed. Please try again.",
            "invalid_image_format": "🖼️ Invalid image format. Supported: PNG, JPEG, WebP.",
            "image_too_large": "🖼️ Image is too large for processing.",
            
            # General errors
            "unknown_error": "❌ An unexpected error occurred. Please try again.",
            "configuration_error": "⚙️ Configuration error. Please contact administrator.",
            "permission_denied": "🔒 Permission denied.",
            "feature_disabled": "🚫 This feature is disabled.",
        }
    
    def handle_openai_error(self, error: Exception) -> GPT5AssistantError:
        """Convert OpenAI errors to GPT5Assistant errors with user-friendly messages"""
        
        if isinstance(error, OpenAIRateLimitError):
            return RateLimitError(
                f"Rate limit exceeded: {error}",
                self.error_messages["rate_limit_exceeded"]
            )
        
        if isinstance(error, AuthenticationError):
            return ConfigurationError(
                f"Authentication failed: {error}",
                self.error_messages["invalid_api_key"]
            )
        
        if isinstance(error, APIConnectionError):
            return APIError(
                f"API connection error: {error}",
                self.error_messages["connection_error"]
            )
        
        if isinstance(error, OpenAIError):
            error_str = str(error).lower()
            
            # Check for specific error types in message
            if "quota" in error_str or "billing" in error_str:
                return QuotaExceededError(
                    f"Quota exceeded: {error}",
                    self.error_messages["quota_exceeded"]
                )
            
            if "content_policy" in error_str or "policy" in error_str:
                return ContentPolicyError(
                    f"Content policy violation: {error}",
                    self.error_messages["content_policy"]
                )
            
            if "model" in error_str and ("unavailable" in error_str or "not found" in error_str):
                return ModelUnavailableError(
                    f"Model unavailable: {error}",
                    self.error_messages["model_unavailable"]
                )
            
            if "timeout" in error_str:
                return APIError(
                    f"Timeout error: {error}",
                    self.error_messages["timeout"]
                )
            
            if "invalid" in error_str:
                return APIError(
                    f"Invalid request: {error}",
                    self.error_messages["invalid_request"]
                )
            
            # Generic OpenAI error
            return APIError(
                f"OpenAI API error: {error}",
                self.error_messages["server_error"]
            )
        
        # Not an OpenAI error, re-raise
        raise error
    
    def handle_http_error(self, error: httpx.HTTPStatusError) -> GPT5AssistantError:
        """Handle HTTP errors from external services"""
        
        status_code = error.response.status_code
        
        if status_code == 429:
            return RateLimitError(
                f"Rate limit exceeded: {error}",
                self.error_messages["rate_limit_exceeded"]
            )
        elif status_code == 401:
            return ConfigurationError(
                f"Authentication failed: {error}",
                self.error_messages["invalid_api_key"]
            )
        elif status_code == 403:
            return APIError(
                f"Permission denied: {error}",
                self.error_messages["permission_denied"]
            )
        elif status_code == 413:
            return FileTooLargeError(
                f"File too large: {error}",
                self.error_messages["file_too_large"]
            )
        elif 400 <= status_code < 500:
            return APIError(
                f"Client error: {error}",
                self.error_messages["invalid_request"]
            )
        elif 500 <= status_code < 600:
            return APIError(
                f"Server error: {error}",
                self.error_messages["server_error"]
            )
        else:
            return APIError(
                f"HTTP error: {error}",
                self.error_messages["unknown_error"]
            )
    
    def handle_file_error(self, error: Exception, file_size: Optional[int] = None) -> GPT5AssistantError:
        """Handle file-related errors"""
        
        if file_size and file_size > 32 * 1024 * 1024:
            return FileTooLargeError(
                f"File too large: {file_size} bytes",
                self.error_messages["file_too_large"]
            )
        
        error_str = str(error).lower()
        
        if "unsupported" in error_str or "invalid format" in error_str:
            return UnsupportedFileError(
                f"Unsupported file: {error}",
                self.error_messages["unsupported_file"]
            )
        
        if "upload" in error_str or "save" in error_str:
            return GPT5AssistantError(
                f"Upload failed: {error}",
                self.error_messages["upload_failed"]
            )
        
        return GPT5AssistantError(
            f"File error: {error}",
            self.error_messages["unknown_error"]
        )
    
    def handle_discord_error(self, error: discord.DiscordException) -> GPT5AssistantError:
        """Handle Discord API errors"""
        
        if isinstance(error, discord.Forbidden):
            return GPT5AssistantError(
                f"Discord permission error: {error}",
                "🔒 Bot lacks required permissions for this action."
            )
        
        if isinstance(error, discord.HTTPException):
            if error.status == 413:
                return GPT5AssistantError(
                    f"Message too large: {error}",
                    "📝 Response is too long for Discord."
                )
            
            return GPT5AssistantError(
                f"Discord API error: {error}",
                "🔌 Discord API error. Please try again."
            )
        
        return GPT5AssistantError(
            f"Discord error: {error}",
            self.error_messages["unknown_error"]
        )
    
    def get_user_message(self, error: Exception) -> str:
        """Get user-friendly error message"""
        
        if isinstance(error, GPT5AssistantError):
            return error.user_message or error.args[0]
        
        # Try to handle other error types
        try:
            if isinstance(error, OpenAIError):
                mapped_error = self.handle_openai_error(error)
                return mapped_error.user_message
            
            if isinstance(error, httpx.HTTPStatusError):
                mapped_error = self.handle_http_error(error)
                return mapped_error.user_message
            
            if isinstance(error, discord.DiscordException):
                mapped_error = self.handle_discord_error(error)
                return mapped_error.user_message
        
        except Exception:
            # If mapping fails, use generic message
            pass
        
        return self.error_messages["unknown_error"]
    
    async def send_error_message(
        self,
        target: discord.Message | discord.Interaction,
        error: Exception,
        ephemeral: bool = False
    ) -> None:
        """Send user-friendly error message to Discord"""
        
        user_message = self.get_user_message(error)
        
        # Log the actual error for debugging
        logger.error(f"Error in GPT5Assistant: {error}", exc_info=True)
        
        try:
            if isinstance(target, discord.Interaction):
                if not target.response.is_done():
                    await target.response.send_message(user_message, ephemeral=ephemeral)
                else:
                    await target.followup.send(user_message, ephemeral=ephemeral)
            else:
                await target.reply(user_message)
        
        except Exception as send_error:
            logger.error(f"Failed to send error message: {send_error}")


# Global error handler instance
error_handler = ErrorHandler()