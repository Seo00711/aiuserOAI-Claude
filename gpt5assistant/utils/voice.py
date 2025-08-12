import asyncio
import tempfile
import io
from pathlib import Path
from typing import Optional, List, Dict, Any
import discord
import aiofiles
import logging

logger = logging.getLogger("red.gpt5assistant.voice")


class VoiceProcessor:
    """Handles voice message transcription using OpenAI Whisper API"""
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
        self.supported_formats = {
            'audio/mpeg', 'audio/mp3', 'audio/wav', 'audio/m4a', 
            'audio/ogg', 'audio/webm', 'audio/flac'
        }
        self.max_file_size = 25 * 1024 * 1024  # 25MB limit for Whisper API
    
    def is_voice_message(self, message: discord.Message) -> bool:
        """Check if message contains voice/audio attachments"""
        if not message.attachments:
            return False
        
        return any(
            att.content_type and att.content_type.lower() in self.supported_formats
            for att in message.attachments
        )
    
    def get_voice_attachments(self, message: discord.Message) -> List[discord.Attachment]:
        """Get all voice/audio attachments from a message"""
        return [
            att for att in message.attachments
            if att.content_type and att.content_type.lower() in self.supported_formats
            and att.size <= self.max_file_size
        ]
    
    async def transcribe_voice_message(
        self, 
        attachment: discord.Attachment,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Transcribe a voice attachment using OpenAI Whisper"""
        
        if attachment.size > self.max_file_size:
            raise ValueError(f"File too large: {attachment.size} bytes (max: {self.max_file_size})")
        
        if not attachment.content_type or attachment.content_type.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported audio format: {attachment.content_type}")
        
        temp_files = []
        try:
            # Download the audio file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{attachment.filename}") as temp_file:
                await attachment.save(temp_file)
                audio_path = Path(temp_file.name)
                temp_files.append(audio_path)
            
            # Transcribe using OpenAI Whisper
            with open(audio_path, "rb") as audio_file:
                transcription_params = {
                    "file": (attachment.filename, audio_file, attachment.content_type),
                    "model": "whisper-1",
                    "response_format": "verbose_json"
                }
                
                if language:
                    transcription_params["language"] = language
                
                response = await self.openai_client.client.audio.transcriptions.create(**transcription_params)
            
            # Extract transcription data
            result = {
                "text": response.text,
                "language": getattr(response, 'language', 'unknown'),
                "duration": getattr(response, 'duration', 0),
                "filename": attachment.filename,
                "file_size": attachment.size,
                "format": attachment.content_type
            }
            
            # Add segments if available (for detailed analysis)
            if hasattr(response, 'segments') and response.segments:
                result["segments"] = [
                    {
                        "start": seg.get('start', 0),
                        "end": seg.get('end', 0),
                        "text": seg.get('text', '')
                    }
                    for seg in response.segments
                ]
            
            logger.info(f"Transcribed voice message: {attachment.filename} ({attachment.size} bytes)")
            return result
            
        except Exception as e:
            logger.error(f"Error transcribing voice message {attachment.filename}: {e}")
            raise
        
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    temp_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file {temp_file}: {e}")
    
    async def transcribe_multiple_attachments(
        self,
        attachments: List[discord.Attachment],
        language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Transcribe multiple voice attachments"""
        
        results = []
        
        for attachment in attachments:
            try:
                result = await self.transcribe_voice_message(attachment, language)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to transcribe {attachment.filename}: {e}")
                results.append({
                    "text": f"[Error transcribing {attachment.filename}: {str(e)}]",
                    "filename": attachment.filename,
                    "error": True
                })
        
        return results
    
    def format_transcription_for_chat(self, transcription_result: Dict[str, Any]) -> str:
        """Format transcription result for use in chat context"""
        
        if transcription_result.get("error"):
            return transcription_result["text"]
        
        text = transcription_result.get("text", "").strip()
        filename = transcription_result.get("filename", "audio")
        duration = transcription_result.get("duration", 0)
        
        if not text:
            return f"[Voice message: {filename} - no speech detected]"
        
        # Format with metadata
        metadata = f"[Voice message from {filename}"
        if duration > 0:
            metadata += f", {duration:.1f}s"
        metadata += "]"
        
        return f"{metadata}: {text}"
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported audio formats"""
        return list(self.supported_formats)
    
    def get_max_file_size(self) -> int:
        """Get maximum supported file size in bytes"""
        return self.max_file_size
    
    async def detect_language(self, text: str) -> Optional[str]:
        """Detect language of text (for future language-specific processing)"""
        # This could be enhanced with a proper language detection library
        # For now, we'll rely on Whisper's automatic language detection
        return None


class VoiceMessageHandler:
    """Handles voice message processing in Discord context"""
    
    def __init__(self, voice_processor: VoiceProcessor):
        self.voice_processor = voice_processor
    
    async def process_voice_message(
        self,
        message: discord.Message,
        language: Optional[str] = None
    ) -> Optional[str]:
        """Process voice message and return transcribed text for chat context"""
        
        if not self.voice_processor.is_voice_message(message):
            return None
        
        voice_attachments = self.voice_processor.get_voice_attachments(message)
        
        if not voice_attachments:
            return None
        
        try:
            # Transcribe all voice attachments
            transcriptions = await self.voice_processor.transcribe_multiple_attachments(
                voice_attachments, 
                language
            )
            
            # Format for chat context
            formatted_parts = []
            for transcription in transcriptions:
                formatted_text = self.voice_processor.format_transcription_for_chat(transcription)
                formatted_parts.append(formatted_text)
            
            return "\n".join(formatted_parts)
            
        except Exception as e:
            logger.error(f"Error processing voice message: {e}")
            return f"[Error processing voice message: {str(e)}]"
    
    async def create_transcription_embed(
        self,
        message: discord.Message,
        transcriptions: List[Dict[str, Any]]
    ) -> discord.Embed:
        """Create a Discord embed showing transcription results"""
        
        embed = discord.Embed(
            title="🎤 Voice Message Transcription",
            description=f"Transcribed {len(transcriptions)} voice message(s)",
            color=0x3498db
        )
        
        embed.set_author(
            name=message.author.display_name,
            icon_url=message.author.avatar.url if message.author.avatar else None
        )
        
        for i, transcription in enumerate(transcriptions):
            if transcription.get("error"):
                embed.add_field(
                    name=f"❌ {transcription['filename']}",
                    value=transcription["text"][:1000],
                    inline=False
                )
            else:
                filename = transcription.get("filename", f"Voice {i+1}")
                text = transcription.get("text", "No speech detected")
                duration = transcription.get("duration", 0)
                language = transcription.get("language", "unknown")
                
                field_name = f"🎵 {filename}"
                if duration > 0:
                    field_name += f" ({duration:.1f}s)"
                
                field_value = text[:1000]
                if len(text) > 1000:
                    field_value += "..."
                
                if language != "unknown":
                    field_value += f"\n*Language: {language}*"
                
                embed.add_field(
                    name=field_name,
                    value=field_value,
                    inline=False
                )
        
        embed.set_footer(text="Transcribed using OpenAI Whisper")
        return embed