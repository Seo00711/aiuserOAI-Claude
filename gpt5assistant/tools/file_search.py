import logging
import tempfile
import aiofiles
from typing import List, Dict, Any
from pathlib import Path
import discord

from ..openai_client import OpenAIClient, OpenAIClientError

logger = logging.getLogger("red.gpt5assistant.tools.file_search")


class FileSearchTool:
    def __init__(self, openai_client: OpenAIClient):
        self.client = openai_client
        self.supported_types = {
            '.txt', '.md', '.csv', '.tsv', '.json', '.xml', '.html', '.pdf', '.doc', '.docx',
            '.py', '.js', '.ts', '.java', '.cpp', '.c', '.cs', '.php', '.rb', '.go', '.rs',
            '.xlsx', '.xls', '.ppt', '.pptx', '.rtf', '.odt', '.ods', '.odp', '.pages', '.epub'
        }
    
    async def upload_files(
        self,
        attachments: List[discord.Attachment],
        guild_id: int
    ) -> Dict[str, Any]:
        if not attachments:
            raise ValueError("No attachments provided")
        
        temp_files = []
        valid_files = []
        
        try:
            for attachment in attachments:
                if not self._is_supported_file(attachment):
                    logger.warning(f"Skipping unsupported file: {attachment.filename}")
                    continue
                
                if attachment.size > 32 * 1024 * 1024:  # 32MB limit
                    logger.warning(f"File too large: {attachment.filename} ({attachment.size} bytes)")
                    continue
                
                # Download to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{attachment.filename}") as temp_file:
                    await attachment.save(temp_file)
                    file_path = Path(temp_file.name)
                    temp_files.append(file_path)
                    valid_files.append(file_path)
            
            if not valid_files:
                raise ValueError("No valid files to upload")
            
            kb_id = await self.client.upload_files_for_search(valid_files, guild_id)
            
            logger.info(f"Uploaded {len(valid_files)} files to knowledge base {kb_id} for guild {guild_id}")
            
            return {
                "knowledge_base_id": kb_id,
                "uploaded_files": [f.name.split('_', 1)[-1] for f in valid_files],
                "file_count": len(valid_files)
            }
        
        except OpenAIClientError as e:
            logger.error(f"File upload failed: {e}")
            raise
        
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    temp_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file {temp_file}: {e}")
    
    def _is_supported_file(self, attachment: discord.Attachment) -> bool:
        file_extension = Path(attachment.filename).suffix.lower()
        
        # Check by extension
        if file_extension in self.supported_types:
            return True
        
        # Check by content type
        if attachment.content_type:
            content_type = attachment.content_type.lower()
            
            # Text files
            if content_type.startswith('text/'):
                return True
            
            # Application files
            supported_apps = [
                'application/json',
                'application/xml',
                'application/pdf',
                'application/msword',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'application/vnd.ms-excel',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                'application/vnd.ms-powerpoint',
                'application/vnd.openxmlformats-officedocument.presentationml.presentation',
                'application/rtf',
                'application/vnd.oasis.opendocument.text',
                'application/vnd.oasis.opendocument.spreadsheet',
                'application/vnd.oasis.opendocument.presentation',
                'application/epub+zip'
            ]
            
            if any(content_type.startswith(app) for app in supported_apps):
                return True
        
        return False
    
    def get_supported_extensions(self) -> List[str]:
        return sorted(list(self.supported_types))
    
    def get_file_size_limit(self) -> int:
        return 32 * 1024 * 1024  # 32MB
    
    async def get_knowledge_base_info(self, guild_id: int) -> Dict[str, Any]:
        kb_id = self.client._kb_ids.get(guild_id)
        
        if not kb_id:
            return {
                "has_knowledge_base": False,
                "file_count": 0,
                "knowledge_base_id": None
            }
        
        try:
            # Note: This would require additional OpenAI API calls to get detailed info
            # For now, return basic info
            return {
                "has_knowledge_base": True,
                "knowledge_base_id": kb_id,
                "file_count": "unknown"  # Would need API call to get exact count
            }
        
        except Exception as e:
            logger.error(f"Failed to get knowledge base info: {e}")
            return {
                "has_knowledge_base": False,
                "file_count": 0,
                "knowledge_base_id": None,
                "error": str(e)
            }