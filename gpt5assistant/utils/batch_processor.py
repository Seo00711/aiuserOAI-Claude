import asyncio
import tempfile
import mimetypes
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import discord
import aiofiles
import logging

logger = logging.getLogger("red.gpt5assistant.batch_processor")


class FileMetadata:
    """Metadata for processed files"""
    
    def __init__(
        self,
        filename: str,
        file_type: str,
        size: int,
        content_type: Optional[str] = None,
        processed: bool = False,
        error: Optional[str] = None
    ):
        self.filename = filename
        self.file_type = file_type
        self.size = size
        self.content_type = content_type
        self.processed = processed
        self.error = error
        self.summary: Optional[str] = None
        self.key_points: List[str] = []
        self.word_count: Optional[int] = None


class BatchFileProcessor:
    """Handles batch processing of multiple file uploads"""
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
        
        # Supported file types for different processing methods
        self.text_types = {
            '.txt', '.md', '.csv', '.json', '.xml', '.html', '.py', '.js', 
            '.ts', '.java', '.cpp', '.c', '.cs', '.php', '.rb', '.go', '.rs'
        }
        
        self.document_types = {
            '.pdf', '.doc', '.docx'
        }
        
        self.spreadsheet_types = {
            '.xls', '.xlsx', '.csv', '.tsv', '.ods'
        }
        
        self.presentation_types = {
            '.ppt', '.pptx', '.odp'
        }
        
        self.archive_types = {
            '.zip', '.tar', '.gz', '.rar', '.7z'
        }
        
        self.video_types = {
            '.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv'
        }
        
        self.advanced_document_types = {
            '.rtf', '.odt', '.pages', '.epub'
        }
        
        self.image_types = {
            '.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'
        }
        
        # File size limits (in bytes)
        self.max_file_size = 100 * 1024 * 1024  # 100MB
        self.max_batch_size = 50  # Maximum files per batch
        self.max_total_size = 500 * 1024 * 1024  # 500MB total per batch
    
    async def process_batch(
        self,
        attachments: List[discord.Attachment],
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a batch of file attachments"""
        
        if not attachments:
            raise ValueError("No attachments provided")
        
        if len(attachments) > self.max_batch_size:
            raise ValueError(f"Too many files: {len(attachments)} (max: {self.max_batch_size})")
        
        # Validate total size
        total_size = sum(att.size for att in attachments)
        if total_size > self.max_total_size:
            raise ValueError(f"Total batch size too large: {total_size} bytes (max: {self.max_total_size})")
        
        options = options or {}
        generate_summaries = options.get("generate_summaries", True)
        extract_key_points = options.get("extract_key_points", True)
        include_content = options.get("include_content", False)
        
        # Process files in parallel (with concurrency limit)
        semaphore = asyncio.Semaphore(5)  # Limit concurrent file processing
        tasks = []
        
        for attachment in attachments:
            task = self._process_single_file(
                attachment, 
                semaphore, 
                generate_summaries, 
                extract_key_points,
                include_content
            )
            tasks.append(task)
        
        # Wait for all files to process
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Organize results
        processed_files = []
        errors = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append({
                    "filename": attachments[i].filename,
                    "error": str(result)
                })
            else:
                processed_files.append(result)
        
        # Generate batch summary if requested and we have multiple files
        batch_summary = None
        if generate_summaries and len(processed_files) > 1:
            try:
                batch_summary = await self._generate_batch_summary(processed_files)
            except Exception as e:
                logger.error(f"Failed to generate batch summary: {e}")
        
        return {
            "processed_files": processed_files,
            "errors": errors,
            "batch_summary": batch_summary,
            "stats": {
                "total_files": len(attachments),
                "processed_successfully": len(processed_files),
                "failed": len(errors),
                "total_size": total_size,
                "processing_time": None  # Could add timing
            }
        }
    
    async def _process_single_file(
        self,
        attachment: discord.Attachment,
        semaphore: asyncio.Semaphore,
        generate_summary: bool,
        extract_key_points: bool,
        include_content: bool
    ) -> FileMetadata:
        """Process a single file attachment"""
        
        async with semaphore:
            file_path = None
            try:
                # Validate file
                if attachment.size > self.max_file_size:
                    raise ValueError(f"File too large: {attachment.size} bytes")
                
                # Create metadata
                metadata = FileMetadata(
                    filename=attachment.filename,
                    file_type=self._get_file_type(attachment),
                    size=attachment.size,
                    content_type=attachment.content_type
                )
                
                # Download file to temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{attachment.filename}") as temp_file:
                    await attachment.save(temp_file)
                    file_path = Path(temp_file.name)
                
                # Process based on file type
                if metadata.file_type in ["text", "code"]:
                    content = await self._read_text_file(file_path)
                    if include_content:
                        metadata.content = content
                    metadata.word_count = len(content.split()) if content else 0
                    
                elif metadata.file_type == "document":
                    # For documents, we'd use the file search API
                    content = await self._extract_document_content(file_path, attachment)
                    if include_content:
                        metadata.content = content
                    metadata.word_count = len(content.split()) if content else 0
                    
                elif metadata.file_type == "spreadsheet":
                    # For spreadsheets, extract summary information
                    content = await self._analyze_spreadsheet(file_path, attachment)
                    if include_content:
                        metadata.content = content
                    
                elif metadata.file_type == "presentation":
                    # For presentations, extract slide content/structure
                    content = await self._analyze_presentation(file_path, attachment)
                    if include_content:
                        metadata.content = content
                    
                elif metadata.file_type == "image":
                    # For images, we can use GPT-5 vision capabilities
                    content = await self._analyze_image(file_path, attachment)
                    if include_content:
                        metadata.content = content
                    
                elif metadata.file_type == "archive":
                    # For archives, list contents and basic analysis
                    content = await self._analyze_archive(file_path, attachment)
                    if include_content:
                        metadata.content = content
                    
                elif metadata.file_type == "video":
                    # For videos, extract basic metadata
                    content = await self._analyze_video(file_path, attachment)
                    if include_content:
                        metadata.content = content
                    
                else:
                    # Unsupported file type
                    metadata.error = f"Unsupported file type: {metadata.file_type}"
                    return metadata
                
                # Generate summary if requested and we have content
                if generate_summary and hasattr(metadata, 'content') or metadata.file_type == "image":
                    try:
                        metadata.summary = await self._generate_file_summary(metadata, content)
                    except Exception as e:
                        logger.error(f"Failed to generate summary for {attachment.filename}: {e}")
                
                # Extract key points if requested
                if extract_key_points and hasattr(metadata, 'content'):
                    try:
                        metadata.key_points = await self._extract_key_points(metadata, content)
                    except Exception as e:
                        logger.error(f"Failed to extract key points for {attachment.filename}: {e}")
                
                metadata.processed = True
                return metadata
                
            except Exception as e:
                metadata = FileMetadata(
                    filename=attachment.filename,
                    file_type="unknown",
                    size=attachment.size,
                    error=str(e)
                )
                return metadata
            
            finally:
                # Clean up temporary file
                if file_path and file_path.exists():
                    try:
                        file_path.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to clean up temp file {file_path}: {e}")
    
    def _get_file_type(self, attachment: discord.Attachment) -> str:
        """Determine the type of file for processing"""
        
        extension = Path(attachment.filename).suffix.lower()
        content_type = attachment.content_type or ""
        
        if extension in self.text_types or content_type.startswith('text/'):
            return "text"
        elif extension in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.cs', '.php', '.rb', '.go', '.rs']:
            return "code"
        elif extension in self.document_types or extension in self.advanced_document_types or 'document' in content_type:
            return "document"
        elif extension in self.spreadsheet_types or 'spreadsheet' in content_type or 'sheet' in content_type:
            return "spreadsheet"
        elif extension in self.presentation_types or 'presentation' in content_type:
            return "presentation"
        elif extension in self.image_types or content_type.startswith('image/'):
            return "image"
        elif extension in self.archive_types or 'archive' in content_type or 'zip' in content_type:
            return "archive"
        elif extension in self.video_types or content_type.startswith('video/'):
            return "video"
        else:
            return "unknown"
    
    async def _read_text_file(self, file_path: Path) -> str:
        """Read content from a text file"""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                return await f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            async with aiofiles.open(file_path, 'r', encoding='latin-1') as f:
                return await f.read()
    
    async def _extract_document_content(self, file_path: Path, attachment: discord.Attachment) -> str:
        """Extract content from document files (PDF, Word, etc.)"""
        
        # For now, we'll return a placeholder - in production, you'd use
        # libraries like PyPDF2, python-docx, or send to OpenAI file API
        
        extension = Path(attachment.filename).suffix.lower()
        
        if extension == '.pdf':
            return f"[PDF content from {attachment.filename} - {attachment.size} bytes]"
        elif extension in ['.doc', '.docx']:
            return f"[Word document content from {attachment.filename} - {attachment.size} bytes]"
        else:
            return f"[Document content from {attachment.filename} - {attachment.size} bytes]"
    
    async def _analyze_spreadsheet(self, file_path: Path, attachment: discord.Attachment) -> str:
        """Analyze spreadsheet content"""
        
        extension = Path(attachment.filename).suffix.lower()
        
        if extension == '.csv':
            # For CSV files, we can read the first few rows
            try:
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    lines = []
                    for i in range(min(10, 100)):  # Read first 10 lines or 100 chars per line max
                        line = await f.readline()
                        if not line:
                            break
                        lines.append(line.strip()[:100])
                    
                    if lines:
                        return f"CSV file with {len(lines)} rows (sample):\\n" + "\\n".join(lines[:5])
                    else:
                        return "Empty CSV file"
            except Exception as e:
                return f"CSV file - could not read content: {str(e)}"
        
        elif extension in ['.xlsx', '.xls']:
            return f"Excel spreadsheet: {attachment.filename} - {attachment.size} bytes. Contains worksheets with data tables, formulas, and charts."
        
        elif extension == '.ods':
            return f"OpenDocument Spreadsheet: {attachment.filename} - {attachment.size} bytes. LibreOffice/OpenOffice format with data tables."
        
        else:
            return f"Spreadsheet file: {attachment.filename} - {attachment.size} bytes"
    
    async def _analyze_presentation(self, file_path: Path, attachment: discord.Attachment) -> str:
        """Analyze presentation content"""
        
        extension = Path(attachment.filename).suffix.lower()
        
        if extension in ['.pptx', '.ppt']:
            return f"PowerPoint presentation: {attachment.filename} - {attachment.size} bytes. Contains slides with text, images, and multimedia content."
        
        elif extension == '.odp':
            return f"OpenDocument Presentation: {attachment.filename} - {attachment.size} bytes. LibreOffice/OpenOffice presentation format."
        
        else:
            return f"Presentation file: {attachment.filename} - {attachment.size} bytes"
    
    async def _analyze_archive(self, file_path: Path, attachment: discord.Attachment) -> str:
        """Analyze archive content"""
        
        extension = Path(attachment.filename).suffix.lower()
        
        # For archives, we could potentially list contents, but for security reasons,
        # we'll just provide basic information
        
        if extension == '.zip':
            return f"ZIP archive: {attachment.filename} - {attachment.size} bytes. Compressed file collection."
        
        elif extension in ['.tar', '.gz']:
            return f"TAR archive: {attachment.filename} - {attachment.size} bytes. Unix/Linux archive format."
        
        elif extension == '.rar':
            return f"RAR archive: {attachment.filename} - {attachment.size} bytes. WinRAR compressed archive."
        
        elif extension == '.7z':
            return f"7-Zip archive: {attachment.filename} - {attachment.size} bytes. High compression archive format."
        
        else:
            return f"Archive file: {attachment.filename} - {attachment.size} bytes"
    
    async def _analyze_video(self, file_path: Path, attachment: discord.Attachment) -> str:
        """Analyze video content"""
        
        extension = Path(attachment.filename).suffix.lower()
        
        # For videos, we could potentially extract frames or metadata,
        # but for now, provide basic information
        
        if extension == '.mp4':
            return f"MP4 video: {attachment.filename} - {attachment.size} bytes. MPEG-4 video format."
        
        elif extension == '.mov':
            return f"QuickTime video: {attachment.filename} - {attachment.size} bytes. Apple video format."
        
        elif extension == '.avi':
            return f"AVI video: {attachment.filename} - {attachment.size} bytes. Audio Video Interleave format."
        
        elif extension == '.mkv':
            return f"Matroska video: {attachment.filename} - {attachment.size} bytes. Open container format."
        
        elif extension == '.webm':
            return f"WebM video: {attachment.filename} - {attachment.size} bytes. Open web video format."
        
        else:
            return f"Video file: {attachment.filename} - {attachment.size} bytes"
    
    async def _analyze_image(self, file_path: Path, attachment: discord.Attachment) -> str:
        """Analyze image content using GPT-5 vision"""
        
        try:
            import base64
            
            # Read and encode the image
            async with aiofiles.open(file_path, "rb") as image_file:
                image_data = await image_file.read()
            
            # Encode to base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Determine image format
            image_format = self._get_image_format(attachment.filename)
            data_url = f"data:image/{image_format};base64,{base64_image}"
            
            # Create vision analysis prompt
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Analyze this image ({attachment.filename}) and provide a detailed description including:\n1. What objects, people, or scenes are visible\n2. Colors, composition, and style\n3. Any text or symbols present\n4. The apparent purpose or context\n5. Technical aspects (if relevant)"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url,
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]
            
            # Call GPT-5 with vision
            response = await self.openai_client.client.chat.completions.create(
                model="gpt-5",
                messages=messages,
                max_tokens=500,
                temperature=0.3
            )
            
            analysis = response.choices[0].message.content.strip()
            
            # Add metadata
            file_info = f"Image file: {attachment.filename} ({self._format_bytes(attachment.size)}, {image_format.upper()})"
            
            return f"{file_info}\n\nVisual Analysis:\n{analysis}"
            
        except Exception as e:
            logger.error(f"Error analyzing image {attachment.filename}: {e}")
            # Fallback to basic info
            return f"Image file: {attachment.filename} - {self._format_bytes(attachment.size)} ({self._get_image_format(attachment.filename).upper()} format). Analysis failed: {str(e)}"
    
    def _get_image_format(self, filename: str) -> str:
        """Get image format from filename"""
        extension = Path(filename).suffix.lower()
        format_map = {
            '.jpg': 'jpeg',
            '.jpeg': 'jpeg',
            '.png': 'png',
            '.gif': 'gif',
            '.webp': 'webp',
            '.bmp': 'bmp'
        }
        return format_map.get(extension, 'jpeg')
    
    def _format_bytes(self, bytes_size: int) -> str:
        """Format bytes into human-readable size"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024
        return f"{bytes_size:.1f} TB"
    
    async def _generate_file_summary(self, metadata: FileMetadata, content: str) -> str:
        """Generate a summary for a single file"""
        
        if not content or len(content.strip()) < 50:
            return "File too short to summarize"
        
        # Create prompt based on file type
        if metadata.file_type == "code":
            prompt = f"Analyze this code file ({metadata.filename}) and provide a brief summary of its purpose, main functions, and key features:\\n\\n{content[:4000]}"
        elif metadata.file_type == "document":
            prompt = f"Summarize the main points and key information from this document ({metadata.filename}):\\n\\n{content[:4000]}"
        elif metadata.file_type == "spreadsheet":
            prompt = f"Analyze this spreadsheet ({metadata.filename}) and summarize its structure, data types, and apparent purpose:\\n\\n{content[:4000]}"
        elif metadata.file_type == "presentation":
            prompt = f"Summarize this presentation ({metadata.filename}) including its topic, structure, and key content:\\n\\n{content[:4000]}"
        elif metadata.file_type == "archive":
            prompt = f"Describe this archive file ({metadata.filename}) including its format and potential contents:\\n\\n{content[:4000]}"
        elif metadata.file_type == "video":
            prompt = f"Describe this video file ({metadata.filename}) including its format and basic properties:\\n\\n{content[:4000]}"
        elif metadata.file_type == "image":
            prompt = f"Analyze this image file ({metadata.filename}) and describe its content, format, and potential use case:\\n\\n{content[:4000]}"
        else:
            prompt = f"Provide a brief summary of this {metadata.file_type} file ({metadata.filename}):\\n\\n{content[:4000]}"
        
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant that creates concise, informative summaries of files. Focus on the main points and key information."},
                {"role": "user", "content": prompt}
            ]
            
            # Use OpenAI API to generate summary
            response = await self.openai_client.client.chat.completions.create(
                model="gpt-5",
                messages=messages,
                max_tokens=200,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return f"Summary generation failed: {str(e)}"
    
    async def _extract_key_points(self, metadata: FileMetadata, content: str) -> List[str]:
        """Extract key points from file content"""
        
        if not content or len(content.strip()) < 100:
            return []
        
        prompt = f"Extract 3-5 key points or main takeaways from this {metadata.file_type} file ({metadata.filename}). Return as a simple list:\\n\\n{content[:4000]}"
        
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant that extracts key points from documents. Return key points as a numbered list."},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.openai_client.client.chat.completions.create(
                model="gpt-5",
                messages=messages,
                max_tokens=300,
                temperature=0.3
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse numbered list into array
            points = []
            for line in content.split('\\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    # Remove number/bullet and clean up
                    point = line.split('.', 1)[-1].strip() if '.' in line else line[1:].strip()
                    if point:
                        points.append(point)
            
            return points[:5]  # Limit to 5 points
            
        except Exception as e:
            logger.error(f"Failed to extract key points: {e}")
            return []
    
    async def _generate_batch_summary(self, processed_files: List[FileMetadata]) -> str:
        """Generate a summary of the entire batch of files"""
        
        if not processed_files:
            return "No files processed successfully"
        
        # Create overview of files
        file_overview = []
        for file_meta in processed_files:
            summary = file_meta.summary or "No summary available"
            file_overview.append(f"- {file_meta.filename} ({file_meta.file_type}): {summary}")
        
        overview_text = "\\n".join(file_overview)
        
        prompt = f"Analyze this batch of {len(processed_files)} files and provide a comprehensive summary that highlights:\\n1. Common themes or patterns\\n2. Overall purpose or project scope\\n3. Key insights or important information\\n\\nFiles processed:\\n{overview_text}"
        
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant that analyzes collections of files and provides insightful batch summaries. Focus on patterns, themes, and overall insights."},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.openai_client.client.chat.completions.create(
                model="gpt-5",
                messages=messages,
                max_tokens=400,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate batch summary: {e}")
            return f"Batch summary generation failed: {str(e)}"
    
    def get_supported_file_types(self) -> Dict[str, List[str]]:
        """Get all supported file types organized by category"""
        return {
            "text": list(self.text_types),
            "documents": list(self.document_types | self.advanced_document_types),
            "spreadsheets": list(self.spreadsheet_types),
            "presentations": list(self.presentation_types),
            "images": list(self.image_types),
            "archives": list(self.archive_types),
            "videos": list(self.video_types)
        }
    
    def get_batch_limits(self) -> Dict[str, int]:
        """Get batch processing limits"""
        return {
            "max_files": self.max_batch_size,
            "max_file_size": self.max_file_size,
            "max_total_size": self.max_total_size
        }