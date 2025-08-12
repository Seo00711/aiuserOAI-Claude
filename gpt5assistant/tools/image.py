import logging
import base64
from typing import Dict, Any, Optional, List
from pathlib import Path
import tempfile
import discord
import aiofiles

from ..openai_client import OpenAIClient, OpenAIClientError

logger = logging.getLogger("red.gpt5assistant.tools.image")


class ImageTool:
    def __init__(self, openai_client: OpenAIClient):
        self.client = openai_client
    
    async def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "standard",
        style: str = "natural"
    ) -> Dict[str, Any]:
        try:
            logger.info(f"Generating image with prompt: {prompt[:100]}...")
            
            result = await self.client.generate_image(
                prompt=prompt,
                size=size,
                quality=quality,
                style=style
            )
            
            logger.info(f"Image generated successfully: {result['url']}")
            return result
        
        except OpenAIClientError as e:
            logger.error(f"Image generation failed: {e}")
            raise
    
    async def edit_image(
        self,
        attachment: discord.Attachment,
        prompt: str,
        size: str = "1024x1024",
        mask_attachment: Optional[discord.Attachment] = None
    ) -> Dict[str, Any]:
        temp_files = []
        
        try:
            logger.info(f"Editing image {attachment.filename} with prompt: {prompt[:100]}...")
            
            # Download the original image
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{attachment.filename}") as temp_image:
                await attachment.save(temp_image)
                image_path = Path(temp_image.name)
                temp_files.append(image_path)
            
            mask_path = None
            if mask_attachment:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_mask_{mask_attachment.filename}") as temp_mask:
                    await mask_attachment.save(temp_mask)
                    mask_path = Path(temp_mask.name)
                    temp_files.append(mask_path)
            
            result = await self.client.edit_image(
                image_path=image_path,
                prompt=prompt,
                mask_path=mask_path,
                size=size
            )
            
            logger.info(f"Image edited successfully: {result['url']}")
            return result
        
        except OpenAIClientError as e:
            logger.error(f"Image editing failed: {e}")
            raise
        
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    temp_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file {temp_file}: {e}")
    
    def validate_image_attachment(self, attachment: discord.Attachment) -> bool:
        if not attachment.content_type:
            return False
        
        valid_types = ["image/png", "image/jpeg", "image/jpg", "image/webp"]
        return attachment.content_type.lower() in valid_types
    
    def get_supported_sizes(self) -> List[str]:
        return ["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"]
    
    def get_supported_qualities(self) -> List[str]:
        return ["standard", "hd"]
    
    def get_supported_styles(self) -> List[str]:
        return ["natural", "vivid"]
    
    async def analyze_image(
        self,
        attachment: discord.Attachment,
        custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze an image using GPT-5 vision capabilities"""
        
        temp_files = []
        
        try:
            logger.info(f"Analyzing image {attachment.filename}...")
            
            # Download the image
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{attachment.filename}") as temp_image:
                await attachment.save(temp_image)
                image_path = Path(temp_image.name)
                temp_files.append(image_path)
            
            # Read and encode the image
            async with aiofiles.open(image_path, "rb") as image_file:
                image_data = await image_file.read()
            
            # Encode to base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Determine image format
            image_format = self._get_image_format(attachment.filename)
            data_url = f"data:image/{image_format};base64,{base64_image}"
            
            # Create analysis prompt
            if custom_prompt:
                analysis_text = custom_prompt
            else:
                analysis_text = """Analyze this image in detail. Please describe:

1. **Visual Content**: What objects, people, scenes, or subjects are visible?
2. **Composition & Style**: Colors, lighting, composition, artistic style or photographic technique
3. **Text & Symbols**: Any visible text, signs, logos, symbols, or written content
4. **Context & Purpose**: What appears to be the purpose, setting, or context of this image?
5. **Technical Aspects**: Image quality, format, any notable technical characteristics
6. **Mood & Atmosphere**: The overall mood, emotion, or atmosphere conveyed

Be thorough but concise in your analysis."""
            
            # Create vision analysis request
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": analysis_text
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
            response = await self.client.client.chat.completions.create(
                model="gpt-5",
                messages=messages,
                max_tokens=1000,
                temperature=0.3
            )
            
            analysis = response.choices[0].message.content.strip()
            
            # Extract image metadata
            metadata = {
                "filename": attachment.filename,
                "size_bytes": attachment.size,
                "size_formatted": self._format_bytes(attachment.size),
                "format": image_format.upper(),
                "content_type": attachment.content_type,
                "url": attachment.url
            }
            
            logger.info(f"Image analysis completed for {attachment.filename}")
            
            return {
                "analysis": analysis,
                "metadata": metadata,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Image analysis failed for {attachment.filename}: {e}")
            return {
                "analysis": f"Failed to analyze image: {str(e)}",
                "metadata": {
                    "filename": attachment.filename,
                    "size_bytes": attachment.size,
                    "error": str(e)
                },
                "success": False
            }
        
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    temp_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file {temp_file}: {e}")
    
    async def compare_images(
        self,
        image1: discord.Attachment,
        image2: discord.Attachment,
        comparison_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compare two images using GPT-5 vision"""
        
        temp_files = []
        
        try:
            logger.info(f"Comparing images {image1.filename} and {image2.filename}...")
            
            # Download both images
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_1_{image1.filename}") as temp1:
                await image1.save(temp1)
                image1_path = Path(temp1.name)
                temp_files.append(image1_path)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_2_{image2.filename}") as temp2:
                await image2.save(temp2)
                image2_path = Path(temp2.name)
                temp_files.append(image2_path)
            
            # Read and encode both images
            async with aiofiles.open(image1_path, "rb") as f1:
                image1_data = await f1.read()
            async with aiofiles.open(image2_path, "rb") as f2:
                image2_data = await f2.read()
            
            # Encode to base64
            base64_image1 = base64.b64encode(image1_data).decode('utf-8')
            base64_image2 = base64.b64encode(image2_data).decode('utf-8')
            
            # Get formats
            format1 = self._get_image_format(image1.filename)
            format2 = self._get_image_format(image2.filename)
            
            data_url1 = f"data:image/{format1};base64,{base64_image1}"
            data_url2 = f"data:image/{format2};base64,{base64_image2}"
            
            # Create comparison prompt
            if comparison_prompt:
                prompt_text = comparison_prompt
            else:
                prompt_text = f"""Compare these two images ({image1.filename} and {image2.filename}):

1. **Visual Similarities**: What elements, objects, or themes do they share?
2. **Key Differences**: What are the main differences in content, style, or composition?
3. **Quality & Technical**: Compare image quality, resolution, and technical aspects
4. **Style & Mood**: How do the artistic styles, colors, and moods compare?
5. **Content Analysis**: Detailed comparison of subjects, scenes, or objects shown
6. **Overall Assessment**: Which aspects make each image unique or notable?

Provide a comprehensive comparison analysis."""
            
            # Create comparison request
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_text
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url1,
                                "detail": "high"
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url2,
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]
            
            # Call GPT-5 with vision
            response = await self.client.client.chat.completions.create(
                model="gpt-5",
                messages=messages,
                max_tokens=1200,
                temperature=0.3
            )
            
            comparison = response.choices[0].message.content.strip()
            
            logger.info(f"Image comparison completed")
            
            return {
                "comparison": comparison,
                "image1_metadata": {
                    "filename": image1.filename,
                    "size": self._format_bytes(image1.size),
                    "format": format1.upper()
                },
                "image2_metadata": {
                    "filename": image2.filename,
                    "size": self._format_bytes(image2.size),
                    "format": format2.upper()
                },
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Image comparison failed: {e}")
            return {
                "comparison": f"Failed to compare images: {str(e)}",
                "success": False
            }
        
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    temp_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file {temp_file}: {e}")
    
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