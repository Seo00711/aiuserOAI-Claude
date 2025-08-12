import asyncio
import logging
from typing import Any, Dict, List, Optional, AsyncGenerator, Union
import json
import aiofiles
from pathlib import Path

import httpx
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config_schemas import ModelConfig, ToolConfig
from .errors import error_handler, APIError, RateLimitError, GPT5AssistantError


logger = logging.getLogger("red.gpt5assistant.openai_client")


class OpenAIClient:
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key, timeout=httpx.Timeout(60.0))
        self._kb_ids: Dict[int, str] = {}  # guild_id -> knowledge_base_id
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError,))
    )
    async def respond_chat(
        self,
        messages: List[Dict[str, Any]],
        model_config: ModelConfig,
        tool_config: ToolConfig,
        guild_id: Optional[int] = None,
        previous_response_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        tools = self._build_tools_list(tool_config, guild_id)
        
        try:
            request_data = {
                "model": model_config.name,
                "input": messages[-1]["content"] if messages else "",
                "tools": tools,
                "reasoning": {"effort": model_config.reasoning.effort},
                "text": {"verbosity": model_config.text.verbosity}
            }
            
            if model_config.max_tokens:
                request_data["max_tokens"] = model_config.max_tokens
            if model_config.temperature != 0.7:
                request_data["temperature"] = model_config.temperature
            if previous_response_id:
                request_data["previous_response_id"] = previous_response_id
            if len(messages) > 1:
                request_data["messages"] = messages[:-1]
            
            logger.debug(f"OpenAI request: {json.dumps(request_data, indent=2)}")
            
            response = await self.client.responses.create(**request_data)
            
            if hasattr(response, 'content') and response.content:
                for chunk in response.content:
                    if hasattr(chunk, 'text') and chunk.text:
                        yield chunk.text
            else:
                yield response.text if hasattr(response, 'text') else str(response)
                
        except Exception as e:
            raise error_handler.handle_openai_error(e)
    
    async def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "standard",
        style: str = "natural"
    ) -> Dict[str, Any]:
        try:
            response = await self.client.images.generate(
                model="gpt-image-1",
                prompt=prompt,
                size=size,
                quality=quality,
                style=style,
                n=1
            )
            
            image_data = response.data[0]
            return {
                "url": image_data.url,
                "revised_prompt": getattr(image_data, "revised_prompt", None),
                "size": size,
                "quality": quality,
                "style": style
            }
        except Exception as e:
            raise error_handler.handle_openai_error(e)
    
    async def edit_image(
        self,
        image_path: Path,
        prompt: str,
        mask_path: Optional[Path] = None,
        size: str = "1024x1024"
    ) -> Dict[str, Any]:
        try:
            async with aiofiles.open(image_path, "rb") as image_file:
                image_data = await image_file.read()
            
            kwargs = {
                "model": "gpt-image-1",
                "image": image_data,
                "prompt": prompt,
                "size": size,
                "n": 1
            }
            
            if mask_path and mask_path.exists():
                async with aiofiles.open(mask_path, "rb") as mask_file:
                    kwargs["mask"] = await mask_file.read()
            
            response = await self.client.images.edit(**kwargs)
            
            image_result = response.data[0]
            return {
                "url": image_result.url,
                "revised_prompt": getattr(image_result, "revised_prompt", None),
                "size": size
            }
        except Exception as e:
            raise error_handler.handle_openai_error(e)
    
    async def upload_files_for_search(
        self,
        file_paths: List[Path],
        guild_id: int
    ) -> str:
        try:
            file_ids = []
            
            for file_path in file_paths:
                async with aiofiles.open(file_path, "rb") as f:
                    file_content = await f.read()
                
                file_obj = await self.client.files.create(
                    file=file_content,
                    purpose="assistants"
                )
                file_ids.append(file_obj.id)
            
            if guild_id in self._kb_ids:
                kb_id = self._kb_ids[guild_id]
                await self.client.beta.assistants.files.create(
                    assistant_id=kb_id,
                    file_ids=file_ids
                )
            else:
                assistant = await self.client.beta.assistants.create(
                    name=f"Guild {guild_id} Knowledge Base",
                    tools=[{"type": "file_search"}],
                    file_ids=file_ids
                )
                kb_id = assistant.id
                self._kb_ids[guild_id] = kb_id
            
            logger.info(f"Uploaded {len(file_ids)} files to knowledge base {kb_id}")
            return kb_id
            
        except Exception as e:
            raise error_handler.handle_openai_error(e)
    
    def _build_tools_list(self, tool_config: ToolConfig, guild_id: Optional[int] = None) -> List[Dict[str, Any]]:
        tools = []
        
        if tool_config.web_search:
            tools.append({"type": "web_search"})
        
        if tool_config.file_search and guild_id and guild_id in self._kb_ids:
            tools.append({
                "type": "file_search",
                "file_search": {"knowledge_base_id": self._kb_ids[guild_id]}
            })
        
        if tool_config.code_interpreter:
            tools.append({
                "type": "code_interpreter",
                "container": {"type": "auto"}
            })
        
        return tools
    
    async def close(self) -> None:
        if hasattr(self.client, 'close'):
            await self.client.close()