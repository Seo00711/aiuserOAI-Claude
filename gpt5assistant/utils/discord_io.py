import asyncio
import io
from typing import List, Optional, AsyncGenerator
import discord
import httpx
import logging
from pathlib import Path

logger = logging.getLogger("red.gpt5assistant.discord_io")


class DiscordStreamer:
    def __init__(self, message_or_interaction: discord.Message | discord.Interaction):
        self.message = message_or_interaction
        self.is_interaction = isinstance(message_or_interaction, discord.Interaction)
        self._current_message: Optional[discord.Message] = None
        self._typing_task: Optional[asyncio.Task] = None
    
    async def start_typing(self) -> None:
        if self.is_interaction:
            return  # Interactions don't support typing
        
        if self._typing_task and not self._typing_task.done():
            return
        
        self._typing_task = asyncio.create_task(self._typing_loop())
    
    async def stop_typing(self) -> None:
        if self._typing_task and not self._typing_task.done():
            self._typing_task.cancel()
            try:
                await self._typing_task
            except asyncio.CancelledError:
                pass
    
    async def _typing_loop(self) -> None:
        try:
            while True:
                await self.message.channel.trigger_typing()
                await asyncio.sleep(5)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in typing loop: {e}")
    
    async def stream_response(
        self,
        content_generator: AsyncGenerator[str, None],
        ephemeral: bool = False
    ) -> Optional[discord.Message]:
        await self.start_typing()
        
        try:
            chunks = []
            current_chunk = ""
            
            async for chunk in content_generator:
                current_chunk += chunk
                
                # Check if we need to split the chunk
                if len(current_chunk) >= 1800:  # Leave buffer for code blocks
                    split_point = self._find_split_point(current_chunk, 1800)
                    ready_chunk = current_chunk[:split_point]
                    current_chunk = current_chunk[split_point:]
                    
                    chunks.append(ready_chunk)
                    await self._send_chunk(ready_chunk, len(chunks) == 1, ephemeral)
            
            # Send remaining chunk
            if current_chunk.strip():
                chunks.append(current_chunk)
                await self._send_chunk(current_chunk, len(chunks) == 1, ephemeral)
            
            return self._current_message
        
        finally:
            await self.stop_typing()
    
    async def _send_chunk(self, chunk: str, is_first: bool, ephemeral: bool = False) -> None:
        try:
            if is_first:
                if self.is_interaction:
                    if not self.message.response.is_done():
                        await self.message.response.send_message(chunk, ephemeral=ephemeral)
                        self._current_message = await self.message.original_response()
                    else:
                        await self.message.followup.send(chunk, ephemeral=ephemeral)
                else:
                    self._current_message = await self.message.reply(chunk)
            else:
                # Follow-up message
                if self.is_interaction:
                    await self.message.followup.send(chunk, ephemeral=ephemeral)
                else:
                    await self.message.channel.send(chunk)
        
        except discord.HTTPException as e:
            logger.error(f"Failed to send chunk: {e}")
            # Try to send error message
            error_msg = "⚠️ Response too long or failed to send."
            try:
                if self.is_interaction and not self.message.response.is_done():
                    await self.message.response.send_message(error_msg, ephemeral=True)
                elif self.is_interaction:
                    await self.message.followup.send(error_msg, ephemeral=True)
                else:
                    await self.message.reply(error_msg)
            except Exception:
                pass
    
    def _find_split_point(self, text: str, max_length: int) -> int:
        if len(text) <= max_length:
            return len(text)
        
        # Try to split at code block boundaries
        code_block_pattern = "```"
        if code_block_pattern in text[:max_length]:
            # Count code blocks to avoid splitting inside one
            blocks_before = text[:max_length].count(code_block_pattern)
            if blocks_before % 2 == 1:  # Inside a code block
                # Find the end of the code block
                end_pos = text.find(code_block_pattern, max_length)
                if end_pos != -1 and end_pos < len(text) - 100:  # Don't create tiny remaining chunk
                    return end_pos + 3
        
        # Split at paragraph breaks
        for i in range(max_length - 1, max_length // 2, -1):
            if text[i] == '\n' and i < len(text) - 1 and text[i + 1] == '\n':
                return i + 2
        
        # Split at sentence breaks
        for i in range(max_length - 1, max_length // 2, -1):
            if text[i] in '.!?' and i < len(text) - 1 and text[i + 1] == ' ':
                return i + 1
        
        # Split at word boundaries
        for i in range(max_length - 1, max_length // 2, -1):
            if text[i] == ' ':
                return i + 1
        
        # Fallback: split at max length
        return max_length


async def download_image(url: str, filename: str) -> Optional[Path]:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            
            file_path = Path(f"/tmp/{filename}")
            with open(file_path, "wb") as f:
                f.write(response.content)
            
            return file_path
    
    except Exception as e:
        logger.error(f"Failed to download image from {url}: {e}")
        return None


async def send_image_result(
    target: discord.Message | discord.Interaction,
    image_url: str,
    prompt: str,
    metadata: dict,
    ephemeral: bool = False
) -> None:
    try:
        # Download the image
        filename = f"generated_image_{asyncio.get_event_loop().time()}.png"
        image_path = await download_image(image_url, filename)
        
        if not image_path:
            error_msg = "❌ Failed to download generated image."
            if isinstance(target, discord.Interaction):
                if not target.response.is_done():
                    await target.response.send_message(error_msg, ephemeral=ephemeral)
                else:
                    await target.followup.send(error_msg, ephemeral=ephemeral)
            else:
                await target.reply(error_msg)
            return
        
        # Create embed with metadata
        embed = discord.Embed(
            title="🎨 Generated Image",
            description=f"**Prompt:** {prompt[:1000]}{'...' if len(prompt) > 1000 else ''}",
            color=0x00ff00
        )
        
        # Add metadata in a collapsed format
        details = []
        if metadata.get("size"):
            details.append(f"Size: {metadata['size']}")
        if metadata.get("quality"):
            details.append(f"Quality: {metadata['quality']}")
        if metadata.get("style"):
            details.append(f"Style: {metadata['style']}")
        if metadata.get("revised_prompt"):
            details.append(f"Revised: {metadata['revised_prompt'][:200]}{'...' if len(metadata.get('revised_prompt', '')) > 200 else ''}")
        
        if details:
            embed.add_field(
                name="Details",
                value="```\n" + "\n".join(details) + "\n```",
                inline=False
            )
        
        # Send with file attachment
        with open(image_path, "rb") as f:
            file = discord.File(f, filename=filename)
            
            if isinstance(target, discord.Interaction):
                if not target.response.is_done():
                    await target.response.send_message(embed=embed, file=file, ephemeral=ephemeral)
                else:
                    await target.followup.send(embed=embed, file=file, ephemeral=ephemeral)
            else:
                await target.reply(embed=embed, file=file)
        
        # Clean up
        try:
            image_path.unlink()
        except Exception:
            pass
    
    except Exception as e:
        logger.error(f"Failed to send image result: {e}")
        error_msg = f"❌ Failed to send generated image: {str(e)}"
        
        try:
            if isinstance(target, discord.Interaction):
                if not target.response.is_done():
                    await target.response.send_message(error_msg, ephemeral=ephemeral)
                else:
                    await target.followup.send(error_msg, ephemeral=ephemeral)
            else:
                await target.reply(error_msg)
        except Exception:
            pass


def create_allowed_mentions() -> discord.AllowedMentions:
    return discord.AllowedMentions(
        everyone=False,
        users=True,
        roles=False,
        replied_user=True
    )