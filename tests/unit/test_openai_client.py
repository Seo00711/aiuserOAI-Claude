import pytest
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
import tempfile
import aiofiles

from gpt5assistant.openai_client import OpenAIClient
from gpt5assistant.config_schemas import ModelConfig, ToolConfig, ReasoningConfig, TextConfig
from gpt5assistant.errors import APIError


@pytest.fixture
def openai_client():
    return OpenAIClient("test-api-key")


@pytest.fixture
def model_config():
    return ModelConfig(
        name="gpt-5",
        temperature=0.7,
        max_tokens=1000,
        reasoning=ReasoningConfig(effort="medium"),
        text=TextConfig(verbosity="medium")
    )


@pytest.fixture
def tool_config():
    return ToolConfig(
        web_search=True,
        file_search=True,
        code_interpreter=True,
        image=True
    )


@pytest.mark.asyncio
async def test_openai_client_initialization():
    client = OpenAIClient("test-key")
    assert client.client is not None
    assert client._kb_ids == {}


@pytest.mark.asyncio
async def test_respond_chat_success(openai_client, model_config, tool_config, mock_openai_client):
    messages = [{"role": "user", "content": "Hello"}]
    
    with patch.object(openai_client, 'client', mock_openai_client):
        chunks = []
        async for chunk in openai_client.respond_chat(messages, model_config, tool_config):
            chunks.append(chunk)
        
        assert len(chunks) == 2
        assert chunks[0] == "Test response chunk 1"
        assert chunks[1] == " chunk 2"
        
        # Verify the request was made with correct parameters
        mock_openai_client.responses.create.assert_called_once()
        call_args = mock_openai_client.responses.create.call_args[1]
        assert call_args["model"] == "gpt-5"
        assert call_args["reasoning"]["effort"] == "medium"
        assert call_args["text"]["verbosity"] == "medium"


@pytest.mark.asyncio
async def test_respond_chat_with_tools(openai_client, model_config, tool_config, mock_openai_client):
    messages = [{"role": "user", "content": "Search for information about AI"}]
    
    with patch.object(openai_client, 'client', mock_openai_client):
        async for _ in openai_client.respond_chat(messages, model_config, tool_config, guild_id=123):
            pass
        
        call_args = mock_openai_client.responses.create.call_args[1]
        tools = call_args["tools"]
        
        # Should include web_search and code_interpreter (file_search requires KB)
        tool_types = [tool["type"] for tool in tools]
        assert "web_search" in tool_types
        assert "code_interpreter" in tool_types


@pytest.mark.asyncio
async def test_generate_image_success(openai_client, mock_openai_client):
    with patch.object(openai_client, 'client', mock_openai_client):
        result = await openai_client.generate_image("A beautiful sunset")
        
        assert result["url"] == "https://example.com/image.png"
        assert result["revised_prompt"] == "Test image"
        assert result["size"] == "1024x1024"
        
        mock_openai_client.images.generate.assert_called_once_with(
            model="gpt-image-1",
            prompt="A beautiful sunset",
            size="1024x1024",
            quality="standard",
            style="natural",
            n=1
        )


@pytest.mark.asyncio
async def test_edit_image_success(openai_client, mock_openai_client):
    with tempfile.NamedTemporaryFile() as temp_file:
        temp_file.write(b"fake image data")
        temp_file.flush()
        
        image_path = Path(temp_file.name)
        
        with patch.object(openai_client, 'client', mock_openai_client):
            result = await openai_client.edit_image(image_path, "Make it colorful")
            
            assert result["url"] == "https://example.com/image.png"
            assert result["size"] == "1024x1024"
            
            mock_openai_client.images.edit.assert_called_once()


@pytest.mark.asyncio
async def test_upload_files_for_search_new_kb(openai_client, mock_openai_client):
    with tempfile.NamedTemporaryFile() as temp_file:
        temp_file.write(b"test file content")
        temp_file.flush()
        
        file_paths = [Path(temp_file.name)]
        guild_id = 123
        
        with patch.object(openai_client, 'client', mock_openai_client):
            kb_id = await openai_client.upload_files_for_search(file_paths, guild_id)
            
            assert kb_id == "asst-123"
            assert openai_client._kb_ids[guild_id] == "asst-123"
            
            mock_openai_client.files.create.assert_called_once()
            mock_openai_client.beta.assistants.create.assert_called_once()


@pytest.mark.asyncio
async def test_build_tools_list_all_enabled(openai_client, tool_config):
    tools = openai_client._build_tools_list(tool_config)
    
    tool_types = [tool["type"] for tool in tools]
    assert "web_search" in tool_types
    assert "code_interpreter" in tool_types
    # file_search not included without guild KB


@pytest.mark.asyncio
async def test_build_tools_list_with_kb(openai_client, tool_config):
    guild_id = 123
    openai_client._kb_ids[guild_id] = "asst-123"
    
    tools = openai_client._build_tools_list(tool_config, guild_id)
    
    tool_types = [tool["type"] for tool in tools]
    assert "web_search" in tool_types
    assert "file_search" in tool_types
    assert "code_interpreter" in tool_types


@pytest.mark.asyncio
async def test_close(openai_client):
    mock_client = Mock()
    mock_client.close = AsyncMock()
    openai_client.client = mock_client
    
    await openai_client.close()
    mock_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_error_handling_openai_error(openai_client, model_config, tool_config, mock_openai_client):
    from openai import AuthenticationError
    
    mock_openai_client.responses.create.side_effect = AuthenticationError("Invalid API key")
    
    with patch.object(openai_client, 'client', mock_openai_client):
        with pytest.raises(APIError):
            async for _ in openai_client.respond_chat([], model_config, tool_config):
                pass