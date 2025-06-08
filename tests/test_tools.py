import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from neo.tools import Tool
from neo.tools.image_gen.openai import agenerate_image
from neo.tools.speech.openai import aspeech2text
from neo.types.contents import AudioContent, ImageContent, TextContent

from neo.models.providers.openai import OpenAIResponseModel
from neo.models.providers.anthropic import AnthropicModel
from neo.tools.internal_tools.openai import WebSearch as OpenAIWebSearch
from neo.tools.internal_tools.anthropic import WebSearch as AnthropicWebSearch
from neo.types.modalities import Modality

from neo.agentic import Neo, Task, Instruction, ModelConfigs, OtherConfigs
from neo.types.tool_codes import StandardToolCode


class TestToolCreation:
    """Test basic tool creation and functionality."""

    def test_tool_creation(self):
        """Test basic tool creation."""
        def example_function(param1: str, param2: int) -> str:
            """Example function docstring."""
            return f"{param1}: {param2}"

        tool = Tool(func=example_function)

        # Check attributes
        assert tool.name == "example_function"
        assert tool.description == "Example function docstring."
        assert len(tool.params["properties"]) == 2

        # Check function call
        result = tool.func("test", 123)
        assert result == "test: 123"


class TestOpenAITools:
    """Test OpenAI-specific tools."""

    @pytest.mark.asyncio
    @patch("neo.tools.image_gen.openai.AsyncOpenAI")
    async def test_agenerate_image(self, mock_async_openai):
        """Test the agenerate_image function with mocked OpenAI API."""
        # Set up the mock client and response
        mock_client = MagicMock()
        mock_async_openai.return_value = mock_client

        mock_images = MagicMock()
        mock_client.images = mock_images

        mock_generate = AsyncMock()
        mock_images.generate = mock_generate

        mock_generate.return_value = MagicMock(
            data=[MagicMock(b64_json=None, url="http://example.com/image.png")]
        )

        result = await agenerate_image(prompt="a cat", model="dall-e-3")

        # Check API was called with correct params
        mock_generate.assert_called_once()
        call_args = mock_generate.call_args[1]
        assert call_args["prompt"] == "a cat"
        assert call_args["model"] == "dall-e-3"

        # Check returned content
        assert isinstance(result, ImageContent)

    @pytest.mark.asyncio
    @patch("neo.tools.speech.openai.speech2text.AsyncOpenAI")
    async def test_aspeech2text(self, mock_async_openai):
        """Test the aspeech2text function with mocked OpenAI API."""
        mock_client = MagicMock()
        mock_async_openai.return_value = mock_client

        mock_audio = MagicMock()
        mock_client.audio = mock_audio

        mock_transcriptions = MagicMock()
        mock_audio.transcriptions = mock_transcriptions

        mock_create = AsyncMock()
        mock_transcriptions.create = mock_create

        transcription = "This is a transcription of the audio."
        mock_create.return_value = MagicMock(text=transcription)

        # Create audio content
        audio = AudioContent(
            data=b"audio data", mime_type="audio/wav", file_name="test.wav"
        )

        result = await aspeech2text(audio=audio)

        # Check API was called
        mock_create.assert_called_once()

        # Check result
        assert isinstance(result, TextContent)
        assert result.data == transcription


class TestInternalToolClassInstanceHandling:
    """Test that both tool classes and instances work for internal tools."""

    def test_openai_websearch_is_internal_tool_class(self):
        """Test that OpenAI WebSearch class is recognized as internal tool."""
        model = OpenAIResponseModel(
            model="gpt-4.1",
            input_modalities=[Modality.TEXT],
        )
        
        assert model.is_internal_tool(OpenAIWebSearch) is True

    def test_openai_websearch_is_internal_tool_instance(self):
        """Test that OpenAI WebSearch instance is recognized as internal tool."""
        model = OpenAIResponseModel(
            model="gpt-4.1",
            input_modalities=[Modality.TEXT],
        )
        
        websearch_instance = OpenAIWebSearch()
        assert model.is_internal_tool(websearch_instance) is True

    def test_anthropic_websearch_is_internal_tool_class(self):
        """Test that Anthropic WebSearch class is recognized as internal tool."""
        model = AnthropicModel(
            model="claude-3-5-haiku-latest",
            input_modalities=[Modality.TEXT],
        )
        
        assert model.is_internal_tool(AnthropicWebSearch) is True

    def test_anthropic_websearch_is_internal_tool_instance(self):
        """Test that Anthropic WebSearch instance is recognized as internal tool."""
        model = AnthropicModel(
            model="claude-3-5-haiku-latest",
            input_modalities=[Modality.TEXT],
        )
        
        websearch_instance = AnthropicWebSearch()
        assert model.is_internal_tool(websearch_instance) is True

    def test_cross_provider_tool_not_internal(self):
        """Test that OpenAI tools are not internal to Anthropic models and vice versa."""
        openai_model = OpenAIResponseModel(
            model="gpt-4.1",
            input_modalities=[Modality.TEXT],
        )
        anthropic_model = AnthropicModel(
            model="claude-3-5-haiku-latest",
            input_modalities=[Modality.TEXT],
        )
        
        # OpenAI WebSearch should not be internal to Anthropic model
        assert anthropic_model.is_internal_tool(OpenAIWebSearch) is False
        assert anthropic_model.is_internal_tool(OpenAIWebSearch()) is False
        
        # Anthropic WebSearch should not be internal to OpenAI model
        assert openai_model.is_internal_tool(AnthropicWebSearch) is False
        assert openai_model.is_internal_tool(AnthropicWebSearch()) is False

    def test_register_tool_with_class(self):
        """Test that tool registration works with tool classes."""
        model = OpenAIResponseModel(
            model="gpt-4.1",
            input_modalities=[Modality.TEXT],
        )
        
        # Should not raise an exception
        schema = model.register_tool(OpenAIWebSearch)
        assert schema is not None

    def test_register_tool_with_instance(self):
        """Test that tool registration works with tool instances."""
        model = OpenAIResponseModel(
            model="gpt-4.1",
            input_modalities=[Modality.TEXT],
        )
        
        websearch_instance = OpenAIWebSearch()
        
        # Should not raise an exception
        schema = model.register_tool(websearch_instance)
        assert schema is not None

    def test_callable_function_not_internal(self):
        """Test that regular callable functions are not considered internal tools."""
        model = OpenAIResponseModel(
            model="gpt-4.1",
            input_modalities=[Modality.TEXT],
        )
        
        def dummy_function():
            return "test"
        
        assert model.is_internal_tool(dummy_function) is False


class TestToolCodeResolution:
    """Test that StandardToolCode enums are resolved to actual tool instances."""

    def test_tool_code_resolution_creation(self):
        """Test that Neo can be created with StandardToolCode in tools."""
        instruction = Instruction(
            content="Act like a helpful assistant.",
            model_configs=ModelConfigs(model="gpt-4.1"), 
            other_configs=OtherConfigs(
                timeaware=False, 
                tools=[StandardToolCode.WEB_SEARCH]
            )
        )
        
        task = Task(user_input="Hello", instruction=instruction)
        neo = Neo(tasks=[task])
        
        # Should not raise an exception
        assert neo is not None
        assert task.instruction.other_configs.tools == [StandardToolCode.WEB_SEARCH]

    def test_multiple_tool_codes(self):
        """Test that multiple StandardToolCode enums work together."""
        instruction = Instruction(
            content="Act like a helpful assistant.",
            model_configs=ModelConfigs(model="gpt-4.1"), 
            other_configs=OtherConfigs(
                timeaware=False, 
                tools=[
                    StandardToolCode.WEB_SEARCH,  # Tool code enum
                    StandardToolCode.IMAGE_GEN,   # Another tool code enum
                ]
            )
        )
        
        task = Task(user_input="Hello", instruction=instruction)
        neo = Neo(tasks=[task])
        
        # Should not raise an exception
        assert neo is not None
        assert len(task.instruction.other_configs.tools) == 2

    def test_mixed_enum_and_string_tool_codes(self):
        """Test that a mix of StandardToolCode enum and string tool codes work together."""
        instruction = Instruction(
            content="Act like a helpful assistant.",
            model_configs=ModelConfigs(model="gpt-4.1"), 
            other_configs=OtherConfigs(
                timeaware=False, 
                tools=[
                    StandardToolCode.WEB_SEARCH,               # Tool code enum
                    StandardToolCode.IMAGE_GEN.value,          # Tool code string
                ]
            )
        )
        
        task = Task(user_input="Hello", instruction=instruction)
        neo = Neo(tasks=[task])
        
        # Should not raise an exception
        assert neo is not None
        assert len(task.instruction.other_configs.tools) == 2

    def test_empty_tools_list(self):
        """Test that empty tools list is handled correctly."""
        instruction = Instruction(
            content="Act like a helpful assistant.",
            model_configs=ModelConfigs(model="gpt-4.1"), 
            other_configs=OtherConfigs(
                timeaware=False, 
                tools=[]
            )
        )
        
        task = Task(user_input="Hello", instruction=instruction)
        neo = Neo(tasks=[task])
        
        # Should not raise an exception
        assert neo is not None
        assert task.instruction.other_configs.tools == []

    def test_no_tools_specified(self):
        """Test that no tools specified is handled correctly."""
        instruction = Instruction(
            content="Act like a helpful assistant.",
            model_configs=ModelConfigs(model="gpt-4.1"), 
            other_configs=OtherConfigs(timeaware=False)
        )
        
        task = Task(user_input="Hello", instruction=instruction)
        neo = Neo(tasks=[task])
        
        # Should not raise an exception
        assert neo is not None

    def test_string_tool_codes(self):
        """Test that string tool codes work as well as enum values."""
        instruction = Instruction(
            content="Act like a helpful assistant.",
            model_configs=ModelConfigs(model="gpt-4.1"), 
            other_configs=OtherConfigs(
                timeaware=False, 
                tools=[StandardToolCode.WEB_SEARCH.value]  # String version of tool code
            )
        )
        
        task = Task(user_input="Hello", instruction=instruction)
        neo = Neo(tasks=[task])
        
        # Should not raise an exception
        assert neo is not None
        assert task.instruction.other_configs.tools == [StandardToolCode.WEB_SEARCH.value]

    def test_all_standard_tool_codes_available(self):
        """Test that all StandardToolCode values are available for gpt-4.1."""
        instruction = Instruction(
            content="Act like a helpful assistant.",
            model_configs=ModelConfigs(model="gpt-4.1"), 
            other_configs=OtherConfigs(
                timeaware=False, 
                tools=[
                    StandardToolCode.WEB_SEARCH,
                    StandardToolCode.IMAGE_GEN,
                    StandardToolCode.TRANSCRIBE,
                ]
            )
        )
        
        task = Task(user_input="Hello", instruction=instruction)
        neo = Neo(tasks=[task])
        
        # Should not raise an exception
        assert neo is not None
        assert len(task.instruction.other_configs.tools) == 3


class TestToolRegistryIntegration:
    """Test tool registry functionality and integration."""

    def test_tool_registry_enum_to_string_conversion(self):
        """Test that tool registry converts enums to strings properly."""
        from neo.agentic.tool_registry import ToolRegistry
        from neo.models.providers.openai import OpenAIResponseModel
        
        tr = ToolRegistry()
        available_tools = tr.list_tools(OpenAIResponseModel)
        
        # Tools should be registered by their string values, not enum representations
        assert StandardToolCode.WEB_SEARCH.value in available_tools
        assert StandardToolCode.IMAGE_GEN.value in available_tools
        assert StandardToolCode.TRANSCRIBE.value in available_tools
        
        # Should not contain the full enum string representation
        assert "StandardToolCode.WEB_SEARCH" not in available_tools

    def test_tool_registry_get_tool_by_enum(self):
        """Test that tools can be retrieved by enum."""
        from neo.agentic.tool_registry import ToolRegistry
        from neo.models.providers.openai import OpenAIResponseModel
        
        tr = ToolRegistry()
        
        # Should be able to get tool by enum
        tool = tr.get_tool(OpenAIResponseModel, StandardToolCode.WEB_SEARCH)
        assert tool is not None
        assert tool.name == "WebSearch"

    def test_tool_registry_get_tool_by_string(self):
        """Test that tools can be retrieved by string."""
        from neo.agentic.tool_registry import ToolRegistry
        from neo.models.providers.openai import OpenAIResponseModel
        
        tr = ToolRegistry()
        
        # Should be able to get tool by string
        tool = tr.get_tool(OpenAIResponseModel, StandardToolCode.WEB_SEARCH.value)
        assert tool is not None
        assert tool.name == "WebSearch"

    def test_tool_registry_cross_provider_isolation(self):
        """Test that tools are properly isolated between providers."""
        from neo.agentic.tool_registry import ToolRegistry
        from neo.models.providers.openai import OpenAIResponseModel
        from neo.models.providers.anthropic import AnthropicModel
        
        tr = ToolRegistry()
        
        # Both should have web search, but different implementations
        openai_tool = tr.get_tool(OpenAIResponseModel, StandardToolCode.WEB_SEARCH)
        anthropic_tool = tr.get_tool(AnthropicModel, StandardToolCode.WEB_SEARCH)
        
        assert openai_tool is not None
        assert anthropic_tool is not None
        
        # Check that they are different tool instances with different underlying classes
        assert openai_tool != anthropic_tool
        assert type(openai_tool).__name__ == "WebSearch"
        assert type(anthropic_tool).__name__ == "WebSearch"
        
        # Verify they come from different modules/providers
        assert "openai" in str(type(openai_tool))
        assert "anthropic" in str(type(anthropic_tool))