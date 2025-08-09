import pytest
from neo.tools.code_gen import (
    code_execution_tool, 
    CodeGenHelper, 
    CodeLanguage,
    UnsupportedLanguageError,
    CodeExecutionTimeoutError,
    CodeExecutionError
)
from neo.tools.tool import Tool
from neo.types.contents import TextContent


@pytest.fixture
def tool():
    """Test fixture for code execution tool."""
    return Tool(func=code_execution_tool)


def test_tool_metadata(tool):
    """Test that the tool has correct metadata."""
    assert tool.name == "code_execution_tool"
    assert tool.description is not None
    assert "python" in tool.description.lower()
    
    # Check parameters schema
    params = tool.params
    assert params["type"] == "object"
    assert "language" in params["properties"]
    assert "code" in params["properties"]
    assert "timeout" in params["properties"]
    assert params["required"] == ["language", "code", "timeout"]
    assert not params["additionalProperties"]

@pytest.mark.asyncio
async def test_python_execution_success():
    """Test successful Python code execution."""
    result = await code_execution_tool(
        language=CodeLanguage.PYTHON,
        code="print('Hello World')\nprint(2 + 3)",
        timeout=10
    )
    
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    
    output = result[0].data
    assert "Success: True" in output
    assert "Hello World" in output
    assert "5" in output
    assert "Return Code: 0" in output

@pytest.mark.asyncio
async def test_python_execution_error():
    """Test Python code execution with syntax error."""
    result = await code_execution_tool(
        language=CodeLanguage.PYTHON,
        code="print('unterminated string",
        timeout=10
    )
    
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    
    output = result[0].data
    assert "Success: False" in output
    assert "SyntaxError" in output

@pytest.mark.asyncio
async def test_javascript_execution():
    """Test JavaScript code execution."""
    result = await code_execution_tool(
        language=CodeLanguage.JAVASCRIPT,
        code="console.log('Hello from JS'); console.log(10 + 5);",
        timeout=10
    )
    
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    
    output = result[0].data
    if "Success: True" in output:
        assert "Hello from JS" in output
        assert "15" in output
    else:
        # Skip if Node.js is not available
        pytest.skip("Node.js not available for testing")

@pytest.mark.asyncio
async def test_unsupported_language():
    """Test handling of unsupported language."""
    # Test that the function raises UnsupportedLanguageError for invalid string input
    with pytest.raises(UnsupportedLanguageError) as exc_info:
        await code_execution_tool("unsupported_lang", "print('test')", 10)
    
    assert "Unsupported language 'unsupported_lang'" in str(exc_info.value)
    assert "Supported languages:" in str(exc_info.value)

@pytest.mark.asyncio
async def test_shell_execution():
    """Test shell script execution."""
    result = await code_execution_tool(
        language=CodeLanguage.SHELL,
        code="echo 'Hello from shell'\necho 'Current directory:'\npwd",
        timeout=10
    )
    
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    
    output = result[0].data
    assert "Success: True" in output
    assert "Hello from shell" in output

@pytest.mark.asyncio
async def test_timeout_parameter():
    """Test that timeout parameter is handled correctly."""
    # Test with a very short timeout (this should work quickly)
    result = await code_execution_tool(
        language=CodeLanguage.PYTHON,
        code="print('quick test')",
        timeout=1
    )
    
    assert isinstance(result, list)
    assert len(result) == 1
    output = result[0].data
    assert "Success: True" in output


def test_get_file_extension():
    """Test file extension mapping."""
    assert CodeGenHelper.get_file_extension(CodeLanguage.PYTHON) == ".py"
    assert CodeGenHelper.get_file_extension(CodeLanguage.JAVASCRIPT) == ".js"
    assert CodeGenHelper.get_file_extension(CodeLanguage.SHELL) == ".sh"
    assert CodeGenHelper.get_file_extension(CodeLanguage.GO) == ".go"
    assert CodeGenHelper.get_file_extension(CodeLanguage.TYPESCRIPT) == ".ts"

@pytest.mark.asyncio
async def test_execute_code_direct():
    """Test direct code execution through helper."""
    helper = CodeGenHelper()
    result = await helper.execute_code("print('direct test')", CodeLanguage.PYTHON, 10)
    
    assert isinstance(result, dict)
    assert "success" in result
    assert "output" in result
    assert "stderr" in result
    assert "return_code" in result
    assert "language" in result
    
    assert result["success"] is True
    assert "direct test" in result["output"]
    assert result["language"] == "python"

@pytest.mark.asyncio
async def test_execute_code_timeout():
    """Test code execution timeout handling."""
    helper = CodeGenHelper()
    # Test with a sleep command that should timeout
    with pytest.raises(CodeExecutionTimeoutError) as exc_info:
        await helper.execute_code(
            "import time; time.sleep(5)", 
            CodeLanguage.PYTHON, 
            timeout=1
        )
    
    assert "timed out" in str(exc_info.value)


@pytest.mark.asyncio
async def test_code_execution_tool_timeout():
    """Test code execution tool timeout handling."""
    # Test with a sleep command that should timeout
    with pytest.raises(CodeExecutionTimeoutError) as exc_info:
        await code_execution_tool(
            CodeLanguage.PYTHON,
            "import time; time.sleep(5)",
            timeout=1
        )
    
    assert "timed out" in str(exc_info.value)


@pytest.mark.asyncio
async def test_enum_string_representation():
    """Test handling of enum string representation like 'CodeLanguage.PYTHON'."""
    result = await code_execution_tool(
        "CodeLanguage.PYTHON",  # This is what AI systems sometimes pass
        "print('Hello from enum string!')",
        timeout=10
    )
    
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    
    output = result[0].data
    assert "Success: True" in output
    assert "Hello from enum string!" in output


@pytest.mark.asyncio
async def test_different_language_formats():
    """Test different ways to specify languages."""
    test_cases = [
        ("python", "print('plain string')"),
        ("PYTHON", "print('uppercase string')"),
        ("CodeLanguage.PYTHON", "print('enum string representation')"),
        (CodeLanguage.PYTHON, "print('actual enum')"),
    ]
    
    for language_input, code in test_cases:
        result = await code_execution_tool(language_input, code, timeout=10)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert "Success: True" in result[0].data