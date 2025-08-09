import asyncio
import os
import subprocess
import tempfile
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from neo.tools.tool import BaseTool
from neo.types.contents import TextContent
from neo.types.errors import ToolError


class UnsupportedLanguageError(ToolError):
    """Raised when an unsupported programming language is specified."""

    pass


class CodeExecutionError(ToolError):
    """Raised when code execution fails or encounters an error."""

    pass


class CodeExecutionTimeoutError(CodeExecutionError):
    """Raised when code execution times out."""

    pass


class CodeLanguage(str, Enum):
    """Supported programming languages for code execution."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    SHELL = "shell"
    TYPESCRIPT = "typescript"
    GO = "go"


class CodeGenHelper:
    """
    ⚠️  DEVELOPMENT ONLY - NOT SECURE FOR PRODUCTION ⚠️

    Helper class for code execution operations.

    SECURITY WARNING: This implementation lacks proper sandboxing and should
    ONLY be used in development environments. Production use requires:
    - Docker containerization
    - Resource limits (CPU, memory, network)
    - Filesystem isolation
    - Privilege dropping
    - Comprehensive audit logging
    """

    @staticmethod
    def get_file_extension(language: CodeLanguage) -> str:
        """Get file extension for a programming language."""
        extensions = {
            CodeLanguage.PYTHON: ".py",
            CodeLanguage.JAVASCRIPT: ".js",
            CodeLanguage.TYPESCRIPT: ".ts",
            CodeLanguage.SHELL: ".sh",
            CodeLanguage.GO: ".go",
        }
        return extensions.get(language, ".txt")

    @staticmethod
    async def execute_code(
        code: str, language: CodeLanguage, timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Execute code safely in a temporary environment.

        Args:
            code: Code to execute
            language: Programming language
            timeout: Execution timeout in seconds

        Returns:
            Dictionary containing execution results
        """
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=CodeGenHelper.get_file_extension(language),
                delete=False,
            ) as temp_file:
                temp_file.write(code)
                temp_file_path = temp_file.name

            # Define execution commands for different languages
            execution_commands = {
                CodeLanguage.PYTHON: ["python3", temp_file_path],
                CodeLanguage.JAVASCRIPT: ["node", temp_file_path],
                CodeLanguage.TYPESCRIPT: ["npx", "ts-node", temp_file_path],
                CodeLanguage.SHELL: ["bash", temp_file_path],
                CodeLanguage.GO: ["go", "run", temp_file_path],
            }

            if language not in execution_commands:
                raise UnsupportedLanguageError(
                    f"Execution not supported for {language.value}"
                )

            # Execute the code
            process = await asyncio.create_subprocess_exec(
                *execution_commands[language],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.path.dirname(temp_file_path),
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
                result_returncode = process.returncode
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise CodeExecutionTimeoutError(
                    f"Code execution timed out after {timeout} seconds"
                )

            return {
                "success": result_returncode == 0,
                "output": stdout.decode() if stdout else "",
                "stderr": stderr.decode() if stderr else "",
                "return_code": result_returncode,
                "language": language.value,
            }
        except (
            UnsupportedLanguageError,
            CodeExecutionTimeoutError,
            CodeExecutionError,
        ):
            raise
        except Exception as e:
            raise CodeExecutionError(f"Code execution failed: {str(e)}")
        finally:
            # Clean up temporary files
            try:
                os.unlink(temp_file_path)
            except:
                pass


async def code_execution_tool(
    language: CodeLanguage, code: str, timeout: int
) -> TextContent:
    """
    Execute code in various programming languages and return formatted results.

    ⚠️  DEVELOPMENT ONLY - NOT SECURE FOR PRODUCTION ⚠️

    This tool executes code in a temporary file environment and returns the output.
    Supports Python, JavaScript, TypeScript, Shell scripts, and Go.

    Args:
        language (CodeLanguage): Programming language to execute
        code (str): The source code to execute
        timeout (int): Maximum execution time in seconds

    Returns:
        TextContent: The output of the executed code

    Raises:
        UnsupportedLanguageError: If an unsupported language is provided
        CodeExecutionTimeoutError: If code execution exceeds the timeout limit
        CodeExecutionError: For other execution errors or failures

    SECURITY WARNING: No sandboxing - development environments only.
    """
    # Handle different input formats
    try:
        if isinstance(language, str):
            if language.startswith("CodeLanguage."):
                enum_name = language.split(".", 1)[1]
                lang = CodeLanguage[enum_name]
            else:
                lang = CodeLanguage(language.lower())
        else:
            lang = language
    except (ValueError, KeyError):
        supported = ", ".join([lang.value for lang in CodeLanguage])
        lang_str = language if isinstance(language, str) else str(language)
        raise UnsupportedLanguageError(
            f"Unsupported language '{lang_str}'. Supported languages: {supported}"
        )

    # Execute code - exceptions will bubble up
    result = await CodeGenHelper.execute_code(code, lang, timeout)
    return TextContent(data=result["output"])
