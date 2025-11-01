import asyncio
import os
import socket
import subprocess
import tempfile
from enum import Enum
from typing import Any, Dict, List

from neo.types.contents import TextContent
from neo.types.errors import ToolError


class UnsupportedLanguageError(ToolError):
    """Raised when an unsupported programming language is specified."""


class CodeExecutionError(ToolError):
    """Raised when code execution fails or encounters an error."""


class CodeExecutionTimeoutError(CodeExecutionError):
    """Raised when code execution times out."""


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
        code: str,
        language: CodeLanguage,
        timeout: int = 30,
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
                CodeLanguage.TYPESCRIPT: ["npx", "tsx", temp_file_path],
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
            except asyncio.TimeoutError as e:
                process.kill()
                await process.wait()
                raise CodeExecutionTimeoutError(
                    f"Code execution timed out after {timeout} seconds"
                ) from e

            return {
                "success": result_returncode == 0,
                "output": stdout.decode() if stdout else "",
                "stderr": stderr.decode() if stderr else "",
                "return_code": result_returncode,
                "language": language.value,
            }
        # Allow custom tool errors to propagate without modification
        except (
            OSError,
            ValueError,
            RuntimeError,
            UnicodeError,
            subprocess.SubprocessError,
        ) as e:
            raise CodeExecutionError(f"Code execution failed: {str(e)}") from e
        finally:
            # Clean up temporary files
            try:
                os.unlink(temp_file_path)
            except (FileNotFoundError, PermissionError):
                pass


def _is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        try:
            sock.connect((host, port))
            return True
        except OSError:
            return False


async def code_execution_tool(
    language: CodeLanguage,
    code: str,
    timeout: int,
    start_server: bool = False,
    server_host: str = "127.0.0.1",
    server_port: int = 3000,
) -> List[Any]:
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
    except (ValueError, KeyError) as e:
        supported = ", ".join([lang.value for lang in CodeLanguage])
        lang_str = language if isinstance(language, str) else str(language)
        raise UnsupportedLanguageError(
            f"Unsupported language '{lang_str}'. Supported languages: {supported}"
        ) from e

    # If requested, start a long-running server in background and emit URL
    if start_server:
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=CodeGenHelper.get_file_extension(lang),
                delete=False,
            ) as temp_file:
                temp_file.write(code)
                temp_file_path = temp_file.name

            execution_commands = {
                CodeLanguage.PYTHON: ["python3", temp_file_path],
                CodeLanguage.JAVASCRIPT: ["node", temp_file_path],
                CodeLanguage.TYPESCRIPT: ["npx", "tsx", temp_file_path],
                CodeLanguage.SHELL: ["bash", temp_file_path],
                CodeLanguage.GO: ["go", "run", temp_file_path],
            }

            if lang not in execution_commands:
                raise UnsupportedLanguageError(
                    f"Server start not supported for {lang.value}"
                )

            # Start background process
            log_dir = tempfile.mkdtemp(prefix="neo-codegen-")
            stdout_path = os.path.join(log_dir, "stdout.log")
            stderr_path = os.path.join(log_dir, "stderr.log")
            stdout_f = open(stdout_path, "wb")
            stderr_f = open(stderr_path, "wb")

            process = subprocess.Popen(
                execution_commands[lang],
                cwd=os.path.dirname(temp_file_path),
                stdout=stdout_f,
                stderr=stderr_f,
                start_new_session=True,
            )

            # Best-effort readiness wait
            total_wait = 0.0
            while total_wait < 5.0:
                if _is_port_open(server_host, server_port):
                    break
                await asyncio.sleep(0.2)
                total_wait += 0.2

            server_url = f"http://{server_host}:{server_port}"
            summary = (
                f"Started server (pid {process.pid})\n"
                f"Language: {lang.value}\n"
                f"Command: {' '.join(execution_commands[lang])}\n"
                f"Logs: {stdout_path} (stdout), {stderr_path} (stderr)\n"
                f"URL: {server_url}\n"
                f'NEO_SIGNAL: {{"type": "open_url", "url": "{server_url}", "target": "browser"}}'
            )

            return [TextContent(data=summary)]
        except (
            OSError,
            ValueError,
            RuntimeError,
            UnicodeError,
            subprocess.SubprocessError,
        ) as e:
            raise CodeExecutionError(f"Failed to start server: {str(e)}") from e

    # Execute code - exceptions will bubble up
    result = await CodeGenHelper.execute_code(code, lang, timeout)
    formatted = (
        f"Success: {result['success']}\n"
        f"Language: {result['language']}\n"
        f"Return Code: {result['return_code']}\n"
        f"Stdout:\n{result['output'] if result['output'] else '(empty)'}\n"
        f"Stderr:\n{result['stderr'] if result['stderr'] else '(empty)'}\n"
    )
    return [TextContent(data=formatted)]
