# %%
import logging


def get_logger(name):
    logger = CustomLogger(name=name)

    # Clear existing handlers
    logger.handlers.clear()
    logging.getLogger().handlers.clear()

    # Set level and formatter
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    # Add handler to logger
    logger.addHandler(ch)
    return logger


import textwrap
import traceback
from typing import Iterable

import requests


def group_by_len(text, max_len: int = 1024) -> Iterable[str]:
    """optimized for discord message and showing markdown code block"""
    items = text.splitlines()

    start, count = 0, 0
    for end, item in enumerate(items):
        n = len(item) + 2
        if n + count >= max_len and count > 0:
            r = "\n".join(items[start:end])
            # maybe one line is too long
            if len(r) > max_len:
                _r = textwrap.wrap(r, width=max_len)
            else:
                _r = [r]
            for __r in _r:
                yield __r
            count = 0
            start = end
        count += n

    if count > 0:
        r = "\n".join(items[start:])
        if len(r) > max_len:
            _r = textwrap.wrap(r, width=max_len)
        else:
            _r = [r]
        for __r in _r:
            yield __r


import asyncio
import os
from concurrent.futures import ThreadPoolExecutor


class CustomLogger(logging.Logger):
    _executor = None

    def __init__(
        self, name: str, level: int | str = 0, send_webhook: bool = True
    ) -> None:
        self.send_webhook = send_webhook

        self.webhook_url = os.getenv("NEO_LOGGER_WEBHOOK_URL")
        if self.webhook_url is None:
            self.send_webhook = False

        super().__init__(name, level)

    @classmethod
    def _get_executor(cls):
        if cls._executor is None:
            cls._executor = ThreadPoolExecutor(
                max_workers=5, thread_name_prefix="LoggerWebhook"
            )
        return cls._executor

    def _send_webhook_message(self, log_type: str, message: str):
        if not self.send_webhook or not self.webhook_url:
            return

        # This function will be executed in a separate thread by the executor
        def _blocking_post(url, json_payload):
            try:
                requests.post(url, json=json_payload, timeout=10)
            except requests.exceptions.RequestException as e:
                # Log to stderr or a fallback mechanism to avoid recursion and further blocking
                print(
                    f"CustomLogger: Webhook request failed for {self.name} ({log_type}): {e}"
                )
            except Exception as e:
                # Catch any other unexpected errors in the thread
                print(
                    f"CustomLogger: Unexpected error in webhook sending thread for {self.name} ({log_type}): {e}"
                )

        starter = True
        log_type_capitalized = log_type.capitalize()
        for r_chunk in group_by_len(message, max_len=1900):
            current_chunk_content_with_header = r_chunk
            if starter:
                header = "=" * 50
                header += f"\n# {self.name}: \t{log_type_capitalized}\n"
                current_chunk_content_with_header = header + r_chunk
                starter = False

            payload = {"content": current_chunk_content_with_header}

            try:
                loop = asyncio.get_running_loop()
                # If we are in an asyncio event loop, run the blocking call in an executor
                executor = CustomLogger._get_executor()
                loop.run_in_executor(
                    executor, _blocking_post, self.webhook_url, payload
                )
            except RuntimeError:  # No running event loop
                # Not in an asyncio event loop (e.g., synchronous script), call directly
                _blocking_post(self.webhook_url, payload)

    def info(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.INFO):
            super().info(msg, *args, **kwargs)

            if self.send_webhook:
                if args:
                    msg = msg % args
                self._send_webhook_message("INFO", msg)

    def debug(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.DEBUG):
            super().info(msg, *args, **kwargs)

            if self.send_webhook:
                if args:
                    msg = msg % args
                self._send_webhook_message("DEBUG", msg)

    def warning(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.WARNING):
            super().info(msg, *args, **kwargs)

            if self.send_webhook:
                if args:
                    msg = msg % args
                self._send_webhook_message("WARNING", msg)

    def error(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.ERROR):
            super().info(msg, *args, **kwargs)

            if self.send_webhook:
                if args:
                    msg = msg % args
                self._send_webhook_message("ERROR", msg)

    def server_tool(self, tool_name: str, tool_data: dict, *args, **kwargs):
        """Log server tool usage (like web search results) with structured format"""
        if self.isEnabledFor(logging.INFO):
            info_type = "SERVER_TOOL"
            formatted_msg = f"'{tool_name}' executed: {tool_data}"
            super().info(f"[{info_type}] - {formatted_msg}", *args, **kwargs)

            if self.send_webhook:
                self._send_webhook_message(info_type, formatted_msg)

    def thinking(self, thinking_content: str, *args, **kwargs):
        """Log model thinking/reasoning content with special formatting"""
        if self.isEnabledFor(logging.INFO):
            info_type = "THINKING"
            formatted_msg = f"{thinking_content}"
            super().info(f"[{info_type}] - {formatted_msg}", *args, **kwargs)

            if self.send_webhook:
                self._send_webhook_message(info_type, formatted_msg)


# %%
