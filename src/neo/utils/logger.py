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


import os
import time
import threading
from collections import deque


class CustomLogger(logging.Logger):
    _message_queue = deque()
    _rate_limiter_thread = None
    _rate_limit_lock = threading.Lock()

    def __init__(
        self, name: str, level: int | str = 0, send_webhook: bool = True
    ) -> None:
        self.send_webhook = send_webhook

        self.webhook_url = os.getenv("NEO_LOGGER_WEBHOOK_URL")
        if self.webhook_url is None:
            self.send_webhook = False

        super().__init__(name, level)

        # Start rate limiter thread if webhook is enabled
        if self.send_webhook:
            self._start_rate_limiter()

    @classmethod
    def _start_rate_limiter(cls):
        """Start the background thread that processes the message queue"""
        if cls._rate_limiter_thread is None or not cls._rate_limiter_thread.is_alive():
            cls._rate_limiter_thread = threading.Thread(
                target=cls._process_queue, daemon=True, name="WebhookRateLimiter"
            )
            cls._rate_limiter_thread.start()

    @classmethod
    def _process_queue(cls):
        """Background thread that sends messages from the queue at a controlled rate"""
        while True:
            try:
                with cls._rate_limit_lock:
                    if cls._message_queue:
                        url, payload = cls._message_queue.popleft()
                    else:
                        url, payload = None, None

                if url and payload:
                    cls._blocking_post_with_retry(url, payload)

                # Discord limit: 5 requests per 2 seconds
                # Sleep 0.5s = 2 requests/sec = well under the limit
                time.sleep(0.5)
            except Exception as e:
                print(f"CustomLogger: Rate limiter error: {e}")
                time.sleep(1)

    @staticmethod
    def _blocking_post_with_retry(url, json_payload, max_retries=3):
        """Send webhook with exponential backoff on 429 errors"""
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=json_payload, timeout=10)

                if response.status_code == 429:
                    # Rate limited - check Retry-After header
                    retry_after = int(response.headers.get('Retry-After', 5))
                    print(f"CustomLogger: Rate limited (429), waiting {retry_after}s")
                    time.sleep(retry_after)
                    continue

                if response.status_code not in (200, 204):
                    print(f"CustomLogger: Webhook failed: HTTP {response.status_code}")

                return  # Success

            except requests.exceptions.RequestException as e:
                print(f"CustomLogger: Webhook request error: {e}")
                if attempt < max_retries - 1:
                    backoff = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    time.sleep(backoff)
            except Exception as e:
                print(f"CustomLogger: Unexpected webhook error: {e}")
                return

    def _send_webhook_message(self, log_type: str, message: str):
        """Queue a message for webhook delivery instead of sending immediately"""
        if not self.send_webhook or not self.webhook_url:
            return

        log_type_capitalized = log_type.capitalize()

        # Add header to message before chunking
        header = "=" * 50
        header += f"\n# {self.name}: \t{log_type_capitalized}\n"
        full_message = header + message

        # Add each chunk to the queue for rate-limited sending
        for r_chunk in group_by_len(full_message, max_len=1900):
            payload = {"content": r_chunk}
            with self._rate_limit_lock:
                self._message_queue.append((self.webhook_url, payload))

    def info(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.INFO):
            super().info(msg, *args, **kwargs)

            if self.send_webhook:
                if args:
                    msg = msg % args
                self._send_webhook_message("INFO", msg)

    def debug(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.DEBUG):
            super().debug(msg, *args, **kwargs)

            if self.send_webhook:
                if args:
                    msg = msg % args
                self._send_webhook_message("DEBUG", msg)

    def warning(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.WARNING):
            super().warning(msg, *args, **kwargs)

            if self.send_webhook:
                if args:
                    msg = msg % args
                self._send_webhook_message("WARNING", msg)

    def error(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.ERROR):
            super().error(msg, *args, **kwargs)

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
