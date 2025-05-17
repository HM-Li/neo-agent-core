# %%
import asyncio
import json
import pprint
from collections import deque
from typing import List

from neo.contexts.context import Context
from neo.types.contents import BaseContent
from neo.types.roles import Role
from neo.utils.logger import get_logger

LOGGER = get_logger("Thread")


class Thread:
    """A thread is a collection of contexts. A model can turn a thread into it's own API format, such as openai chat completion messages."""

    def __init__(self, contexts: List[Context] | Context | str = None):
        self._contexts = deque([], maxlen=100)
        self._lock = asyncio.Lock()
        if contexts is not None:
            if not isinstance(contexts, list):
                contexts = [contexts]

            # Add contexts synchronously in constructor
            for context in contexts:
                self._append_context(context)

    async def __aiter__(self):
        """Asynchronously iterate over a snapshot of contexts."""
        async with self._lock:
            snapshot = list(self._contexts)
        for context in snapshot:
            yield context

    async def aget_context(self, index: int):
        async with self._lock:
            snapshot = list(self._contexts)
        try:
            return snapshot[index]
        except IndexError as e:
            raise IndexError("Context index out of range.") from e

    def get_context(self, index: int):
        """Get a context at a specific index."""
        try:
            return self._contexts[index]
        except IndexError as e:
            raise IndexError("Context index out of range.") from e

    def __len__(self):
        return len(self._contexts)

    def is_empty(self):
        """Check if the thread is empty."""
        return len(self._contexts) == 0

    @property
    def last_context_time(self):
        """Get the time of the last context in the thread."""
        if self._contexts:
            return self._contexts[-1].time_provided
        return None

    async def append_contexts(self, contexts: List[Context]):
        """Add a list of contexts to the thread."""
        if not isinstance(contexts, list):
            raise TypeError("contexts must be a list of Context objects.")

        async with self._lock:
            for context in contexts:
                self._append_context(context)

    def _append_context(self, context: Context | str):
        """Append a context to the end of the thread.

        Parameters
        ----------
        context : Context | str
            if str, will be converted to a context with an alternative role
        """
        match context:
            case str():
                previous_role = Role.SYSTEM
                if self._contexts:
                    previous_role = self._contexts[-1].provider_role

                if previous_role != Role.USER:
                    role = Role.USER
                else:
                    role = Role.ASSISTANT

                LOGGER.info(f"Converting string to context with role {role}.")
                context = Context(contents=context, provider_role=role)
            case BaseContent():
                context = Context(contents=[context])

        self._contexts.append(context)

    async def append_context(self, context: Context | str):
        """Append a context to the end of the thread.

        Parameters
        ----------
        context : Context | str
            if str, will be converted to a context with an alternative role
        """
        async with self._lock:
            self._append_context(context)

    async def extend_thread(self, thread: "Thread"):
        """Extend the thread with another thread."""
        # First get all contexts from the source thread
        contexts = []
        async for context in thread:
            contexts.append(context)

        # Then add them all under a single lock
        async with self._lock:
            for context in contexts:
                self._append_context(context)

    async def add_context_to_beginning(self, context: Context):
        async with self._lock:
            self._contexts.appendleft(context)

    async def pop_context_from_beginning(self) -> Context:
        async with self._lock:
            return self._contexts.popleft()

    async def pop_context_from_end(self) -> Context:
        async with self._lock:
            return self._contexts.pop()

    async def afork(self) -> "Thread":
        """Fork the thread into a new thread with a new deque containing the same context references."""
        async with self._lock:
            new_thread = Thread()
            new_thread._contexts = deque(self._contexts)
            return new_thread

    @property
    def participants(self):
        """Get the participants in the thread."""
        participants = set()
        for context in self._contexts:
            if context.provider_name:
                participants.add(context.provider_name)
        return participants

    def __repr__(self):
        return f"Thread({len(self._contexts)} contexts)"

    def __str__(self):
        return pprint.pformat(self._contexts)

    def display(self):
        if not self._contexts:
            return "Empty Thread"

        messages = []
        for ctx in self._contexts:
            content = str(ctx).replace("\n", "\n    ")  # indent content
            role = ctx.provider_role.value
            messages.append(f"{role}: {content}")
            messages.append("---")  # Add a separator line

        # Remove the last separator
        if messages:
            messages.pop()

        print("\n".join(messages))

    def dumps(self):
        """Serialize the thread to a string."""
        l = [c.model_dump_json(indent=4) for c in self._contexts]
        json_str = "[" + ",\n".join(l) + "]"
        return json_str

    def dump(self, file_path: str):
        """Serialize the thread to a file."""
        with open(file_path, "w") as f:
            f.write(self.dumps())


# %%
# %%
