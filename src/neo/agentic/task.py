from enum import Enum
from typing import Any, ClassVar, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from neo.agentic.instruction import Instruction
from neo.contexts import Context, Thread
from neo.utils.ids import IDMixin


class TaskStatus(str, Enum):
    """Status of a task during execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Task(IDMixin, BaseModel):
    """The Task class encapsulates a task with user input, instruction code, instruction,
    and dependent tasks.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_input: Optional[str | Context] = Field(
        default=None,
        description="User input for the task. If None, no additional input besides upstream task outputs.",
    )

    instruction: Optional[Instruction | str] = Field(
        default=None,
        description="Instruction for the task. If string, it matches with the Instruction code.",
    )

    subsequent_tasks: Optional[List["Task"]] = Field(
        default=None,
        description="List of dependent tasks that depend on this task.",
        exclude=True,
        repr=False,
    )

    id: str = Field(
        default_factory=lambda: Task.generate_id(),
        description="Unique identifier for the task.",
    )

    name: Optional[str] = Field(default=None, description="The name of the task.")

    SHORT_ID_LENGTH: ClassVar[int] = 5

    deliverable: Optional[Thread] = Field(
        default=None, description="The deliverable of the task.", exclude=True
    )

    base_thread_snapshot: Optional[Thread] = Field(
        default=None,
        description="The base thread snapshot for the task.",
        exclude=True,
    )

    status: TaskStatus = Field(
        default=TaskStatus.PENDING, description="Current status of the task execution."
    )

    @field_validator("user_input")
    @classmethod
    def _validate_user_input(cls, v):
        if isinstance(v, str):
            v = v.strip()
            if not v:
                raise ValueError("User input must be a non-empty string.")
            v = Context(contents=v)
        return v

    @field_validator("instruction")
    @classmethod
    def _validate_instruction(cls, v):
        if isinstance(v, str):
            v = v.strip().lower()
            if not v:
                raise ValueError("Instruction code must be a non-empty string.")
            v = Instruction(content=v)

        return v

    def model_post_init(self, __context: Any) -> None:
        if not self.user_input and not self.instruction:
            raise ValueError("Either user_input or instruction must be provided.")

    def add_subsequent_task(self, task: "Task") -> None:
        """Add a dependent task to the current task."""
        if self.subsequent_tasks is None:
            self.subsequent_tasks = []
        self.subsequent_tasks.append(task)

    def reset(self) -> None:
        """Reset the task to its initial state."""
        self.status = TaskStatus.PENDING
        self.deliverable = None
        self.base_thread_snapshot = None

    def __str__(self) -> str:
        """String representation of the Task."""

        _id = self.id[-self.SHORT_ID_LENGTH :]
        status_emoji = {
            TaskStatus.PENDING: "⏳",
            TaskStatus.RUNNING: "🔄",
            TaskStatus.COMPLETED: "✅",
            TaskStatus.FAILED: "❌",
            TaskStatus.CANCELLED: "🚫",
        }.get(self.status, "❓")

        if self.name:
            return f'<Task | Name: "{self.name}", Status: {status_emoji}{self.status.value}, ID: {_id}>'

        def get_truncated_string(s: Optional[str], max_len: int = 20) -> str:
            if not s:
                return "N/A"
            s = str(s).strip()
            if len(s) > max_len:
                return s[: max_len - 3] + "..."
            return s

        user_input_str = get_truncated_string(
            str(self.user_input) if self.user_input else None
        )
        instruction_str = get_truncated_string(
            str(self.instruction) if self.instruction else None
        )

        parts = []
        if self.user_input:
            parts.append(f'Input: "{user_input_str}"')
        if self.instruction:
            parts.append(f'Instr: "{instruction_str}"')

        parts.append(f"Status: {status_emoji}{self.status.value}")
        parts.append(f"ID: {_id}")

        return f"<Task | {', '.join(parts)}>"
