from enum import Enum
from abc import ABC
from typing import Any, Callable, ClassVar, List, Optional, Dict

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from neo.agentic.instruction import Instruction
from neo.contexts import Context, Thread
from neo.utils.ids import IDMixin
from neo.models.base import BaseChatModel

class TaskStatus(str, Enum):
    """Status of a task during execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class BaseTask(BaseModel, ABC, IDMixin):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    subsequent_tasks: Optional[
        List["BaseTask"] | Dict[str, "BaseTask"]
    ] = Field(
        default=None,
        description=(
            "Dependent tasks that depend on this task. If a list, tasks will be "
            "triggered once this task is complete. If a dict, tasks will be triggered "
            "based on this task's output (the last content of the last context of the deliverable thread casted as a string) matching the dict keys."
        ),
        exclude=True,
        repr=False,
    )

    id: str = Field(
        default=None,
        description="Unique identifier for the task. Can be a UUID or a name of the task.",
    )

    SHORT_ID_LENGTH: ClassVar[int] = 5

    @model_validator(mode="before")
    @classmethod
    def _generate_id_if_not_provided(cls, values):
        """Generate ID using the actual class name if not provided."""
        if isinstance(values, dict) and values.get("id") is None:
            values["id"] = cls.generate_id()
        return values

    deliverable: Optional[Thread | Context] = Field(
        default=None, description="The deliverable of the task.", exclude=True
    )
    
    unfinished_deliverable: Optional[Thread | Context] = Field(
        default=None, description="The unfinished deliverable of the task during execution.", exclude=True
    )

    base_thread_snapshot: Optional[Thread] = Field(
        default=None,
        description="The base thread snapshot for the task.",
        exclude=True,
    )
    
    done_by: Optional[BaseChatModel] = Field(
        default=None,
        description="The model that completed the task.",
        exclude=True,
    )

    status: TaskStatus = Field(
        default=TaskStatus.PENDING, description="Current status of the task execution."
    )

    def add_subsequent_task(self, task: "BaseTask", trigger: str = None) -> None:
        """Add a dependent task to the current task."""
        if self.subsequent_tasks is None:
            self.subsequent_tasks = [] if trigger is None else {}
        
        if trigger is not None:
            if not isinstance(self.subsequent_tasks, dict):
                raise ValueError("subsequent_tasks must be a dict when using triggers.")
            self.subsequent_tasks[trigger] = task
        else:
            if not isinstance(self.subsequent_tasks, list):
                raise ValueError("subsequent_tasks must be a list when not using triggers.")
            self.subsequent_tasks.append(task)
            
    def list_subsequent_tasks(self) -> List["BaseTask"]:
        """List all dependent tasks of the current task."""
        if self.subsequent_tasks is None:
            return []
        
        if isinstance(self.subsequent_tasks, list):
            return self.subsequent_tasks
        else:
            return list(self.subsequent_tasks.values())

    def reset(self) -> None:
        """Reset the task to its initial state."""
        self.status = TaskStatus.PENDING
        self.deliverable = None
        self.base_thread_snapshot = None
        
    def __str__(self) -> str:
        """String representation of the Task."""

        status_emoji = {
            TaskStatus.PENDING: "⏳",
            TaskStatus.RUNNING: "🔄",
            TaskStatus.COMPLETED: "✅",
            TaskStatus.FAILED: "❌",
            TaskStatus.CANCELLED: "🚫",
        }.get(self.status, "❓")

        return f'<Task | Name: "{self.id}", Status: {status_emoji}{self.status.value}>'
        


class ModelTask(BaseTask):
    """The Task class encapsulates a task with user input, instruction code, instruction,
    and dependent tasks.
    """

    user_input: Optional[str | Context] = Field(
        default=None,
        description="User input for the task. If None, no additional input besides upstream task outputs.",
    )

    instruction: Optional[Instruction | str] = Field(
        default=None,
        description="Instruction for the task. If string, it matches with the Instruction code.",
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


class FunctionTask(BaseTask):
    """A task that executes a Python callable function.

    The callable receives the Neo state dict (mapping task IDs to BaseTask objects)
    and must return a Context object.
    """

    func: Callable[[Dict[str, BaseTask]], Context] = Field(
        ...,
        description="Callable that receives state dict and returns a Context object.",
    )

    @field_validator("func")
    @classmethod
    def _validate_func(cls, v):
        if not callable(v):
            raise ValueError("func must be callable.")
        return v