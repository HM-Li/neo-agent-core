import pytest
import pydantic
from unittest.mock import patch, MagicMock, AsyncMock
import uuid

from neo.contexts import Thread, Context
from neo.types import contents as C
from neo.types.roles import Role
from neo.agentic import Neo, Task, Instruction, ModelConfigs, OtherConfigs


@pytest.fixture
def basic_instruction():
    return Instruction(
        content="Act like a helpful assistant.",
        model_configs=ModelConfigs(model="gpt-4o"),
        other_configs=OtherConfigs(timeaware=False),
    )


@pytest.fixture
def basic_task(basic_instruction):
    return Task(
        user_input="What is the capital of France?", instruction=basic_instruction
    )


@pytest.fixture
def mock_model_response():
    thread = Thread(
        contexts=[
            "What is the capital of France?",
            Context(role=Role.ASSISTANT, contents="The capital of France is Paris."),
        ]
    )
    return thread


# Patch uuid to ensure deterministic IDs for testing
@patch("uuid.uuid4")
def test_task_creation(mock_uuid):
    _id_str = "12345678-1234-5678-1234-567812345678"
    mock_uuid.return_value = uuid.UUID("12345678-1234-5678-1234-567812345678")

    instruction = Instruction(content="Be helpful")
    task = Task(user_input="Hello", instruction=instruction)

    assert task.id == f"task_{_id_str}"
    assert isinstance(task.user_input, Context)
    assert task.user_input.contents[0].data == "Hello"
    assert task.instruction == instruction
    assert task.subsequent_tasks == None
    assert task.base_thread_snapshot is None


def test_instruction_defaults():
    # Test with minimal params
    instr = Instruction(content="Be helpful")
    assert instr.content == "Be helpful"
    assert instr.model_configs is None
    assert instr.other_configs is None

    # Test with custom configs
    model_config = ModelConfigs(model="gpt-4o", temperature=0.7)
    other_config = OtherConfigs(timeaware=True)
    instr = Instruction(
        content="Be creative", model_configs=model_config, other_configs=other_config
    )
    assert instr.model_configs.model == "gpt-4o"
    assert instr.model_configs.temperature == 0.7
    assert instr.other_configs.timeaware is True


def test_model_configs():
    # Test default initialization
    with pytest.raises(pydantic.ValidationError):
        config = ModelConfigs()

    # Test with custom values
    config = ModelConfigs(model="gpt-4o", temperature=0.8, max_tokens=1000, top_p=0.9)
    assert config.model == "gpt-4o"
    assert config.temperature == 0.8
    assert config.max_tokens == 1000
    assert config.top_p == 0.9


def test_other_configs():
    # Test with custom values
    config = OtherConfigs(
        timeaware=True,
    )
    assert config.timeaware is True


def test_task_dependencies():
    task3 = Task(user_input="Task 3")
    task2 = Task(user_input="Task 2", subsequent_tasks=[task3])
    task1 = Task(user_input="Task 1", subsequent_tasks=[task2])

    assert len(task1.subsequent_tasks) == 1
    assert task1.subsequent_tasks[0] == task2
    assert len(task2.subsequent_tasks) == 1
    assert task2.subsequent_tasks[0] == task3


@pytest.mark.asyncio
@patch("neo.models.providers.openai.OpenAIResponseModel.acreate")
async def test_neo_run_all(mock_acreate, basic_task, mock_model_response):
    mock_acreate.return_value = mock_model_response

    # Create Neo instance with tasks
    task1 = basic_task
    task2 = Task(
        user_input="What is the population of Paris?",
        instruction=task1.instruction,
        subsequent_tasks=[task1],
    )

    neo = Neo(tasks=[task1, task2])

    # Run all tasks
    final_thread = await neo.run_all()
