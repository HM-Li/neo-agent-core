import asyncio
import copy
from collections import defaultdict
from typing import Dict, List, Set

from neo.agentic.instruction import ModelConfigs
from neo.agentic.model_registry import ModelRegistry
from neo.agentic.task import Task, TaskStatus
from neo.agentic.tool_registry import ToolRegistry
from neo.contexts import Context, Thread
from neo.types.contents import ToolOutputContent
from neo.types.errors import TaskRuntimeError
from neo.types.roles import Role
from neo.types.tool_codes import StandardToolCode
from neo.utils.logger import get_logger


class Neo:

    def __init__(
        self,
        tasks: List[Task] | Task = None,
        default_model_configs: ModelConfigs = None,
        max_tool_execution_rounds: int = 5,
        default_user_input: str = "continue",
        model_registry_fuzzy_mode: bool = False,
    ) -> None:
        """
        A class to manage the execution of tasks in a directed acyclic graph (DAG).
        Manages the execution of Task nodes in a directed acyclic graph (DAG) while respecting dependencies.

        Tasks are executed in topological order; tasks with no dependencies are run concurrently, and
        tasks are scheduled as soon as all their prerequisites have completed.

        Parameters
        ----------
        tasks : List[Task] | Task, optional
            A list of Task objects or a single Task object to start with.
                Any dependent tasks will be added automatically.
            If None, no tasks are added initially.

        default_model_configs : ModelConfigs, optional
            Default model configurations to use for tasks that don't specify their own.
            If provided as a dict, it will be converted to a ModelConfigs instance.
            If None, tasks must provide their own model configurations.

        max_tool_execution_rounds : int, optional
            Maximum number of tool execution rounds per task to prevent infinite loops.
            Default is 5.

        default_user_input : str, optional
            Default user input to use if a task does not provide any.
            This ensures the conversation can continue even without explicit input.
            Default is "continue the conversation".
            This is important to prevent empty responses from the model when the previous response was from another assistant.

        model_registry_fuzzy_mode : bool, optional
            If True, the model registry will use fuzzy matching to find models.
            This is useful when you want to allow different versions or slight variations of model names. e.g. gpt-3.5-turbo vs gpt-3.5-turbo-0301.
        """
        self.model_registry = ModelRegistry()
        self.tool_registry = ToolRegistry()

        if (
            not isinstance(default_model_configs, ModelConfigs)
            and default_model_configs is not None
        ):
            default_model_configs = ModelConfigs(**default_model_configs)

        self.default_model_configs = default_model_configs
        self.max_tool_execution_rounds = max_tool_execution_rounds
        self.default_user_input = default_user_input
        self.model_registry_fuzzy_mode = model_registry_fuzzy_mode

        self.tasks: Dict[str, Task] = {}
        self.head_task_ids: Set[str] = set()
        self.end_task_ids: Set[str] = set()
        self.in_degree: Dict[str, int] = defaultdict(int)
        self.logger = get_logger(__name__)

        self._init_tasks(tasks)

    def _init_tasks(self, tasks: List[Task] | Task) -> Dict[str, Task]:
        """
        Initializes the task manager with a list of task heads or a single task head.

        Parameters
        ----------
        tasks : List[Task] | Task
            A list of Task objects or a single Task object to start with.

        Returns
        -------
        Dict[str, Task]
            A dictionary mapping task IDs to Task objects.
        """
        if isinstance(tasks, Task):
            tasks = [tasks]

        # get head tasks and check for cyclic dependencies
        dependent_tasks = set()
        for task in tasks:
            dependent_tasks.update(self._register_and_fetch_dependents(task))

        self.head_task_ids = self.tasks.keys() - dependent_tasks
        if not self.head_task_ids:
            raise ValueError("No head tasks found; possible cyclic dependency exists.")

        self.logger.info(
            f"Head tasks: {tuple(str(self.tasks[i]) for i in self.head_task_ids)}"
        )

        # Build the graph in a single optimized pass
        self._build_graph()

        end_tasks = [str(self.tasks[task_id]) for task_id in self.end_task_ids]
        self.logger.info(f"End tasks: {', '.join(end_tasks)}")

        # Warn if graph doesn't converge to single task
        if len(self.end_task_ids) > 1:
            self.logger.warning(
                f"Task graph has {len(self.end_task_ids)} end tasks and does not converge to a single task. Consider reviewing task dependencies."
            )

    def _register_and_fetch_dependents(
        self, task: Task, visited: Set[str] = None
    ) -> Set[str]:
        """recursively get all dependent tasks with cycle detection"""
        if visited is None:
            visited = set()

        # Cycle detection
        if task.id in visited:
            raise ValueError(
                f"Cyclic dependency detected: Task {task} forms a cycle in the task graph."
            )

        visited.add(task.id)
        self.tasks[task.id] = task

        if task.subsequent_tasks is None:
            visited.remove(task.id)  # Backtrack
            return set()

        dependent_tasks = set()
        for dep in task.subsequent_tasks:
            dependent_tasks.add(dep.id)
            dependent_tasks.update(self._register_and_fetch_dependents(dep, visited))

        visited.remove(task.id)  # Backtrack
        return dependent_tasks

    def _build_graph(self) -> None:
        """
        Optimized single-pass graph building that:
        1. Calculates in-degrees efficiently
        2. Validates structure
        3. Finds end tasks
        4. Validates tools only once per task
        """
        processed = set()  # Tasks that have been fully processed

        # Process each head task
        for task_id in self.head_task_ids:
            self._process_task(self.tasks[task_id], processed)

    def _process_task(self, task: Task, processed: set) -> None:
        """Process a single task: validate, calculate in-degrees."""
        # Skip if already processed (shared task from another head)
        if task.id in processed:
            return

        processed.add(task.id)
        self.logger.info(f"Registering task {task}...")

        # Validate tools for this task (only once)
        self._validate_task_tools(task)

        # Process dependencies and calculate in-degrees
        if task.subsequent_tasks is not None:
            for dep in task.subsequent_tasks:
                self.in_degree[dep.id] += 1  # Calculate in-degree as we go
                self._process_task(dep, processed)
        else:
            # Leaf node - add as end task
            self.end_task_ids.add(task.id)

    def create_model_for_task(self, task: Task):
        """
        Create a model instance configured for the given task.

        This method handles:
        - Model configuration resolution (task-specific vs default)
        - System instruction setup
        - Tool code resolution to actual tool instances
        - Model instantiation with all configurations

        Parameters
        ----------
        task : Task
            The task for which to create a model

        Returns
        -------
        Model instance configured for the task

        Raises
        ------
        ValueError
            If no model configs are provided for the task
        """
        # prepare model configs
        model_configs = self.default_model_configs
        system_instruction = None
        kwargs = {}

        instruction = task.instruction
        if instruction is not None:
            if instruction.model_configs is not None:
                model_configs = instruction.model_configs

            # get instruction content
            if instruction.content is not None:
                system_instruction = instruction.content

            if instruction.other_configs is not None:
                kwargs = instruction.other_configs.model_dump()

                # replace tool codes with actual tool instances
                if "tools" in kwargs and kwargs["tools"]:
                    resolved_tools = []
                    model_class = self.model_registry.get_model_registry(
                        model_configs.model, fuzzy_mode=self.model_registry_fuzzy_mode
                    )["class"]

                    for tool_item in kwargs["tools"]:
                        if isinstance(tool_item, (str, StandardToolCode)):
                            # It's a tool code, resolve it to actual tool
                            try:
                                tool = self.tool_registry.get_tool(
                                    model_class, tool_item
                                )
                                resolved_tools.append(tool)
                            except KeyError:
                                self.logger.warning(
                                    f"Tool '{tool_item}' not found for model class '{model_class.__name__}'. Skipping."
                                )
                        else:
                            # It's already a tool instance or callable
                            resolved_tools.append(tool_item)
                    kwargs["tools"] = resolved_tools

        if model_configs is None:
            raise ValueError(f"No model configs provided for {task}.")

        configs = model_configs.model_dump(exclude={"model"}, exclude_none=True)

        return self.model_registry.create_model(
            model=model_configs.model,
            instruction=system_instruction,
            configs=configs,
            **kwargs,
        )

    def _validate_task_tools(self, task: Task) -> None:
        """
        Validates that all tools specified in task's other_configs exist in the tool registry
        for the configured model.
        """
        if task.instruction is None or task.instruction.other_configs is None:
            return

        tools = task.instruction.other_configs.tools
        if tools is None:
            return

        # Get model configs to determine model class
        model_configs = task.instruction.model_configs or self.default_model_configs
        if model_configs is None:
            raise ValueError(
                f"No model configs provided for task {task}. Cannot validate tools "
                f"{tools} without knowing the target model."
            )

        entry = self.model_registry.get_model_registry(model_configs.model)
        model_class = entry["class"]

        # Validate each tool exists in registry
        for tool_code in tools:
            try:
                self.tool_registry.get_tool(model_class, tool_code)
            except KeyError as e:
                raise ValueError(
                    f"Tool '{tool_code}' specified in task {task} is not available "
                    f"for model '{model_configs.model}' (class: {model_class.__name__}). "
                    f"Available tools: {self.tool_registry.list_tools(model_class)}"
                ) from e

    async def run_single_task(
        self, task: Task, thread: Thread, start_fresh: bool
    ) -> None:
        """
        Executes a single task.
        """
        try:
            # Set task status to running
            task.status = TaskStatus.RUNNING
            self.logger.info(f"Starting task execution: {task}")
            if start_fresh is False:
                # use deliverable if available
                if task.deliverable is not None:
                    await thread.extend_thread(task.deliverable)
                    if task.status != TaskStatus.COMPLETED:
                        task.status = TaskStatus.COMPLETED
                    self.logger.info(f"Task {task} already completed.")
                    return

            # Create isolated thread fork for this task to prevent race conditions
            task.base_thread_snapshot = await thread.afork()
            task_thread = await thread.afork()

            model = self.create_model_for_task(task)

            # run the task with model, handling tool execution rounds
            round_count = 0
            generated_thread = Thread()

            last_context_is_user = False
            if len(task_thread) > 0:
                last_context_is_user = (
                    task_thread.get_context(-1).provider_role == Role.USER
                )

            while round_count < self.max_tool_execution_rounds:
                round_count += 1
                self.logger.debug(f"Task {task} round {round_count}")

                # Use default_user_input if task.user_input is None to prevent empty responses
                user_input_for_round = task.user_input if round_count == 1 else None
                if (
                    user_input_for_round is None
                    and round_count == 1
                    and not last_context_is_user
                ):
                    user_input_for_round = self.default_user_input

                output_thread = await model.acreate(
                    user_input=user_input_for_round,
                    base_thread=task_thread,
                    return_generated_thread=True,
                )

                # Check for empty response thread and error out
                if len(output_thread) == 0:
                    raise TaskRuntimeError(
                        f"Task {task} received empty response from model in round {round_count}. "
                        f"This may indicate the conversation ended with assistant message or model refused to respond."
                    )

                # Check if the response thread has contexts with no contents
                for context in output_thread:
                    if not context.contents:
                        raise TaskRuntimeError(
                            f"Task {task} received context with no contents in round {round_count}. "
                            f"This indicates an invalid response from the model."
                        )

                # Collect all generated contexts from this round
                await generated_thread.extend_thread(output_thread)

                # Check if the last context has tool output content
                if len(output_thread) > 0:
                    last_context = output_thread.get_context(-1)
                    if last_context.contents and len(last_context.contents) > 0:
                        last_content = last_context.contents[-1]
                        if isinstance(last_content, ToolOutputContent):
                            # Continue the conversation for tool execution
                            await task_thread.extend_thread(output_thread)
                            continue

                # No tool output found, task completed
                break
            else:
                # Maximum rounds reached - this is an error condition
                raise TaskRuntimeError(
                    f"Task {task} exceeded maximum tool execution rounds "
                    f"({self.max_tool_execution_rounds}). This may indicate infinite tool loops."
                )

            # Create deliverable with all generated contexts from all rounds
            task.deliverable = generated_thread

            # Update shared thread for task handshaking
            await thread.extend_thread(task.deliverable)

            # Mark task as completed
            task.status = TaskStatus.COMPLETED
            self.logger.info(f"Task {task} completed in {round_count} rounds.")
        except Exception as e:
            # Mark task as failed
            task.status = TaskStatus.FAILED
            self.logger.error(f"Task {task} failed with error: {str(e)}")
            raise TaskRuntimeError(f"Task {task} failed with error: {str(e)}") from e

    async def run_all(
        self,
        start_fresh: bool = False,
        base_thread: Thread = None,
        initial_user_input: str | Context = None,
    ) -> Thread:
        """
        Executes all registered tasks following their dependency order in the graph.

        Returns:
            List[str]: A list of messages indicating the completion of each task.

        Raises:
            ValueError: If a dependency is declared for a task that has not been added, or if a cycle is detected.
        """
        if base_thread is not None and initial_user_input is not None:
            raise ValueError(
                "Cannot provide both base_thread and initial_user_input. "
                "Use one or the other."
            )

        in_degree = copy.deepcopy(self.in_degree)
        ready = [self.tasks[task_id] for task_id in self.head_task_ids]

        if base_thread is None:
            if initial_user_input is not None:
                base_thread = Thread(contexts=[initial_user_input])
            else:
                base_thread = Thread()

        try:
            # Process tasks in topological order.
            while ready:
                self.logger.info(f"Running ready tasks: {tuple(map(str, ready))}")
                current_tasks = [
                    asyncio.create_task(
                        self.run_single_task(
                            task, thread=base_thread, start_fresh=start_fresh
                        )
                    )
                    for task in ready
                ]

                try:
                    await asyncio.gather(*current_tasks)
                except Exception as e:
                    # Cancel all running tasks and mark them as cancelled
                    for i, asyncio_task in enumerate(current_tasks):
                        if not asyncio_task.done():
                            asyncio_task.cancel()
                            # Mark the corresponding Task object as cancelled
                            if i < len(ready):
                                ready[i].status = TaskStatus.CANCELLED
                    # Wait for tasks to be cancelled
                    await asyncio.gather(*current_tasks, return_exceptions=True)
                    raise RuntimeError(
                        "Task execution failed. Cancelled all running tasks."
                    ) from e

                # Update dependent tasks' in-degrees and prepare new ready tasks.
                new_ready = []
                for task in ready:
                    # Update in-degree of dependent tasks and prepare new ready tasks.
                    if task.subsequent_tasks is not None:
                        for dep in task.subsequent_tasks:
                            in_degree[dep.id] -= 1
                            if in_degree[dep.id] == 0:
                                new_ready.append(dep)

                ready = new_ready

            # Check for uncompleted tasks—this would indicate a cyclic dependency.
            uncompleted_tasks = [
                task_id
                for task_id, task in self.tasks.items()
                if task.status != TaskStatus.COMPLETED
            ]
            if uncompleted_tasks:
                raise ValueError(f"Tasks not completed: {set(uncompleted_tasks)}")

            return base_thread

        except Exception as e:
            self.logger.error(f"Error during task execution: {str(e)}")
            raise

    def display_dependencies(self) -> str:
        """
        Visualize task dependencies in markdown format.
        Dependencies are represented using arrows (->) and each task is represented with str(task).
        Tasks with no downstream tasks are shown as a separate row only if they have no upstream tasks.
        """

        def dfs(task, chain):
            new_chain = chain + [str(task)]
            if not task.subsequent_tasks:
                # If the task has no downstream tasks, return the current chain as a completed branch
                return [" -> ".join(new_chain)]
            chains = []
            for child in task.subsequent_tasks:
                chains.extend(dfs(child, new_chain))
            return chains

        result_lines = []
        # Process each head task (tasks without upstream dependencies)
        for head_id in self.head_task_ids:
            head_task = self.tasks[head_id]
            result_lines.extend(dfs(head_task, []))

        print("\n".join([f"- {line}" for line in result_lines]))

    def get_task_by_id(self, task_id: str) -> Task | None:
        """
        Get a task by its ID.

        Parameters:
            task_id (str): The ID of the task to retrieve. or the last 5 digits of the task id

        Returns:
            Task | None: The Task object if found, otherwise None.
        """
        if len(task_id) > Task.SHORT_ID_LENGTH:
            return self.tasks.get(task_id)
        return next(
            (task for task in self.tasks.values() if task.id.endswith(task_id)), None
        )

    def get_task_status_summary(self) -> dict:
        """
        Get a summary of task statuses.

        Returns:
            dict: A dictionary with status counts and task lists by status.
        """
        status_counts = {status.value: 0 for status in TaskStatus}
        tasks_by_status = {status.value: [] for status in TaskStatus}

        for task in self.tasks.values():
            status_counts[task.status.value] += 1
            tasks_by_status[task.status.value].append(task)

        return {
            "counts": status_counts,
            "tasks": tasks_by_status,
            "total_tasks": len(self.tasks),
        }

    def display_task_status(self) -> None:
        """
        Display a formatted overview of all task statuses.
        """
        summary = self.get_task_status_summary()

        print("📊 Task Status Overview:")
        print(f"   Total Tasks: {summary['total_tasks']}")
        print()

        status_emojis = {
            TaskStatus.PENDING.value: "⏳",
            TaskStatus.RUNNING.value: "🔄",
            TaskStatus.COMPLETED.value: "✅",
            TaskStatus.FAILED.value: "❌",
            TaskStatus.CANCELLED.value: "🚫",
        }

        for status_value, count in summary["counts"].items():
            if count > 0:
                emoji = status_emojis.get(status_value, "❓")
                print(f"   {emoji} {status_value.title()}: {count}")
                for task in summary["tasks"][status_value]:
                    print(f"      • {task}")
                print()

        if (
            summary["counts"][TaskStatus.PENDING.value] == 0
            and summary["counts"][TaskStatus.RUNNING.value] == 0
        ):
            if (
                summary["counts"][TaskStatus.FAILED.value] > 0
                or summary["counts"][TaskStatus.CANCELLED.value] > 0
            ):
                print("🚨 Execution completed with errors!")
            else:
                print("🎉 All tasks completed successfully!")
