import asyncio
import copy
from collections import defaultdict
from typing import Dict, List, Set

from neo.agentic.instruction import ModelConfigs
from neo.agentic.model_registry import ModelRegistry
from neo.agentic.task import Task
from neo.contexts import Thread
from neo.types.errors import TaskRuntimeError
from neo.utils.logger import get_logger


class Neo:

    def __init__(
        self,
        tasks: List[Task] | Task = None,
        default_model_configs: ModelConfigs = None,
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
        """
        self.model_registry = ModelRegistry()

        if (
            not isinstance(default_model_configs, ModelConfigs)
            and default_model_configs is not None
        ):
            default_model_configs = ModelConfigs(**default_model_configs)

        self.default_model_configs = default_model_configs

        self.tasks: Dict[str, Task] = {}
        self.head_task_ids: Set[str] = set()
        self.end_task_id: str = None
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

        for task_id in self.head_task_ids:
            _visited = set()
            self._build_graph(self.tasks[task_id], visited=_visited)

        self.logger.info(f"End task: {self.tasks[self.end_task_id]}")

    def _register_and_fetch_dependents(self, task: Task) -> Set[str]:
        """recursively get all dependent tasks"""
        self.tasks[task.id] = task

        if task.subsequent_tasks is None:
            return set()

        dependent_tasks = set()
        for dep in task.subsequent_tasks:
            dependent_tasks.add(dep.id)
            dependent_tasks.update(self._register_and_fetch_dependents(dep))

        return dependent_tasks

    def _build_graph(self, task: Task, visited: set) -> None:
        """
        Adds a new task node to the manager.
        """
        if task.id in visited:
            raise ValueError(
                f"Cyclic dependency detected: Task {task} is already registered."
            )
        visited.add(task.id)

        self.logger.info(f"Registering task {task}...")

        # recursively add task and ensure no cyclic dependency
        if task.subsequent_tasks is not None:
            for dep in task.subsequent_tasks:
                self.in_degree[dep.id] += 1
                self._build_graph(dep, visited)
        else:
            # this is a leaf node
            if self.end_task_id is not None and self.end_task_id != task.id:
                _task = self.tasks[self.end_task_id]
                raise ValueError(
                    f"Graph must converge to a single leaf node. Found: {_task} and {task}"
                )
            self.end_task_id = task.id

    async def run_single_task(
        self, task: Task, thread: Thread, start_fresh: bool
    ) -> None:
        """
        Executes a single task.
        """
        try:
            if start_fresh is False:
                # use deliverable if available
                if task.deliverable is not None:
                    await thread.extend_thread(task.deliverable)
                    self.logger.info(f"Task {task} already completed.")
                    return

            task.base_thread_snapshot = await thread.afork()

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

            if model_configs is None:
                raise ValueError(f"No model configs provided for {task}. ")

            configs = model_configs.model_dump(exclude={"model"}, exclude_none=True)

            model = self.model_registry.create_model(
                model=model_configs.model,
                instruction=system_instruction,
                configs=configs,
                **kwargs,
            )

            # run the task with model
            output_thread = await model.acreate(
                user_input=task.user_input,
                base_thread=thread,
                return_generated_thread=True,  # only track newly generated contents as deliverables
            )
            task.deliverable = output_thread
            self.logger.info(f"Task {task} completed.")
        except Exception as e:
            raise TaskRuntimeError(f"Task {task} failed with error: {str(e)}") from e

    async def run_all(
        self, start_fresh: bool = False, base_thread: Thread = None
    ) -> Thread:
        """
        Executes all registered tasks following their dependency order in the graph.

        Returns:
            List[str]: A list of messages indicating the completion of each task.

        Raises:
            ValueError: If a dependency is declared for a task that has not been added, or if a cycle is detected.
        """

        in_degree = copy.deepcopy(self.in_degree)
        ready = [self.tasks[task_id] for task_id in self.head_task_ids]

        completed_tasks = set()
        if base_thread is None:
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
                    # Cancel all running tasks
                    for task in current_tasks:
                        if not task.done():
                            task.cancel()
                    # Wait for tasks to be cancelled
                    await asyncio.gather(*current_tasks, return_exceptions=True)
                    raise RuntimeError(
                        "Task execution failed. Cancelled all running tasks."
                    ) from e

                # Mark tasks as completed and update dependent tasks' in-degrees.
                new_ready = []
                for task in ready:
                    completed_tasks.add(task.id)

                    # Update in-degree of dependent tasks and prepare new ready tasks.
                    if task.subsequent_tasks is not None:
                        for dep in task.subsequent_tasks:
                            in_degree[dep.id] -= 1
                            if in_degree[dep.id] == 0:
                                new_ready.append(dep)

                ready = new_ready

            # Check for uncompleted tasks—this would indicate a cyclic dependency.
            if len(completed_tasks) != len(self.tasks):
                missing = set(self.tasks.keys()) - completed_tasks
                raise ValueError(f"Tasks not completed: {missing}")

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
        
