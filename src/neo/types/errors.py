class ModelServiceError(Exception):
    pass


class ModelModalityError(Exception):
    pass


class ContextLengthExceededError(Exception):
    pass


class ToolError(Exception):
    pass


class ToolNotFoundError(ToolError):
    pass


class ToolRuntimeError(ToolError):
    pass


class TaskRuntimeError(Exception):
    pass
