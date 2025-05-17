from neo.tools.tool import BaseTool
from typing import ClassVar
from typing import Optional, Any


class BaseInternalTool(BaseTool):
    name: ClassVar[str]
    provider: ClassVar[Optional[Any]] = None
