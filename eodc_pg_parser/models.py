from typing import Any, Dict, Optional, List, Union, TypeVar
from pydantic import BaseModel, Extra, Field


class ProcessArgument(BaseModel, extra=Extra.allow):
    from_node: Optional[str]
    from_parameter: Optional[str]
    reducer: Optional[Union['ProcessGraph', str]]


class ProcessNode(BaseModel, extra=Extra.allow):
    process_id: str
    arguments: Dict[str, Optional[Union[ProcessArgument, str, List]]]
    result: Optional[bool]
        

class ProcessGraph(BaseModel):
    process_graph: Dict[str, ProcessNode]
