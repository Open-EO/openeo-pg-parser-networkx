import json

import pytest
from yprov4wfs.datamodel.data import Data
from yprov4wfs.datamodel.task import Task
from yprov4wfs.datamodel.workflow import Workflow

from openeo_pg_parser_networkx import OpenEOProcessGraph
from openeo_pg_parser_networkx.process_registry import Process, ProcessRegistry


def test_execute_returns_result_and_workflow(process_graph_path):
    """
    Test that OpenEOProcessGraph returns result and workflow correctly
    for all sample graphs, using a mock registry based on required processes.
    """

    with open(process_graph_path) as f:
        flat_pg = json.load(f)

    pg = OpenEOProcessGraph(flat_pg)

    mock_registry = ProcessRegistry(wrap_funcs=[])
    for process_id in pg.required_processes:
        mock_registry[process_id] = Process(
            spec={},
            implementation=lambda *args, **kwargs: args[0] if args else None,
            namespace="predefined",
        )

    # Create callable and execute
    result = pg.to_callable(mock_registry)()
    workflow = pg.workflow

    # Assertions
    assert result is not None, "Result should not be None"
    assert workflow is not None, "Workflow should not be None"
    assert isinstance(
        workflow, Workflow
    ), "Workflow should be a yprov4wfs.Workflow instance"
    assert len(workflow._tasks) > 0, "Workflow should have at least one task"
    assert workflow._status in ["Ok", "Error"], "Workflow status should be Ok or Error"

    # Test the tasks
    assert isinstance(workflow._tasks, list), "Workflow._tasks should be a list"
    for task in workflow._tasks:
        # Each task should be a Task instance
        assert isinstance(
            task, Task
        ), f"Each task should be a Task instance but got {type(task)}"
        assert hasattr(task, "_id"), "Task must have an _id"
        assert hasattr(task, "_name"), "Task must have a _name"
        assert hasattr(task, "_start_time"), "Task must have a start_time"
        assert hasattr(task, "_end_time"), "Task must have an end_time"
        assert hasattr(task, "_status"), "Task must have a status"
        assert hasattr(task, "_inputs"), "Task must have _inputs"
        assert hasattr(task, "_outputs"), "Task must have _outputs"

    # Test the data
    assert isinstance(workflow._data, list), "Workflow._data should be a list"
    for data in workflow._data:
        assert isinstance(
            data, Data
        ), f"Each data node should be a Data instance but got {type(data)}"
        assert hasattr(data, "_id"), "Data must have an _id"
        assert hasattr(data, "_name"), "Data must have a _name"
