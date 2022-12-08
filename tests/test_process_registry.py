import pytest

from openeo_pg_parser_networkx.pg_schema import ParameterReference
from openeo_pg_parser_networkx.process_registry import ProcessParameterMissing


def test_process_registry(process_registry):
    assert "max" in process_registry
    assert "_max" in process_registry

    assert not any(
        [
            process_id.startswith("_") or process_id.endswith("_")
            for process_id in process_registry.store.keys()
        ]
    )


def test_process_registry_aliases(process_registry):
    size_before = len(process_registry)

    assert "test_max" not in process_registry
    process_registry.add_alias("max", "test_max")
    assert "test_max" in process_registry
    assert process_registry["test_max"] == process_registry["max"]

    size_after = len(process_registry)
    assert size_after == size_before


def test_process_decorator(process_registry):
    def test_process(param1, param2=6, **kwarg):
        return param1 * param2

    process_registry["test_process"] = test_process

    result = process_registry["test_process"](
        param1=ParameterReference(from_parameter="test_param_ref"),
        parameters={"test_param_ref": 2},
    )
    assert result == 12


def test_process_decorator_missing_parameter(process_registry):
    def test_process(param1, param2=6, **kwarg):
        return param1 * param2

    process_registry["test_process"] = test_process

    with pytest.raises(ProcessParameterMissing):
        process_registry["test_process"](
            param1=ParameterReference(from_parameter="test_param_ref"),
            parameters={"wrong_param": 2},
        )
