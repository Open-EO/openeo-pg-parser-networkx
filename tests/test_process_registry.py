import json
from functools import wraps

import pytest

from openeo_pg_parser_networkx.process_registry import Process, ProcessRegistry
from tests.conftest import TEST_DATA_DIR


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


def test_process_registry_alias_for_missing_base_process(process_registry):
    with pytest.raises(ValueError):
        process_registry.add_alias("not_registered", "test_max")


def test_process_registry_delete(process_registry):
    size_before = len(process_registry)
    process_registry.add_alias("max", "test_max")

    del process_registry["max"]
    assert len(process_registry) == size_before - 1

    with pytest.raises(KeyError):
        process_registry["max"]

    assert "test_max" not in process_registry


def test_process_registry_wrap_func(process_registry):
    def test_wrapper(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            return "wrapped"

        return wrapper

    process_registry.add_wrap_func(test_wrapper)
    assert process_registry["max"].implementation() == "wrapped"


def test_process_spec(process_registry):
    process = process_registry["max"]
    assert isinstance(process.spec, dict)


def test_storing_process_without_spec():
    process_registry = ProcessRegistry()

    process = Process(spec=json.load(open(TEST_DATA_DIR / "max.json")))
    process_registry["max"] = process
    assert isinstance(process.spec, dict)
    assert process.namespace == "predefined"
    assert process.implementation is None


def test_storing_process_with_namespace():
    process_registry = ProcessRegistry()

    process_registry['test_namespace', 'test_process_id'] = Process(
        spec={}, implementation="test", namespace="test_namespace"
    )

    assert ('test_namespace', 'test_process_id') in process_registry
    assert 'test_process_id' in process_registry['test_namespace', None]
    assert process_registry['test_namespace', 'test_process_id'].implementation == "test"


def test_deleting_process_with_namespace():
    process_registry = ProcessRegistry()

    process_registry['test_namespace', 'test_process_id'] = Process(
        spec={}, implementation="test", namespace="test_namespace"
    )

    del process_registry['test_namespace', 'test_process_id']

    assert 'test_process_id' not in process_registry['test_namespace', None]


def test_deleting_namespace():

    process_registry = ProcessRegistry()

    process_registry['test_namespace', 'test_process_id'] = Process(
        spec={}, implementation="test", namespace="test_namespace"
    )

    del process_registry['test_namespace', None]

    assert 'test_namespace' not in process_registry
