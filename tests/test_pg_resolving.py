import json

import pytest

from openeo_pg_parser_networkx import resolving_utils
from openeo_pg_parser_networkx.process_registry import Process, ProcessRegistry


def get_udp(process_id: str, namespace: str) -> dict:
    process_id = process_id.lower()
    with open(f'tests/data/res_tests/udps/{process_id}.json') as f:
        return dict(json.load(f))


def error_get_udp(process_id: str, namespace: str) -> dict:
    raise KeyError("Couldn't find UDP.")


def get_predefined_process_registry():
    predefined_process_registry = ProcessRegistry()

    predefined_processes_specs = [
        ('add', {}),
        ('apply', {}),
        ('load_collection', {}),
        ('save_result', {}),
        ('sum', {}),
        ('reduce_dimension', {}),
    ]

    for process_id, spec in predefined_processes_specs:
        predefined_process_registry[("predefined", process_id)] = Process(spec)

    return predefined_process_registry


def get_full_process_registry() -> ProcessRegistry:
    full_process_registry = get_predefined_process_registry()

    for udp in ['w_add', 'valid_load', 'nested_add', 'gfm']:
        full_process_registry['user', udp] = Process(
            get_udp(udp, "user"), implementation=None, namespace="user"
        )

    return full_process_registry


@pytest.fixture
def predefined_process_registry() -> ProcessRegistry:
    return get_predefined_process_registry()


@pytest.fixture
def full_process_registry() -> ProcessRegistry:
    return get_full_process_registry()


@pytest.fixture
def unresolved_pg() -> dict:
    with open('tests/data/res_tests/unresolved/unresolved_complex.json') as f:
        return dict(json.loads(f.read()))


@pytest.fixture
def unresolved_gfm_pg() -> dict:
    with open('tests/data/res_tests/unresolved/unresolved_gfm.json') as f:
        return dict(json.loads(f.read()))


@pytest.fixture
def correctly_resolved_pg() -> dict:
    with open('tests/data/res_tests/resolved/resolved_complex.json') as f:
        return dict(json.loads(f.read()))


@pytest.fixture
def correctly_resolved_gfm_pg() -> dict:
    with open('tests/data/res_tests/resolved/resolved_gfm.json') as f:
        return dict(json.loads(f.read()))


def test_resolve_graph_with_predefined_process_registry(
    predefined_process_registry: ProcessRegistry,
    unresolved_pg: dict,
    correctly_resolved_pg: dict,
):
    resolved_pg = resolving_utils.resolve_process_graph(
        process_graph=unresolved_pg,
        process_registry=predefined_process_registry,
        get_udp_spec=get_udp,
    )

    assert correctly_resolved_pg == resolved_pg


def test_resolve_graph_with_full_process_registry(
    full_process_registry: ProcessRegistry,
    unresolved_pg: dict,
    correctly_resolved_pg: dict,
):
    resolved_pg = resolving_utils.resolve_process_graph(
        process_graph=unresolved_pg,
        process_registry=full_process_registry,
    )

    assert correctly_resolved_pg == resolved_pg


def test_resolve_graph_with_faulty_process_registry(
    predefined_process_registry: ProcessRegistry, unresolved_pg: dict
):
    with pytest.raises(KeyError):
        resolving_utils.resolve_process_graph(
            process_graph=unresolved_pg,
            process_registry=predefined_process_registry,
        )


def test_resolve_graph_with_faulty_get_udp_spec(
    predefined_process_registry: ProcessRegistry, unresolved_pg: dict
):
    with pytest.raises(ValueError):
        resolving_utils.resolve_process_graph(
            process_graph=unresolved_pg,
            process_registry=predefined_process_registry,
            get_udp_spec=lambda x, y: {},
        )


def test_resolve_graph_with_error_get_udp_spec(
    predefined_process_registry: ProcessRegistry, unresolved_pg: dict
):
    with pytest.raises(ValueError):
        resolving_utils.resolve_process_graph(
            process_graph=unresolved_pg,
            process_registry=predefined_process_registry,
            get_udp_spec=error_get_udp,
        )


def test_resolve_graph_with_none_get_udp_spec(
    predefined_process_registry: ProcessRegistry, unresolved_pg: dict
):
    with pytest.raises(ValueError):
        resolving_utils.resolve_process_graph(
            process_graph=unresolved_pg,
            process_registry=predefined_process_registry,
            get_udp_spec=lambda x, y: None,
        )


def test_resolve_gfm_graph_with_predefined_process_registry(
    predefined_process_registry: ProcessRegistry,
    unresolved_gfm_pg: dict,
    correctly_resolved_gfm_pg: dict,
):
    resolved_pg = resolving_utils.resolve_process_graph(
        process_graph=unresolved_gfm_pg,
        process_registry=predefined_process_registry,
        get_udp_spec=get_udp,
    )

    with open('resolved_gfm_graph.json', 'w') as f:
        json.dump(resolved_pg, f)
    assert correctly_resolved_gfm_pg == resolved_pg
