import json

import pytest

from openeo_pg_parser_networkx.graph import OpenEOProcessGraph
from openeo_pg_parser_networkx.process_registry import Process, ProcessRegistry


def get_udp(process_id: str) -> dict:
    with open(f'tests/data/res_tests/udps/{process_id}.json') as f:
        return dict(json.load(f))


def fake_get_udp(process_id: str) -> dict:
    return {}


def get_predefined_process_registry():
    predefined_process_registry = ProcessRegistry()

    predefined_processes_specs = [
        ('add', {}),
        ('apply', {}),
        ('load_collection', {}),
        ('save_result', {})
    ]

    for process_id, spec in predefined_processes_specs:
        predefined_process_registry[("predefined", process_id)] = Process(spec)

    return predefined_process_registry


def get_full_process_registry() -> ProcessRegistry:
    full_process_registry = get_predefined_process_registry()

    for udp in ['w_add', 'valid_load', 'nested_add']:
        full_process_registry['user', udp] = Process(
            get_udp(udp), implementation=None, namespace="user"
        )

    return full_process_registry


@pytest.fixture
def predefined_process_registry() -> ProcessRegistry:
    return get_predefined_process_registry()


@pytest.fixture
def full_process_registry() -> ProcessRegistry:
    return get_full_process_registry()


@pytest.fixture
def unresolved_pg() -> OpenEOProcessGraph:
    with open('tests/data/res_tests/unresolved/unresolved_complex.json') as f:
        return OpenEOProcessGraph(dict(json.loads(f.read())))


@pytest.fixture
def correctly_resolved_pg() -> OpenEOProcessGraph:
    with open('tests/data/res_tests/resolved/resolved_complex.json') as f:
        return OpenEOProcessGraph(dict(json.loads(f.read())))


def test_resolve_graph_withpredefined_process_registr(
    predefined_process_registry: ProcessRegistry,
    unresolved_pg: OpenEOProcessGraph,
    correctly_resolved_pg: OpenEOProcessGraph,
):
    resolved_pg = unresolved_pg.resolve_process_graph(
        process_registry=predefined_process_registry, get_udp_spec=get_udp
    )

    assert correctly_resolved_pg.pg_data == resolved_pg.pg_data


def test_resolve_graph_with_full_process_registry(
    full_process_registry: ProcessRegistry,
    unresolved_pg: OpenEOProcessGraph,
    correctly_resolved_pg: OpenEOProcessGraph,
):
    resolved_pg = unresolved_pg.resolve_process_graph(
        process_registry=full_process_registry
    )

    assert correctly_resolved_pg.pg_data == resolved_pg.pg_data


def test_resolve_graph_with_faulty_process_registry(
    predefined_process_registry: ProcessRegistry,
    unresolved_pg: OpenEOProcessGraph,
    correctly_resolved_pg: OpenEOProcessGraph,
):
    with pytest.raises(KeyError):
        unresolved_pg.resolve_process_graph(process_registry=predefined_process_registry)


def test_resolve_graph_with_faulty_get_udp_spec(
    predefined_process_registry: ProcessRegistry,
    unresolved_pg: OpenEOProcessGraph,
    correctly_resolved_pg: OpenEOProcessGraph,
):
    with pytest.raises(KeyError):
        unresolved_pg.resolve_process_graph(
            process_registry=predefined_process_registry, get_udp_spec=fake_get_udp
        )
