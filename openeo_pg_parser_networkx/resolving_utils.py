import copy
from typing import Any, Callable

from openeo_pg_parser_networkx.process_registry import Process, ProcessRegistry


def resolve_process_graph(
    process_graph: dict[str, Any],
    process_registry: ProcessRegistry,
    get_udp_spec: Callable[[str], dict] = None,
):
    '''
    Recursively resolves a process graph until all nodes are predefined processes,
    this applies to both the given process graph and any sub-graphs found in nodes
    like apply and reduce.

    Parameters:
        process_graph (dict[str,Any]): The Process Graph which is to be resolved.
        process_registry (ProcessRegistry): The process registry containing all predefined processes,
                                            and optionally all relevant UDPs in the 'user' namespace
        get_udp_spec (Callable[[str], dict]): A callable which returns the spec of a given udp, takes process_id as a parameter.
                                              It being a "spec" just requires it to be a dictionary containing the key 'process_graph' which
                                              contains the relevant Process-Graph as its value.

    Returns:
        resolved_process_graph (dict[str,Any]): The resolved process graph.
    '''
    process_graph = _unpack_process_graph(
        process_graph=process_graph,
        process_registry=process_registry,
        get_udp_spec=get_udp_spec,
    )
    process_graph = _resolve_sub_process_graphs(
        process_graph=process_graph,
        process_registry=process_registry,
        get_udp_spec=get_udp_spec,
    )
    return process_graph


def _unpack_process_graph(
    process_graph: dict[str, Any],
    process_registry: ProcessRegistry,
    get_udp_spec: Callable[[str], dict] = None,
):
    '''
    Wraps method calls necessary for resolving a process graph without sub-graphs within
    nodes like apply and reduce.
    '''
    _fill_in_processes(process_graph, process_registry, get_udp_spec)
    root_result = _adjust_references(process_graph)
    process_graph = _flatten_graph(process_graph)
    _remove_non_root_result(process_graph, root_result)
    return process_graph


'''
Finds and resolves any sub process graphs within nodes like apply and reduce.
'''


def _resolve_sub_process_graphs(
    process_graph: dict,
    process_registry: ProcessRegistry,
    get_udp_spec: Callable[[str], dict] = None,
):
    for _, node in process_graph.items():
        if 'process' in node['arguments']:
            node['arguments']['process']['process_graph'] = resolve_process_graph(
                node['arguments']['process']['process_graph'],
                process_registry,
                get_udp_spec,
            )
    return process_graph


def _fill_in_processes(
    process_graph: dict[str, Any],
    process_registry: ProcessRegistry,
    get_udp_spec: Callable[[str], dict] = None,
):
    """
    Recursively fill in UDPs with their respective definitions until
    only predefined processes remain.

    Ignores subgraphs, those get resolved individually.
    """
    for process_replacement_id, process in process_graph.items():
        process_id = process['process_id']

        if ('predefined', process_id) not in process_registry:
            if ('user', process_id) not in process_registry and get_udp_spec is not None:
                process_registry[('user', process_id)] = Process(
                    get_udp_spec(process_id), implementation=None, namespace='user'
                )

            process_graph[process_replacement_id] = copy.deepcopy(
                process_registry[('user', process_id)].spec['process_graph']
            )

            _remap_names(
                process_graph=process_graph, process_replacement_id=process_replacement_id
            )
            _adjust_parameters(
                process_graph=process_graph,
                process_replacement_id=process_replacement_id,
                arguments=process['arguments'],
            )
            _fill_in_processes(
                process_graph=process_graph[process_replacement_id],
                process_registry=process_registry,
                get_udp_spec=get_udp_spec,
            )


def _remap_names(process_graph, process_replacement_id):
    '''
    Renames process nodes so that uniqueness is guaranteed, new names
    follow the schema root_subnode_subsubnode etc.
    '''
    name_remapping = [
        (f"{process_replacement_id}_{key}", key)
        for key, _ in process_graph[process_replacement_id].items()
    ]

    for new_key, old_key in name_remapping:
        process_graph[process_replacement_id][new_key] = process_graph[
            process_replacement_id
        ].pop(old_key)


def _adjust_parameters(process_graph, process_replacement_id, arguments):
    '''
    "Rewires" the from_parameter connections of filled in UDPs to be consistent with
    the parameters available from parent nodes.
    '''
    if len(arguments) > 0:
        for node_key, node in process_graph[process_replacement_id].items():
            for arg_key, from_param in node['arguments'].items():
                # Find from_parameter value in arguments list and replace with arguments[from_parameter value] value
                if "from_parameter" in from_param:
                    process_graph[process_replacement_id][node_key]['arguments'][
                        arg_key
                    ] = arguments[from_param['from_parameter']]


def _adjust_references(input_graph):
    '''
    "Rewires" the from_node connections of filled in UDPs to be consistent with
    the nodes available on the level of the parent nodes.

    Returns the result_node of the specified level,
    so that _remove_non_result_node can be called later.
    '''
    for key, node in input_graph.items():
        if _has_children(node):
            return_key = _adjust_references(node)
            for _, node in input_graph.items():
                if not _has_children(node) and 'arguments' in node:
                    for _, argument in node['arguments'].items():
                        stripped_key = key.split('_')[-1]
                        if (
                            isinstance(argument, dict)
                            and 'from_node' in argument
                            and (
                                argument['from_node'] == stripped_key
                                or argument['from_node'] == key
                            )
                        ):
                            argument['from_node'] = return_key
    return _get_result_node(input_graph)


def _get_result_node(process_graph):
    '''
    Finds a result node in a process graph, if none can be found, recursively
    enters sub-levels to find one.

    If UDPs are nested singularily result nodes cannot be found on the first level and
    thus recursion is necessary
    '''
    for key, node in process_graph.items():
        if 'result' in node:
            return key
    for value in process_graph.values():
        return _get_result_node(value)


def _has_children(node):
    '''
    Helper function to determine if a node has children or not
    '''
    return 'process_id' not in node


def _flatten_graph(process_graph):
    '''
    Recursively flattens the process graph,
    bringing all nodes without children to root level.
    '''
    flattened_graph = {}
    for key, value in process_graph.items():
        if isinstance(value, dict) and _has_children(value):
            flattened_graph.update(_flatten_graph(value))
        else:
            flattened_graph[key] = value
    return flattened_graph


def _remove_non_root_result(input_graph, root_result):
    '''
    Removes all result fields from nodes which aren't the root_result node.
    '''
    if root_result is not None:
        for id, node in input_graph.items():
            if 'result' in node and id != root_result:
                node.pop('result')
