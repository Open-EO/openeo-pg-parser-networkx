import copy
import logging
from timeit import default_timer as timer
from typing import Any, Callable, Optional

from openeo_pg_parser_networkx.process_registry import (
    DEFAULT_NAMESPACE,
    Process,
    ProcessRegistry,
)

logger = logging.getLogger(__name__)


def resolve_process_graph(
    process_graph: dict[str, Any],
    process_registry: ProcessRegistry,
    get_udp_spec: Optional[Callable[[str, str], dict]] = None,
    namespace: Optional[str] = "user",
):
    '''
    This function resolves a process graph.

    process_registry only needs to be populated with all predefined processes unless
    "get_udp_spec" is ommitted or set to None deliberately, in that case process_registry
    needs to already be populated with all UDPs that will be encountered in addition to
    all predefined processes. In this case, please also give the namespace in which the
    UDPs can be found inside of the process_registry (default is "user").

    otherwise:

    The "get_udp_spec" Callable needs to take process_id and namespace as parameters
    and return the spec* of the given process_id's UDP.

    Simplest Example:
    {"process_graph":{....}}

    The resolving logic then automatically fetches all relevant UDP definitions through the
    "get_udp_spec" Callable.

    Implementation Note (get_udp_spec == None):

    Populating the process_registry with every UDP of the given user is the simplest approach.
    If that would mean loading too many UDPs and cannot be done, either implement a
    more optimized means of populating the process_registry yourself, or use the
    optional get_udp_spec Callable.

    Implementation Note (get_udp_spec != None):

    get_udp_spec takes process_id and namespace, you can use namespace to pass in
    the user_id to find the specific process_id wherever you implement this function.

    If you use get_udp_spec, take note of the fact that using namespace like this is
    recommended as ommitting it would just give you the string "user" in the get_udp_spec
    call as a namespace, which probably isn't useful information.

    Parameters:
        process_graph (dict[str, Any]):
            The process graph that should be resolved

        process_registry (ProcessRegistry):
            fully populated process_registry with predefined processes

        get_udp_spec (Optional[Callable[[str, str], dict]]) = None:
            Optional Callable which needs to take process_id as a parameter
            and return the spec* of the given process_id's UDP.

        namespace (Optional[str]) = "user":
            the namespace in which the UDPs are located in the process_registry
            might be used if the process_registry is given pre-loaded with
            UDPs from a specific user, saved under a specific user_id

    Returns:
        resolved_process_graph (dict[str, Any]):
            the resolved process graph


    spec* (dict): The only REQUIRED part of a "spec" is one key named "process_graph",
    which contains the given UDPs process_graph as its value.
    '''

    start_time = timer()

    process_graph = _unpack_process_graph(
        process_graph=process_graph,
        process_registry=process_registry,
        get_udp_spec=get_udp_spec,
        namespace=namespace,
    )
    process_graph = _resolve_sub_process_graphs(
        process_graph=process_graph,
        process_registry=process_registry,
        get_udp_spec=get_udp_spec,
        namespace=namespace,
    )

    end_time = timer()

    logger.info(f"Resolving this process graph took {end_time - start_time} seconds.")
    return process_graph


def _unpack_process_graph(
    process_graph: dict[str, Any],
    process_registry: ProcessRegistry,
    get_udp_spec: Optional[Callable[[str, str], dict]] = None,
    namespace: str = "user",
):
    '''
    Wraps method calls necessary for resolving a process graph without sub-graphs within
    nodes like apply and reduce.
    '''
    _fill_in_processes(process_graph, process_registry, get_udp_spec, namespace)
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
    get_udp_spec: Optional[Callable[[str, str], dict]] = None,
    namespace: str = "user",
):
    for _, node in process_graph.items():
        if 'process' in node['arguments']:
            node['arguments']['process']['process_graph'] = resolve_process_graph(
                process_graph=node['arguments']['process']['process_graph'],
                process_registry=process_registry,
                get_udp_spec=get_udp_spec,
                namespace=namespace,
            )
    return process_graph


def _fill_in_processes(
    process_graph: dict[str, Any],
    process_registry: ProcessRegistry,
    get_udp_spec: Optional[Callable[[str, str], dict]] = None,
    namespace: str = "user",
):
    """
    Recursively fill in UDPs with their respective definitions until
    only predefined processes remain.

    Ignores subgraphs, those get resolved individually.
    """
    for process_replacement_id, process in process_graph.items():
        process_id = process['process_id']

        if (DEFAULT_NAMESPACE, process_id) not in process_registry:
            if (
                namespace,
                process_id,
            ) not in process_registry and get_udp_spec is not None:
                process_registry[(namespace, process_id)] = Process(
                    get_udp_spec(process_id, namespace),
                    implementation=None,
                    namespace=namespace,
                )

            process_graph[process_replacement_id] = copy.deepcopy(
                process_registry[(namespace, process_id)].spec['process_graph']
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
                namespace=namespace,
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
        if not _is_process(node):
            return_key = _adjust_references(node)
            for _, node in input_graph.items():
                if _is_process(node) and 'arguments' in node:
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


def _is_process(node):
    '''
    Helper function to determine if a node has children or not
    '''
    return 'process_id' in node


def _flatten_graph(process_graph):
    '''
    Recursively flattens the process graph,
    bringing all nodes without children to root level.
    '''
    flattened_graph = {}
    for key, value in process_graph.items():
        if isinstance(value, dict) and not _is_process(value):
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
