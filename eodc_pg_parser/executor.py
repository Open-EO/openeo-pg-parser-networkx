from typing import Callable, Dict

from eodc_pg_parser.graph import OpenEOProcessGraph
from eodc_pg_parser.pg_schema import PGEdgeType


class OpenEOExecutor:
    def __init__(self, parsed_graph: OpenEOProcessGraph, process_registry: Dict) -> None:
        self.parsed_graph = parsed_graph
        self.process_registry = process_registry
        self.intermediate_results = {}

    def _map_node_to_callable(self, node: str) -> Callable:
        node_with_data = self.parsed_graph.G.nodes(data=True)[node]
        process_impl = self.process_registry[node_with_data["process_id"]]

        def run():
            for _, source_node, data in self.parsed_graph.G.out_edges(node, data=True):
                source_node_process_callable = self._map_node_to_callable(source_node)
                if data["reference_type"] == PGEdgeType.ResultReference:
                    if source_node not in self.intermediate_results:
                        source_node_process_callable()
                    
                    for arg_sub in data["arg_substitutions"]:
                        arg_sub.access_func(
                            new_value=self.intermediate_results[source_node], set_bool=True
                        )
                elif data["reference_type"] == PGEdgeType.Callback:
                    self.parsed_graph.G.nodes[node]["resolved_kwargs"][
                        data["arg_name"]
                    ] = source_node_process_callable
            resolved_kwargs = self.parsed_graph.G.nodes[node]["resolved_kwargs"]
            result = process_impl(**resolved_kwargs)
            self.intermediate_results[node] = result

        return run
























    #     for node in self.parsed_graph:
    #         print(f"Executing node {node}.")
    #         node_with_data = self.parsed_graph.G.nodes(data=True)[node]
    #         resolved_kwargs = self._populate_result_references(node)

    #         process_impl = self.process_registry[node_with_data["process_id"]]
    #         parameterized_process_impl = partial(process_impl, **resolved_kwargs)

    #         # Only execute nodes within the top-level process graph, all callbacks will be called elsewhere.
    #         if node_with_data["process_graph_uid"] == self.parsed_graph.uid:
    #             parameterized_process_impl()
    #         self.parameterized_previous_nodes_impl[node] = parameterized_process_impl

    # def _populate_result_references(self, node: str):
    #     for _, source_node, data in self.parsed_graph.G.out_edges(node, data=True):
    #         source_node_process_impl = self.parameterized_previous_nodes_impl[source_node]
    #         if data["reference_type"] == PGEdgeType.ResultReference:
    #             for arg_sub in data["arg_substitutions"]:
    #                 arg_sub.access_func(
    #                     new_value=source_node_process_impl(), set_bool=True
    #                 )
    #         elif data["reference_type"] == PGEdgeType.Callback:
    #             self.parsed_graph.G.nodes[node]["resolved_kwargs"][
    #                 data["arg_name"]
    #             ] = source_node_process_impl

    #     resolved_kwargs = self.parsed_graph.G.nodes(data=True)[node]["resolved_kwargs"]
    #     return freeze(resolved_kwargs)
