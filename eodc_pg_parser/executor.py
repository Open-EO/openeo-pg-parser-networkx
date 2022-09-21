from functools import partial
from typing import Callable, Dict

from eodc_pg_parser.graph import OpenEOProcessGraph
from eodc_pg_parser.pg_schema import PGEdgeType, ParameterReference


class OpenEOExecutor:
    def __init__(self, parsed_graph: OpenEOProcessGraph, process_registry: Dict) -> None:
        self.parsed_graph = parsed_graph
        self.process_registry = process_registry
        self.results_cache = {}

    def _map_node_to_callable(self, node: str) -> Callable:
        node_with_data = self.parsed_graph.G.nodes(data=True)[node]
        process_impl = self.process_registry[node_with_data["process_id"]]

        static_parameters = node_with_data["resolved_kwargs"]
        parent_callables = []

        for _, source_node, data in self.parsed_graph.G.out_edges(node, data=True):
            if data["reference_type"] == PGEdgeType.ResultReference:
                parent_callables.append(self._map_node_to_callable(source_node))
            elif data["reference_type"] == PGEdgeType.Callback:
                callback = self._map_node_to_callable(source_node)
                static_parameters[data["arg_name"]] = callback

        if isinstance(static_parameters['data'], ParameterReference):
            del static_parameters['data']

        prebaked_process_impl = partial(process_impl, **static_parameters)

        def node_callable(parent_callables, **kwargs):
            # Get dynamic data sources that need to be filled insitu
            for func in parent_callables:
                func(**kwargs)

            try:
                return self.results_cache.__getitem__(node)
            except KeyError:
                dynamic_parameters = {}
                
                for _, source_node, data in self.parsed_graph.G.out_edges(node, data=True):
                    if data["reference_type"] == PGEdgeType.ResultReference:
                        for arg_sub in data["arg_substitutions"]:
                            arg_sub.access_func(
                                new_value=self.results_cache[source_node], set_bool=True
                            )
                        
                        dynamic_parameters[arg_sub.arg_name] = self.parsed_graph.G.nodes(data=True)[node]["resolved_kwargs"].__getitem__(arg_sub.arg_name)

                # resolve to needed xarray
                if not dynamic_parameters:
                    dynamic_parameters = kwargs

                result = prebaked_process_impl(**dynamic_parameters)

                self.results_cache[node] = result

        return partial(node_callable, parent_callables=parent_callables)
