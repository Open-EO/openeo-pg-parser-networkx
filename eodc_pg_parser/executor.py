from functools import partial
from typing import Callable, Dict

from eodc_pg_parser.graph import OpenEOProcessGraph
from eodc_pg_parser.pg_schema import PGEdgeType, ParameterReference
from eodc_pg_parser.utils import out_of_graph_predecessors


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

        if 'data' in static_parameters.keys():
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

                # If we have no dynamic parameters, we need to resolve to the insitu xarray. I.e, get data into first subgraph nodes
                if not dynamic_parameters:
                    dynamic_parameters = kwargs

                result = prebaked_process_impl(**dynamic_parameters)
                
                # Set value if it does not exist. If the value exists it was set by the last node in a subgraph.
                if node not in self.results_cache.keys():
                    self.results_cache[node] = result

                # See if there are any predecessors these results will need to be given to. I.e, get data out of last subgraph node.
                unrelated_preds = out_of_graph_predecessors(self, node)
                if unrelated_preds:
                    for pred in unrelated_preds:
                        self.results_cache[pred] = self.results_cache[node]

        return partial(node_callable, parent_callables=parent_callables)
