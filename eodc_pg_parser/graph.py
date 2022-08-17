from typing import List
import networkx as nx
import random
import logging
from eodc_pg_parser.pg_schema import PGEdgeType, ProcessGraph, ProcessNode, ResultReference, ParameterReference
from eodc_pg_parser.utils import ProcessGraphUnflattener


logger = logging.getLogger(__name__)


class ProcessParameterMissing(Exception):
    pass


class OpenEOProcessGraph(object):
    def __init__(self, pg_data):
        self.G = nx.DiGraph()
        nested_raw_graph = self._unflatten_raw_process_graph(pg_data)
        self.nested_graph = self._parse_datamodel(nested_raw_graph)

        # Start parsing the graph at the result node of the top-level graph.
        self._parse_process_graph(self.nested_graph)

    @staticmethod
    def _unflatten_raw_process_graph(raw_flat_graph: dict) -> dict:
        """
        Translates a flat process graph into a nested structure by resolving the from_node references.
        """
        nested_graph = {"process_graph": {"root": ProcessGraphUnflattener.unflatten(raw_flat_graph["process_graph"])}}
        logger.warning("Deserialised process graph into nested structure")
        return nested_graph

    @staticmethod
    def _parse_datamodel(nested_graph: dict) -> ProcessGraph:
        """
        Parses a nested process graph into the Pydantic datamodel for ProcessGraph. 
        """

        return ProcessGraph.parse_obj(nested_graph)

    def _resolve_parameter_reference(self):
        pass

    def _parse_process_graph(self, process_graph: ProcessGraph):
        """
        Make sure each process graph operates within its own namespace so that nodes are unique.
        """
        for node_name, node in process_graph.process_graph.items():
            if node.result:
                self._walk_node(node, node_name, process_graph.uid)
                return node_name

    def _walk_node(self, node: ProcessNode, node_name: str, process_graph_uid: str):
        unique_node_id = f"{node_name}-{process_graph_uid}"
        
        self.G.add_node(unique_node_id, resolved_kwargs={}, node_name=node_name)

        # ALl this does is split the arguments into sub dicts by the type of argument. This is done because the order in which we resolve these matters.
        simple_args = {arg_name: getattr(arg_wrapper, "__root__", None) for arg_name, arg_wrapper in node.arguments.items() if not isinstance(getattr(arg_wrapper, "__root__", None), (ResultReference, ProcessGraph, ParameterReference))}
        parameter_references = {arg_name: getattr(arg_wrapper, "__root__", None) for arg_name, arg_wrapper in node.arguments.items() if isinstance(getattr(arg_wrapper, "__root__", None), ParameterReference)}
        result_references = {arg_name: getattr(arg_wrapper, "__root__", None) for arg_name, arg_wrapper in node.arguments.items() if isinstance(getattr(arg_wrapper, "__root__", None), ResultReference)}
        callbacks = {arg_name: getattr(arg_wrapper, "__root__", None) for arg_name, arg_wrapper in node.arguments.items() if isinstance(getattr(arg_wrapper, "__root__", None), ProcessGraph)}

        # For all simple arguments, just add the value into the resolved kwargs to be passed on
        for arg_name, arg in simple_args.items():
            self.G.nodes[unique_node_id]["resolved_kwargs"][arg_name] = arg

        # Resolve parameter references by looking for parameters in parent graphs
        arg: ParameterReference
        for arg_name, arg in parameter_references.items():
            self.G.nodes[unique_node_id]["resolved_kwargs"][arg_name] = arg

            # if not resolved_param:
            #     raise ProcessParameterMissing(f"ParameterReference {arg_name} on ProcessNode {node_id} could not be resolved")
                    
        arg: ResultReference
        for arg_name, arg in result_references.items():
            self.G.add_edge(unique_node_id, f"{arg.from_node}-{process_graph_uid}", reference_type=PGEdgeType.ResultReference, arg_name=arg_name, access_function=arg.access_function)
            self._walk_node(arg.node, node_name=arg.from_node, process_graph_uid=process_graph_uid)

        arg: ProcessGraph
        for arg_name, arg in callbacks.items():
            callback_result_node_name = self._parse_process_graph(arg)
            self.G.add_edge(unique_node_id, f"{callback_result_node_name}-{arg.uid}", reference_type=PGEdgeType.Callback, arg_name=arg_name)
            
    @property
    def nodes(self) -> List:
        return list(self.G.nodes(data=True))

    @property
    def edges(self) -> List:
        return list(self.G.edges(data=True))

    @property
    def in_edges(self, node: str) -> List:
        return list(self.G.in_edges(node, data=True))

    def plot(self):
        if self.G.number_of_nodes() < 1:
            logger.warning("Graph has no nodes, nothing to plot.")
            return

        n_colours = (
            max(nx.shortest_path_length(self.G, source=self.get_root_node()).values())
            + 1
        )
        node_colour_palette = [random.randint(0, 255) for _ in range(n_colours)]
        edge_colour_palette = {PGEdgeType.ResultReference: "blue", PGEdgeType.Callback: "red"}
        node_colours = [node_colour_palette[self.get_node_depth(node)] for node in self.G.nodes]
        edge_colors = [edge_colour_palette.get(self.G.edges[edge]["reference_type"], "green") for edge in self.G.edges]
        
        nx.draw_planar(
            self.G,
            labels=nx.get_node_attributes(self.G, "node_name"),
            horizontalalignment="right",
            verticalalignment="top",
            node_color=node_colours,
            edge_color=edge_colors
        )
        # nx.draw_networkx_edge_labels(
        #     G=self.G,
        #     pos=nx.kamada_kawai_layout(self.G),
        #     # edge_labels=nx.get_edge_attributes(self.G, "reference_type"),
        #     font_color="#00211e",
        # )

    def get_root_node(self):
        return next(nx.topological_sort(self.G))

    def get_node_depth(self, node):
        return nx.shortest_path_length(self.G, source=self.get_root_node(), target=node)

