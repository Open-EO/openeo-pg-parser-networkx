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
        root_node = self.nested_graph.process_graph["root"]
        self._walk_node(root_node, root_node.process_id)

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


    def _walk_node(self, node: ProcessNode, node_id: str):
        self.G.add_node(node_id, resolved_kwargs={})

        # ALl this does is split the arguments into sub dicts by the type of argument. This is done because the order in which we resolve these matters.
        simple_args = {arg_name: getattr(arg_wrapper, "__root__", None) for arg_name, arg_wrapper in node.arguments.items() if not isinstance(getattr(arg_wrapper, "__root__", None), (ResultReference, ProcessGraph, ParameterReference))}
        parameter_references = {arg_name: getattr(arg_wrapper, "__root__", None) for arg_name, arg_wrapper in node.arguments.items() if isinstance(getattr(arg_wrapper, "__root__", None), ParameterReference)}
        result_references = {arg_name: getattr(arg_wrapper, "__root__", None) for arg_name, arg_wrapper in node.arguments.items() if isinstance(getattr(arg_wrapper, "__root__", None), ResultReference)}
        callbacks = {arg_name: getattr(arg_wrapper, "__root__", None) for arg_name, arg_wrapper in node.arguments.items() if isinstance(getattr(arg_wrapper, "__root__", None), ProcessGraph)}

        # For all simple arguments, just add the value into the resolved kwargs to be passed on
        for arg_name, arg in simple_args.items():
            self.G.nodes[node_id]["resolved_kwargs"][arg_name] = arg

        # Resolve parameter references by looking for parameters in parent graphs
        for arg_name, arg in parameter_references.items():
            # Recursively search through parent Process nodes to resolve parameter references.
            def search_parents_for_parameter(child_node_id, arg_name, origin_node_id):
                for parent_node, _, child_edge_data in self.G.in_edges(child_node_id, data=True):
                    # TODO: Replace these with an EdgeType Enum
                    if child_edge_data["reference_type"] == PGEdgeType.Callback:
                        # First check whether the parameter is already resolved
                        if arg_name in self.G.nodes[parent_node]["resolved_kwargs"]:

                            # Need to check whether any
                            self.G.nodes[origin_node_id]["resolved_kwargs"][arg_name] = self.G.nodes[parent_node]["resolved_kwargs"][arg_name]
                            return True
                        
                        # If not, check the result references of the parent node for this parameter
                        for parent_node, grand_parent_node, parent_edge_data in self.G.out_edges(parent_node, data=True):
                            if parent_edge_data["reference_type"] == PGEdgeType.ResultReference:
                                if parent_edge_data["arg_name"] == arg_name:
                                    self.G.add_edge(origin_node_id, grand_parent_node, reference_type=PGEdgeType.ResultReference, arg_name=arg_name)
                                    return True

                        return search_parents_for_parameter(child_node_id=parent_node, arg_name=arg_name, origin_node_id=origin_node_id)
                    
                    # Search Result references for forther callback nodes to search in 
                    if child_edge_data["reference_type"] == PGEdgeType.ResultReference:
                        return search_parents_for_parameter(child_node_id=parent_node, arg_name=arg_name, origin_node_id=origin_node_id)
                    
                    return False

            resolved_param = search_parents_for_parameter(child_node_id=node_id, arg_name=arg_name, origin_node_id=node_id)
            if not resolved_param:
                raise ProcessParameterMissing(f"ParameterReference {arg_name} on ProcessNode {node_id} could not be resolved")
                    
            # TODO: If it's a result reference, add it to the list of result refernce that need to be resolve beneath!

        for arg_name, arg in result_references.items():
            # TODO: Pass the parameter object down
            self.G.add_edge(node_id, arg.from_node, reference_type=PGEdgeType.ResultReference, arg_name=arg_name)
            self._walk_node(arg.node, node_id=arg.from_node)

        for arg_name, arg in callbacks.items():
            # TODO: These aren't ordered, so I need to find the result node
            for sub_node_id, sub_node in arg.process_graph.items():
                if sub_node.result:
                    self.G.add_edge(node_id, sub_node_id, reference_type=PGEdgeType.Callback, arg_name=arg_name)
                    self._walk_node(sub_node, sub_node_id)

    
        # TODO: Handle reducers

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
            labels={node: node for node in self.G.nodes},
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

