from ctypes import Union
from typing import Any, Callable, Dict, List
import networkx as nx
import random
import logging
from eodc_pg_parser.pg_schema import (
    PGEdgeType,
    ProcessArgument,
    ProcessGraph,
    ProcessNode,
    ResultReference,
    ParameterReference,
)
from eodc_pg_parser.utils import ProcessGraphUnflattener
import pydantic
from collections import defaultdict, namedtuple
from functools import partial


logger = logging.getLogger(__name__)

ArgSubstitution = namedtuple("ArgSubstitution", ["arg_name", "setter_func"])

MISSING_VALUE = "__MISSING__"


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
        nested_graph = {
            "process_graph": {
                "root": ProcessGraphUnflattener.unflatten(raw_flat_graph["process_graph"])
            }
        }
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
        raise Exception("Process graph has no return node!")

    def _resolve_result_reference(
        self, unique_node_id: str, from_node: str, arg_name, process_graph_uid
    ):
        self.G.nodes[unique_node_id]["resolved_kwargs"][arg_name] = MISSING_VALUE

        # This just points to the resolved_kwarg itself!
        setter_func = partial(
            lambda unique_node_id, arg_name, new_value: self.G.nodes[unique_node_id][
                "resolved_kwargs"
            ].__setitem__(arg_name, new_value),
            unique_node_id=unique_node_id,
            arg_name=arg_name,
        )

        self._add_result_reference_edge(
            unique_node_id,
            OpenEOProcessGraph._get_unique_node_id(from_node, process_graph_uid),
            arg_name=arg_name,
            setter_func=setter_func,
        )

    def _add_result_reference_edge(
        self,
        unique_node_id: str,
        from_node_unique_id: str,
        arg_name: str,
        setter_func: Callable,
    ):
        self.G.add_edge(
            unique_node_id, from_node_unique_id, reference_type=PGEdgeType.ResultReference
        )
        if not self.G.edges[unique_node_id, from_node_unique_id].get(
            "arg_substitutions", False
        ):
            self.G.edges[unique_node_id, from_node_unique_id]["arg_substitutions"] = []

        self.G.edges[unique_node_id, from_node_unique_id]["arg_substitutions"].append(
            ArgSubstitution(arg_name=arg_name, setter_func=setter_func)
        )

    @staticmethod
    def _get_unique_node_id(node_name: str, process_graph_uid: str):
        return f"{node_name}-{process_graph_uid}"

    def _walk_node(self, node: ProcessNode, node_name: str, process_graph_uid: str):
        """
        Parse all the required information from the current node into self.G and recursively walk child nodes.
        """
        print(f"Walking node {node_name}")

        unique_node_id = OpenEOProcessGraph._get_unique_node_id(
            node_name, process_graph_uid
        )

        if self.G.nodes.get(unique_node_id, False):
            # Only walk a node once!
            return

        self.G.add_node(unique_node_id, resolved_kwargs={}, node_name=node_name)
        result_references_to_walk = {}  # type: Dict["str", ProcessNode]

        arg: ProcessArgument
        for arg_name, arg in node.arguments.items():
            unpacked_arg = arg.__root__

            self.G.nodes[unique_node_id]["resolved_kwargs"][arg_name] = unpacked_arg

            if isinstance(unpacked_arg, ResultReference):
                self.G.nodes[unique_node_id]["resolved_kwargs"][arg_name] = MISSING_VALUE
                self._resolve_result_reference(
                    unique_node_id=unique_node_id,
                    from_node=unpacked_arg.from_node,
                    arg_name=arg_name,
                    process_graph_uid=process_graph_uid,
                )
                result_references_to_walk[unpacked_arg.from_node] = unpacked_arg.node

            elif isinstance(unpacked_arg, dict):
                self.G.nodes[unique_node_id]["resolved_kwargs"][arg_name] = unpacked_arg

                for k, v in unpacked_arg.items():
                    try:
                        sub_result_reference = ResultReference.parse_obj(v)
                        setter_func = partial(
                            lambda unique_node_id, arg_name, new_value, key: self.G.nodes[
                                unique_node_id
                            ]["resolved_kwargs"][arg_name].__setitem__(key, new_value),
                            unique_node_id=unique_node_id,
                            arg_name=arg_name,
                            key=k,
                        )
                        result_references_to_walk[
                            sub_result_reference.from_node
                        ] = sub_result_reference.node
                        self._add_result_reference_edge(
                            unique_node_id,
                            OpenEOProcessGraph._get_unique_node_id(
                                sub_result_reference.from_node, process_graph_uid
                            ),
                            arg_name=arg_name,
                            setter_func=setter_func,
                        )
                    except pydantic.error_wrappers.ValidationError:
                        pass

            elif isinstance(unpacked_arg, list):
                self.G.nodes[unique_node_id]["resolved_kwargs"][arg_name] = unpacked_arg
                for i, element in enumerate(unpacked_arg):
                    try:
                        sub_result_reference = ResultReference.parse_obj(element)
                        setter_func = partial(
                            lambda unique_node_id, arg_name, new_value, key: self.G.nodes[
                                unique_node_id
                            ]["resolved_kwargs"][arg_name].__setitem__(key, new_value),
                            unique_node_id=unique_node_id,
                            arg_name=arg_name,
                            key=i,
                        )
                        result_references_to_walk[
                            sub_result_reference.from_node
                        ] = sub_result_reference.node
                        self._add_result_reference_edge(
                            unique_node_id,
                            OpenEOProcessGraph._get_unique_node_id(
                                sub_result_reference.from_node, process_graph_uid
                            ),
                            arg_name=arg_name,
                            setter_func=setter_func,
                        )
                    except pydantic.error_wrappers.ValidationError:
                        pass

            elif isinstance(unpacked_arg, ProcessGraph):
                self.G.nodes[unique_node_id]["resolved_kwargs"][arg_name] = MISSING_VALUE
                callback_result_node_name = self._parse_process_graph(unpacked_arg)
                self.G.add_edge(
                    unique_node_id,
                    OpenEOProcessGraph._get_unique_node_id(
                        callback_result_node_name, unpacked_arg.uid
                    ),
                    reference_type=PGEdgeType.Callback,
                    arg_name=arg_name,
                )

        sub_node: ProcessNode
        for sub_node_name, sub_node in result_references_to_walk.items():
            self._walk_node(
                sub_node, node_name=sub_node_name, process_graph_uid=process_graph_uid
            )

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
            max(nx.shortest_path_length(self.G, source=self.get_root_node()).values()) + 1
        )
        node_colour_palette = [random.randint(0, 255) for _ in range(n_colours)]
        edge_colour_palette = {
            PGEdgeType.ResultReference: "blue",
            PGEdgeType.Callback: "red",
        }
        node_colours = [
            node_colour_palette[self.get_node_depth(node)] for node in self.G.nodes
        ]
        edge_colors = [
            edge_colour_palette.get(self.G.edges[edge]["reference_type"], "green")
            for edge in self.G.edges
        ]

        nx.draw_planar(
            self.G,
            labels=nx.get_node_attributes(self.G, "node_name"),
            horizontalalignment="right",
            verticalalignment="top",
            node_color=node_colours,
            edge_color=edge_colors,
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
