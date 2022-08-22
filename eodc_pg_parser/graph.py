from __future__ import annotations

import logging
import random
from collections import namedtuple
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Dict, List, Optional

import networkx as nx
import pydantic

from eodc_pg_parser.pg_schema import (ParameterReference, PGEdgeType,
                                      ProcessArgument, ProcessGraph,
                                      ProcessNode, ResultReference)
from eodc_pg_parser.utils import ProcessGraphUnflattener

logger = logging.getLogger(__name__)

ArgSubstitution = namedtuple("ArgSubstitution", ["arg_name", "setter_func"])

@dataclass
class EvalEnv:
    """
    Object to pass which parameter references are available for each node throughout walking the graph.
    """
    parent: Optional[EvalEnv]
    parameters: Dict[str, str] = field(default_factory=dict)  # Mapping arg_name to node where to get it from 

    def search_for_parameter(self, arg_name: str):
        """
        Recursively search for a parameter in a node's lineage. The most specific parameter (i.e. from the closest ancestor) is used.
        """
        if arg_name in self.parameters:
            return self.parameters[arg_name]
        if self.parent:
            return self.parent.search_for_parameter(arg_name)
        raise ProcessParameterMissing(f"ProcessParameter {arg_name} missing.")

UNRESOLVED_RESULT_REFERENCE_VALUE = "__UNRESOLVED_RESULT_REFERENCE__"
UNRESOLVED_CALLBACK_VALUE = "__UNRESOLVED_CALLBACK__"


class ProcessParameterMissing(Exception):
    pass


class OpenEOProcessGraph(object):
    def __init__(self, pg_data: Dict):
        self.G = nx.DiGraph()
        nested_raw_graph = self._unflatten_raw_process_graph(pg_data)
        self.nested_graph = self._parse_datamodel(nested_raw_graph)

        # Start parsing the graph at the result node of the top-level graph.
        self._parse_process_graph(self.nested_graph, eval_env=EvalEnv(parent=None, parameters={}))

    @staticmethod
    def _unflatten_raw_process_graph(raw_flat_graph: Dict) -> Dict:
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
    def _parse_datamodel(nested_graph: Dict) -> ProcessGraph:
        """
        Parses a nested process graph into the Pydantic datamodel for ProcessGraph.
        """

        return ProcessGraph.parse_obj(nested_graph)

    def _parse_process_graph(self, process_graph: ProcessGraph, eval_env: EvalEnv):
        """
        Start recursively walking a process graph from its result node and parse its information into self.G. 
        This step passes process_graph.uid to make sure that each process graph operates within its own namespace so that nodes are unique.
        """
        
        for node_name, node in process_graph.process_graph.items():
            if node.result:
                self._walk_node(node, node_name, process_graph.uid, eval_env=eval_env)
                return node_name
        raise Exception("Process graph has no return node!")

    def _resolve_result_reference(
        self, node_uid: str, from_node_uid: str, arg_name):
        self.G.nodes[node_uid]["resolved_kwargs"][arg_name] = UNRESOLVED_RESULT_REFERENCE_VALUE

        # This just points to the resolved_kwarg itself!
        setter_func = partial(
            lambda node_uid, arg_name, new_value: self.G.nodes[node_uid][
                "resolved_kwargs"
            ].__setitem__(arg_name, new_value),
            node_uid=node_uid,
            arg_name=arg_name,
        )

        self._add_result_reference_edge(
            node_uid,
            from_node_uid,
            arg_name=arg_name,
            setter_func=setter_func,
        )

    def _add_result_reference_edge(
        self,
        node_uid: str,
        from_node_uid: str,
        arg_name: str,
        setter_func: Callable,
    ):
        self.G.add_edge(
            node_uid, from_node_uid, reference_type=PGEdgeType.ResultReference
        )
        if "arg_substitutions" not in self.G.edges[node_uid, from_node_uid]:
            self.G.edges[node_uid, from_node_uid]["arg_substitutions"] = []

        self.G.edges[node_uid, from_node_uid]["arg_substitutions"].append(
            ArgSubstitution(arg_name=arg_name, setter_func=setter_func)
        )

    @staticmethod
    def _get_node_uid(node_name: str, process_graph_uid: str):
        return f"{node_name}-{process_graph_uid}"

    def _walk_node(self, node: ProcessNode, node_name: str, process_graph_uid: str, eval_env: EvalEnv):
        """
        Parse all the required information from the current node into self.G and recursively walk child nodes.
        """
        print(f"Walking node {node_name}")

        node_uid = OpenEOProcessGraph._get_node_uid(
            node_name, process_graph_uid
        )

        # Only walk a node once!
        if self.G.nodes.get(node_uid, False):
            return

        self.G.add_node(node_uid, resolved_kwargs={}, node_name=node_name)
        result_references_to_walk = {}  # type: Dict[str, ProcessNode]

        arg: ProcessArgument
        for arg_name, arg in node.arguments.items():
            unpacked_arg = arg.__root__

            # Put the raw arg into the resolved_kwargs dict. If there are no further references within, that's already the right kwarg to pass on.
            # If there are further references, doing this will ensure that the container for these references is already there
            # and the setter_functions can inject the resolved parameters later.
            self.G.nodes[node_uid]["resolved_kwargs"][arg_name] = unpacked_arg

            if isinstance(unpacked_arg, ParameterReference):
                # Search parent nodes for the referenced parameter.
                parameter_provider_node_id = eval_env.search_for_parameter(unpacked_arg.from_parameter)
                assert unpacked_arg.from_parameter in self.G.nodes[parameter_provider_node_id]["resolved_kwargs"]

                resolved_parameter = self.G.nodes[parameter_provider_node_id]["resolved_kwargs"][unpacked_arg.from_parameter]
                if resolved_parameter == UNRESOLVED_RESULT_REFERENCE_VALUE:
                    # Find out which node the result reference edge on parameter_provider_node_id is pointing to
                    # Create a new result_reference edge from unique_node_id to that origin node
                    self.G.nodes[node_uid]["resolved_kwargs"][arg_name] = UNRESOLVED_RESULT_REFERENCE_VALUE
                    for _, potential_origin_node, data in self.G.out_edges(parameter_provider_node_id, data=True):
                        if data["reference_type"] == PGEdgeType.ResultReference:
                            arg_substitution: ArgSubstitution
                            for arg_substitution in data["arg_substitutions"]:
                                if arg_substitution.arg_name == unpacked_arg.from_parameter:
                                    self._resolve_result_reference(
                                        node_uid=node_uid,
                                        from_node_uid=potential_origin_node,
                                        arg_name=arg_name
                                    )
                                    self.G.nodes[node_uid]["resolved_kwargs"][arg_name] = UNRESOLVED_RESULT_REFERENCE_VALUE
                else:
                    self.G.nodes[node_uid]["resolved_kwargs"][unpacked_arg.from_parameter] = resolved_parameter
                

            elif isinstance(unpacked_arg, ResultReference):
                self.G.nodes[node_uid]["resolved_kwargs"][arg_name] = UNRESOLVED_RESULT_REFERENCE_VALUE
                self._resolve_result_reference(
                    node_uid=node_uid,
                    from_node_uid=OpenEOProcessGraph._get_node_uid(unpacked_arg.from_node, process_graph_uid),
                    arg_name=arg_name
                )
                result_references_to_walk[unpacked_arg.from_node] = unpacked_arg.node

            elif isinstance(unpacked_arg, dict):
                self.G.nodes[node_uid]["resolved_kwargs"][arg_name] = unpacked_arg

                for k, v in unpacked_arg.items():
                    try:
                        sub_result_reference = ResultReference.parse_obj(v)
                        setter_func = partial(
                            lambda node_uid, arg_name, new_value, key: self.G.nodes[
                                node_uid
                            ]["resolved_kwargs"][arg_name].__setitem__(key, new_value),
                            node_uid=node_uid,
                            arg_name=arg_name,
                            key=k,
                        )
                        result_references_to_walk[
                            sub_result_reference.from_node
                        ] = sub_result_reference.node
                        self._add_result_reference_edge(
                            node_uid,
                            OpenEOProcessGraph._get_node_uid(
                                sub_result_reference.from_node, process_graph_uid
                            ),
                            arg_name=arg_name,
                            setter_func=setter_func,
                        )
                    except pydantic.error_wrappers.ValidationError:
                        pass

            elif isinstance(unpacked_arg, list):
                self.G.nodes[node_uid]["resolved_kwargs"][arg_name] = unpacked_arg
                for i, element in enumerate(unpacked_arg):
                    try:
                        sub_result_reference = ResultReference.parse_obj(element)
                        setter_func = partial(
                            lambda node_uid, arg_name, new_value, key: self.G.nodes[
                                node_uid
                            ]["resolved_kwargs"][arg_name].__setitem__(key, new_value),
                            node_uid=node_uid,
                            arg_name=arg_name,
                            key=i,
                        )
                        result_references_to_walk[
                            sub_result_reference.from_node
                        ] = sub_result_reference.node
                        self._add_result_reference_edge(
                            node_uid,
                            OpenEOProcessGraph._get_node_uid(
                                sub_result_reference.from_node, process_graph_uid
                            ),
                            arg_name=arg_name,
                            setter_func=setter_func,
                        )
                    except pydantic.error_wrappers.ValidationError:
                        pass

        new_eval_env = EvalEnv(parent=eval_env, parameters={k: node_uid for k, _ in self.G.nodes[node_uid]["resolved_kwargs"].items()})

        arg: ProcessArgument
        for arg_name, arg in node.arguments.items():
            unpacked_arg = arg.__root__
            if isinstance(unpacked_arg, ProcessGraph):
                self.G.nodes[node_uid]["resolved_kwargs"][arg_name] = UNRESOLVED_CALLBACK_VALUE
                callback_result_node_name = self._parse_process_graph(unpacked_arg, eval_env=new_eval_env)
                self.G.add_edge(
                    node_uid,
                    OpenEOProcessGraph._get_node_uid(
                        callback_result_node_name, unpacked_arg.uid
                    ),
                    reference_type=PGEdgeType.Callback,
                    arg_name=arg_name,
                )
        
        sub_node: ProcessNode
        for sub_node_name, sub_node in result_references_to_walk.items():
            self._walk_node(
                sub_node, node_name=sub_node_name, process_graph_uid=process_graph_uid, eval_env=eval_env
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

        nx.draw_kamada_kawai(
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
