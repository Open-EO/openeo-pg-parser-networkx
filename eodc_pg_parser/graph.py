import networkx as nx
import random
import logging
from openeo.internal.process_graph_visitor import ProcessGraphUnflattener
from eodc_pg_parser.pg_schema import ProcessGraph, ProcessNode, ResultReference, ParameterReference


logger = logging.getLogger(__name__)


class OpenEOProcessGraph(object):
    def __init__(self, pg_data):
        self.G = nx.DiGraph()
        root_node = self._unflatten_process_graph(pg_data).process_graph["root"]
        self._walk_node(root_node, root_node.process_id)

    def _unflatten_process_graph(self, flat_graph: dict) -> ProcessGraph:
        graph = {"process_graph": {"root": ProcessGraphUnflattener.unflatten(flat_graph["process_graph"])}}
        process_graph = ProcessGraph.parse_obj(graph)
        logger.warning("Deserialised process graph into nested structure")
        return process_graph

    def _walk_node(self, node: ProcessNode, node_id: str):
        # 1. Find the connected nodes. These can either be ResultReferences or ParameterReferences 
        # (or UDFs I guess)

        self.G.add_node(node_id, node=node, resolved_kwargs={})

        # Walk any dependencies first!
        for arg_name, arg_wrapper in node.arguments.items():
            # This is a consequence of using __root__ in the Pydantic model for ProcessNode
            arg = arg_wrapper.__root__

            # Create edges for result references
            if isinstance(arg, ResultReference):
                self._walk_node(arg.node, node_id=arg.from_node)
                self.G.add_edge(node_id, arg.from_node, reference_type="ResultReference", arg_name=arg_name)

            elif isinstance(arg, ProcessGraph):
                root_node_id = next(iter(arg.process_graph))
                root_node = arg.process_graph.get(root_node_id)
                self._walk_node(root_node, root_node_id)
                self.G.add_edge(node_id, root_node_id, reference_type="Callback", arg_name=arg_name)

            # Parameter references need to be resolved from the dependant nodes upwards.
            elif isinstance(arg, ParameterReference):
                pass

            # Construct the argument list
            else:
                self.G.nodes[node_id]["resolved_kwargs"][arg_name] = arg
    
        # TODO: Handle reducers

    def _resolve_parameter_references(self):
        pass

    def plot(self):
        if self.G.number_of_nodes() < 1:
            logger.warning("Graph has no nodes, nothing to plot.")
            return

        n_colours = (
            max(nx.shortest_path_length(self.G, source=self.get_root_node()).values())
            + 1
        )
        node_colour_palette = [random.randint(0, 255) for _ in range(n_colours)]
        edge_colour_palette = {"ResultReference": "blue", "Reducer": "red"}
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

