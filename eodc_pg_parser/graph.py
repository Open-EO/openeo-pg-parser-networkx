import networkx as nx
import random

class OpenEOGraph(object):
    def __init__(self, data):
        self.G = self.parse_process_graph(data)

    def parse_process_graph(self, data, attrs=None, name="name", ident="id"):
        graph = nx.MultiDiGraph()
        nx.add_path(graph, [0, 1, 2, 3])
        return graph

    def plot(self):
        n_colours = max(nx.shortest_path_length(self.G, source=self.get_root_node()).values()) + 1
        colour_palette = [random.randint(0, 255) for _ in range(n_colours)]
        colours = [colour_palette[self.get_node_depth(node)] for node in self.G.nodes]

        nx.draw_kamada_kawai(self.G, labels={node: node for node in self.G.nodes}, horizontalalignment="right", verticalalignment="top", node_color=colours)
        nx.draw_networkx_edge_labels(self.G, nx.kamada_kawai_layout(self.G), nx.get_edge_attributes(self.G,'name'), font_color="#00211e")

    def get_root_node(self):
        return next(nx.topological_sort(self.G))

    def get_node_depth(self, node):
        return nx.shortest_path_length(self.G, source=self.get_root_node(), target=node)

# def cytoscape_graph(selfdata, attrs=None, name="name", ident="id"):
#     """
#     Create a NetworkX graph from a dictionary in cytoscape JSON format.

#     Parameters
#     ----------
#     data : dict
#         A dictionary of data conforming to cytoscape JSON format.
#     attrs : dict or None (default=None)
#         A dictionary containing the keys 'name' and 'ident' which are mapped to
#         the 'name' and 'id' node elements in cyjs format. All other keys are
#         ignored. Default is `None` which results in the default mapping
#         ``dict(name="name", ident="id")``.

#         .. deprecated:: 2.6

#            The `attrs` keyword argument will be replaced with `name` and
#            `ident` in networkx 3.0

#     name : string
#         A string which is mapped to the 'name' node element in cyjs format.
#         Must not have the same value as `ident`.
#     ident : string
#         A string which is mapped to the 'id' node element in cyjs format.
#         Must not have the same value as `name`.

#     Returns
#     -------
#     graph : a NetworkX graph instance
#         The `graph` can be an instance of `Graph`, `DiGraph`, `MultiGraph`, or
#         `MultiDiGraph` depending on the input data.

#     Raises
#     ------
#     NetworkXError
#         If the `name` and `ident` attributes are identical.

#     See Also
#     --------
#     cytoscape_data: convert a NetworkX graph to a dict in cyjs format

#     References
#     ----------
#     .. [1] Cytoscape user's manual:
#        http://manual.cytoscape.org/en/stable/index.html

#     Examples
#     --------
#     >>> data_dict = {
#     ...     'data': [],
#     ...     'directed': False,
#     ...     'multigraph': False,
#     ...     'elements': {'nodes': [{'data': {'id': '0', 'value': 0, 'name': '0'}},
#     ...       {'data': {'id': '1', 'value': 1, 'name': '1'}}],
#     ...      'edges': [{'data': {'source': 0, 'target': 1}}]}
#     ... }
#     >>> G = nx.cytoscape_graph(data_dict)
#     >>> G.name
#     ''
#     >>> G.nodes()
#     NodeView((0, 1))
#     >>> G.nodes(data=True)[0]
#     {'id': '0', 'value': 0, 'name': '0'}
#     >>> G.edges(data=True)
#     EdgeDataView([(0, 1, {'source': 0, 'target': 1})])
#     """
#     if attrs is not None:
#         import warnings

#         msg = (
#             "\nThe `attrs` keyword argument of cytoscape_data is deprecated\n"
#             "and will be removed in networkx 3.0.\n"
#             "It is replaced with explicit `name` and `ident` keyword\n"
#             "arguments.\n"
#             "To make this warning go away and ensure usage is forward\n"
#             "compatible, replace `attrs` with `name` and `ident`,\n"
#             "for example:\n\n"
#             "   >>> cytoscape_data(G, attrs={'name': 'foo', 'ident': 'bar'})\n\n"
#             "should instead be written as\n\n"
#             "   >>> cytoscape_data(G, name='foo', ident='bar')\n\n"
#             "The default values of 'name' and 'id' will not change."
#         )
#         warnings.warn(msg, DeprecationWarning, stacklevel=2)

#         name = attrs["name"]
#         ident = attrs["ident"]
#     # -------------------------------------------------- #

#     if name == ident:
#         raise nx.NetworkXError("name and ident must be different.")

#     multigraph = data.get("multigraph")
#     directed = data.get("directed")
#     if multigraph:
#         graph = nx.MultiGraph()
#     else:
#         graph = nx.Graph()
#     if directed:
#         graph = graph.to_directed()
#     graph.graph = dict(data.get("data"))
#     for d in data["elements"]["nodes"]:
#         node_data = d["data"].copy()
#         node = d["data"]["value"]

#         if d["data"].get(name):
#             node_data[name] = d["data"].get(name)
#         if d["data"].get(ident):
#             node_data[ident] = d["data"].get(ident)

#         graph.add_node(node)
#         graph.nodes[node].update(node_data)

#     for d in data["elements"]["edges"]:
#         edge_data = d["data"].copy()
#         sour = d["data"]["source"]
#         targ = d["data"]["target"]
#         if multigraph:
#             key = d["data"].get("key", 0)
#             graph.add_edge(sour, targ, key=key)
#             graph.edges[sour, targ, key].update(edge_data)
#         else:
#             graph.add_edge(sour, targ)
#             graph.edges[sour, targ].update(edge_data)
#     return graph