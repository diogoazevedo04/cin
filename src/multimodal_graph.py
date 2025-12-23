import networkx as nx
from graph import build_graph
from osm_walk import load_walk_graph
from connect import connect_stops_to_walk


def build_multimodal_graph():
    G_transit = build_graph()
    G_walk = load_walk_graph()

    G = nx.compose(G_transit, G_walk)
    connect_stops_to_walk(G_walk, G)

    return G, G_walk

