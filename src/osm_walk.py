import osmnx as ox
import networkx as nx


def load_walk_graph():
    G = ox.graph_from_place(
        "Porto, Portugal",
        network_type="walk",
        simplify=True
    )

    # ⚠️ CONVERSÃO CRÍTICA
    G = nx.DiGraph(G)

    for u, v, data in G.edges(data=True):
        length_m = data.get("length", 0.0)

        data["distance_km"] = length_m / 1000
        data["time_min"] = (length_m / 1000) / 4.8 * 60  # 4.8 km/h
        data["modo"] = "walk"
        data["co2"] = 0.0

    for _, data in G.nodes(data=True):
        data["modo"] = "walk"

    return G
