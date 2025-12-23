import networkx as nx


def shortest_path_time(G, source, target):
    """
    Caminho mais rápido (Dijkstra) usando time_min como peso.
    """

    path = nx.shortest_path(
        G,
        source=source,
        target=target,
        weight="time_min",
        method="dijkstra"
    )

    total_time = nx.shortest_path_length(
        G,
        source=source,
        target=target,
        weight="time_min",
        method="dijkstra"
    )

    return path, total_time


