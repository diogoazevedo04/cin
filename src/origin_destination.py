import osmnx as ox

WALK_SPEED_KMH = 5.0


def meters_to_minutes(dist_m):
    return (dist_m / 1000) / WALK_SPEED_KMH * 60


def add_origin_destination(G, G_walk, lat, lon, node_id):
    """
    Adiciona um nó genérico (origem ou destino) ligado ao OSM.
    """
    nodes = [
    n for n, d in G_walk.nodes(data=True)
    if "x" in d and "y" in d
    ]

    # encontrar nó pedonal mais próximo
    nearest, dist = ox.distance.nearest_nodes(
        G_walk,
        lon,
        lat,
        return_dist=True
    )

    time_min = meters_to_minutes(dist)

    # adicionar nó ao grafo principal
    G.add_node(
        node_id,
        lat=lat,
        lon=lon,
        modo="walk"
    )

    # origem/destino → rua
    G.add_edge(
        node_id,
        nearest,
        modo="walk",
        time_min=time_min,
        co2=0
    )

    # rua → origem/destino
    G.add_edge(
        nearest,
        node_id,
        modo="walk",
        time_min=time_min,
        co2=0
    )
