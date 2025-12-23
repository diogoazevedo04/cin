import osmnx as ox

WALK_SPEED_KMH = 5.0


def meters_to_minutes(dist_m):
    return (dist_m / 1000) / WALK_SPEED_KMH * 60


def connect_stops_to_walk(G_walk, G_transit, max_dist=300):


    added_edges = 0

    nodes = list(G_transit.nodes(data=True))

    for node, data in nodes:
        lat = data.get("lat")
        lon = data.get("lon")

        if lat is None or lon is None:
            continue

        try:
            nearest, dist = ox.distance.nearest_nodes(
                G_walk,
                lon,
                lat,
                return_dist=True
            )
        except Exception:
            continue

        if dist > max_dist:
            continue

        time_min = meters_to_minutes(dist)

        # paragem → rua
        G_transit.add_edge(
            node,
            nearest,
            modo="walk",
            time_min=time_min,
            co2=0
        )

        # rua → paragem
        G_transit.add_edge(
            nearest,
            node,
            modo="walk",
            time_min=time_min,
            co2=0
        )

        added_edges += 2

    print(f"Arestas pedonais adicionadas: {added_edges}")

