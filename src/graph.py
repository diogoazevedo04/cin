import pandas as pd
import networkx as nx
from math import radians, sin, cos, sqrt, atan2




def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * \
        cos(radians(lat2)) * sin(dlon / 2) ** 2

    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def time_to_minutes(t):
    h, m, s = map(int, t.split(":"))
    return h * 60 + m + s / 60


def load_stops(G, stops_file, modo):
    stops = pd.read_csv(stops_file)

    for _, row in stops.iterrows():
        node_id = f"{modo}_{row['stop_id']}"

        G.add_node(
            node_id,
            name=row["stop_name"],
            lat=float(row["stop_lat"]),
            lon=float(row["stop_lon"]),
            modo=modo
        )


def load_edges_from_gtfs(G, stop_times_file, modo, co2_per_km):
    stop_times = pd.read_csv(stop_times_file)
    stop_times = stop_times.sort_values(["trip_id", "stop_sequence"])

    edge_data = {}

    for _, group in stop_times.groupby("trip_id"):
        rows = group.to_dict("records")

        for i in range(len(rows) - 1):
            s1 = rows[i]
            s2 = rows[i + 1]

            u = f"{modo}_{s1['stop_id']}"
            v = f"{modo}_{s2['stop_id']}"

            if not G.has_node(u) or not G.has_node(v):
                continue

            try:
                t1 = time_to_minutes(s1["departure_time"])
                t2 = time_to_minutes(s2["arrival_time"])
            except Exception:
                continue

            delta = t2 - t1
            if delta <= 0:
                continue

            lat1, lon1 = G.nodes[u]["lat"], G.nodes[u]["lon"]
            lat2, lon2 = G.nodes[v]["lat"], G.nodes[v]["lon"]

            dist_km = haversine(lat1, lon1, lat2, lon2)

            edge_data.setdefault((u, v), []).append((delta, dist_km))

    for (u, v), values in edge_data.items():
        avg_time = sum(t for t, _ in values) / len(values)
        avg_dist = sum(d for _, d in values) / len(values)

        G.add_edge(
            u, v,
            modo=modo,
            time_min=avg_time,
            distance_km=avg_dist,
            co2=avg_dist * co2_per_km
        )


def build_graph():
    G = nx.DiGraph()

    # Nós
    load_stops(G, "data/gtfs/mdp/stops.txt", modo="metro")
    load_stops(G, "data/gtfs/stcp/stops.txt", modo="bus")

    # Arestas GTFS
    load_edges_from_gtfs(
        G,
        "data/gtfs/mdp/stop_times.txt",
        modo="metro",
        co2_per_km=40
    )

    load_edges_from_gtfs(
        G,
        "data/gtfs/stcp/stop_times.txt",
        modo="bus",
        co2_per_km=109.9
    )

    return G





