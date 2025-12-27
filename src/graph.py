import math
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import pandas as pd


# --- Configurações ---

WALK_SPEED_KMH = 5.0  # Velocidade média a pé
WALK_RADIUS_KM = 0.45  # Raio máximo para ligações a pé
TRANSFER_TIME = 2.0  # minutos
CO2_METRO = 40.0 # g/km
CO2_BUS = 109.9 # g/km


# --- Funções Auxiliares ---

def haversine(lat1, lon1, lat2, lon2):
    """Calcula a distância do grande círculo entre dois pontos em km."""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def time_to_minutes(t_str):
    """Converte string de tempo (HH:MM:SS) para minutos totais. 
    É necessária para processar a os horários do ficheiro stop_times.txt"""
    
    try:
        h, m, s = map(int, t_str.split(":"))
        return h * 60 + m + s / 60
    except ValueError:
        raise ValueError(f"Formato de tempo inválido: {t_str}")


def walk_time_minutes(dist_km):
    """Calcula tempo de caminhada baseado na velocidade média."""
    return (dist_km / WALK_SPEED_KMH) * 60


# --- Funções de Construção do Grafo ---

def load_stops(graph, filepath, modo):
    """Carrega paragens de um ficheiro CSV para o grafo."""
    path = Path(filepath)
    if not path.exists():
        return
    
    stops = pd.read_csv(path)
    for _, row in stops.iterrows():
        node_id = f"{modo}_{row['stop_id']}"
        graph.add_node(
            node_id,
            lat=float(row["stop_lat"]),
            lon=float(row["stop_lon"]),
            modo=modo
        )


def load_gtfs_edges(graph, filepath, modo, co2_per_km):
    """Processa stop_times e cria arestas de transporte público."""
    path = Path(filepath)
    if not path.exists():
        return
    
    stop_times = pd.read_csv(path)
    stop_times = stop_times.sort_values(["trip_id", "stop_sequence"])
    
    edge_data: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
    count_edges = 0
    
    for _, group in stop_times.groupby("trip_id"):
        rows = group.to_dict("records")
        
        for i in range(len(rows) - 1):
            s1, s2 = rows[i], rows[i + 1]
            u = f"{modo}_{s1['stop_id']}"
            v = f"{modo}_{s2['stop_id']}"
            
            if not graph.has_node(u) or not graph.has_node(v):
                continue
            
            try:
                t1 = time_to_minutes(s1["departure_time"])
                t2 = time_to_minutes(s2["arrival_time"])
                delta = t2 - t1
                if delta <= 0:
                    continue
            except ValueError:
                continue
            
            lat1, lon1 = graph.nodes[u]["lat"], graph.nodes[u]["lon"]
            lat2, lon2 = graph.nodes[v]["lat"], graph.nodes[v]["lon"]
            dist_km = haversine(lat1, lon1, lat2, lon2)
            
            if (u, v) not in edge_data:
                edge_data[(u, v)] = []
            edge_data[(u, v)].append((delta, dist_km))
    
    for (u, v), values in edge_data.items():
        avg_time = sum(t for t, _ in values) / len(values)
        avg_dist = sum(d for _, d in values) / len(values)
        
        graph.add_edge(
            u, v,
            modo=modo,
            time_min=avg_time,
            distance_km=avg_dist,
            co2=avg_dist * co2_per_km
        )
        count_edges += 1
    
def add_walk_edges(graph):
    """Gera conexões pedonais usando Haversine com fator 1.2."""
    nodes = list(graph.nodes(data=True))
    
    for i, (u, du) in enumerate(nodes):
        for v, dv in nodes[i + 1:]:
            dist_euclidian = haversine(du["lat"], du["lon"], dv["lat"], dv["lon"])
            if dist_euclidian > WALK_RADIUS_KM:
                continue
            
            dist_km = dist_euclidian * 1.2
            modos = {du.get("modo"), dv.get("modo")}
            transfer_time = TRANSFER_TIME if modos == {"metro", "bus"} else 0.0
            time_min = walk_time_minutes(dist_km) + transfer_time
            
            if dist_km <= WALK_RADIUS_KM:
                attrs = {
                    "modo": "walk",
                    "time_min": time_min,
                    "distance_km": dist_km,
                    "co2": 0.0
                }
                graph.add_edge(u, v, **attrs)
                graph.add_edge(v, u, **attrs)
        

def save_graph(graph: nx.MultiDiGraph, output_path: str) -> None:
    """Serializa o grafo para disco usando pickle."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "wb") as f:
        pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)


def build_graph() -> None:
    """Executa o pipeline completo de construção do grafo."""
    graph = nx.MultiDiGraph()
    
    load_stops(graph, "data/gtfs/mdp/stops.txt", "metro")
    load_stops(graph, "data/gtfs/stcp/stops.txt", "bus")
    
    load_gtfs_edges(graph, "data/gtfs/mdp/stop_times.txt", "metro", CO2_METRO)
    load_gtfs_edges(graph, "data/gtfs/stcp/stop_times.txt", "bus", CO2_BUS)
    add_walk_edges(graph)
    
    save_graph(graph, "data/output/graph_base.gpickle")