import csv
import pickle
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path
from typing import Dict, List, Any, Tuple
import networkx as nx

from moead import moead, get_edge

SPEED_KMH = 5.0  # Velocidade média a pé
K_NEIGHBORS = 15  # Número de vizinhos para ligações a pé

# --- Funções de Grafos ---

def load_graph(graph_path):
    """Carrega o grafo a partir de um ficheiro pickle."""
    path = Path(graph_path)
    if not path.exists():
        raise FileNotFoundError(f"Grafo não encontrado em: {path}")
    
    with open(path, "rb") as f:
        return pickle.load(f)


def haversine(lat1, lon1, lat2, lon2):
    """Calcula distância do grande círculo em km."""
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def walking_time(dist_km, speed_kmh = SPEED_KMH):
    """Calcula tempo de caminhada em minutos."""
    return (dist_km / speed_kmh) * 60


def add_virtual_node(graph, node_id, lat, lon, k_neighbors = K_NEIGHBORS):
    """Adiciona um ponto de interesse conectado aos K vizinhos mais próximos."""
    graph.add_node(node_id, lat=lat, lon=lon, modo="walk")
    
    distances = []
    for n, data in graph.nodes(data=True):
        if n == node_id: 
            continue
        d = haversine(lat, lon, data["lat"], data["lon"])
        distances.append((d, n))
    
    distances.sort()
    
    for dist_km, n in distances[:k_neighbors]:
        neighbor_mode = graph.nodes[n].get("modo")
        transfer_penalty = 1.0 if neighbor_mode in {"metro", "bus"} else 0.0
        t = walking_time(dist_km) + transfer_penalty
        attrs = {"modo": "walk", "distance_km": dist_km, "time_min": t, "co2": 0}
        graph.add_edge(node_id, n, **attrs)
        graph.add_edge(n, node_id, **attrs)


# --- Funções de Análise ---

def get_mode_breakdown(path, graph, edges=None):
    """Agrega tempo, distância e contagem de arestas por modo."""
    totals = {
        'walk': {'time': 0.0, 'dist': 0.0, 'edges': 0},
        'metro': {'time': 0.0, 'dist': 0.0, 'edges': 0},
        'bus': {'time': 0.0, 'dist': 0.0, 'edges': 0},
    }
    
    if edges:
        for e in edges:
            m = e.get('modo', 'walk')
            if m not in totals:
                totals[m] = {'time': 0.0, 'dist': 0.0, 'edges': 0}
            totals[m]['time'] += float(e.get('time_min', 0.0))
            totals[m]['dist'] += float(e.get('distance_km', 0.0))
            totals[m]['edges'] += 1
    else:
        for u, v in zip(path[:-1], path[1:]):
            e = get_edge(graph, u, v)
            m = e.get('modo', 'walk')
            if m not in totals:
                totals[m] = {'time': 0.0, 'dist': 0.0, 'edges': 0}
            totals[m]['time'] += float(e.get('time_min', 0.0))
            totals[m]['dist'] += float(e.get('distance_km', 0.0))
            totals[m]['edges'] += 1
    
    return totals

def segment_path(path, graph, edges=None):
    """Divide o caminho em segmentos contíguos por modo."""
    if not path or len(path) < 2:
        return []
    
    segments = []
    current_mode = None
    seg_start = path[0]
    acc_time = 0.0
    acc_dist = 0.0
    
    if edges:
        for i, e in enumerate(edges):
            mode = e.get('modo')
            time_min = float(e.get('time_min', 0.0))
            dist_km = float(e.get('distance_km', 0.0))
            
            if current_mode is None:
                current_mode = mode
            if mode != current_mode:
                segments.append((current_mode, seg_start, path[i], acc_time, acc_dist))
                current_mode = mode
                seg_start = path[i]
                acc_time = 0.0
                acc_dist = 0.0
            acc_time += time_min
            acc_dist += dist_km
        segments.append((current_mode, seg_start, path[-1], acc_time, acc_dist))
    else:
        for u, v in zip(path[:-1], path[1:]):
            edge = get_edge(graph, u, v)
            mode = edge.get('modo')
            time_min = float(edge.get('time_min', 0.0))
            dist_km = float(edge.get('distance_km', 0.0))
            if current_mode is None:
                current_mode = mode
            if mode != current_mode:
                segments.append((current_mode, seg_start, u, acc_time, acc_dist))
                current_mode = mode
                seg_start = u
                acc_time = 0.0
                acc_dist = 0.0
            acc_time += time_min
            acc_dist += dist_km
        segments.append((current_mode, seg_start, path[-1], acc_time, acc_dist))
    
    return segments


def print_solution_details(sol, graph, name = "Solução"):
    """Imprime detalhes de uma solução (dict)."""
    edges = sol.get("edges")
    path = sol.get("path", [])
    print(f"\n{'='*60}\n{name.upper()}\n{'='*60}")
    print(f"Tempo total: {sol['time']:.1f} min")
    print(f"CO₂ total: {sol['co2']:.0f} g")
    print(f"Paragens: {len(path)}")
    
    walk_dist = sum(e.get('distance_km', 0.0) for e in edges if e.get('modo') == 'walk') if edges else 0.0
    breakdown = get_mode_breakdown(path, graph, edges)
    print(f"Distância a pé: {walk_dist:.2f} km")
    print("\nResumo por modo:")
    for m in ['walk', 'metro', 'bus']:
        t = breakdown.get(m, {}).get('time', 0.0)
        d = breakdown.get(m, {}).get('dist', 0.0)
        e = breakdown.get(m, {}).get('edges', 0)
        if e:
            print(f" - {m.upper()}: {t:.1f} min | {d:.2f} km | {e} arestas")
    
    print(f"\nSegmentos:")
    for mode, a, b, tmin, dkm in segment_path(path, graph, edges):
        print(f" - {mode.upper()}: {a} → {b} | {tmin:.1f} min | {dkm:.2f} km")


def print_comparison(solutions_dict: Dict[str, Dict[str, Any]]):
    """Imprime tabela comparativa de soluções."""
    print(f"\n{'='*60}\nCOMPARAÇÃO DE SOLUÇÕES\n{'='*60}\n")
    print(f"{'Critério':<15} {'Tempo':<15} {'CO₂':<15} {'A pé (km)':<15}")
    print("-" * 60)
    
    for name, sol in solutions_dict.items():
        edges = sol.get("edges")
        walk_dist = sum(e.get('distance_km', 0.0) for e in edges if e.get('modo') == 'walk') if edges else 0.0
        print(f"{name:<15} {sol['time']:<15.1f} {sol['co2']:<15.0f} {walk_dist:<15.2f}")


# --- Funções de Persistência ---

def save_pickle(data, filepath):
    """Guarda dados em formato pickle."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)

def save_pareto_csv(pareto_front, graph, filepath):
    """Guarda frente de Pareto em formato CSV."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time_min', 'co2_g', 'path_length', 'walk_distance_km'])
        
        for sol in pareto_front:
            path_nodes = sol.get('path', [])
            walk_dist = sum(
                get_edge(graph, u, v)['distance_km'] 
                for u, v in zip(path_nodes[:-1], path_nodes[1:])
                if get_edge(graph, u, v)['modo'] == 'walk'
            )
            writer.writerow([sol['time'], sol['co2'], len(path_nodes), walk_dist])


# --- Função Principal ---

def get_coordinates(location_name):
    """Pede ao utilizador as coordenadas (latitude, longitude)."""
    while True:
        try:
            coords_input = input(f"Insira as coordenadas de {location_name} (lat, lon): ")
            lat, lon = map(float, coords_input.split(','))
            return (lat, lon)
        except ValueError:
            print("Formato inválido. Use: latitude,longitude (ex: 41.1768,-8.6936)")


def run_optimization():
    """Pipeline completo de otimização."""
    # Pedir coordenadas ao utilizador
    print(f"\n{'='*60}")
    print("OTIMIZAÇÃO DE ROTAS MULTIOBJETIVO")
    print(f"{'='*60}\n")
    
    origin = get_coordinates("origem")
    destination = get_coordinates("destino")
    
    print(f"\n✓ Origem: {origin}")
    print(f"✓ Destino: {destination}\n")
    
    # Carregar grafo
    graph = load_graph("data/output/graph_base.gpickle")
    
    # Adicionar origem e destino
    add_virtual_node(graph, "origin", *origin)
    add_virtual_node(graph, "destination", *destination)
    
    # Executar MOEA/D
    pareto_front, extremes, history = moead(
        graph=graph,
        source="origin",
        target="destination"
    )
    
    if extremes:
        labels = {
            'best_time': "Melhor Tempo",
            'best_co2': "Melhor CO₂",
            'balanced': "Balanceada"
        }
        
        for key, label in labels.items():
            if key in extremes:
                print_solution_details(extremes[key], graph, label)
        
        solutions_to_compare = {labels[k]: v for k, v in extremes.items()}
        print_comparison(solutions_to_compare)
    
    # Exportar resultados
    results = {
        'pareto_front': pareto_front,
        'extremes': extremes,
        'history': history,
        'config': {
            'origin': origin,
            'destination': destination
        }
    }
    
    print(f"\n{'='*60}\nA EXPORTAR RESULTADOS\n{'='*60}")
    save_pickle(results, "output/images/moead_results.pkl")
    save_pareto_csv(pareto_front, graph, "output/images/pareto_front.csv")


if __name__ == "__main__":
    run_optimization()