import csv
import pickle
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path
from typing import Dict, List, Any, Tuple
import networkx as nx

from moead import optimize_moead, get_extreme_solutions, analyze_pareto_front, get_edge

# ==============================
# Funções de Rede
# ==============================

def load_graph(graph_path: str) -> nx.DiGraph:
    """Carrega o grafo a partir de um ficheiro pickle."""
    path = Path(graph_path)
    if not path.exists():
        raise FileNotFoundError(f"Grafo não encontrado em: {path}")
    
    with open(path, "rb") as f:
        return pickle.load(f)


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calcula distância do grande círculo em km."""
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def walking_time(dist_km: float, speed_kmh: float = 5.0) -> float:
    """Calcula tempo de caminhada em minutos."""
    return (dist_km / speed_kmh) * 60


def add_virtual_node(graph: nx.DiGraph, node_id: str, lat: float, lon: float, k_neighbors: int = 15):
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


def add_direct_shortcut(graph: nx.DiGraph, node_a: str, node_b: str):
    """Cria uma ligação direta a pé entre dois nós."""
    if not graph.has_node(node_a) or not graph.has_node(node_b):
        return
    
    lat1, lon1 = graph.nodes[node_a]["lat"], graph.nodes[node_a]["lon"]
    lat2, lon2 = graph.nodes[node_b]["lat"], graph.nodes[node_b]["lon"]
    
    dist = haversine(lat1, lon1, lat2, lon2) * 1.2
    time_walk = walking_time(dist)
    
    attrs = {"modo": "walk", "time_min": time_walk, "distance_km": dist, "co2": 0.0}
    graph.add_edge(node_a, node_b, **attrs)
    graph.add_edge(node_b, node_a, **attrs)


# ==============================
# Funções de Análise
# ==============================

def get_mode_breakdown(path: List[str], graph: nx.DiGraph, edges: List[Dict] = None) -> Dict[str, Dict[str, float]]:
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

def segment_path(path: List[str], graph: nx.DiGraph, edges: List[Dict] = None) -> List[Tuple[str, str, str, float, float]]:
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


def print_solution_details(sol, graph: nx.DiGraph, name: str = "Solução"):
    """Imprime detalhes de uma solução."""
    edges = getattr(sol, "edges", None)
    print(f"\n{'='*60}\n{name.upper()}\n{'='*60}")
    print(f"Tempo total: {sol.time:.1f} min")
    print(f"CO₂ total: {sol.co2:.0f} g")
    print(f"Paragens: {len(sol.path)}")
    
    walk_dist = sum(e.get('distance_km', 0.0) for e in edges if e.get('modo') == 'walk') if edges else 0.0
    breakdown = get_mode_breakdown(sol.path, graph, edges)
    print(f"Distância a pé: {walk_dist:.2f} km")
    print("\nResumo por modo:")
    for m in ['walk', 'metro', 'bus']:
        t = breakdown.get(m, {}).get('time', 0.0)
        d = breakdown.get(m, {}).get('dist', 0.0)
        e = breakdown.get(m, {}).get('edges', 0)
        if e:
            print(f" - {m.upper()}: {t:.1f} min | {d:.2f} km | {e} arestas")
    
    print(f"\nSegmentos:")
    for mode, a, b, tmin, dkm in segment_path(sol.path, graph, edges):
        print(f" - {mode.upper()}: {a} → {b} | {tmin:.1f} min | {dkm:.2f} km")


def print_comparison(solutions_dict: Dict[str, Any]):
    """Imprime tabela comparativa de soluções."""
    print(f"\n{'='*60}\nCOMPARAÇÃO DE SOLUÇÕES\n{'='*60}\n")
    print(f"{'Critério':<15} {'Tempo':<15} {'CO₂':<15} {'A pé (km)':<15}")
    print("-" * 60)
    
    for name, sol in solutions_dict.items():
        edges = getattr(sol, "edges", None)
        walk_dist = sum(e.get('distance_km', 0.0) for e in edges if e.get('modo') == 'walk') if edges else 0.0
        print(f"{name:<15} {sol.time:<15.1f} {sol.co2:<15.0f} {walk_dist:<15.2f}")


# ==============================
# Funções de Persistência
# ==============================

def save_pickle(data: Any, filepath: str):
    """Guarda dados em formato pickle."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)

def save_pareto_csv(pareto_front: List[Any], graph: nx.DiGraph, filepath: str):
    """Guarda frente de Pareto em formato CSV."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time_min', 'co2_g', 'path_length', 'walk_distance_km'])
        
        for sol in pareto_front:
            walk_dist = sum(
                get_edge(graph, u, v)['distance_km'] 
                for u, v in zip(sol.path[:-1], sol.path[1:])
                if get_edge(graph, u, v)['modo'] == 'walk'
            )
            writer.writerow([sol.time, sol.co2, len(sol.path), walk_dist])


# ==============================
# Função Principal
# ==============================

def run_optimization():
    """Pipeline completo de otimização."""
    # Configuração
    origin = (41.1768, -8.6936)  # Aliados
    destination = (41.1297, -8.6065)  # Casa da Música
    pop_size = 100
    generations = 50
    
    # Carregar grafo
    graph = load_graph("data/output/graph_base.gpickle")
    
    # Adicionar origem e destino
    add_virtual_node(graph, "origin", *origin)
    add_virtual_node(graph, "destination", *destination)
    
    # Executar MOEA/D
    pareto_front, extremes, history = optimize_moead(
        graph=graph,
        source="origin",
        target="destination",
        population_size=pop_size,
        n_neighbors=20,
        max_generations=generations
    )
    
    # Análise e relatórios
    analyze_pareto_front(pareto_front, graph)
    
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
    
    print(f"\n{'='*60}\nEXPORTANDO RESULTADOS\n{'='*60}")
    save_pickle(results, "data/output/moead_results.pkl")
    save_pareto_csv(pareto_front, graph, "data/pareto_front.csv")


if __name__ == "__main__":
    run_optimization()