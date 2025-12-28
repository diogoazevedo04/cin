import random
import pickle
import csv
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2
import networkx as nx


def haversine(lat1, lon1, lat2, lon2):
    """Calcula distância em km."""
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def generate_scenarios(
    graph,
    num_scenarios_per_difficulty=3,
    difficulties=None
):
    """
    Gera cenários selecionando pares de nós filtrados por distância aérea.
    Retorna: lista de cenários, cada um com (origem, destino, dificuldade, distância_aérea)
    """
    if difficulties is None:
        difficulties = {
            'facil': (0.5, 4.0),      # 500m - 4km
            'medio': (5.0, 9.0),      # 5km - 9km
            'dificil': (10.0, 17.0),  # 10km - 17km
        }
    
    scenarios = []
    nodes = list(graph.nodes(data=True))
    
    if not nodes:
        print("Grafo vazio!")
        return scenarios
    
    for difficulty_name, (min_dist, max_dist) in difficulties.items():
        print(f"\nA Gerar cenários '{difficulty_name}' ({min_dist:.1f}-{max_dist:.1f} km)...")
        
        attempts = 0
        max_attempts = 1000
        found = 0
        
        while found < num_scenarios_per_difficulty and attempts < max_attempts:
            attempts += 1
            
            origin_node, origin_data = random.choice(nodes)
            destination_node, dest_data = random.choice(nodes)
            
            if origin_node == destination_node:
                continue
            
            origin_lat, origin_lon = origin_data['lat'], origin_data['lon']
            dest_lat, dest_lon = dest_data['lat'], dest_data['lon']
            
            air_distance = haversine(origin_lat, origin_lon, dest_lat, dest_lon)
            
            if not (min_dist <= air_distance <= max_dist):
                continue
            
            try:
                if not nx.has_path(graph, origin_node, destination_node):
                    continue
            except:
                continue
            
            scenario = {
                'origin': (origin_lat, origin_lon),
                'destination': (dest_lat, dest_lon),
                'origin_node': origin_node,
                'destination_node': destination_node,
                'difficulty': difficulty_name,
                'air_distance_km': air_distance
            }
            
            scenarios.append(scenario)
            found += 1
            print(f"Cenário {len(scenarios):2d}: {difficulty_name} | "
                  f"Dist: {air_distance:.2f} km | "
                  f"Origem: ({origin_lat:.4f}, {origin_lon:.4f}) | "
                  f"Destino: ({dest_lat:.4f}, {dest_lon:.4f})")
            attempts = 0
        
        if found < num_scenarios_per_difficulty:
            print(f"Apenas {found}/{num_scenarios_per_difficulty} cenários gerados")
    
    return scenarios


def save_scenarios(scenarios, filepath):
    """Guarda cenários em ficheiro pickle e CSV."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scenarios, f)
    print(f"\n{len(scenarios)} cenários guardados em {filepath}")
    
    csv_path = path.parent / "scenarios.csv"
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'difficulty', 'origin_lat', 'origin_lon', 'destination_lat', 'destination_lon', 'air_distance_km'])
        
        for idx, scenario in enumerate(scenarios, 1):
            origin_lat, origin_lon = scenario['origin']
            dest_lat, dest_lon = scenario['destination']
            writer.writerow([
                idx,
                scenario['difficulty'],
                f"{origin_lat:.6f}",
                f"{origin_lon:.6f}",
                f"{dest_lat:.6f}",
                f"{dest_lon:.6f}",
                f"{scenario['air_distance_km']:.2f}"
            ])
    
    print(f"Coordenadas guardadas em {csv_path} para consulta fácil")


def load_scenarios(filepath):
    """Carrega cenários de um ficheiro pickle."""
    path = Path(filepath)
    if not path.exists():
        print(f"Ficheiro não encontrado: {filepath}")
        return []
    
    with open(path, "rb") as f:
        scenarios = pickle.load(f)
    print(f"{len(scenarios)} cenários carregados de {filepath}")
    return scenarios


def print_scenarios_summary(scenarios):
    """Imprime resumo dos cenários."""
    if not scenarios:
        print("Sem cenários disponíveis.")
        return
    
    print(f"\n{'='*70}")
    print(f"RESUMO DOS CENÁRIOS ({len(scenarios)} total)")
    print(f"{'='*70}\n")
    
    by_difficulty = {}
    for s in scenarios:
        diff = s['difficulty']
        if diff not in by_difficulty:
            by_difficulty[diff] = []
        by_difficulty[diff].append(s)
    
    for difficulty in ['facil', 'medio', 'dificil']:
        if difficulty in by_difficulty:
            count = len(by_difficulty[difficulty])
            distances = [s['air_distance_km'] for s in by_difficulty[difficulty]]
            print(f"{difficulty.upper():15} | {count:2d} cenários | "
                  f"Dist: min={min(distances):.2f}, avg={sum(distances)/len(distances):.2f}, max={max(distances):.2f} km")


def run_scenario_generation():
    graph_path = "output/graph_base.gpickle"
    if not Path(graph_path).exists():
        print(f"Grafo não encontrado em {graph_path}")
        return
    
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)
    
    print(f"\n{'='*70}")
    print(f"GERADOR DE CENÁRIOS")
    print(f"{'='*70}")
    print(f"Grafo carregado: {graph.number_of_nodes()} nós, {graph.number_of_edges()} arestas\n")
    
    scenarios = generate_scenarios(
        graph,
        num_scenarios_per_difficulty=3,
        difficulties={
            'facil': (0.5, 3.5),
            'medio': (4.0, 7.0),
            'dificil': (8.5, 17.0)
        }
    )
    
    save_scenarios(scenarios, "output/scenarios.pkl")
    
    print_scenarios_summary(scenarios)


if __name__ == "__main__":
    run_scenario_generation()
