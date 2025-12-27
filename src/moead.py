import random
from collections import defaultdict
from typing import List, Tuple, Dict, NamedTuple

import networkx as nx
import numpy as np


def get_edge(G, u, v):
    """Helper para MultiDiGraph: retorna a aresta com menor tempo entre u e v."""
    if isinstance(G, nx.MultiDiGraph):
        if G.has_edge(u, v):
            edges = G[u][v]
            if not edges:
                return {}
            best_key = min(edges.keys(), key=lambda k: edges[k].get('time_min', float('inf')))
            return edges[best_key]
        elif G.has_edge(v, u):
            edges = G[v][u]
            if not edges:
                return {}
            best_key = min(edges.keys(), key=lambda k: edges[k].get('time_min', float('inf')))
            return edges[best_key]
        else:
            return {}
    else:
        if G.has_edge(u, v):
            return G[u][v]
        elif G.has_edge(v, u):
            return G[v][u]
        else:
            return {}


class Solution(NamedTuple):
    """Representa um caminho com os seus valores objetivos."""
    path: List[str]
    time: float
    co2: float
    violation: float = 0.0
    edges: List[Dict] = []

    def dominates(self, other: "Solution") -> bool:
        objectives = np.array([self.time, self.co2])
        other_objectives = np.array([other.time, other.co2])
        better = False
        for a, b in zip(objectives, other_objectives):
            if a > b:
                return False
            if a < b:
                better = True
        return better


# ==============================
# Funções de Inicialização
# ==============================

def generate_weights(population_size: int) -> np.ndarray:
    """Gera vetores de peso lineares para decomposição."""
    return np.array(
        [[i / (population_size - 1), 1 - i / (population_size - 1)]
         for i in range(population_size)]
    )


def generate_neighborhoods(weights: np.ndarray, n_neighbors: int) -> List[List[int]]:
    """Gera vizinhanças baseadas em proximidade de pesos."""
    population_size = len(weights)
    neighborhoods = []
    for i in range(population_size):
        distances = [
            (np.linalg.norm(weights[i] - weights[j]), j)
            for j in range(population_size) if i != j
        ]
        distances.sort()
        neighborhoods.append([j for _, j in distances[:n_neighbors]])
    return neighborhoods


# ==============================
# Funções de Avaliação
# ==============================

def evaluate_path(
    path: List[str],
    graph: nx.DiGraph,
    max_transfers: int = 4,
    max_walk_time: float = 90.0,
    transfer_penalty: float = 500.0,
    walk_time_penalty: float = 100.0
) -> Tuple[float, float, float, List[Dict]]:
    """Calcula tempo, CO2 e violação de restrições."""
    time = 0.0
    co2 = 0.0
    walk_time = 0.0
    num_transfers = 0
    prev_mode = None
    edges_used = []

    for u, v in zip(path[:-1], path[1:]):
        edge = get_edge(graph, u, v)
        edge_time = edge.get("time_min", 0.0)
        edge_co2 = edge.get("co2", 0.0)
        mode = edge.get("modo", "walk")

        time += edge_time
        co2 += edge_co2
        edges_used.append(edge.copy())

        if mode == "walk":
            walk_time += edge_time

        if prev_mode is not None and prev_mode != mode:
            num_transfers += 1
        prev_mode = mode

    violation = 0.0
    if num_transfers > max_transfers:
        violation += transfer_penalty * (num_transfers - max_transfers)
    if walk_time > max_walk_time:
        violation += walk_time_penalty * (walk_time - max_walk_time)

    return time, co2, violation, edges_used


def heuristic_initialization(
    graph: nx.DiGraph,
    source: str,
    target: str,
    weights: np.ndarray,
    population_size: int,
    max_transfers: int = 4,
    max_walk_time: float = 90.0,
    transfer_penalty: float = 500.0,
    walk_time_penalty: float = 100.0
) -> List[Solution]:
    """Inicialização heurística com shortest path ponderado."""
    solutions = []

    times = [d["time_min"] for _, _, d in graph.edges(data=True)]
    co2s = [d["co2"] for _, _, d in graph.edges(data=True)]

    max_time = max(times) if times else 1.0
    max_co2 = max(co2s) if co2s else 1.0

    for w_time, w_co2 in weights[:-1]:
        try:
            def cost(u, v, d):
                return (
                    w_time * d.get("time_min", 0.0) / max_time +
                    w_co2 * d.get("co2", 0.0) / max_co2
                )

            path = nx.shortest_path(graph, source, target, weight=cost)
            t, c, viol, edges = evaluate_path(path, graph, max_transfers, max_walk_time, transfer_penalty, walk_time_penalty)
            solutions.append(Solution(path, t, c, viol, edges))
        except nx.NetworkXNoPath:
            continue

    while len(solutions) < population_size - 1 and solutions:
        base = random.choice(solutions)
        mutated = mutate(base.path, graph)
        t, c, viol, edges = evaluate_path(mutated, graph, max_transfers, max_walk_time, transfer_penalty, walk_time_penalty)
        solutions.append(Solution(mutated, t, c, viol, edges))

    return solutions[:population_size - 1]


def create_walk_only_solution(
    graph: nx.DiGraph,
    source: str,
    target: str,
    max_walk_time: float = 90.0,
    walk_time_penalty: float = 100.0
) -> Solution:
    """Cria solução 100% walk se possível."""
    H = nx.DiGraph()
    for n, data in graph.nodes(data=True):
        H.add_node(n, **data)

    if isinstance(graph, nx.MultiDiGraph):
        for u, v, key, data in graph.edges(keys=True, data=True):
            if data.get("modo") != "walk":
                continue
            e = H.get_edge_data(u, v)
            if e is None or data.get("time_min", float("inf")) < e.get("time_min", float("inf")):
                H.add_edge(u, v, **data)
    else:
        for u, v, data in graph.edges(data=True):
            if data.get("modo") == "walk":
                H.add_edge(u, v, **data)

    try:
        path = nx.shortest_path(H, source, target, weight="time_min")
    except nx.NetworkXNoPath:
        return None

    time = 0.0
    co2 = 0.0
    edges_used = []
    for a, b in zip(path[:-1], path[1:]):
        e = H[a][b]
        time += e.get("time_min", 0.0)
        co2 += e.get("co2", 0.0)
        edges_used.append(e.copy())

    violation = 0.0
    if time > max_walk_time:
        violation += walk_time_penalty * (time - max_walk_time)

    return Solution(path, time, co2, violation, edges_used)


# ==============================
# Operadores Genéticos
# ==============================

def mutate(path: List[str], graph: nx.DiGraph, rate: float = 0.8) -> List[str]:
    """Mutação por recombinação de segmentos."""
    if len(path) < 3 or random.random() > rate:
        return path

    i = random.randint(0, len(path) - 2)
    j = random.randint(i + 1, len(path) - 1)

    try:
        weight = random.choice(["time_min", "co2"])
        sub = nx.shortest_path(graph, path[i], path[j], weight=weight)
        new_path = path[:i] + sub + path[j + 1:]
        cleaned = [new_path[0]]
        for n in new_path[1:]:
            if n != cleaned[-1]:
                cleaned.append(n)
        return cleaned
    except nx.NetworkXNoPath:
        return path


def crossover(p1: List[str], p2: List[str]) -> List[str]:
    """Cruzamento por nó comum."""
    common = list(set(p1) & set(p2))
    if len(common) < 2:
        return random.choice([p1, p2])

    n = random.choice(common)
    try:
        return p1[: p1.index(n)] + p2[p2.index(n):]
    except ValueError:
        return p1


# ==============================
# Funções de Otimização
# ==============================

def tchebycheff(obj: np.ndarray, w: np.ndarray, ref: np.ndarray) -> float:
    """Agregação Tchebycheff."""
    return np.max(w * np.abs(obj - ref))


def update_reference(population: List[Solution]) -> np.ndarray:
    """Atualiza ponto de referência ideal."""
    times = [s.time for s in population]
    co2s = [s.co2 for s in population]
    return np.array([min(times), min(co2s)])


def update_pareto(pareto_front: List[Solution], sol: Solution) -> List[Solution]:
    """Atualiza frente de Pareto com penalização por violação."""
    def penalized(obj, viol):
        return np.array([obj[0] + viol, obj[1] + viol])

    penal_sol = penalized(np.array([sol.time, sol.co2]), sol.violation)
    filtered = []
    for s in pareto_front:
        penal_s = penalized(np.array([s.time, s.co2]), s.violation)
        if all(penal_sol <= penal_s) and any(penal_sol < penal_s):
            continue
        if all(penal_s <= penal_sol) and any(penal_s < penal_sol):
            return pareto_front
        filtered.append(s)
    filtered.append(sol)
    return prune_pareto_epsilon(filtered)


def prune_pareto_epsilon(pareto_front: List[Solution], epsilon_time: float = 0.5, epsilon_co2: float = 2.0) -> List[Solution]:
    """Remove soluções redundantes (epsilon-dominance)."""
    if len(pareto_front) <= 1:
        return pareto_front
    
    sorted_front = sorted(pareto_front, key=lambda s: s.time)
    pruned = [sorted_front[0]]
    for sol in sorted_front[1:]:
        keep = True
        for kept in pruned:
            if abs(sol.time - kept.time) <= epsilon_time and abs(sol.co2 - kept.co2) <= epsilon_co2:
                keep = False
                break
        if keep:
            pruned.append(sol)
    return pruned


def hypervolume_2d(front: List[Solution], ref: Tuple[float, float]) -> float:
    """Hipervolume 2D para minimização."""
    if not front:
        return 0.0
    
    pts = sorted([(s.time, s.co2) for s in front], key=lambda x: x[0])
    envelope = []
    best_c = float("inf")
    for t, c in pts:
        if c < best_c:
            envelope.append((t, c))
            best_c = c

    hv = 0.0
    prev_c = ref[1]
    for t, c in envelope:
        width = max(0.0, ref[0] - t)
        height = max(0.0, prev_c - c)
        hv += width * height
        prev_c = c
    return hv


def collect_generation_metrics(generation: int, population: List[Solution], pareto_front: List[Solution], hv_ref: Tuple[float, float]) -> Dict:
    """Coleta métricas de uma geração."""
    times_pop = [s.time for s in population]
    co2s_pop = [s.co2 for s in population]
    violations_pop = [s.violation for s in population]

    times_pareto = [s.time for s in pareto_front] or [0.0]
    co2s_pareto = [s.co2 for s in pareto_front] or [0.0]

    hv = hypervolume_2d(pareto_front, hv_ref)

    return {
        "generation": generation,
        "pareto_size": len(pareto_front),
        "avg_time_pop": float(np.mean(times_pop)) if times_pop else 0.0,
        "avg_co2_pop": float(np.mean(co2s_pop)) if co2s_pop else 0.0,
        "min_time_pareto": float(min(times_pareto)) if times_pareto else 0.0,
        "min_co2_pareto": float(min(co2s_pareto)) if co2s_pareto else 0.0,
        "avg_violation_pop": float(np.mean(violations_pop)) if violations_pop else 0.0,
        "hypervolume": hv,
    }


# ==============================
# Função Principal de Otimização
# ==============================

def optimize_moead(
    graph: nx.DiGraph,
    source: str,
    target: str,
    population_size: int = 100,
    n_neighbors: int = 20,
    max_generations: int = 50,
    max_transfers: int = 4,
    max_walk_time: float = 90.0,
    transfer_penalty: float = 500.0,
    walk_time_penalty: float = 100.0,
) -> Tuple[List[Solution], Dict[str, Solution], List[Dict]]:
    """Executa MOEA/D e retorna (pareto_front, extremes, history)."""
    
    weights = generate_weights(population_size)
    neighborhoods = generate_neighborhoods(weights, n_neighbors)
    
    population = heuristic_initialization(
        graph, source, target, weights, population_size,
        max_transfers, max_walk_time, transfer_penalty, walk_time_penalty
    )

    print(f"\n{'='*70}")
    print(f"MOEA/D Inicialização")
    print(f"{'='*70}")
    print(f"População inicial: {len(population)} soluções (heurística)")
    
    walk_seed = create_walk_only_solution(graph, source, target, max_walk_time, walk_time_penalty)
    if walk_seed:
        population.append(walk_seed)
        print(f"Walk-only seed adicionado. População agora: {len(population)} soluções")

    if population:
        times = [s.time for s in population]
        co2s = [s.co2 for s in population]
        print(f"  Tempo: min={min(times):.1f}, avg={np.mean(times):.1f}, max={max(times):.1f}")
        print(f"  CO₂:   min={min(co2s):.1f}, avg={np.mean(co2s):.1f}, max={max(co2s):.1f}")

    pareto_front = []
    for s in population:
        pareto_front = update_pareto(pareto_front, s)

    print(f"Pareto inicial: {len(pareto_front)} soluções")

    times_all = [s.time for s in population]
    co2s_all = [s.co2 for s in population]
    hv_reference = (
        (max(times_all) if times_all else 1.0) * 1.1,
        (max(co2s_all) if co2s_all else 1.0) * 1.1
    )

    print(f"\n{'='*70}")
    print(f"Evolução por Geração")
    print(f"{'='*70}\n")

    history = []

    for gen in range(max_generations):
        ref = update_reference(population)
        pareto_size_before = len(pareto_front)

        for i in range(population_size):
            neighbors = neighborhoods[i]
            p1 = population[i]
            p2 = population[random.choice(neighbors)]

            child_path = (
                crossover(p1.path, p2.path)
                if random.random() < 0.8 else p1.path
            )
            child_path = mutate(child_path, graph)

            t, c, viol, edges = evaluate_path(child_path, graph, max_transfers, max_walk_time, transfer_penalty, walk_time_penalty)
            child = Solution(child_path, t, c, viol, edges)

            for j in neighbors:
                child_scalar = tchebycheff(np.array([child.time, child.co2]), weights[j], ref) + child.violation
                curr_scalar = tchebycheff(np.array([population[j].time, population[j].co2]), weights[j], ref) + population[j].violation
                if child_scalar < curr_scalar:
                    population[j] = child

            pareto_front = update_pareto(pareto_front, child)

        metrics = collect_generation_metrics(gen + 1, population, pareto_front, hv_reference)
        history.append(metrics)

        pareto_size_after = len(pareto_front)
        new_solutions = pareto_size_after - pareto_size_before
        times_pareto = [s.time for s in pareto_front] or [0.0]
        co2s_pareto = [s.co2 for s in pareto_front] or [0.0]
        print(f"Gen {gen + 1:2d} | Pareto: {pareto_size_after:4d} (+{new_solutions:3d}) | "
              f"Time[{min(times_pareto):.1f}, {max(times_pareto):.1f}] | "
              f"CO₂[{min(co2s_pareto):.1f}, {max(co2s_pareto):.1f}] | "
              f"HV: {metrics['hypervolume']:.6f}")

    print(f"\n{'='*70}")
    print(f"Otimização Concluída")
    print(f"{'='*70}")
    print(f"Tamanho final da Frente de Pareto: {len(pareto_front)}")
    if pareto_front:
        times = [s.time for s in pareto_front]
        co2s = [s.co2 for s in pareto_front]
        print(f"Tempo final: min={min(times):.1f}, max={max(times):.1f}")
        print(f"CO₂ final:   min={min(co2s):.1f}, max={max(co2s):.1f}")
    print(f"{'='*70}\n")

    extremes = get_extreme_solutions(pareto_front)
    return pareto_front, extremes, history


def get_extreme_solutions(pareto_front: List[Solution]) -> Dict[str, Solution]:
    """Retorna soluções extremas do Pareto."""
    if not pareto_front:
        return {}

    best_time = min(pareto_front, key=lambda s: s.time)
    best_co2 = min(pareto_front, key=lambda s: s.co2)

    times = [s.time for s in pareto_front]
    co2s = [s.co2 for s in pareto_front]

    t_min, t_max = min(times), max(times)
    c_min, c_max = min(co2s), max(co2s)

    balanced = min(
        pareto_front,
        key=lambda s: (
            (s.time - t_min) / (t_max - t_min + 1e-9) +
            (s.co2 - c_min) / (c_max - c_min + 1e-9)
        ),
    )

    return {
        "best_time": best_time,
        "best_co2": best_co2,
        "balanced": balanced,
    }


def analyze_pareto_front(pareto: List[Solution], graph: nx.DiGraph) -> None:
    """Análise básica do Pareto."""
    times = [s.time for s in pareto]
    co2s = [s.co2 for s in pareto]

    mode_usage = defaultdict(int)
    for s in pareto:
        for u, v in zip(s.path[:-1], s.path[1:]):
            edge = get_edge(graph, u, v)
            mode_usage[edge["modo"]] += 1

    print("Pareto solutions:", len(pareto))
    print("Time min / avg / max:", min(times), np.mean(times), max(times))
    print("CO2 min / avg / max:", min(co2s), np.mean(co2s), max(co2s))
    print("Mode usage:", dict(mode_usage))
