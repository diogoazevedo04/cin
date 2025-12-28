import random
from collections import defaultdict

import networkx as nx
import numpy as np

MAX_TRANSFERS=4
MAX_WALK_TIME=90.0
TRANSFER_PENALTY=500.0
WALK_TIME_PENALTY=100.0

POPULATION_SIZE=100
N_NEIGHBORS=20
MAX_GENERATIONS=50
MUTATION_RATE=0.8
CROSSOVER_RATE=0.8

def get_edge(G, u, v, score = None):
    """Seleciona a aresta entre u e v no MultiDiGraph.
    Se existir mais do que uma aresta entre o par (u, v),
    escolhe a aresta que minimiza o "score" fornecido. 
    Na ausência de score,
    escolhe a aresta com menor tempo ("time_min").
    """
    def pick_best_edge(edge_dict):
        if not edge_dict:
            return {}
        if score:
            scored = [(k, score(edge_dict[k])) for k in edge_dict.keys()]
            if not scored:
                return {}
            min_score = min(s for _, s in scored)
            tol = 1e-9
            candidates = [k for k, s in scored if s <= min_score + tol]
            chosen_key = random.choice(candidates)
            return edge_dict[chosen_key]
        else:
            best_key = min(
                edge_dict.keys(),
                key=lambda k: edge_dict[k].get('time_min', float('inf'))
            )
            return edge_dict[best_key]

    if G.has_edge(u, v):
        return pick_best_edge(G[u][v])
    elif G.has_edge(v, u):
        return pick_best_edge(G[v][u])
    else:
        return {}


def create_solution(path, time, co2, violation= 0.0, edges = None, weights = None):
    """Cria uma solução como dicionário."""
    if edges is None:
        edges = []
    return {
        "path": path,
        "time": time,
        "co2": co2,
        "violation": violation,
        "edges": edges,
        "weights": weights,
    }


# --- Funções de Inicialização ---

def generate_weights(population_size):
    """Gera vetores de peso lineares para a decomposição."""
    return np.array(
        [[i / (population_size - 1), 1 - i / (population_size - 1)]
         for i in range(population_size)]
    )


def generate_neighborhoods(weights, n_neighbors):
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


# --- Funções de Avaliação ---

def evaluate_path(
    path,
    graph,
    max_transfers=MAX_TRANSFERS,
    max_walk_time=MAX_WALK_TIME,
    transfer_penalty=TRANSFER_PENALTY,
    walk_time_penalty=WALK_TIME_PENALTY,
    edge_score=None
):
    """Calcula tempo, CO2 e violação de restrições."""
    time = 0.0
    co2 = 0.0
    walk_time = 0.0
    num_transfers = 0
    prev_mode = None
    edges_used = []

    for u, v in zip(path[:-1], path[1:]):
        edge = get_edge(graph, u, v, score=edge_score)
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
    graph,
    source,
    target,
    weights,
    population_size,
    max_transfers = MAX_TRANSFERS,
    max_walk_time = MAX_WALK_TIME,
    transfer_penalty = TRANSFER_PENALTY,
    walk_time_penalty = WALK_TIME_PENALTY
):
    """Inicialização heurística com Dijkstra."""
    solutions = []

    times = [d["time_min"] for _, _, d in graph.edges(data=True)]
    co2s = [d["co2"] for _, _, d in graph.edges(data=True)]

    max_time = max(times) if times else 1.0
    max_co2 = max(co2s) if co2s else 1.0

    for w_time, w_co2 in weights:
        try:
            def cost(u, v, d):
                return (
                    w_time * d.get("time_min", 0.0) / max_time +
                    w_co2 * d.get("co2", 0.0) / max_co2
                )

            path = nx.shortest_path(graph, source, target, weight=cost)
            edge_score = lambda d: (w_time * d.get("time_min", 0.0) / max_time +
                                    w_co2 * d.get("co2", 0.0) / max_co2)
            t, c, viol, edges = evaluate_path(
                path, graph,
                max_transfers, max_walk_time, transfer_penalty, walk_time_penalty,
                edge_score=edge_score,
            )
            solutions.append(create_solution(path, t, c, viol, edges, weights=(w_time, w_co2)))
        except nx.NetworkXNoPath:
            continue

    while len(solutions) < population_size and solutions:  # 100 soluções
        base = random.choice(solutions)
        b_w = base.get("weights") or (0.5, 0.5)
        edge_score_b = lambda d: (b_w[0] * d.get("time_min", 0.0) / max_time +
                                  b_w[1] * d.get("co2", 0.0) / max_co2)
        mutated = mutate(base["path"], graph, edge_score=edge_score_b)
        t, c, viol, edges = evaluate_path(
            mutated, graph,
            max_transfers, max_walk_time, transfer_penalty, walk_time_penalty,
            edge_score=edge_score_b,
        )
        solutions.append(create_solution(mutated, t, c, viol, edges, weights=b_w))

    return solutions


def create_walk_only_solution(
    graph,
    source,
    target,
    max_walk_time = MAX_WALK_TIME,
    walk_time_penalty = WALK_TIME_PENALTY
):
    """Cria solução 100% walk se possível."""
    H = nx.DiGraph()
    for n, data in graph.nodes(data=True):
        H.add_node(n, **data)

    # Copia apenas arestas walk (só existe uma aresta walk por par de vértices)
    for u, v, key, data in graph.edges(keys=True, data=True):
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

    return create_solution(path, time, co2, violation, edges_used)


# --- Operadores Genéticos ---

def mutate(path, graph, rate = MUTATION_RATE, edge_score = None):
    """Mutação por recombinação de segmentos.

    edge_score: se fornecido, usa este score como custo na busca do subcaminho
    entre dois nós (em vez de escolher aleatoriamente entre 'time_min' e 'co2').
    """
    if len(path) < 3 or random.random() > rate:
        return path

    i = random.randint(0, len(path) - 2)
    j = random.randint(i + 1, len(path) - 1)

    try:
        if edge_score is None:
            weight = random.choice(["time_min", "co2"])
            sub = nx.shortest_path(graph, path[i], path[j], weight=weight)
        else:
            # Introduz ligeira aleatoriedade no custo para fomentar diversidade
            def cost(u, v, d):
                base = edge_score(d)
                jitter = 1.0 + 0.05 * (random.random() * 2 - 1)  # +/-5%
                return base * jitter
            sub = nx.shortest_path(graph, path[i], path[j], weight=cost)
        new_path = path[:i] + sub + path[j + 1:]
        cleaned = [new_path[0]]
        for n in new_path[1:]:
            if n != cleaned[-1]:
                cleaned.append(n)
        return cleaned
    except nx.NetworkXNoPath:
        return path


def crossover(p1, p2, rate = CROSSOVER_RATE):
    """Cruzamento por nó comum com probabilidade CROSSOVER_RATE."""
    if random.random() > rate:
        return random.choice([p1, p2])

    common = list(set(p1) & set(p2))
    if len(common) < 3:
        return random.choice([p1, p2])

    n = random.choice(common)
    try:
        return p1[: p1.index(n)] + p2[p2.index(n):]
    except ValueError:
        return random.choice([p1, p2])

# --- Funções de Otimização ---

def tchebycheff(obj, w, ref):
    return np.max(w * np.abs(obj - ref))


def update_reference(population):
    """Atualiza ponto de referência ideal."""
    times = [s["time"] for s in population]
    co2s = [s["co2"] for s in population]
    return np.array([min(times), min(co2s)])


def update_pareto(pareto_front, sol):
    """Atualiza frente de Pareto com penalização por violação."""
    def penalized(obj, viol):
        return np.array([obj[0] + viol, obj[1] + viol])

    penal_sol = penalized(np.array([sol["time"], sol["co2"]]), sol["violation"])
    filtered = []
    for s in pareto_front:
        penal_s = penalized(np.array([s["time"], s["co2"]]), s["violation"])
        if all(penal_sol <= penal_s) and any(penal_sol < penal_s):
            continue
        if all(penal_s <= penal_sol) and any(penal_s < penal_sol):
            return pareto_front
        filtered.append(s)
    filtered.append(sol)
    return prune_pareto_epsilon(filtered)


def prune_pareto_epsilon(pareto_front, epsilon_time = 0.3, epsilon_co2 = 1.0):
    """Remove soluções redundantes (epsilon-dominance).
    
    Ordena por compromisso normalizado para evitar viés
    e manter melhor diversidade na frente de Pareto.
    """
    if len(pareto_front) <= 1:
        return pareto_front
    
    times = [s["time"] for s in pareto_front]
    co2s = [s["co2"] for s in pareto_front]
    min_time, max_time = min(times), max(times)
    min_co2, max_co2 = min(co2s), max(co2s)
    
    def compromise_score(sol):
        norm_time = (sol["time"] - min_time) / (max_time - min_time + 1e-9)
        norm_co2 = (sol["co2"] - min_co2) / (max_co2 - min_co2 + 1e-9)
        return norm_time + norm_co2
    
    sorted_front = sorted(pareto_front, key=compromise_score)
    pruned = [sorted_front[0]]
    
    for sol in sorted_front[1:]:
        keep = True
        for kept in pruned:
            if abs(sol["time"] - kept["time"]) <= epsilon_time and abs(sol["co2"] - kept["co2"]) <= epsilon_co2:
                keep = False
                break
        if keep:
            pruned.append(sol)
    return pruned


def hypervolume(front, ref):
    """Hipervolume para minimização."""
    if not front:
        return 0.0
    
    pts = sorted([(s["time"], s["co2"]) for s in front], key=lambda x: x[0])
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


def collect_generation_metrics(generation, pareto_front, hv_ref):
    """Coleta métricas de uma geração."""
    times_pareto = [s["time"] for s in pareto_front] or [0.0]
    co2s_pareto = [s["co2"] for s in pareto_front] or [0.0]
    
    hv = hypervolume(pareto_front, hv_ref)

    return {
        "generation": generation,
        "pareto_size": len(pareto_front),
        "min_time_pareto": float(min(times_pareto)),
        "max_time_pareto": float(max(times_pareto)),
        "min_co2_pareto": float(min(co2s_pareto)),
        "max_co2_pareto": float(max(co2s_pareto)),
        "hypervolume": hv,
    }


# --- Função Principal de Otimização ---

def moead(
    graph,
    source,
    target,
    population_size = POPULATION_SIZE,
    n_neighbors = N_NEIGHBORS,
    max_generations = MAX_GENERATIONS,
    crossover_rate = CROSSOVER_RATE,
    max_transfers = MAX_TRANSFERS,
    max_walk_time = MAX_WALK_TIME,
    transfer_penalty = TRANSFER_PENALTY,
    walk_time_penalty = WALK_TIME_PENALTY,
    verbose = True,
):
    """Executa MOEA/D e retorna (pareto_front, extremes, history)."""
    
    weights = generate_weights(population_size)
    neighborhoods = generate_neighborhoods(weights, n_neighbors)
    
    population = heuristic_initialization(
        graph, source, target, weights, population_size,
        max_transfers, max_walk_time, transfer_penalty, walk_time_penalty
    )

    if verbose:
        print(f"\n{'='*70}")
        print(f"MOEA/D Inicialização")
        print(f"{'='*70}")
        print(f"População inicial: {len(population)} soluções (heurística)")
    
    # Substitui population[0] (weights[0]=[0.0,1.0] minimiza CO₂) por walk-only se disponível
    walk_seed = create_walk_only_solution(graph, source, target, max_walk_time, walk_time_penalty)
    if walk_seed:
        population[0] = walk_seed
        if verbose:
            print(f"Walk-only seed substituiu population[0]. População: {len(population)} soluções")
    else:
        if verbose:
            print(f"Walk-only não disponível. População mantém heurística: {len(population)} soluções")

    if population and verbose:
        times_pop = [s["time"] for s in population]
        co2s_pop = [s["co2"] for s in population]
        print(f"  População - Tempo: min={min(times_pop):.1f}, avg={np.mean(times_pop):.1f}, max={max(times_pop):.1f}")
        print(f"  População - CO₂:   min={min(co2s_pop):.1f}, avg={np.mean(co2s_pop):.1f}, max={max(co2s_pop):.1f}")

    pareto_front = []
    for s in population:
        pareto_front = update_pareto(pareto_front, s)

    if verbose:
        if pareto_front:
            times_pareto = [s["time"] for s in pareto_front]
            co2s_pareto = [s["co2"] for s in pareto_front]
            print(f"Pareto inicial: {len(pareto_front)} soluções")
            print(f"  Pareto - Tempo: min={min(times_pareto):.1f}, max={max(times_pareto):.1f}")
            print(f"  Pareto - CO₂:   min={min(co2s_pareto):.1f}, max={max(co2s_pareto):.1f}")
        else:
            print(f"Pareto inicial: 0 soluções")

    times_all = [s["time"] for s in population]
    co2s_all = [s["co2"] for s in population]
    hv_reference = (
        (max(times_all) if times_all else 1.0) * 1.1,
        (max(co2s_all) if co2s_all else 1.0) * 1.1
    )

    edge_times = [d.get("time_min", 0.0) for _, _, d in graph.edges(data=True)]
    edge_co2s = [d.get("co2", 0.0) for _, _, d in graph.edges(data=True)]
    max_edge_time = max(edge_times) if edge_times else 1.0
    max_edge_co2 = max(edge_co2s) if edge_co2s else 1.0

    if verbose:
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

            child_path = crossover(p1["path"], p2["path"], rate=crossover_rate)
            
            child_path = mutate(child_path, graph, edge_score=None)

            w_time, w_co2 = weights[i]
            edge_score = lambda d: (w_time * d.get("time_min", 0.0) / max_edge_time +
                                    w_co2 * d.get("co2", 0.0) / max_edge_co2)
            t, c, viol, edges = evaluate_path(
                child_path, graph,
                max_transfers, max_walk_time, transfer_penalty, walk_time_penalty,
                edge_score=edge_score,
            )
            child = create_solution(child_path, t, c, viol, edges)

            for j in neighbors:
                child_scalar = tchebycheff(np.array([child["time"], child["co2"]]), weights[j], ref) + child["violation"]
                curr_scalar = tchebycheff(np.array([population[j]["time"], population[j]["co2"]]), weights[j], ref) + population[j]["violation"]
                if child_scalar < curr_scalar:
                    population[j] = child

            pareto_front = update_pareto(pareto_front, child)

        metrics = collect_generation_metrics(gen + 1, pareto_front, hv_reference)
        history.append(metrics)

        if verbose:
            pareto_size_after = len(pareto_front)
            new_solutions = pareto_size_after - pareto_size_before
            print(f"Gen {gen + 1:2d} | Pareto: {pareto_size_after:4d} (+{new_solutions:3d}) | "
                  f"Time[{metrics['min_time_pareto']:.1f}, {metrics['max_time_pareto']:.1f}] | "
                  f"CO₂[{metrics['min_co2_pareto']:.1f}, {metrics['max_co2_pareto']:.1f}] | "
                  f"HV: {metrics['hypervolume']:.6f}")

    if verbose:
        print(f"\n{'='*70}")
        print(f"Otimização Concluída")
        print(f"{'='*70}")
        print(f"Tamanho final da Frente de Pareto: {len(pareto_front)}")
        if pareto_front:
            times = [s["time"] for s in pareto_front]
            co2s = [s["co2"] for s in pareto_front]
            print(f"Tempo final: min={min(times):.1f}, max={max(times):.1f}")
            print(f"CO₂ final:   min={min(co2s):.1f}, max={max(co2s):.1f}")
            
            mode_usage = defaultdict(int)
            for s in pareto_front:
                for edge in s.get("edges", []):
                    mode_usage[edge.get("modo", "unknown")] += 1
            print(f"Uso de modos: {dict(mode_usage)}")
        print(f"{'='*70}\n")

    extremes = get_extreme_solutions(pareto_front)
    return pareto_front, extremes, history


def get_extreme_solutions(pareto_front):
    """Retorna soluções extremas do Pareto(melhor tempo, melhor CO2 e balanceado)."""
    if not pareto_front:
        return {}

    best_time = min(pareto_front, key=lambda s: s["time"])
    best_co2 = min(pareto_front, key=lambda s: s["co2"])

    times = [s["time"] for s in pareto_front]
    co2s = [s["co2"] for s in pareto_front]

    t_min, t_max = min(times), max(times)
    c_min, c_max = min(co2s), max(co2s)

    balanced = min(
        pareto_front,
        key=lambda s: (
            (s["time"] - t_min) / (t_max - t_min + 1e-9) +
            (s["co2"] - c_min) / (c_max - c_min + 1e-9)
        ),
    )

    return {
        "best_time": best_time,
        "best_co2": best_co2,
        "balanced": balanced,
    }