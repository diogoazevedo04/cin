"""
Avaliação do MOEA/D em diferentes cenários.
Compara soluções e coleta métricas de desempenho.
"""

import pickle
import time
import csv
from pathlib import Path
from typing import List, Dict, Any
import networkx as nx

from moead import moead, get_edge
from scenarios import load_scenarios, print_scenarios_summary, generate_scenarios, save_scenarios


def evaluate_scenario(graph, scenario, timeout=300):
    """
    Avalia um cenário único.
    Retorna métricas de desempenho e qualidade da solução.
    """
    origin_node = scenario['origin_node']
    destination_node = scenario['destination_node']
    difficulty = scenario['difficulty']
    air_distance = scenario['air_distance_km']
    
    # Executa MOEA/D com os nós já existentes no grafo
    start_time = time.time()
    try:
        pareto_front, extremes, history = moead(
            graph=graph,
            source=origin_node,
            target=destination_node,
            verbose=False
        )
        elapsed = time.time() - start_time
    except Exception as e:
        print(f"Erro na otimização: {e}")
        return None
    
    if not pareto_front:
        print(f"Sem soluções encontradas")
        return None
    
    # Coleta métricas
    times = [s["time"] for s in pareto_front]
    co2s = [s["co2"] for s in pareto_front]
    
    result = {
        'difficulty': difficulty,
        'air_distance_km': air_distance,
        'pareto_size': len(pareto_front),
        'time_min': min(times),
        'time_max': max(times),
        'time_avg': sum(times) / len(times),
        'co2_min': min(co2s),
        'co2_max': max(co2s),
        'co2_avg': sum(co2s) / len(co2s),
        'elapsed_seconds': elapsed,
        'generations': len(history) if history else 0,
    }
    
    return result


def run_full_evaluation(num_scenarios=None):
    """
    Executa avaliação completa em todos os cenários.
    Se cenários não existem, gera-os automaticamente.
    """
    # Carrega grafo
    graph_path = "output/graph_base.gpickle"
    if not Path(graph_path).exists():
        print(f"Grafo não encontrado em {graph_path}")
        return
    
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)
    
    # Carrega cenários ou gera-os se não existem
    scenarios_path = "output/scenarios.pkl"
    if not Path(scenarios_path).exists():
        print(f"\nCenários não encontrados. Gerando...")
        scenarios = generate_scenarios(
            graph,
            num_scenarios_per_difficulty=3,
            difficulties={
                'fácil': (0.5, 4.0),
                'médio': (5.0, 9.0),
                'difícil': (10.0, 17.0),
            }
        )
        save_scenarios(scenarios, scenarios_path)  # Guarda em output/scenarios.pkl
    else:
        scenarios = load_scenarios(scenarios_path)  # Carrega de output/scenarios.pkl
    
    if num_scenarios:
        scenarios = scenarios[:num_scenarios]
    
    print(f"\n{'='*70}")
    print(f"AVALIAÇÃO DO MOEA/D")
    print(f"{'='*70}")
    print(f"Grafo: {graph.number_of_nodes()} nós, {graph.number_of_edges()} arestas")
    print(f"Cenários: {len(scenarios)}\n")
    
    print_scenarios_summary(scenarios)
    
    results = []
    for idx, scenario in enumerate(scenarios, 1):
        difficulty = scenario['difficulty']
        air_dist = scenario['air_distance_km']
        
        print(f"\n[{idx:2d}/{len(scenarios):2d}] Cenário '{difficulty}' (dist aérea: {air_dist:.2f} km)...")
        
        result = evaluate_scenario(graph, scenario)
        if result:
            results.append(result)
            print(f"Frente de Pareto: {result['pareto_size']} soluções | "
                  f"Tempo: {result['elapsed_seconds']:.1f}s")
    
    save_evaluation_results(results)
    print_evaluation_summary(results)


def save_evaluation_results(results):
    """Guarda resultados em CSV e pickle."""
    path = Path("output/evaluation_results.pkl")
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "wb") as f:
        pickle.dump(results, f)
    
    csv_path = Path("output/evaluation_results.csv")
    with open(csv_path, "w", newline='') as f:
        if not results:
            return
        
        fieldnames = list(results[0].keys())
        fieldnames = [k for k in fieldnames if k not in ['best_time', 'best_co2', 'balanced']]
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {k: v for k, v in result.items() if k in fieldnames}
            writer.writerow(row)
    
    print(f"\nResultados guardados em output/evaluation_results.pkl e .csv")


def print_evaluation_summary(results):
    """Imprime resumo dos resultados da avaliação."""
    if not results:
        print("Sem resultados para mostrar.")
        return
    
    print(f"\n{'='*70}")
    print(f"RESUMO DA AVALIAÇÃO")
    print(f"{'='*70}\n")
    
    by_difficulty = {}
    for r in results:
        diff = r['difficulty']
        if diff not in by_difficulty:
            by_difficulty[diff] = []
        by_difficulty[diff].append(r)
    
    for difficulty in ['fácil', 'médio', 'difícil', 'muito_difícil']:
        if difficulty not in by_difficulty:
            continue
        
        res_list = by_difficulty[difficulty]
        n = len(res_list)
        
        pareto_sizes = [r['pareto_size'] for r in res_list]
        times_elapsed = [r['elapsed_seconds'] for r in res_list]
        time_mins = [r['time_min'] for r in res_list]
        time_maxs = [r['time_max'] for r in res_list]
        co2_mins = [r['co2_min'] for r in res_list]
        co2_maxs = [r['co2_max'] for r in res_list]
        
        print(f"{difficulty.upper():15} ({n} cenários)")
        print(f"  Pareto:        avg={sum(pareto_sizes)/n:.1f} | min={min(pareto_sizes)} | max={max(pareto_sizes)}")
        print(f"  Tempo exec:    avg={sum(times_elapsed)/n:.1f}s | min={min(times_elapsed):.1f}s | max={max(times_elapsed):.1f}s")
        print(f"  Tempo rotas:   min={min(time_mins):.1f}-{max(time_maxs):.1f} min")
        print(f"  CO₂ rotas:     min={min(co2_mins):.0f}-{max(co2_maxs):.0f} g")
        print()


if __name__ == "__main__":
    run_full_evaluation()
