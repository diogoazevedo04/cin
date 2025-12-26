import csv
import pickle
import logging
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path
from typing import Dict, List, Any, Tuple
import networkx as nx

from moead import MOEAD, analyze_pareto_front

# Configuração de Logging simples
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

class TransportNetwork:
    """
    Responsável por gerir a topologia da rede de transportes, 
    incluindo cálculos geográficos e manipulação de nós virtuais.
    """
    
    def __init__(self, graph_path: str):
        self.graph_path = Path(graph_path)
        self.G: nx.DiGraph = self._load_graph()

    def _load_graph(self) -> nx.DiGraph:
        if not self.graph_path.exists():
            raise FileNotFoundError(f"Grafo não encontrado em: {self.graph_path}")
        
        logger.info("A carregar grafo base...")
        with open(self.graph_path, "rb") as f:
            G = pickle.load(f)
        return G

    @staticmethod
    def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calcula distância do grande círculo em km."""
        R = 6371
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c

    @staticmethod
    def _walking_time(dist_km: float, speed_kmh: float = 4.5) -> float:
        """Calcula tempo de caminhada em minutos + penalidade fixa."""
        return (dist_km / speed_kmh) * 60 + 3.0

    def add_virtual_node(self, node_id: str, lat: float, lon: float, k_neighbors: int = 15):
        """Adiciona um ponto de interesse (origem/destino) conectado aos K vizinhos mais próximos."""
        self.G.add_node(node_id, lat=lat, lon=lon, modo="walk")
        
        distances = []
        for n, data in self.G.nodes(data=True):
            if n == node_id: continue
            d = self.haversine(lat, lon, data["lat"], data["lon"])
            distances.append((d, n))
        
        distances.sort()
        
        # Ligar aos k mais próximos
        for dist_km, n in distances[:k_neighbors]:
            t = self._walking_time(dist_km) + 5.0 # Penalização transbordo
            attrs = {"modo": "walk", "distance_km": dist_km, "time_min": t, "co2": 0}
            self.G.add_edge(node_id, n, **attrs)
            self.G.add_edge(n, node_id, **attrs)

    def add_direct_shortcut(self, node_a: str, node_b: str):
        """Cria uma ligação direta a pé entre dois nós (fallback)."""
        if not self.G.has_node(node_a) or not self.G.has_node(node_b):
            return

        lat1, lon1 = self.G.nodes[node_a]["lat"], self.G.nodes[node_a]["lon"]
        lat2, lon2 = self.G.nodes[node_b]["lat"], self.G.nodes[node_b]["lon"]
        
        dist = self.haversine(lat1, lon1, lat2, lon2) * 1.3 # Fator tortuosidade
        time_walk = self._walking_time(dist)
        
        attrs = {"modo": "walk", "time_min": time_walk, "distance_km": dist, "co2": 0.0}
        self.G.add_edge(node_a, node_b, **attrs)
        self.G.add_edge(node_b, node_a, **attrs)
        logger.info(f"Ligação direta criada: {dist:.2f} km")

    def get_stats(self) -> str:
        return f"{self.G.number_of_nodes()} nós | {self.G.number_of_edges()} arestas"


class SolutionAnalyzer:
    """
    Responsável por analisar soluções, gerar relatórios e formatar saídas.
    """
    def __init__(self, graph: nx.DiGraph):
        self.G = graph

    def _get_walk_distance(self, path: List[str]) -> float:
        return sum(
            self.G[u][v]['distance_km'] 
            for u, v in zip(path[:-1], path[1:]) 
            if self.G[u][v]['modo'] == 'walk'
        )

    def print_solution_details(self, sol, name: str = "Solução"):
        print(f"\n{'='*60}\n{name.upper()}\n{'='*60}")
        print(f"Tempo total: {sol.time:.1f} min")
        print(f"CO₂ total: {sol.co2:.0f} g")
        print(f"Paragens: {len(sol.path)}")
        
        walk_dist = self._get_walk_distance(sol.path)
        print(f"Distância a pé: {walk_dist:.2f} km")
        
        # Segmentação simplificada para exibição
        print(f"\nDetalhes da viagem:")
        current_mode = None
        group_start = sol.path[0]
        
        for i, (u, v) in enumerate(zip(sol.path[:-1], sol.path[1:])):
            mode = self.G[u][v]['modo']
            if mode != current_mode:
                if current_mode:
                    print(f"   via {current_mode.upper()} até {u}")
                current_mode = mode
                print(f"{mode.upper()} de {u}...", end="")
        print(f" até {sol.path[-1]}")

    def print_comparison(self, solutions_dict: Dict[str, Any]):
        print(f"\n{'='*60}\nCOMPARAÇÃO DE SOLUÇÕES\n{'='*60}\n")
        print(f"{'Critério':<15} {'Tempo':<15} {'CO₂':<15} {'A pé (km)':<15}")
        print("-" * 60)
        
        for name, sol in solutions_dict.items():
            walk_dist = self._get_walk_distance(sol.path)
            print(f"{name:<15} {sol.time:<15.1f} {sol.co2:<15.0f} {walk_dist:<15.2f}")


class DataManager:
    """
    Responsável pela persistência dos dados (Salvar/Carregar).
    """
    @staticmethod
    def save_pickle(data: Any, filepath: str):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Dados salvos em: {filepath}")

    @staticmethod
    def save_pareto_csv(pareto_front: List[Any], graph: nx.DiGraph, filepath: str):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time_min', 'co2_g', 'path_length', 'walk_distance_km'])
            
            for sol in pareto_front:
                walk_dist = sum(
                    graph[u][v]['distance_km'] 
                    for u, v in zip(sol.path[:-1], sol.path[1:])
                    if graph[u][v]['modo'] == 'walk'
                )
                writer.writerow([sol.time, sol.co2, len(sol.path), walk_dist])
        logger.info(f"CSV salvo em: {filepath}")


class RouteOptimizationApp:
    """
    Classe principal (Facade) que orquestra todo o processo.
    """
    
    def __init__(self):
        # Configurações
        self.origin = (41.3616, -8.7541)  # Trindade
        self.destination = (41.1608, -8.6843)  # FEUP
        self.pop_size = 100
        self.generations = 50
        
        # Inicialização de componentes
        self.network = TransportNetwork("data/output/graph_base.gpickle")
        self.analyzer = SolutionAnalyzer(self.network.G)
        self.data_manager = DataManager()

    def setup_route_endpoints(self):
        """Configura os nós de origem e destino na rede."""
        logger.info("\n Configurando origem e destino...")
        self.network.add_virtual_node("origin", *self.origin)
        self.network.add_virtual_node("destination", *self.destination)
        self.network.add_direct_shortcut("origin", "destination")
        logger.info(f"Estado da Rede: {self.network.get_stats()}")

    def run_optimization(self):
        """Executa o algoritmo MOEA/D."""
        self.moead = MOEAD(
            G=self.network.G,
            source="origin",
            target="destination",
            population_size=self.pop_size,
            n_neighbors=20,
            max_generations=self.generations
        )
        self.pareto_front = self.moead.optimize()
        self.extremes = self.moead.get_extreme_solutions()

    def generate_reports(self):
        """Gera e exibe análises na consola."""
        analyze_pareto_front(self.pareto_front, self.network.G)
        
        if self.extremes:
            labels = {
                'best_time': "Melhor Tempo",
                'best_co2': "Melhor CO₂",
                'balanced': "Balanceada"
            }
            
            # Análise detalhada
            for key, label in labels.items():
                if key in self.extremes:
                    self.analyzer.print_solution_details(self.extremes[key], label)
            
            # Tabela comparativa
            solutions_to_compare = {labels[k]: v for k, v in self.extremes.items()}
            self.analyzer.print_comparison(solutions_to_compare)

    def save_results(self):
        """Exporta os resultados finais."""
        results = {
            'pareto_front': self.pareto_front,
            'extremes': self.extremes,
            'history': self.moead.history,
            'config': {
                'origin': self.origin,
                'destination': self.destination
            }
        }
        
        print(f"\n{'='*60}\nEXPORTANDO RESULTADOS\n{'='*60}")
        self.data_manager.save_pickle(results, "data/output/moead_results.pkl")
        self.data_manager.save_pareto_csv(self.pareto_front, self.network.G, "data/pareto_front.csv")

    def run(self):
        """Pipeline de execução."""
        try:
            self.setup_route_endpoints()
            self.run_optimization()
            self.generate_reports()
            self.save_results()
            logger.info("\n Processo completo com sucesso!")
        except Exception as e:
            logger.error(f"Erro crítico na execução: {e}", exc_info=True)


if __name__ == "__main__":
    app = RouteOptimizationApp()
    app.run()