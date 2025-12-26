import logging
import math
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import networkx as nx
import pandas as pd
import requests

# ==============================
# Configuração de Logging
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ==============================
# Constantes e Configurações
# ==============================
class Config:
    """Configurações globais para a construção do grafo."""
    WALK_SPEED_KMH: float = 4.5
    WALK_RADIUS_KM: float = 0.3
    WALK_PENALTY_MIN: float = 3.0
    TRANSFER_PENALTY_MIN: float = 5.0
    
    # Emissões (g CO2 / km)
    CO2_METRO: float = 40.0
    CO2_BUS: float = 109.9
    
    # Timeout para API OSRM
    OSRM_TIMEOUT: int = 10
    OSRM_BASE_URL: str = "http://router.project-osrm.org/route/v1/foot"

# ==============================
# Classe Principal
# ==============================

class TransportGraphBuilder:
    """
    Classe responsável por construir, processar e guardar o grafo multimodal.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        # Session permite reutilizar a conexão TCP, acelerando múltiplos requests
        self.session = requests.Session()
        self._osrm_cache: Dict[Tuple[float, float, float, float], Optional[Dict[str, float]]] = {}

    @staticmethod
    def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calcula a distância do grande círculo entre dois pontos em km."""
        R = 6371  # Raio da Terra em km
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlon / 2) ** 2)

        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    @staticmethod
    def time_to_minutes(t_str: str) -> float:
        """Converte string de tempo (HH:MM:SS) para minutos totais."""
        try:
            h, m, s = map(int, t_str.split(":"))
            return h * 60 + m + s / 60
        except ValueError:
            raise ValueError(f"Formato de tempo inválido: {t_str}")

    @staticmethod
    def walk_time_minutes(dist_km: float) -> float:
        """Calcula tempo de caminhada baseado na velocidade média e penalização."""
        return (dist_km / Config.WALK_SPEED_KMH) * 60 + Config.WALK_PENALTY_MIN

    def _get_osrm_route(self, lat1: float, lon1: float, lat2: float, lon2: float) -> Optional[Dict[str, float]]:
        """
        Consulta a API OSRM para obter rota a pé.
        Retorna dicionário com distância (km) e tempo (min) ou None em caso de falha.
        """
        url = f"{Config.OSRM_BASE_URL}/{lon1},{lat1};{lon2},{lat2}"
        params = {"overview": "false", "steps": "false"}

        try:
            response = self.session.get(url, params=params, timeout=Config.OSRM_TIMEOUT)
            response.raise_for_status()
            data = response.json()

            if data.get("code") == "Ok" and data.get("routes"):
                route = data["routes"][0]
                return {
                    "distance_km": route["distance"] / 1000,
                    "time_min": route["duration"] / 60
                }
        except requests.RequestException:
            # Falha silenciosa para permitir fallback, conforme lógica original
            pass
        
        return None

    def load_stops(self, filepath: str, modo: str) -> None:
        """Carrega paragens de um ficheiro CSV para o grafo."""
        path = Path(filepath)
        if not path.exists():
            logger.error(f"Ficheiro não encontrado: {path}")
            return

        try:
            stops = pd.read_csv(path)
            for _, row in stops.iterrows():
                node_id = f"{modo}_{row['stop_id']}"
                self.graph.add_node(
                    node_id,
                    lat=float(row["stop_lat"]),
                    lon=float(row["stop_lon"]),
                    modo=modo
                )
            logger.info(f"Carregadas {len(stops)} paragens para o modo: {modo}")
        except Exception as e:
            logger.error(f"Erro ao carregar paragens de {filepath}: {e}")

    def load_gtfs_edges(self, filepath: str, modo: str, co2_per_km: float) -> None:
        """Processa stop_times e cria arestas de transporte público."""
        path = Path(filepath)
        if not path.exists():
            logger.error(f"Ficheiro não encontrado: {path}")
            return

        logger.info(f"A processar rotas para: {modo}...")
        stop_times = pd.read_csv(path)
        stop_times = stop_times.sort_values(["trip_id", "stop_sequence"])

        edge_data: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
        count_edges = 0

        # Agrupar por viagem para processar sequências
        for _, group in stop_times.groupby("trip_id"):
            rows = group.to_dict("records")

            for i in range(len(rows) - 1):
                s1, s2 = rows[i], rows[i + 1]
                u = f"{modo}_{s1['stop_id']}"
                v = f"{modo}_{s2['stop_id']}"

                if not self.graph.has_node(u) or not self.graph.has_node(v):
                    continue

                try:
                    t1 = self.time_to_minutes(s1["departure_time"])
                    t2 = self.time_to_minutes(s2["arrival_time"])
                    delta = t2 - t1
                    if delta <= 0:
                        continue
                except ValueError:
                    continue

                # Obter coordenadas do grafo
                lat1, lon1 = self.graph.nodes[u]["lat"], self.graph.nodes[u]["lon"]
                lat2, lon2 = self.graph.nodes[v]["lat"], self.graph.nodes[v]["lon"]
                
                dist_km = self.haversine(lat1, lon1, lat2, lon2)
                
                if (u, v) not in edge_data:
                    edge_data[(u, v)] = []
                edge_data[(u, v)].append((delta, dist_km))

        # Adicionar arestas agregadas (média)
        for (u, v), values in edge_data.items():
            avg_time = sum(t for t, _ in values) / len(values)
            avg_dist = sum(d for _, d in values) / len(values)

            self.graph.add_edge(
                u, v,
                modo=modo,
                time_min=avg_time,
                distance_km=avg_dist,
                co2=avg_dist * co2_per_km
            )
            count_edges += 1
            
        logger.info(f"Arestas criadas para {modo}: {count_edges}")

    def add_walk_edges(self) -> None:
        """
        Gera conexões pedonais entre paragens de modos diferentes.
        Utiliza OSRM com cache e fallback para Haversine.
        """
        nodes = list(self.graph.nodes(data=True))
        total_nodes = len(nodes)
        
        logger.info("A iniciar geração de arestas pedonais (OSRM + Fallback)...")
        start_time = time.time()
        
        edge_count = 0
        stats = {"osrm_success": 0, "osrm_fail": 0, "cache_hits": 0}

        # Iteração O(N^2) otimizada
        for i, (u, du) in enumerate(nodes):
            for v, dv in nodes[i + 1:]:
                # REGRA: Apenas entre modos diferentes
                if du["modo"] == dv["modo"]:
                    continue

                # Pré-filtro Euclidiano
                dist_euclidian = self.haversine(du["lat"], du["lon"], dv["lat"], dv["lon"])
                if dist_euclidian > Config.WALK_RADIUS_KM * 1.5:
                    continue

                # Chave de cache baseada em coordenadas arredondadas (precisão ~1m)
                cache_key = (
                    round(du["lat"], 5), round(du["lon"], 5),
                    round(dv["lat"], 5), round(dv["lon"], 5)
                )

                if cache_key in self._osrm_cache:
                    result = self._osrm_cache[cache_key]
                    stats["cache_hits"] += 1
                    if result:
                        stats["osrm_success"] += 1 
                    else:
                        stats["osrm_fail"] += 1
                else:
                    result = self._get_osrm_route(du["lat"], du["lon"], dv["lat"], dv["lon"])
                    self._osrm_cache[cache_key] = result
                    if result:
                        stats["osrm_success"] += 1
                    else:
                        stats["osrm_fail"] += 1

                # Determinar valores finais (API ou Fallback)
                if result:
                    dist_km = result["distance_km"]
                    time_min = result["time_min"] + Config.TRANSFER_PENALTY_MIN
                else:
                    # Fallback: Fator de Manhattan (1.3x)
                    dist_km = dist_euclidian * 1.3
                    time_min = self.walk_time_minutes(dist_km) + Config.TRANSFER_PENALTY_MIN

                # Filtro final de distância
                if dist_km <= Config.WALK_RADIUS_KM * 1.5:
                    # Adicionar bidirecionalmente
                    attrs = {
                        "modo": "walk",
                        "time_min": time_min,
                        "distance_km": dist_km,
                        "co2": 0.0
                    }
                    self.graph.add_edge(u, v, **attrs)
                    self.graph.add_edge(v, u, **attrs)
                    edge_count += 2

            # Logging periódico de progresso
            if (i + 1) % 100 == 0:
                elapsed_min = (time.time() - start_time) / 60
                progress = (i + 1) / total_nodes
                remaining_min = (elapsed_min / progress) - elapsed_min
                
                logger.info(
                    f"Progresso: {int(progress*100)}% | Arestas: {edge_count} | "
                    f"OSRM(✓{stats['osrm_success']} ✗{stats['osrm_fail']}) | "
                    f"Restante: ~{remaining_min:.1f} min"
                )

        logger.info(f"Concluído. Total arestas pedonais: {edge_count}")
        logger.info(f"Estatísticas OSRM: Sucessos={stats['osrm_success']}, Falhas={stats['osrm_fail']}, Cache Hits={stats['cache_hits']}")

    def save_graph(self, output_path: str) -> None:
        """Serializa o grafo para disco usando pickle."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(path, "wb") as f:
                pickle.dump(self.graph, f, pickle.HIGHEST_PROTOCOL)
            logger.info(f"Grafo guardado com sucesso em: {path}")
            logger.info(f"Resumo: {self.graph.number_of_nodes()} nós | {self.graph.number_of_edges()} arestas")
        except IOError as e:
            logger.error(f"Erro ao guardar grafo: {e}")

    def run(self) -> None:
        """Executa o pipeline completo de construção."""
        logger.info("A iniciar construção do grafo multimodal...")
        
        # 1. Carregar Nós
        self.load_stops("data/gtfs/mdp/stops.txt", modo="metro")
        self.load_stops("data/gtfs/stcp/stops.txt", modo="bus")
        
        # 2. Carregar Arestas GTFS
        self.load_gtfs_edges("data/gtfs/mdp/stop_times.txt", modo="metro", co2_per_km=Config.CO2_METRO)
        self.load_gtfs_edges("data/gtfs/stcp/stop_times.txt", modo="bus", co2_per_km=Config.CO2_BUS)
        
        # 3. Gerar Transbordos
        self.add_walk_edges()
        
        # 4. Guardar
        self.save_graph("data/output/graph_base.gpickle")


if __name__ == "__main__":
    builder = TransportGraphBuilder()
    builder.run()