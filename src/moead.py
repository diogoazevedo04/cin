import random
from collections import defaultdict
from typing import List, Tuple, Dict

import networkx as nx
import numpy as np


class Solution:
    """Representa um caminho com os seus valores objetivos."""

    def __init__(self, path: List[str], time: float, co2: float):
        self.path = path
        self.time = time
        self.co2 = co2
        self.objectives = np.array([time, co2])

    def dominates(self, other: "Solution") -> bool:
        better = False
        for a, b in zip(self.objectives, other.objectives):
            if a > b:
                return False
            if a < b:
                better = True
        return better

    def __repr__(self) -> str:
        return f"Solution(time={self.time:.2f}, co2={self.co2:.2f}, len={len(self.path)})"


class MOEAD:
    """MOEA/D para otimização multiobjetivo de caminhos em grafos."""

    def __init__(
        self,
        G: nx.DiGraph,
        source: str,
        target: str,
        population_size: int = 100,
        n_neighbors: int = 20,
        max_generations: int = 50,
    ):
        self.G = G
        self.source = source
        self.target = target
        self.population_size = population_size
        self.n_neighbors = n_neighbors
        self.max_generations = max_generations

        self.weights = self._generate_weights()
        self.neighborhoods = self._generate_neighborhoods()

        self.population: List[Solution] = []
        self.pareto_front: List[Solution] = []
        self.history: List[Dict] = []

    def _generate_weights(self) -> np.ndarray:
        return np.array(
            [[i / (self.population_size - 1), 1 - i / (self.population_size - 1)]
             for i in range(self.population_size)]
        )

    def _generate_neighborhoods(self) -> List[List[int]]:
        neighborhoods = []
        for i in range(self.population_size):
            distances = [
                (np.linalg.norm(self.weights[i] - self.weights[j]), j)
                for j in range(self.population_size) if i != j
            ]
            distances.sort()
            neighborhoods.append([j for _, j in distances[:self.n_neighbors]])
        return neighborhoods

    def _evaluate_path(self, path: List[str]) -> Tuple[float, float]:
        time = 0.0
        co2 = 0.0
        for u, v in zip(path[:-1], path[1:]):
            edge = self.G[u][v]
            time += edge.get("time_min", 0.0)
            co2 += edge.get("co2", 0.0)
        return time, co2

    def _heuristic_initialization(self) -> List[Solution]:
        solutions = []

        times = [d["time_min"] for _, _, d in self.G.edges(data=True)]
        co2s = [d["co2"] for _, _, d in self.G.edges(data=True)]

        max_time = max(times) if times else 1.0
        max_co2 = max(co2s) if co2s else 1.0

        for w_time, w_co2 in self.weights:
            try:
                def cost(u, v, d):
                    return (
                        w_time * d.get("time_min", 0.0) / max_time +
                        w_co2 * d.get("co2", 0.0) / max_co2
                    )

                path = nx.shortest_path(
                    self.G, self.source, self.target, weight=cost
                )
                t, c = self._evaluate_path(path)
                solutions.append(Solution(path, t, c))
            except nx.NetworkXNoPath:
                continue

        while len(solutions) < self.population_size and solutions:
            base = random.choice(solutions)
            mutated = self._mutate(base.path)
            t, c = self._evaluate_path(mutated)
            solutions.append(Solution(mutated, t, c))

        return solutions[:self.population_size]

    def _mutate(self, path: List[str], rate: float = 0.3) -> List[str]:
        if len(path) < 3 or random.random() > rate:
            return path

        i = random.randint(0, len(path) - 2)
        j = random.randint(i + 1, len(path) - 1)

        try:
            weight = random.choice(["time_min", "co2"])
            sub = nx.shortest_path(self.G, path[i], path[j], weight=weight)
            new_path = path[:i] + sub + path[j + 1:]
            cleaned = [new_path[0]]
            for n in new_path[1:]:
                if n != cleaned[-1]:
                    cleaned.append(n)
            return cleaned
        except nx.NetworkXNoPath:
            return path

    def _crossover(self, p1: List[str], p2: List[str]) -> List[str]:
        common = list(set(p1) & set(p2))
        if len(common) < 2:
            return random.choice([p1, p2])

        n = random.choice(common)
        try:
            return p1[: p1.index(n)] + p2[p2.index(n):]
        except ValueError:
            return p1

    @staticmethod
    def _tchebycheff(obj: np.ndarray, w: np.ndarray, ref: np.ndarray) -> float:
        return np.max(w * np.abs(obj - ref))

    def _update_reference(self) -> np.ndarray:
        times = [s.time for s in self.population]
        co2s = [s.co2 for s in self.population]
        return np.array([min(times), min(co2s)])

    def _update_pareto(self, sol: Solution) -> None:
        self.pareto_front = [s for s in self.pareto_front if not sol.dominates(s)]
        if not any(s.dominates(sol) for s in self.pareto_front):
            self.pareto_front.append(sol)

    def optimize(self) -> List[Solution]:
        self.population = self._heuristic_initialization()

        for s in self.population:
            self._update_pareto(s)

        for gen in range(self.max_generations):
            ref = self._update_reference()

            for i in range(self.population_size):
                neighbors = self.neighborhoods[i]
                p1 = self.population[i]
                p2 = self.population[random.choice(neighbors)]

                child_path = (
                    self._crossover(p1.path, p2.path)
                    if random.random() < 0.8 else p1.path
                )
                child_path = self._mutate(child_path)

                t, c = self._evaluate_path(child_path)
                child = Solution(child_path, t, c)

                for j in neighbors:
                    if self._tchebycheff(
                        child.objectives, self.weights[j], ref
                    ) < self._tchebycheff(
                        self.population[j].objectives, self.weights[j], ref
                    ):
                        self.population[j] = child

                self._update_pareto(child)

            self.history.append({
                "generation": gen + 1,
                "pareto_size": len(self.pareto_front),
            })

        return self.pareto_front

    def get_extreme_solutions(self) -> Dict[str, Solution]:
        if not self.pareto_front:
            return {}

        best_time = min(self.pareto_front, key=lambda s: s.time)
        best_co2 = min(self.pareto_front, key=lambda s: s.co2)

        times = [s.time for s in self.pareto_front]
        co2s = [s.co2 for s in self.pareto_front]

        t_min, t_max = min(times), max(times)
        c_min, c_max = min(co2s), max(co2s)

        balanced = min(
            self.pareto_front,
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


def analyze_pareto_front(pareto: List[Solution], G: nx.DiGraph) -> None:
    times = [s.time for s in pareto]
    co2s = [s.co2 for s in pareto]

    mode_usage = defaultdict(int)
    for s in pareto:
        for u, v in zip(s.path[:-1], s.path[1:]):
            mode_usage[G[u][v]["modo"]] += 1

    print("Pareto solutions:", len(pareto))
    print("Time min / avg / max:", min(times), np.mean(times), max(times))
    print("CO2 min / avg / max:", min(co2s), np.mean(co2s), max(co2s))
    print("Mode usage:", dict(mode_usage))
