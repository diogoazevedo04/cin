import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# =========================
# Configurações Globais
# =========================
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

RESULTS_PKL = Path("data/output/moead_results.pkl")
PARETO_CSV = Path("data/pareto_front.csv")


# =========================
# Utilitários
# =========================
def load_results():
    if not RESULTS_PKL.exists():
        raise FileNotFoundError(f"Resultados não encontrados: {RESULTS_PKL}")

    with open(RESULTS_PKL, "rb") as f:
        return pickle.load(f)


# =========================
# Gráfico 1 — Frente de Pareto
# =========================
def plot_pareto_front():
    if not PARETO_CSV.exists():
        raise FileNotFoundError(f"CSV da frente de Pareto não encontrado: {PARETO_CSV}")

    df = pd.read_csv(PARETO_CSV)

    plt.figure()
    plt.scatter(df["time_min"], df["co2_g"])
    plt.xlabel("Tempo de Viagem (min)")
    plt.ylabel("Emissões de CO₂ (g)")
    plt.title("Frente de Pareto Aproximada")
    plt.grid(True)

    output = FIGURES_DIR / "pareto_front.png"
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Frente de Pareto guardada em {output}")


# =========================
# Gráfico 2 — Evolução da Frente de Pareto
# =========================
def plot_pareto_size_over_generations(results):
    history = results.get("history", [])
    if not history:
        print("[WARN] Histórico vazio — gráfico de convergência ignorado")
        return

    generations = [h["generation"] for h in history]
    pareto_sizes = [h["pareto_size"] for h in history]

    plt.figure()
    plt.plot(generations, pareto_sizes)
    plt.xlabel("Geração")
    plt.ylabel("Tamanho da Frente de Pareto")
    plt.title("Evolução da Frente de Pareto ao Longo das Gerações")
    plt.grid(True)

    output = FIGURES_DIR / "pareto_convergence.png"
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Convergência da Frente de Pareto guardada em {output}")


# =========================
# Gráfico 3 — Distribuição dos Tempos
# =========================
def plot_time_distribution():
    if not PARETO_CSV.exists():
        return

    df = pd.read_csv(PARETO_CSV)

    plt.figure()
    plt.hist(df["time_min"], bins=12)
    plt.xlabel("Tempo de Viagem (min)")
    plt.ylabel("Número de Soluções")
    plt.title("Distribuição dos Tempos na Frente de Pareto")
    plt.grid(True)

    output = FIGURES_DIR / "time_distribution.png"
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Distribuição de tempos guardada em {output}")


# =========================
# Gráfico 4 — Compromisso Tempo vs CO₂ (com extremos)
# =========================
def plot_tradeoff_with_extremes(results):
    pareto = results["pareto_front"]
    extremes = results.get("extremes", {})

    times = [s.time for s in pareto]
    co2s = [s.co2 for s in pareto]

    plt.figure()
    plt.scatter(times, co2s, label="Soluções Pareto")

    if "best_time" in extremes:
        s = extremes["best_time"]
        plt.scatter(s.time, s.co2, marker="x", s=100, label="Melhor Tempo")

    if "best_co2" in extremes:
        s = extremes["best_co2"]
        plt.scatter(s.time, s.co2, marker="x", s=100, label="Melhor CO₂")

    if "balanced" in extremes:
        s = extremes["balanced"]
        plt.scatter(s.time, s.co2, marker="x", s=100, label="Equilibrada")

    plt.xlabel("Tempo de Viagem (min)")
    plt.ylabel("Emissões de CO₂ (g)")
    plt.title("Compromisso Tempo vs CO₂")
    plt.legend()
    plt.grid(True)

    output = FIGURES_DIR / "tradeoff_extremes.png"
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Gráfico de compromisso guardado em {output}")


# =========================
# Execução Principal
# =========================
def main():
    print("\n=== A gerar gráficos para o relatório ===\n")

    results = load_results()

    plot_pareto_front()
    plot_pareto_size_over_generations(results)
    plot_time_distribution()
    plot_tradeoff_with_extremes(results)

    print("\n=== Todos os gráficos foram gerados com sucesso ===\n")


if __name__ == "__main__":
    main()
