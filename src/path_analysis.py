def analyze_path(G, path):
    """
    Analisa um caminho devolvido pelo Dijkstra.
    Retorna métricas agregadas e sequência de modos.
    """

    total_time = 0.0
    total_co2 = 0.0

    time_by_mode = {}
    modes_sequence = []

    transfers = 0
    last_mode = None

    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]

        edge = G[u][v]

        # como é DiGraph simples, há só uma aresta
        time = edge.get("time_min", 0.0)
        co2 = edge.get("co2", 0.0)
        mode = edge.get("modo", "unknown")

        total_time += time
        total_co2 += co2

        time_by_mode[mode] = time_by_mode.get(mode, 0.0) + time

        if last_mode is not None and mode != last_mode:
            transfers += 1

        if not modes_sequence or modes_sequence[-1] != mode:
            modes_sequence.append(mode)

        last_mode = mode

    return {
        "total_time": total_time,
        "total_co2": total_co2,
        "time_by_mode": time_by_mode,
        "transfers": transfers,
        "modes_sequence": modes_sequence
    }
