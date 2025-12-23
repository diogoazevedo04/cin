from multimodal_graph import build_multimodal_graph
from origin_destination import add_origin_destination
from shortest_path import shortest_path_time
from path_analysis import analyze_path

if __name__=="__main__":

    G, G_walk = build_multimodal_graph()

    add_origin_destination(
        G_walk,
        G,
        lat=41.1495,
        lon=-8.6108,
        node_id="origin"
    )

    add_origin_destination(
        G_walk,
        G,
        lat=41.1780,
        lon=-8.5950,
        node_id="destination"
    )

    source="origin"
    target="destination"



    path, total_time = shortest_path_time(G, source, target)

    analysis = analyze_path(G, path)

    print("\n--- Análise do percurso ---")
    print(f"Tempo total: {analysis['total_time']:.2f} min")
    print(f"CO₂ total: {analysis['total_co2']:.1f} g")
    print(f"Transbordos: {analysis['transfers']}")

    print("\nTempo por modo:")
    for mode, t in analysis["time_by_mode"].items():
        print(f"  {mode}: {t:.2f} min")

    print("\nSequência de modos:")
    print(" → ".join(analysis["modes_sequence"]))