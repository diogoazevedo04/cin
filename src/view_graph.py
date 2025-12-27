import argparse
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx

def load_graph():
    """Carrega o grafo multimodal."""
    graph_path = Path("data/output/graph_base.gpickle")
    if not graph_path.exists():
        print(f"❌ Grafo não encontrado em: {graph_path}")
        print("Execute primeiro: python src/graph.py")
        return None
    
    with open(graph_path, "rb") as f:
        G = pickle.load(f)
    return G

def print_graph_stats(G):
    """Imprime estatísticas do grafo."""
    print("\n" + "="*60)
    print("ESTATÍSTICAS DO GRAFO MULTIMODAL")
    print("="*60)
    
    # Contagem por modo
    metro_nodes = [n for n, d in G.nodes(data=True) if d.get('modo') == 'metro']
    bus_nodes = [n for n, d in G.nodes(data=True) if d.get('modo') == 'bus']
    
    print(f"\n📍 Nós (Paragens):")
    print(f"   Metro: {len(metro_nodes)}")
    print(f"   Autocarro: {len(bus_nodes)}")
    print(f"   TOTAL: {G.number_of_nodes()}")
    
    # Contagem de arestas por modo
    edges_by_mode = {'metro': 0, 'bus': 0, 'walk': 0}
    for u, v, data in G.edges(data=True):
        modo = data.get('modo', 'unknown')
        edges_by_mode[modo] = edges_by_mode.get(modo, 0) + 1
    
    print(f"\n🔗 Arestas (Ligações):")
    print(f"   Metro: {edges_by_mode.get('metro', 0)}")
    print(f"   Autocarro: {edges_by_mode.get('bus', 0)}")
    print(f"   Pedonais: {edges_by_mode.get('walk', 0)}")
    print(f"   TOTAL: {G.number_of_edges()}")
    
    # Amostra de nós
    print(f"\n📋 Amostra de Nós:")
    for i, (node, data) in enumerate(list(G.nodes(data=True))[:5]):
        print(f"   {node}: {data.get('modo')} @ ({data.get('lat'):.4f}, {data.get('lon'):.4f})")
    
    # Amostra de arestas
    print(f"\n📋 Amostra de Arestas:")
    for i, (u, v, data) in enumerate(list(G.edges(data=True))[:5]):
        print(f"   {u} → {v}: {data.get('modo')} | "
              f"Tempo: {data.get('time_min', 0):.1f}min | "
              f"Dist: {data.get('distance_km', 0):.2f}km | "
              f"CO₂: {data.get('co2', 0):.1f}g")
    
    print("\n" + "="*60 + "\n")

def plot_graph_spatial(G, show=True):
    """Visualiza o grafo no espaço geográfico (lat/lon)."""
    print("🎨 A gerar visualização espacial do grafo...")
    
    # Separar nós por modo
    metro_nodes = [(n, d) for n, d in G.nodes(data=True) if d.get('modo') == 'metro']
    bus_nodes = [(n, d) for n, d in G.nodes(data=True) if d.get('modo') == 'bus']
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Plot Arestas (primeiro, para ficarem atrás dos nós)
    print("   A desenhar arestas...")
    edges_by_mode = {'metro': [], 'bus': [], 'walk': []}
    
    for u, v, data in G.edges(data=True):
        modo = data.get('modo', 'unknown')
        if u in G.nodes and v in G.nodes:
            u_data = G.nodes[u]
            v_data = G.nodes[v]
            
            x_coords = [u_data['lon'], v_data['lon']]
            y_coords = [u_data['lat'], v_data['lat']]
            
            if modo in edges_by_mode:
                edges_by_mode[modo].append((x_coords, y_coords))
    
    # Desenhar arestas por modo
    if edges_by_mode['metro']:
        for x, y in edges_by_mode['metro']:
            ax.plot(x, y, c='blue', linewidth=0.5, alpha=0.3, zorder=1)
        ax.plot([], [], c='blue', linewidth=1, alpha=0.3, label='Ligações Metro')
    
    if edges_by_mode['bus']:
        for x, y in edges_by_mode['bus']:
            ax.plot(x, y, c='red', linewidth=0.3, alpha=0.2, zorder=1)
        ax.plot([], [], c='red', linewidth=1, alpha=0.3, label='Ligações Autocarro')
    
    if edges_by_mode['walk']:
        for x, y in edges_by_mode['walk']:
            ax.plot(x, y, c='green', linewidth=0.5, alpha=0.15, zorder=1)
        ax.plot([], [], c='green', linewidth=1, alpha=0.3, label='Ligações Pedonais')
    
    print(f"   Desenhadas: {len(edges_by_mode['metro'])} Metro | "
          f"{len(edges_by_mode['bus'])} Bus | "
          f"{len(edges_by_mode['walk'])} Pedonais")
    
    # Plot Nós (por cima das arestas)
    if metro_nodes:
        metro_lons = [d['lon'] for n, d in metro_nodes]
        metro_lats = [d['lat'] for n, d in metro_nodes]
        ax.scatter(metro_lons, metro_lats, c='blue', s=30, alpha=0.8, 
                  label='Paragens Metro', marker='o', zorder=2, edgecolors='darkblue', linewidths=0.5)
    
    if bus_nodes:
        bus_lons = [d['lon'] for n, d in bus_nodes]
        bus_lats = [d['lat'] for n, d in bus_nodes]
        ax.scatter(bus_lons, bus_lats, c='red', s=15, alpha=0.6, 
                  label='Paragens Autocarro', marker='s', zorder=2, edgecolors='darkred', linewidths=0.3)
    
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.set_title('Rede de Transportes Multimodal - Grande Porto\n(Nós e Arestas)', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    output = Path("figures/graph_spatial.png")
    output.parent.mkdir(exist_ok=True)
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"✅ Visualização guardada em: {output}")
    if show:
        plt.show()
    else:
        plt.close()

def main():
    """Menu principal."""
    parser = argparse.ArgumentParser(description="Visualizar grafo multimodal")
    parser.add_argument("--no-show", action="store_true", help="Não abrir janela gráfica (apenas guardar PNG)")
    parser.add_argument("--auto", action="store_true", help="Gerar visualização sem pergunta interativa")
    args = parser.parse_args()

    G = load_graph()
    if G is None:
        return

    print_graph_stats(G)

    if args.auto:
        plot_graph_spatial(G, show=not args.no_show)
        return

    print("Deseja visualizar o grafo? (s/n): ", end="")
    try:
        choice = input().strip().lower()
        if choice == 's':
            plot_graph_spatial(G, show=not args.no_show)
    except:
        print("\nVisualizá-lo depois com: python src/view_graph.py --auto")

if __name__ == "__main__":
    main()
