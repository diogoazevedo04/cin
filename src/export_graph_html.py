import json
import pickle
from pathlib import Path


def load_graph(graph_path: Path):
    if not graph_path.exists():
        raise FileNotFoundError(f"Grafo não encontrado: {graph_path}")
    with open(graph_path, "rb") as f:
        return pickle.load(f)


def graph_to_data(G):
    nodes = []
    edges = []

    # Map numeric index for JS
    id_map = {}
    for idx, (n, d) in enumerate(G.nodes(data=True)):
        id_map[n] = idx
        nodes.append({
            "id": idx,
            "name": str(n),
            "lat": float(d.get("lat", 0.0)),
            "lon": float(d.get("lon", 0.0)),
            "modo": str(d.get("modo", "unknown")),
        })

    for u, v, data in G.edges(data=True):
        if u not in id_map or v not in id_map:
            continue
        edges.append({
            "source": id_map[u],
            "target": id_map[v],
            "from": str(u),
            "to": str(v),
            "modo": str(data.get("modo", "unknown")),
            "time_min": float(data.get("time_min", 0.0)),
            "distance_km": float(data.get("distance_km", 0.0)),
            "co2": float(data.get("co2", 0.0)),
        })

    # Center map around average
    if nodes:
        avg_lat = sum(n["lat"] for n in nodes) / len(nodes)
        avg_lon = sum(n["lon"] for n in nodes) / len(nodes)
    else:
        avg_lat, avg_lon = 41.15, -8.61

    return {"nodes": nodes, "edges": edges, "center": [avg_lat, avg_lon]}


def render_html(data, output_path: Path):
    html = f"""<!doctype html>
<html lang=\"pt\">
<head>
  <meta charset=\"utf-8\"> 
  <title>Rede Multimodal — Visualização Interativa</title>
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"> 
  <link rel=\"stylesheet\" href=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.css\" />
  <script src=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.js\"></script>
  <style>
    html, body {{ height: 100%; margin: 0; }}
    #map {{ height: calc(100vh - 70px); }}
    #toolbar {{ height: 70px; padding: 8px 12px; background: #f7f7f7; border-bottom: 1px solid #ddd; display: flex; align-items: center; gap: 16px; font-family: system-ui, sans-serif; }}
    .badge {{ padding: 4px 8px; border-radius: 6px; font-size: 12px; }}
    .metro {{ background: #dbeafe; color: #1e40af; }}
    .bus {{ background: #fee2e2; color: #991b1b; }}
    .walk {{ background: #dcfce7; color: #14532d; }}
  </style>
</head>
<body>
  <div id=\"toolbar\">
    <strong>Rede Multimodal — Grande Porto</strong>
    <span class=\"badge metro\">Metro</span>
    <span class=\"badge bus\">Autocarro</span>
    <span class=\"badge walk\">A pé</span>
    <label><input type=\"checkbox\" id=\"showMetro\" checked> Mostrar ligações Metro</label>
    <label><input type=\"checkbox\" id=\"showBus\" checked> Mostrar ligações Autocarro</label>
    <label><input type=\"checkbox\" id=\"showWalk\" checked> Mostrar ligações A pé</label>
  </div>
  <div id=\"map\"></div>

  <script>
    const data = {json.dumps(data)};

    const MODE_COLORS = {{
      metro: '#2563eb',
      bus:   '#dc2626',
      walk:  '#16a34a',
      unknown: '#64748b'
    }};

    const map = L.map('map').setView([data.center[0], data.center[1]], 12);
    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap contributors'
    }}).addTo(map);

    // Layer groups by mode for easy toggling
    const nodeLayer = L.layerGroup().addTo(map);
    const edgeLayers = {{ metro: L.layerGroup().addTo(map), bus: L.layerGroup().addTo(map), walk: L.layerGroup().addTo(map) }};

    // Draw nodes
    for (const n of data.nodes) {{
      const color = MODE_COLORS[n.modo] || MODE_COLORS.unknown;
      const marker = L.circleMarker([n.lat, n.lon], {{
        radius: n.modo === 'metro' ? 5 : (n.modo === 'bus' ? 4 : 3),
        color: color,
        fillColor: color,
        fillOpacity: 0.8,
        weight: 1
      }});
      marker.bindPopup(`<b>${{n.name}}</b><br/>Modo: ${{n.modo}}`);
      marker.addTo(nodeLayer);
    }}

    // Draw edges
    const formatEdgeLabel = (e) => {{
      const t = e.time_min.toFixed(1);
      const d = e.distance_km.toFixed(2);
      return `<b>${{e.from}}</b> → <b>${{e.to}}</b><br/>Modo: <b>${{e.modo}}</b><br/>Tempo: ${{t}} min<br/>Distância: ${{d}} km`;
    }};

    for (const e of data.edges) {{
      const a = data.nodes[e.source];
      const b = data.nodes[e.target];
      const color = MODE_COLORS[e.modo] || MODE_COLORS.unknown;
      const line = L.polyline([[a.lat, a.lon], [b.lat, b.lon]], {{
        color: color,
        weight: e.modo === 'walk' ? 2 : (e.modo === 'bus' ? 2.5 : 3),
        opacity: e.modo === 'walk' ? 0.35 : 0.6
      }});
      line.bindPopup(formatEdgeLabel(e));
      const group = edgeLayers[e.modo] || edgeLayers.walk;
      line.addTo(group);
    }}

    // Toggling
    const metroChk = document.getElementById('showMetro');
    const busChk   = document.getElementById('showBus');
    const walkChk  = document.getElementById('showWalk');

    const toggleLayer = (chk, layer) => {{
      chk.addEventListener('change', () => {{
        if (chk.checked) layer.addTo(map); else layer.remove();
      }});
    }};

    toggleLayer(metroChk, edgeLayers.metro);
    toggleLayer(busChk, edgeLayers.bus);
    toggleLayer(walkChk, edgeLayers.walk);
  </script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"[OK] HTML gerado em: {output_path}")


def main():
    graph_path = Path("data/output/graph_base.gpickle")
    output_path = Path("output/graph.html")
    G = load_graph(graph_path)
    data = graph_to_data(G)
    render_html(data, output_path)


if __name__ == "__main__":
    main()
