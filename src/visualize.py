import matplotlib.pyplot as plt
from graph import build_graph

G = build_graph()

metro_x, metro_y = [], []
bus_x, bus_y = [], []

for _, data in G.nodes(data=True):
    if data["modo"] == "metro":
        metro_x.append(data["lon"])
        metro_y.append(data["lat"])
    elif data["modo"] == "bus":
        bus_x.append(data["lon"])
        bus_y.append(data["lat"])

plt.figure(figsize=(8, 8))
plt.scatter(bus_x, bus_y, s=1, c="blue", label="Bus")
plt.scatter(metro_x, metro_y, s=5, c="red", label="Metro")
plt.legend()
plt.title("Paragens de Transporte Público – Grande Porto")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
