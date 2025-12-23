import pandas as pd
import matplotlib.pyplot as plt


def plot_metro_lines():
    stops = pd.read_csv("data/gtfs/mdp/stops.txt")
    stop_times = pd.read_csv("data/gtfs/mdp/stop_times.txt")
    trips = pd.read_csv("data/gtfs/mdp/trips.txt")

    # mapa stop_id -> (lat, lon)
    coords = {
        row["stop_id"]: (row["stop_lat"], row["stop_lon"])
        for _, row in stops.iterrows()
    }

    # trip_id -> route_id
    trip_to_route = dict(zip(trips["trip_id"], trips["route_id"]))

    # ordenar
    stop_times = stop_times.sort_values(["trip_id", "stop_sequence"])

    colors = {
        "A": "green",
        "B": "red",
        "C": "blue",
        "D": "orange",
        "E": "purple",
        "F": "brown"
    }

    plt.figure(figsize=(8, 8))

    for trip_id, group in stop_times.groupby("trip_id"):
        route = trip_to_route.get(trip_id)
        if route not in colors:
            continue

        stops_seq = group["stop_id"].tolist()

        for i in range(len(stops_seq) - 1):
            s1, s2 = stops_seq[i], stops_seq[i + 1]
            if s1 not in coords or s2 not in coords:
                continue

            y = [coords[s1][0], coords[s2][0]]
            x = [coords[s1][1], coords[s2][1]]

            plt.plot(x, y, color=colors[route], linewidth=1)

    plt.title("Linhas do Metro do Porto")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()


if __name__ == "__main__":
    plot_metro_lines()
