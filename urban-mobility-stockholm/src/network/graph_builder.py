"""
Transport network graph builder.
Build a weighted directed graph from transit stop connections using NetworkX.
"""
import pandas as pd
import networkx as nx
from typing import Optional


def build_transit_graph(
    stops: pd.DataFrame,
    stop_times: pd.DataFrame,
    trips: Optional[pd.DataFrame] = None,
) -> nx.DiGraph:
    """
    Build a weighted directed graph from GTFS stop_times.
    Nodes = stops, Edges = consecutive stops on the same trip weighted by stop sequence distance.
    """
    G = nx.DiGraph()

    # Add nodes
    for _, row in stops.iterrows():
        G.add_node(
            str(row["stop_id"]),
            name=row.get("stop_name", ""),
            lat=float(row.get("stop_lat", 0)),
            lon=float(row.get("stop_lon", 0)),
        )

    # Add edges from consecutive stops in the same trip
    sorted_times = stop_times.sort_values(["trip_id", "stop_sequence"])
    for trip_id, group in sorted_times.groupby("trip_id"):
        stop_ids = group["stop_id"].astype(str).tolist()
        for i in range(len(stop_ids) - 1):
            src = stop_ids[i]
            dst = stop_ids[i + 1]
            # Default weight is 1 minute; could be replaced with actual travel time
            weight = 1.0
            if G.has_edge(src, dst):
                # Keep the minimum weight (fastest connection)
                G[src][dst]["weight"] = min(G[src][dst]["weight"], weight)
            else:
                G.add_edge(src, dst, weight=weight, trip_id=str(trip_id))

    return G


def compute_shortest_path(
    G: nx.DiGraph,
    source: str,
    target: str,
    weight: str = "weight",
) -> tuple[list[str], float]:
    """
    Compute the shortest path between two stops using Dijkstra's algorithm.
    Returns (path, total_weight).
    """
    try:
        path = nx.dijkstra_path(G, source, target, weight=weight)
        length = nx.dijkstra_path_length(G, source, target, weight=weight)
        return path, length
    except nx.NetworkXNoPath:
        return [], float("inf")
    except nx.NodeNotFound as e:
        raise ValueError(f"Stop not found in graph: {e}")


def get_network_stats(G: nx.DiGraph) -> dict:
    """Return basic statistics about the transit graph."""
    return {
        "num_stops": G.number_of_nodes(),
        "num_connections": G.number_of_edges(),
        "is_weakly_connected": nx.is_weakly_connected(G) if G.number_of_nodes() > 0 else False,
        "average_degree": (
            sum(d for _, d in G.degree()) / G.number_of_nodes()
            if G.number_of_nodes() > 0
            else 0.0
        ),
    }
