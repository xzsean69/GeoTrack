"""
Route optimization engine.
Detect congested edges in a transport graph and propose alternative routes.
"""
import itertools
import pandas as pd
import networkx as nx
from typing import Optional


def compute_edge_loads(
    G: nx.DiGraph,
    demand_df: pd.DataFrame,
    zone_stop_map: dict[str, list[str]],
) -> nx.DiGraph:
    """
    Assign demand loads to graph edges.
    demand_df: DataFrame with columns zone_id, demand.
    zone_stop_map: maps zone_id to list of stop_ids in that zone.
    """
    G = G.copy()
    nx.set_edge_attributes(G, 0.0, "load")

    zone_demand = demand_df.set_index("zone_id")["demand"].to_dict()

    for zone_id, stop_ids in zone_stop_map.items():
        demand = zone_demand.get(zone_id, 0.0)
        for stop_id in stop_ids:
            for neighbor in G.successors(stop_id):
                G[stop_id][neighbor]["load"] = G[stop_id][neighbor].get("load", 0.0) + demand

    return G


def identify_congested_edges(
    G: nx.DiGraph,
    threshold: float = 500.0,
) -> list[tuple[str, str, dict]]:
    """Return edges where load exceeds the given threshold."""
    congested = []
    for u, v, data in G.edges(data=True):
        if data.get("load", 0.0) >= threshold:
            congested.append((u, v, data))
    return congested


def suggest_alternative_routes(
    G: nx.DiGraph,
    congested_edges: list[tuple[str, str, dict]],
    k: int = 2,
) -> dict[tuple[str, str], list[list[str]]]:
    """
    For each congested edge (u, v), suggest up to k alternative paths
    by temporarily removing the edge and finding k shortest paths.
    """
    alternatives: dict[tuple[str, str], list[list[str]]] = {}
    for u, v, _ in congested_edges:
        G_temp = G.copy()
        G_temp.remove_edge(u, v)
        try:
            paths = list(
                itertools.islice(nx.shortest_simple_paths(G_temp, u, v, weight="weight"), k)
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            paths = []
        alternatives[(u, v)] = paths
    return alternatives


def simulate_optimized_travel_time(
    G: nx.DiGraph,
    source: str,
    target: str,
    removed_edges: Optional[list[tuple[str, str]]] = None,
) -> float:
    """
    Simulate travel time on an optimized graph where congested edges are removed.
    Returns total weight of the shortest path, or inf if no path exists.
    """
    G_temp = G.copy()
    if removed_edges:
        for u, v in removed_edges:
            if G_temp.has_edge(u, v):
                G_temp.remove_edge(u, v)
    try:
        return nx.dijkstra_path_length(G_temp, source, target, weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return float("inf")


def compare_network_efficiency(
    G_original: nx.DiGraph,
    G_optimized: nx.DiGraph,
    sample_pairs: list[tuple[str, str]],
) -> pd.DataFrame:
    """
    Compare average travel time between original and optimized graphs
    for a set of source-target pairs.
    """
    records = []
    for src, tgt in sample_pairs:
        try:
            orig = nx.dijkstra_path_length(G_original, src, tgt, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            orig = float("inf")
        try:
            opt = nx.dijkstra_path_length(G_optimized, src, tgt, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            opt = float("inf")
        records.append({"source": src, "target": tgt, "original": orig, "optimized": opt})
    df = pd.DataFrame(records)
    df["improvement"] = df["original"] - df["optimized"]
    return df
