"""Orbital cargo station modeled as a networkx graph.

17 modules (nodes, attr `capacity` = cargo mass limit), 20 corridors (edges,
attrs `weight` = distance/time, `airlock`, `cluttered`; never both flags on one
edge), 2 single-lane airlocks, 5 cluttered corridors. The graph is connected.
"""

import networkx as nx


def build_default_station() -> nx.Graph:
    """Return the default station graph (see module docstring)."""
    station = nx.Graph()

    # Core ring (the original 6-module station), including the first airlock
    # on docking_bay-module_a. Routes are computed by routing.find_path, not
    # hardcoded here.
    station.add_node("docking_bay", capacity=5000)
    station.add_node("module_a", capacity=2000)
    station.add_node("module_b", capacity=2000)
    station.add_node("module_c", capacity=1500)
    station.add_node("cargo_bay", capacity=8000)
    station.add_node("module_d", capacity=1500)

    station.add_edge("docking_bay", "module_a", weight=1, airlock=True, cluttered=False)
    station.add_edge("module_a", "module_b", weight=2, airlock=False, cluttered=False)
    station.add_edge("module_b", "module_c", weight=3, airlock=False, cluttered=True)
    station.add_edge("module_c", "cargo_bay", weight=2, airlock=False, cluttered=False)
    station.add_edge("cargo_bay", "module_d", weight=4, airlock=False, cluttered=True)
    station.add_edge("module_d", "docking_bay", weight=3, airlock=False, cluttered=False)

    # Expansion: additional modules and corridors branching off the core ring
    # (see the expansion note in the design spec, section 2).
    station.add_node("module_e", capacity=1800)
    station.add_node("module_f", capacity=1800)
    station.add_node("storage_bay", capacity=6000)
    station.add_node("lab_module", capacity=1200)
    station.add_node("hab_ring", capacity=2500)
    station.add_node("solar_array", capacity=800)
    station.add_node("comms_module", capacity=900)
    station.add_node("airlock_2_bay", capacity=1000)
    station.add_node("workshop", capacity=1400)
    station.add_node("greenhouse", capacity=1600)
    station.add_node("observation_deck", capacity=700)

    # Branch off module_c toward a secondary cluster (storage/lab/hab).
    station.add_edge("module_c", "module_e", weight=2, airlock=False, cluttered=False)
    station.add_edge("module_e", "storage_bay", weight=3, airlock=False, cluttered=True)
    station.add_edge("storage_bay", "lab_module", weight=2, airlock=False, cluttered=False)
    station.add_edge("lab_module", "hab_ring", weight=3, airlock=False, cluttered=False)
    station.add_edge("hab_ring", "module_e", weight=4, airlock=False, cluttered=False)  # cross-link

    # Branch off cargo_bay / module_d toward exterior modules.
    station.add_edge("cargo_bay", "solar_array", weight=5, airlock=False, cluttered=False)
    station.add_edge("solar_array", "comms_module", weight=2, airlock=False, cluttered=True)
    station.add_edge("comms_module", "observation_deck", weight=2, airlock=False, cluttered=False)
    station.add_edge("module_d", "module_f", weight=3, airlock=False, cluttered=False)
    station.add_edge("module_f", "workshop", weight=2, airlock=False, cluttered=True)
    station.add_edge("workshop", "greenhouse", weight=2, airlock=False, cluttered=False)
    station.add_edge("greenhouse", "hab_ring", weight=4, airlock=False, cluttered=False)  # cross-link

    # Second airlock: a secondary chokepoint far from the first, on the
    # exterior cluster. The default simulation scenarios cross it in opposite
    # directions.
    station.add_edge("observation_deck", "airlock_2_bay", weight=2, airlock=True, cluttered=False)
    station.add_edge("airlock_2_bay", "module_f", weight=3, airlock=False, cluttered=False)

    return station


if __name__ == "__main__":
    g = build_default_station()
    airlock_edges = [(u, v) for u, v, d in g.edges(data=True) if d.get("airlock")]
    cluttered_edges = [(u, v) for u, v, d in g.edges(data=True) if d.get("cluttered")]

    print(f"nodes: {g.number_of_nodes()}")
    print(f"edges: {g.number_of_edges()}")
    print(f"connected: {nx.is_connected(g)}")
    print(f"airlock edges: {airlock_edges}")
    print(f"cluttered edges: {cluttered_edges}")
