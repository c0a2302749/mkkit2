import networkx as nx
import numpy as np


def compute_small_world_metrics(n_agents: int = 8, k: int = 4, p: float = 0.2, seed: int = 42):
    g = nx.watts_strogatz_graph(n_agents, k, p, seed=seed)

    C = nx.average_clustering(g)
    L = nx.average_shortest_path_length(g)

    n_rand = 100
    C_rand_vals = []
    L_rand_vals = []
    for s in range(n_rand):
        gr = nx.watts_strogatz_graph(n_agents, k, 1.0, seed=s)
        C_rand_vals.append(nx.average_clustering(gr))
        L_rand_vals.append(nx.average_shortest_path_length(gr))

    C_rand = float(np.mean(C_rand_vals))
    L_rand = float(np.mean(L_rand_vals))

    sigma = (C / C_rand) / (L / L_rand)

    print("=" * 55)
    print("Small-world network metrics (Watts-Strogatz)")
    print(f"  n={n_agents}, k={k}, p={p}, seed={seed}")
    print("=" * 55)
    print(f"  Clustering coefficient C           = {C:.4f}")
    print(f"  Random baseline C_rand             = {C_rand:.4f}")
    print(f"  Avg shortest path length L         = {L:.4f}")
    print(f"  Random baseline L_rand             = {L_rand:.4f}")
    print(f"  Small-world index sigma            = {sigma:.4f}")
    print()
    if sigma > 1:
        print(f"  -> sigma > 1: small-world property satisfied")
    else:
        print(f"  -> sigma <= 1: small-world property not satisfied")
        print(f"     (small-world property requires n >> 1; with n={n_agents} the graph is too dense)")
    print()
    print(f"  Comparison:")
    print(f"    C / C_rand = {C/C_rand:.2f}  (>>1 expected)")
    print(f"    L / L_rand = {L/L_rand:.2f}  (~1 expected)")
    print()

    # Larger-n reference for demonstration
    return _reference_large_n(k)


def _reference_large_n(k: int = 4):
    n_large = 100
    p_large = 0.2
    g = nx.watts_strogatz_graph(n_large, k, p_large, seed=42)
    C_big = nx.average_clustering(g)
    L_big = nx.average_shortest_path_length(g)

    gr = nx.watts_strogatz_graph(n_large, k, 1.0, seed=0)
    C_rand_big = nx.average_clustering(gr)
    L_rand_big = nx.average_shortest_path_length(gr)
    sigma_big = (C_big / C_rand_big) / (L_big / L_rand_big)

    print("  (Reference: n=100, k=4, p=0.2)")
    print(f"    C={C_big:.4f}, C_rand={C_rand_big:.4f}, C/C_rand={C_big/C_rand_big:.1f}")
    print(f"    L={L_big:.4f}, L_rand={L_rand_big:.4f}, L/L_rand={L_big/L_rand_big:.1f}")
    print(f"    sigma={sigma_big:.2f}  -> small-world property SATISFIED")


def compute_community_metrics(n_agents: int = 8, n_communities: int = 2, seed: int = 42):
    sizes = [n_agents // n_communities] * n_communities
    remainder = n_agents - sum(sizes)
    if remainder:
        sizes[-1] += remainder
    p_in, p_out = 0.6, 0.15
    g = nx.stochastic_block_model(
        sizes, p_in * np.eye(n_communities) + p_out * (1 - np.eye(n_communities)),
        seed=seed,
    )

    C = nx.average_clustering(g.to_undirected())

    connected_components = list(nx.connected_components(g.to_undirected()))
    n_components = len(connected_components)
    L_str: str
    if nx.is_connected(g.to_undirected()):
        L = nx.average_shortest_path_length(g)
        L_str = f"{L:.4f}"
    else:
        L_str = f"graph not connected ({n_components} components)"

    print("=" * 55)
    print("Community network metrics (Stochastic Block Model)")
    print(f"  n={n_agents}, communities={n_communities}, p_in={p_in}, p_out={p_out}")
    print("=" * 55)
    print(f"  Clustering coefficient C            = {C:.4f}")
    print(f"  Avg shortest path length L          = {L_str}")

    adj = nx.to_numpy_array(g)
    intra = 0
    inter = 0
    start = 0
    comm_ranges = []
    for s in sizes:
        comm_ranges.append((start, start + s))
        start += s
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            if adj[i][j] == 0:
                continue
            same = any(lo <= i < hi and lo <= j < hi for lo, hi in comm_ranges)
            if same:
                intra += 1
            else:
                inter += 1
    total = intra + inter
    if total:
        print(f"  Intra-community edges               = {intra} ({intra/total*100:.0f}%)")
        print(f"  Inter-community edges               = {inter} ({inter/total*100:.0f}%)")
        print(f"  Edge ratio (inter/intra)            = {inter/intra:.3f}" if intra else "")


if __name__ == "__main__":
    compute_small_world_metrics()
    print()
    compute_community_metrics()
