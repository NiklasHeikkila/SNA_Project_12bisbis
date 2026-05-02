import networkx as nx
import numpy as np
import random
import pandas as pd

# --- Load the dataset ---
# Download: https://snap.stanford.edu/data/facebook_combined.txt.gz
G = nx.read_edgelist("facebook_combined.txt.gz", nodetype=int)

# Convert to directed graph (so in/out degree centrality exist explicitly)
G = G.to_directed()

# --- Create samples A and B ---
nodes = list(G.nodes())
random.seed(42)

A_nodes = set(random.sample(nodes, 2000))
B_nodes = set(nodes) - A_nodes

# Induced subgraphs (keep only edges within each sample)
G_A = G.subgraph(A_nodes)
G_B = G.subgraph(B_nodes)

# --- Function to compute metrics ---
def compute_metrics(G_sub):
    in_deg = list(nx.in_degree_centrality(G_sub).values())
    out_deg = list(nx.out_degree_centrality(G_sub).values())
    closeness = list(nx.closeness_centrality(G_sub).values())
    betweenness = list(nx.betweenness_centrality(G_sub).values())

    return {
        "in_degree_mean": np.mean(in_deg),
        "in_degree_std": np.std(in_deg),
        "out_degree_mean": np.mean(out_deg),
        "out_degree_std": np.std(out_deg),
        "closeness_mean": np.mean(closeness),
        "closeness_std": np.std(closeness),
        "betweenness_mean": np.mean(betweenness),
        "betweenness_std": np.std(betweenness),
    }

# --- Compute for A and B ---
metrics_A = compute_metrics(G_A)
metrics_B = compute_metrics(G_B)

# --- Display as table ---
df = pd.DataFrame([metrics_A, metrics_B], index=["Sample A", "Sample B"])
print(df)