import sys
import numpy as np
import networkx as nx
from collections import defaultdict, Counter
import itertools
import math

def read_graphs(filename):
    graphs = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        if lines[i].startswith('#'):
            i += 1
            vertices = []
            edges = []
            
            while i < len(lines) and lines[i].startswith('v'):
                parts = lines[i].strip().split()
                vertices.append((int(parts[1]), int(parts[2])))
                i += 1
            
            while i < len(lines) and lines[i].startswith('e'):
                parts = lines[i].strip().split()
                edges.append((int(parts[1]), int(parts[2]), int(parts[3])))
                i += 1
            
            G = nx.Graph()
            for node_id, label in vertices:
                G.add_node(node_id, label=label)
            for u, v, label in edges:
                G.add_edge(u, v, label=label)
            
            graphs.append(G)
        else:
            i += 1
    
    return graphs

def extract_fragments(graph, max_nodes=4):
    fragments = []
    all_nodes = list(graph.nodes())
    
    for size in range(1, max_nodes + 1):
        for nodes in itertools.combinations(all_nodes, size):
            subg = graph.subgraph(nodes)
            if nx.is_connected(subg):
                frag_info = extract_fragment_info(subg)
                fragments.append(frag_info)
    
    return fragments

def extract_fragment_info(subgraph):
    node_labels = tuple(sorted(subgraph.nodes[n]['label'] for n in subgraph.nodes()))
    edge_labels = []
    
    for u, v in subgraph.edges():
        edge_labels.append(subgraph[u][v]['label'])
    edge_labels = tuple(sorted(edge_labels))
    
    num_nodes = subgraph.number_of_nodes()
    num_edges = subgraph.number_of_edges()
    
    return (node_labels, edge_labels, num_nodes, num_edges)

def select_discriminative_features(fragment_counts, total_graphs, k=50):
    fragment_list = list(fragment_counts.items())
    
    scores = []
    for frag, count in fragment_list:
        freq = count / total_graphs
        
        if freq > 0.95 or freq < 0.05:
            discriminative_score = abs(freq - 0.5) * 100
        else:
            discriminative_score = 0
        
        scores.append((frag, count, discriminative_score))
    
    scores.sort(key=lambda x: (-x[2], -x[1]))
    
    selected = scores[:k]
    return [item[0] for item in selected]

def identify_features_optimized(database_file, output_file):

    database_graphs = read_graphs(database_file)

    
    fragment_counter = Counter()
    
    for idx, graph in enumerate(database_graphs):
        if idx % 500 == 0:
            print(f"Processing graph {idx}/{len(database_graphs)}")
        
        fragments = extract_fragments(graph, max_nodes=3)
        unique_frags = set(fragments)
        
        for frag in unique_frags:
            fragment_counter[frag] += 1
    
    # print(f"Total unique fragments: {len(fragment_counter)}")
    
    selected_fragments = select_discriminative_features(fragment_counter, len(database_graphs), k=50)
    
    # print(f"Selected {len(selected_fragments)} discriminative fragments")
    
    with open(output_file, 'w') as f:
        f.write("# Discriminative fragments (format: node_labels|edge_labels|num_nodes|num_edges)\n")
        for idx, frag in enumerate(selected_fragments):
            node_labels, edge_labels, num_nodes, num_edges = frag
            f.write(f"{node_labels}|{edge_labels}|{num_nodes}|{num_edges}\n")
    


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(1)
    
    identify_features_optimized(sys.argv[1], sys.argv[2])