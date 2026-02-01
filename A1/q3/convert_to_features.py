import sys
import numpy as np
from collections import Counter

def read_graphs_fast(filename):
    graphs = []
    current_graph = {'vertices': [], 'edges': []}
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('#'):
                if current_graph['vertices'] or current_graph['edges']:
                    graphs.append(current_graph)
                current_graph = {'vertices': [], 'edges': []}
            elif line.startswith('v'):
                parts = line.split()
                current_graph['vertices'].append(int(parts[2]))
            elif line.startswith('e'):
                parts = line.split()
                current_graph['edges'].append(int(parts[3]))
    
    if current_graph['vertices'] or current_graph['edges']:
        graphs.append(current_graph)
    
    return graphs

def read_fragments_simple(filename):
    fragments = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if '|' in line:
                parts = line.split('|')
                if len(parts) >= 4:
                    node_part = parts[0]
                    edge_part = parts[1]
                    
                    if node_part:
                        node_labels = list(map(int, node_part.split(',')))
                    else:
                        node_labels = []
                    
                    if edge_part:
                        edge_labels = list(map(int, edge_part.split(',')))
                    else:
                        edge_labels = []
                    
                    num_nodes = int(parts[2])
                    num_edges = int(parts[3])
                    
                    fragments.append({
                        'node_labels': node_labels,
                        'edge_labels': edge_labels,
                        'num_nodes': num_nodes,
                        'num_edges': num_edges
                    })
    
    return fragments

def check_fragment_simple(graph_data, fragment):
    graph_nodes = sorted(graph_data['vertices'])
    graph_edges = sorted(graph_data['edges'])
    
    frag_nodes = sorted(fragment['node_labels'])
    frag_edges = sorted(fragment['edge_labels'])
    
    if len(frag_nodes) > len(graph_nodes):
        return False
    
    if len(frag_edges) > len(graph_edges):
        return False
    
    graph_node_counter = Counter(graph_nodes)
    graph_edge_counter = Counter(graph_edges)
    
    frag_node_counter = Counter(frag_nodes)
    frag_edge_counter = Counter(frag_edges)
    
    for label, count in frag_node_counter.items():
        if graph_node_counter.get(label, 0) < count:
            return False
    
    for label, count in frag_edge_counter.items():
        if graph_edge_counter.get(label, 0) < count:
            return False
    
    return True

def convert_to_features_simple(graphs_file, fragments_file, output_file):
    print(f"Reading graphs from {graphs_file}")
    graphs = read_graphs_fast(graphs_file)
    print(f"Read {len(graphs)} graphs")
    
    print(f"Reading fragments from {fragments_file}")
    fragments = read_fragments_simple(fragments_file)
    print(f"Read {len(fragments)} fragments")
    
    features = np.zeros((len(graphs), len(fragments)), dtype=int)
    
    for i, graph_data in enumerate(graphs):
        if i % 100 == 0:
            print(f"Processing graph {i}/{len(graphs)}")
        
        for j, fragment in enumerate(fragments):
            if check_fragment_simple(graph_data, fragment):
                features[i, j] = 1
    
    np.save(output_file, features, allow_pickle=False)
    print(f"Saved features to {output_file}.npy")
    print(f"Feature matrix shape: {features.shape}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 convert_to_features.py <graphs_file> <fragments_file> <output_file>")
        sys.exit(1)
    
    convert_to_features_simple(sys.argv[1], sys.argv[2], sys.argv[3])